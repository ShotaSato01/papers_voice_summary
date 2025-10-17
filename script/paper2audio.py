# file: pdf2voice_free_localtts.py  （VOICEVOX ENGINE + Gemini Files API 対応・フル版）
import os
import re
import json
import time
import random
import argparse
import unicodedata
import tempfile
import datetime as dt
from pathlib import Path

import requests
from slugify import slugify
from pydub import AudioSegment   
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai

# =========================
# 設定
# =========================
load_dotenv()

PROMPT = """あなたは研究者向けのナレーターです。
与えられた論文について、その分野を専門とする大学院生向けの日本語の口語解説の原稿を作成してください。
原稿の構成は以下のとおりです。
1) 問題設定 (その論文の新規性・有用性など)
2) 手法の詳細 (どのような技術を用いているか)
3) 手法の有効性の検証方法 (タスクの設定、評価指標、実験結果)
4) 結論 (ここまでのまとめ)

[注意事項]
・全文を読み上げた時とき、5~10分になるような分量にすること
・日本語の'音'をそのまま文字に起こしたもの作成すること
・英語の場合は「読み」をひらがなで記載すること
"""

# =========================
# ユーティリティ（プレイリスト）
# =========================
def write_m3u(tracks, out_path: Path):
    lines = ["#EXTM3U"]
    for rel, dur, title in tracks:
        lines.append(f"#EXTINF:{int(round(dur))},{title}")
        lines.append(rel)
    out_path.write_text("\n".join(lines), encoding="utf-8")

def write_podcast_rss(tracks, out_path: Path, title: str, desc: str):
    now = dt.datetime.utcnow()
    items = []
    for rel, dur, t, pubdate in tracks:
        guid = slugify(t) + "-" + str(abs(hash(rel)))
        items.append(f"""
        <item>
          <title>{t}</title>
          <guid isPermaLink="false">{guid}</guid>
          <pubDate>{pubdate}</pubDate>
          <enclosure url="{Path(rel).resolve().as_uri()}" type="audio/mpeg" />
        </item>
        """.strip())
    rss = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<title>{title}</title><description>{desc}</description><language>ja</language>
<lastBuildDate>{now.strftime('%a, %d %b %Y %H:%M:%S +0000')}</lastBuildDate>
{''.join(items)}
</channel></rss>"""
    out_path.write_text(rss, encoding="utf-8")

# =========================
# テキスト前処理（500対策）
# =========================
MD_TOKEN_RE = re.compile(r"[#*_>\-\[\]\(\)`~|]| {2,}")
EMOJI_RE = re.compile("[\U00010000-\U0010FFFF]", flags=re.UNICODE)

def sanitize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = MD_TOKEN_RE.sub(" ", s)        # Markdown記号の除去
    s = EMOJI_RE.sub("", s)            # 絵文字などの除去
    s = re.sub(r"\s{2,}", " ", s)      # 連続空白の圧縮
    return s.strip()

def split_to_sentences(text: str, max_chars: int = 180) -> list[str]:
    # 句点や改行で一次分割
    parts = re.split(r"(?<=[。！？\?])\s*|\n+", text)
    parts = [p.strip() for p in parts if p.strip()]

    # 長い文は読点や空白でさらに分割
    chunks = []
    for p in parts:
        if len(p) <= max_chars:
            chunks.append(p)
            continue
        tmp = re.split(r"(?<=、)\s*|\s{1,}", p)
        buf = ""
        for t in tmp:
            if not t:
                continue
            if len(buf) + len(t) <= max_chars:
                buf += t
            else:
                if buf:
                    chunks.append(buf)
                buf = t
        if buf:
            chunks.append(buf)
    return [c for c in chunks if c.strip()]

# =========================
# VOICEVOX ENGINE 呼び出し
# =========================
def voicevox_query_and_synth(
    engine_url: str,
    speaker: int,
    text: str,
    params: dict,
    timeout: float = 30.0,
    retries: int = 2,
) -> bytes:
    """
    1チャンク分を audio_query → synthesis。
    500系が出たら軽くパラメータ（抑揚・速度）を弱めて再試行。
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            aq = requests.post(
                f"{engine_url.rstrip('/')}/audio_query",
                params={"text": text, "speaker": speaker},
                timeout=timeout,
            )
            if aq.status_code >= 500 and attempt < retries:
                # 形態素で落ちた可能性 → 少し待って再試行
                time.sleep(0.4 + random.random() * 0.4)
                # 抑揚をほんの少し弱める
                params = {
                    **params,
                    "intonationScale": max(0.8, params.get("intonationScale", 1.0) * 0.97),
                }
                continue
            aq.raise_for_status()
            query = aq.json()
            query.update(params)

            syn = requests.post(
                f"{engine_url.rstrip('/')}/synthesis",
                params={"speaker": speaker},
                data=json.dumps(query),
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            if syn.status_code == 200:
                return syn.content

            # 失敗時：パラメータを少し弱めて再試行
            if attempt < retries:
                params = {
                    **params,
                    "intonationScale": max(0.8, params.get("intonationScale", 1.0) * 0.95),
                    "speedScale": max(0.9, params.get("speedScale", 1.0) * 0.98),
                }
                time.sleep(0.4 + random.random() * 0.4)
                continue
            syn.raise_for_status()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.4 + random.random() * 0.4)
                continue
            raise
    if last_err:
        raise last_err
    raise RuntimeError("VOICEVOX synthesis failed unexpectedly.")

def synth_with_voicevox(script: str, engine_url: str, speaker: int, **kw) -> AudioSegment:
    clean = sanitize_text(script)
    max_chars = int(kw.pop("max_chars", 180))
    sentences = split_to_sentences(clean, max_chars=max_chars)
    if not sentences:
        raise RuntimeError("音声化できるテキストが空です（サニタイズ後に空になりました）")

    params = {
        "speedScale": kw.get("speed_scale", 1.0),
        "volumeScale": kw.get("volume_scale", 1.0),
        "pitchScale": kw.get("pitch_scale", 0.0),
        "intonationScale": kw.get("intonation_scale", 1.0),
        "prePhonemeLength": kw.get("pre_phoneme_length", 0.1),
        "postPhonemeLength": kw.get("post_phoneme_length", 0.5),
        # "enableInterrogativeUpspeak": True,  # 必要なら
    }

    pieces: list[AudioSegment] = []
    tmpdir = Path(tempfile.mkdtemp(prefix="vvx_"))

    for i, sent in enumerate(sentences, 1):
        raw = voicevox_query_and_synth(engine_url, speaker, sent, params, timeout=30.0, retries=2)
        wav_path = tmpdir / f"p{i:04d}.wav"
        wav_path.write_bytes(raw)
        pieces.append(AudioSegment.from_wav(wav_path))

    # 文間に少し無音を挟んで自然に
    gap = AudioSegment.silent(duration=200)
    out = pieces[0]
    for p in pieces[1:]:
        out += gap + p
    return out

# =========================
# メイン
# =========================
def main():
    ap = argparse.ArgumentParser(description="Gemini Files API（upload）+ VOICEVOX ENGINE でPDF→音声")
    ap.add_argument("--pdf_dir", default="../papers")
    ap.add_argument("-o","--out_dir", default="../voice_out")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--force", action="store_true")

    # VOICEVOX ENGINE
    ap.add_argument("--vvx_url", default="http://127.0.0.1:50021", help="VOICEVOX ENGINE のURL")
    ap.add_argument("--speaker_id", type=int, default=1, help="話者ID（/speakers で確認）")
    ap.add_argument("--speed", type=float, default=1.25)
    ap.add_argument("--volume", type=float, default=1.0)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--intonation", type=float, default=1.0)
    ap.add_argument("--pre_phoneme", type=float, default=0.1)
    ap.add_argument("--post_phoneme", type=float, default=0.5)
    ap.add_argument("--max_chars", type=int, default=180, help="1チャンクの最大文字数")

    ap.add_argument("--playlist", default="playlist.m3u8")
    ap.add_argument("--rss", default="podcast.xml")
    ap.add_argument("--title", default="論文解説プレイリスト（Files API版）")
    ap.add_argument("--desc", default="Gemini Files APIでPDF解析＋VOICEVOX音声合成")
    args = ap.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY が未設定（.envを確認）")

    client = genai.Client(api_key=api_key)

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError("PDFが見つかりません")

    # VOICEVOX ENGINE 起動確認
    try:
        r = requests.get(f"{args.vvx_url.rstrip('/')}/speakers", timeout=5)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"VOICEVOX ENGINE に接続できませんでした: {args.vvx_url}\n"
            "Docker やアプリが起動しているか、ポート設定を確認してください。"
        ) from e

    tracks_m3u, tracks_rss = [], []
    for pdf in tqdm(pdfs, desc="Processing PDFs"):
        print(f"\n=== {pdf.name} ===")
        title = pdf.stem
        slug = slugify(title) or slugify(pdf.name)
        mp3_path = out_dir / f"{slug}.mp3"

        if mp3_path.exists() and not args.force:
            print("  -> 既存MP3を使用")
            seg = AudioSegment.from_file(mp3_path, format="mp3")
        else:
            # ---- Files API: アップロード → 処理完了待ち ----
            print(f"  -> アップロード中: {pdf.name}")
            uploaded = client.files.upload(file=str(pdf))
            while getattr(uploaded, "state", None) and uploaded.state.name == "PROCESSING":
                time.sleep(1.5)
                uploaded = client.files.get(name=uploaded.name)

            # ---- Gemini でスクリプト生成 ----
            print("  -> 解析/要約生成")
            resp = client.models.generate_content(
                model=args.model,
                contents=[uploaded, PROMPT],
                config=genai.types.GenerateContentConfig(temperature=args.temperature),
            )
            script = (getattr(resp, "text", "") or "").strip()
            if not script:
                print("  -> 要約生成に失敗（空応答）。スキップ")
                continue

            # ---- VOICEVOX 合成 ----
            audio = synth_with_voicevox(
                script,
                engine_url=args.vvx_url,
                speaker=args.speaker_id,
                speed_scale=args.speed,
                volume_scale=args.volume,
                pitch_scale=args.pitch,
                intonation_scale=args.intonation,
                pre_phoneme_length=args.pre_phoneme,
                post_phoneme_length=args.post_phoneme,
                max_chars=args.max_chars,
            )
            audio.export(mp3_path, format="mp3")
            seg = audio

        rel = str(mp3_path.relative_to(out_dir))
        mtime = dt.datetime.utcfromtimestamp(mp3_path.stat().st_mtime)
        tracks_m3u.append((rel, seg.duration_seconds, title))
        tracks_rss.append((rel, seg.duration_seconds, title, mtime.strftime('%a, %d %b %Y %H:%M:%S +0000')))

    write_m3u(tracks_m3u, out_dir / args.playlist)
    write_podcast_rss(tracks_rss, out_dir / args.rss, args.title, args.desc)
    print(f"\n出力完了: {out_dir}\n- {args.playlist}\n- {args.rss}")

if __name__ == "__main__":
    main()
