import requests, json, soundfile as sf, io

BASE = "http://127.0.0.1:50021"
text = "こんにちは。ローカルでVOICEVOXを使っています。"
speaker = 1  # 話者IDは /speakers で確認

# 1) audio_query
q = requests.post(f"{BASE}/audio_query", params={"text": text, "speaker": speaker})
query = q.json()

# 2) synthesis
audio = requests.post(f"{BASE}/synthesis", params={"speaker": speaker}, data=json.dumps(query))
sf.write("output.wav", sf.read(io.BytesIO(audio.content))[0], 24000)
