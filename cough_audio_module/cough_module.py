from pathlib import Path
import numpy as np, torch, librosa, joblib
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

SR, N, LAYER = 16000, 16000, 3
_dir = Path(__file__).parent
_fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
_enc = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").eval()
_clf = joblib.load(_dir / "cough_lr.joblib")

def _load(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    if len(y) >= N:
        s = (len(y) - N) // 2
        y = y[s:s + N]
    else:
        y = np.pad(y, (0, N - len(y)))
    return y.astype(np.float32)

def predict_cough(paths):
    """paths: ek mareez ke cough wav files ki list. Return: TB score (0 se 1)."""
    waves = [_load(p) for p in paths]
    inp = _fe(waves, sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        hs = _enc(inp.input_values, output_hidden_states=True).hidden_states[LAYER]
    return float(_clf.predict_proba(hs.mean(1).numpy())[:, 1].mean())
