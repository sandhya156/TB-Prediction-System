
import numpy as np, torch, librosa, joblib
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

SR, N, LAYER = 16000, 16000, 3
_fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
_enc = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").eval()
_clf = joblib.load("cough_lr.joblib")

def _load(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    if len(y) >= N:
        s = (len(y) - N) // 2          # center 1 s crop, same as training
        y = y[s:s + N]
    else:
        y = np.pad(y, (0, N - len(y)))
    return y.astype(np.float32)

def predict_cough(paths):
    """paths: list of cough wav files from ONE patient. Returns TB score in [0, 1]."""
    waves = [_load(p) for p in paths]
    inp = _fe(waves, sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        hs = _enc(inp.input_values, output_hidden_states=True).hidden_states[LAYER]
    emb = hs.mean(1).numpy()
    return float(_clf.predict_proba(emb)[:, 1].mean())   # mean over clips
