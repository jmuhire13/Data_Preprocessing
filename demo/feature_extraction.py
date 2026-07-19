"""
Shared feature-extraction logic for the Task 5 CLI demo.

Mirrors the preprocessing done in notebooks/02_image_processing.ipynb and
notebooks/03_audio_processing.ipynb exactly, so a brand-new face image or
voice recording is turned into the same feature representation the models
were trained on.
"""
import os
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
import librosa
import numpy as np

STANDARD_FACE_SIZE = (128, 128)
TARGET_SR = 22050

_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
_ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()


def detect_and_crop_face(img_bgr, margin_ratio=0.2):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    faces = _face_cascade.detectMultiScale(eq, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60))
    if len(faces) == 0:
        raise RuntimeError('No face detected in image.')

    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    mx, my = int(w * margin_ratio), int(h * margin_ratio)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(img_bgr.shape[1], x + w + mx), min(img_bgr.shape[0], y + h + my)
    crop = img_bgr[y0:y1, x0:x1]
    return cv2.resize(crop, STANDARD_FACE_SIZE)


def extract_image_features(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    embed = cv2.resize(gray, (32, 32)).flatten().astype(np.float32) / 255.0

    gray_hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    gray_hist = gray_hist / (gray_hist.sum() + 1e-8)

    color_hists = []
    for ch in range(3):
        h = cv2.calcHist([img_bgr], [ch], None, [32], [0, 256]).flatten()
        color_hists.append(h / (h.sum() + 1e-8))

    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges > 0)

    features = {}
    for i, v in enumerate(embed):
        features[f'embed_{i}'] = v
    for i, v in enumerate(gray_hist):
        features[f'gray_hist_{i}'] = v
    for ch_name, hist in zip(['b', 'g', 'r'], color_hists):
        for i, v in enumerate(hist):
            features[f'{ch_name}_hist_{i}'] = v
    features['mean_intensity'] = float(np.mean(gray))
    features['std_intensity'] = float(np.std(gray))
    features['edge_density'] = float(edge_density)
    return features


def image_path_to_features(path):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise RuntimeError(f'Could not read image: {path}')
    face_crop = detect_and_crop_face(img_bgr)
    return extract_image_features(face_crop)


def load_audio(path, target_sr=TARGET_SR):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.mp4', '.m4a', '.mov'):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.run(
            [_ffmpeg_exe, '-y', '-i', path, '-ar', str(target_sr), '-ac', '1', tmp_path],
            capture_output=True, check=True,
        )
        y, sr = librosa.load(tmp_path, sr=target_sr)
        os.remove(tmp_path)
        return y, sr
    return librosa.load(path, sr=target_sr)


def extract_audio_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    features = {}
    for i in range(mfcc.shape[0]):
        features[f'mfcc_{i}_mean'] = float(np.mean(mfcc[i]))
        features[f'mfcc_{i}_std'] = float(np.std(mfcc[i]))
    features['rolloff_mean'] = float(np.mean(rolloff))
    features['rolloff_std'] = float(np.std(rolloff))
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std'] = float(np.std(rms))
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_std'] = float(np.std(zcr))
    features['centroid_mean'] = float(np.mean(centroid))
    features['centroid_std'] = float(np.std(centroid))
    # NOTE: 'duration' is deliberately NOT included - the voice model was
    # trained without it (see notebooks/04_model_creation.ipynb).
    return features


def audio_path_to_features(path):
    y, sr = load_audio(path)
    return extract_audio_features(y, sr)
