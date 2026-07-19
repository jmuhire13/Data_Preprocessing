"""
Task 5: System Demonstration - CLI simulation of the full authentication +
product recommendation flow.

Flow (matches the assignment's system diagram):
    face image -> Facial Recognition Model
        fail -> Access Denied
        pass -> run Product Recommendation Model (result held, not shown yet)
                -> voice sample -> Voiceprint Verification Model
                    fail -> Access Denied
                    pass -> display the predicted product

Usage:
    python run_demo.py --face <path_to_image> --voice <path_to_audio>

Examples:
    python run_demo.py --face ../image_data/M1_1.jpeg --voice ../audio_data/M1_approve.mp4
    python run_demo.py --face ../unauthorized_data/unauthorized_face.jpeg --voice ../audio_data/M1_approve.mp4
"""
import argparse
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extraction import image_path_to_features, audio_path_to_features

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'processed')

CONFIDENCE_THRESHOLD = 0.70

# Arbitrary member -> customer_id mapping for demo purposes only (see Task 5
# discussion: the transaction dataset has no real connection to the group's
# identities, so each member stands in for one real customer record).
MEMBER_CUSTOMER_MAP = {
    'M1': 161,
    'M2': 187,
    'M3': 159,
    'M4': 114,
}


def load_models():
    return {
        'face_model': joblib.load(os.path.join(MODELS_DIR, 'face_model.joblib')),
        'face_cols': joblib.load(os.path.join(MODELS_DIR, 'face_feature_columns.joblib')),
        'voice_model': joblib.load(os.path.join(MODELS_DIR, 'voice_model.joblib')),
        'voice_cols': joblib.load(os.path.join(MODELS_DIR, 'voice_feature_columns.joblib')),
        'product_model': joblib.load(os.path.join(MODELS_DIR, 'product_model.joblib')),
        'product_cols': joblib.load(os.path.join(MODELS_DIR, 'product_feature_columns.joblib')),
    }


def recognize_face(face_path, models):
    features = image_path_to_features(face_path)
    X = pd.DataFrame([features])[models['face_cols']]
    proba = models['face_model'].predict_proba(X)[0]
    classes = models['face_model'].classes_
    best_idx = proba.argmax()
    return classes[best_idx], proba[best_idx]


def verify_voice(voice_path, models):
    features = audio_path_to_features(voice_path)
    X = pd.DataFrame([features])[models['voice_cols']]
    proba = models['voice_model'].predict_proba(X)[0]
    classes = models['voice_model'].classes_
    best_idx = proba.argmax()
    return classes[best_idx], proba[best_idx]


def run_product_recommendation(member, models):
    """Builds a feature row for the member's mapped customer (their most recent
    transaction) and predicts a product category."""
    customer_id = MEMBER_CUSTOMER_MAP[member]
    merged = pd.read_csv(os.path.join(DATA_DIR, 'merged_customer_data.csv'))
    merged['purchase_date'] = pd.to_datetime(merged['purchase_date'])

    customer_rows = merged[merged['customer_id'] == customer_id]
    if customer_rows.empty:
        raise RuntimeError(f'No transaction data found for customer_id={customer_id}')
    latest = customer_rows.sort_values('purchase_date').iloc[[-1]]

    drop_cols = ['customer_id', 'transaction_id', 'purchase_date', 'customer_rating', 'product_category']
    X_row = latest.drop(columns=drop_cols)
    X_row = pd.get_dummies(X_row, columns=['review_sentiment', 'primary_platform'])
    X_row = X_row.reindex(columns=models['product_cols'], fill_value=0)

    prediction = models['product_model'].predict(X_row)[0]
    return customer_id, prediction


def main():
    parser = argparse.ArgumentParser(description='Identity & Product Recommendation System demo')
    parser.add_argument('--face', required=True, help='Path to a face image')
    parser.add_argument('--voice', required=True, help='Path to a voice recording')
    args = parser.parse_args()

    print('=== User Identity & Product Recommendation System ===')
    models = load_models()

    # --- Step 1: Facial Recognition ---
    print('\n[1/3] Facial Recognition...')
    try:
        face_member, face_conf = recognize_face(args.face, models)
    except Exception as e:
        print(f'  ERROR during face processing: {e}')
        print('\nACCESS DENIED: could not process face image.')
        sys.exit(1)

    print(f'  Closest match: {face_member} (confidence {face_conf:.3f}, threshold {CONFIDENCE_THRESHOLD})')
    if face_conf < CONFIDENCE_THRESHOLD:
        print('\nACCESS DENIED: face not recognized as an authorized user.')
        sys.exit(1)
    print(f'  -> Face recognized as {face_member}.')

    # --- Step 2: Run Product Recommendation Model (result held until voice passes) ---
    customer_id, predicted_product = run_product_recommendation(face_member, models)

    # --- Step 3: Voice Verification ---
    print('\n[2/3] Voice Verification...')
    try:
        voice_member, voice_conf = verify_voice(args.voice, models)
    except Exception as e:
        print(f'  ERROR during voice processing: {e}')
        print('\nACCESS DENIED: could not process voice sample.')
        sys.exit(1)

    print(f'  Closest match: {voice_member} (confidence {voice_conf:.3f}, threshold {CONFIDENCE_THRESHOLD})')
    if voice_conf < CONFIDENCE_THRESHOLD:
        print('\nACCESS DENIED: voice not verified as an authorized user.')
        sys.exit(1)
    if voice_member != face_member:
        print(f'  -> Voice identity ({voice_member}) does not match face identity ({face_member}).')
        print('\nACCESS DENIED: face and voice identities do not match.')
        sys.exit(1)
    print(f'  -> Voice verified as {voice_member}.')

    # --- Step 4: Display predicted product ---
    print('\n[3/3] Transaction approved.')
    print(f'  Member: {face_member} (customer_id={customer_id})')
    print(f'\nACCESS GRANTED. Predicted product recommendation: {predicted_product}')


if __name__ == '__main__':
    main()
