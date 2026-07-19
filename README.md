# User Identity & Product Recommendation System

A multimodal ML pipeline that authenticates a user via facial recognition and voice
verification before displaying a personalized product recommendation.

## Flow

```text
face image -> Facial Recognition Model
    fail -> Access Denied
    pass -> Product Recommendation Model (result held)
            -> voice sample -> Voiceprint Verification Model
                fail -> Access Denied
                pass -> display predicted product
```

## Project structure

```text
data/
  raw/                       Original provided datasets, untouched
    customer_transactions.csv
    customer_social_profiles.csv
  processed/                 Engineered outputs produced by the notebooks
    merged_customer_data.csv
    image_features.csv
    audio_features.csv
image_data/                4 members x 3 face photos (neutral/smiling/surprised)
audio_data/                 4 members x 2 voice clips ("approve" / "confirm")
unauthorized_data/          Impostor face photo + voice clip (for Task 5's rejection test)
notebooks/
  01_data_merge.ipynb        EDA, merges transactions + social profiles -> data/processed/merged_customer_data.csv
  02_image_processing.ipynb  Face crop, augmentation, feature extraction -> data/processed/image_features.csv
  03_audio_processing.ipynb  Waveform/spectrogram (plotted and interpreted), augmentation, feature extraction -> data/processed/audio_features.csv
  04_model_creation.ipynb    Trains and evaluates all three models, explains and demonstrates the multimodal decision logic
models/                     Saved models (joblib) + their feature column lists
demo/
  feature_extraction.py      Shared preprocessing, reused by the CLI demo
  run_demo.py                CLI: full authentication + recommendation simulation
requirements.txt
report_corrections.md       Fixes for a few numbers in the written project report, checked against the actual data
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the notebooks

Run in order (01 -> 04); each one reads the previous notebook's output from `data/`:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_merge.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_image_processing.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_audio_processing.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_model_creation.ipynb
```

## Running the demo

```bash
cd demo
python run_demo.py --face PATH_TO_FACE_IMAGE --voice PATH_TO_VOICE_RECORDING
```

Replace both paths with real files, for example:

```bash
python run_demo.py --face ../image_data/M1_1.jpeg --voice ../audio_data/M1_approve.mp4
python run_demo.py --face ../unauthorized_data/unauthorized_face.jpeg --voice ../audio_data/M1_approve.mp4
```

Members are matched against a confidence threshold of 0.70 on both face and voice, and
the two identities must agree with each other before a recommendation is shown.

For a successful run, use one of the files already in `image_data/` or `audio_data/`,
not a freshly recorded photo or clip - see Known caveats for why.

## Model results

| Model                   | Method                                                    | Result                             |
| ----------------------- | --------------------------------------------------------- | ---------------------------------- |
| Facial Recognition      | `StratifiedGroupKFold` (k=3, grouped by source photo)     | 97.9% avg accuracy, 0.979 macro F1 |
| Voiceprint Verification | `StratifiedGroupKFold` (k=2, grouped by source recording) | 84.4% avg accuracy, 0.824 macro F1 |
| Product Recommendation  | Stratified train/test split                               | 16.7% test accuracy                |

**The Product Recommendation Model's low accuracy is a verified negative result, not a
bug.** It was checked against dummy baselines (which scored 23-30%, higher than the
trained model), feature importances (evenly spread, no dominant/leaking feature), and a
permutation test on the two features with the strongest nominal association
(p=0.105, indistinguishable from chance). The underlying data has no statistically
detectable relationship between the available customer/social features and the product
category actually purchased.

## Known caveats

- `primary_platform` / `review_sentiment` in the merged dataset use an alphabetical
  tie-break for customers with equally-frequent categories (~30% and ~18% of customers
  respectively) - documented in `01_data_merge.ipynb`.
- `customer_rating` is kept in `merged_customer_data.csv` for reference but excluded from
  the Product Recommendation Model's features, since it's a post-purchase value that
  wouldn't exist yet at recommendation time.
- The confidence threshold (0.70) cleanly separates the impostor samples from members'
  own submitted files, but does **not** generalize well to brand-new samples (verified
  with held-out cross-validation): ~23% of correctly-identified held-out face samples,
  and ~96% of correctly-identified held-out voice samples, would be wrongly denied at
  0.70. For voice, no threshold fixes this - the unauthorized sample scores 0.565, right
  inside the range where genuine new recordings also score, so any threshold strict
  enough to reject impostors also rejects most new legitimate ones. This comes from
  having only 2 training recordings per person, not a tunable parameter, which is why
  demo runs should use the originally submitted files (see above).
- The member -> `customer_id` mapping used by the demo (M1-M4 to specific transaction
  customers) is arbitrary, for demonstration purposes only.
