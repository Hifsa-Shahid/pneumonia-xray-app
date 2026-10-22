# Chest X-ray Pneumonia Screening

A deep-learning demo that estimates the probability of pneumonia from a frontal chest X-ray, using three
fine-tuned CNNs (ResNet-50, DenseNet-121, EfficientNet-B0) and a calibrated ensemble.

**Live demo:** 'https://pneumonia-xray-screening.streamlit.app'

> **Research project, not a medical device.** Do not use it to diagnose, treat or make decisions about any patient.

## What is in this project

- **Data:** Kaggle *Chest X-Ray Images (Pneumonia)* (paediatric, frontal X-rays).
- **Training:** ImageNet-pretrained ResNet-50, DenseNet-121 and EfficientNet-B0, fully fine-tuned with class-weighted loss,
  mixed precision and early stopping.
- **Leak-free split:** the original validation folder has only 16 images, so train/validation were rebuilt with a
  patient-grouped split (overlap checked). The official test split was used once, at the end.
- **Calibration and thresholds:** temperature scaling and decision thresholds were fitted on validation only.
- **Uncertainty:** 95% confidence intervals from a patient-level bootstrap, plus McNemar tests between models.
- **Explainability (notebook only):** Score-CAM implemented from scratch, with an empty-map diagnostic and
  faithfulness metrics. The web app shows the screening result only.
- **Shortcut check:** test AUC when only the image centre, or only the outer frame, is visible.

## Results (official test split, 624 images)

Operating points were chosen on validation. Brackets are 95% patient-level bootstrap confidence intervals.

| Model | Threshold | AUC | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|---|
| ResNet-50 | 0.081 | 0.973 [0.958, 0.985] | 0.997 [0.991, 1.000] | 0.611 [0.549, 0.675] | 0.853 [0.821, 0.884] |
| DenseNet-121 | 0.554 | 0.975 [0.959, 0.987] | 0.997 [0.992, 1.000] | 0.611 [0.548, 0.675] | 0.853 [0.820, 0.883] |
| EfficientNet-B0 | 0.161 | 0.977 [0.964, 0.987] | 0.997 [0.991, 1.000] | 0.590 [0.527, 0.650] | 0.845 [0.810, 0.874] |
| Ensemble | 0.415 | 0.979 [0.966, 0.989] | 0.997 [0.991, 1.000] | 0.650 [0.586, 0.709] | 0.867 [0.834, 0.896] |

The AUC intervals overlap almost completely, so the models cannot be ranked from this test set. With a default threshold of 0.5
on uncalibrated probabilities, sensitivity was about 99.5% and specificity 61–72%.

**Shortcut check (test AUC):**

| Model | Full image | Centre only | Frame only |
|---|---|---|---|
| ResNet-50 | 0.973 | 0.895 | 0.924 |
| DenseNet-121 | 0.975 | 0.958 | 0.929 |
| EfficientNet-B0 | 0.977 | 0.958 | 0.835 |

## Limitations

- One public paediatric dataset, no external validation.
- Specificity of roughly 60–70%: many healthy X-rays are flagged. A threshold chosen on validation did not transfer well to
  the test split, which points to a distribution shift between the two.
- Regions outside the central chest carry enough signal to classify, so the models may partly use non-anatomical cues. The frame
  still includes outer lung edges, so this is an indication, not proof.
- Score-CAM maps are normalised per image (relative importance, not absolute evidence), and the method cannot explain
  "normal" predictions with a zero-image baseline.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Optional tests: `pip install pytest && pytest -q`

## Model files

The app needs these in `models/` (see `models/README.md`):

```
models/resnet50.pt   models/densenet121.pt   models/efficientnet_b0.pt   models/app_config.json
```

The fp32 ResNet-50 checkpoint is about 94 MB, close to GitHub's 100 MB file limit. `scripts/shrink_checkpoints.py`
stores the weights as float16 (about half the size, loaded back as float32), so no Git LFS is needed.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (public).
2. Open [share.streamlit.io](https://share.streamlit.io), choose **New app**, select the repo, branch `main`, main file `app.py`.
3. Under **Advanced settings**, choose Python 3.12 (or 3.11), then **Deploy**.

`requirements.txt` installs the CPU build of PyTorch, which is much smaller than the default CUDA build.

## Repository layout

```
app.py                     Streamlit UI
src/inference.py           model loading and CPU inference
src/logic.py               verdict logic and image checks (unit-tested)
models/                    checkpoints + app_config.json
samples/normal|pneumonia/  example X-rays for the in-app "Try a sample" tab
scripts/                   checkpoint shrinking helper
tests/                     unit tests
notebooks/                 training + Score-CAM notebook (add your .ipynb here)
```

## Data and credits

Chest X-ray images: Kermany et al., *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, Cell, 2018
(Kaggle mirror: `paultimothymooney/chest-xray-pneumonia`). Check the dataset page for its license terms before redistributing images.
Sample images in `samples/` were drawn at random from the test split.

## Disclaimer

For education and research only. No clinical validation or regulatory clearance.
