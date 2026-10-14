# Model Card: 1D CNN for 12-Lead ECG Abnormality Detection

*Format follows Mitchell et al., "Model Cards for Model Reporting" (2019).*

## Model details

| | |
|---|---|
| Developer | Midhili Asok (independent portfolio project) |
| Model | 1D convolutional neural network: 5 conv blocks, 422,149 parameters; ensemble of 5 seeds |
| Inputs | 10-second, 12-lead ECG at 100 Hz, bandpass-filtered 0.5–40 Hz and standardized with PTB-XL training statistics |
| Outputs | Task A: probability that the ECG is abnormal. Task B: independent probabilities for NORM, MI, STTC, CD and HYP |
| Training | AdamW with cosine decay, early stopping on validation AUROC, and augmentation (amplitude scaling, noise, time shifts, lead dropout); Platt scaling on validation |
| Framework | TensorFlow 2 / Keras 3 |
| Comparators | 1D ResNet (3.2M parameters), XGBoost and logistic regression on 88 handcrafted features |

## Intended use

- **Intended:** research, education and demonstrating evaluation practice for clinical machine-learning models.
- **Not intended:** diagnosis, triage or any decision about a real patient. This is not a regulated medical device, and it has not been validated prospectively or reviewed by clinicians.

## Training and evaluation data

- **PTB-XL v1.0.3:** 21,388 ECGs with at least one diagnostic label, from about 18,900 patients, recorded in Germany between 1989 and 1996.
- **Splits:** official stratified folds. Training: folds 1–8 (17,084 ECGs). Validation: fold 9 (2,146). Test: fold 10 (2,158). No patient appears in more than one split.
- **Label definition:** "Normal" means NORM is the only diagnostic superclass; any other superclass counts as abnormal. Rhythm statements are not part of the label.
- **External test set:** CPSC 2018 (11 Chinese hospitals, 500 Hz, resampled to 100 Hz). Evaluation set: 4,349 ECGs, 918 normal and 3,431 with 1st-degree AV block, LBBB, RBBB, ST depression or ST elevation. 2,518 ECGs with rhythm diagnoses (AF, PAC, PVC) are excluded, because the label does not cover rhythm.

## Performance (PTB-XL test set)

| Metric | Value |
|---|---|
| Task A AUROC (95% CI) | 0.944 (0.934–0.952) |
| Sensitivity / specificity at validation-chosen threshold (0.462) | 0.819 / 0.918 |
| Task B macro AUROC (5-seed ensemble) | 0.933 |
| Brier score after Platt scaling | 0.094 (calibration slope 1.01) |

**Task B per-class AUROC (mean over seeds):** NORM 0.947 · MI 0.924 · STTC 0.937 · CD 0.923 · HYP 0.902

## External validation (CPSC 2018, Task A)

| Metric | PTB-XL, matched conditions | CPSC 2018 |
|---|---|---|
| AUROC (95% CI) | 0.978 (0.966–0.987) | 0.920 (0.911–0.928) |
| Sensitivity / specificity at PTB-XL threshold (0.462) | 0.900 / 0.927 | 0.760 / 0.891 |
| Best threshold on this dataset | 0.628 | 0.230 |

| Condition (CPSC 2018) | n | Detected at PTB-XL threshold |
|---|---|---|
| LBBB | 190 | 99.5% |
| RBBB | 1,576 | 93.2% |
| 1st-degree AV block | 713 | 74.3% |
| ST depression | 811 | 48.1% |
| ST elevation | 211 | 45.0% |

Task B outputs on CPSC: conduction-disturbance output AUROC 0.965; ST/T-change output AUROC 0.832.

## Robustness (PTB-XL test, Task A AUROC)

| Condition | AUROC |
|---|---|
| Clean (reference) | 0.943 |
| Gaussian noise, SNR 14 dB / 0 dB | 0.943 / 0.920 |
| Strong baseline wander, with / without bandpass filter | 0.927 / 0.692 |
| CNN retrained on 6 limb leads | 0.919 (0.908–0.930) |
| CNN retrained on leads I + II | 0.921 (0.910–0.932) |
| CNN retrained on lead I only | 0.880 (0.866–0.893); sensitivity 0.667 |

The 12-lead model must not be used with missing leads: blanking all but lead I drops AUROC to 0.682.

## Performance by subgroup (Task A AUROC)

| Group | n | AUROC |
|---|---|---|
| Male | 1,110 | 0.950 |
| Female | 1,048 | 0.937 |
| Age < 40 | 284 | **0.861** |
| Age 40–59 | 639 | 0.914 |
| Age 60–74 | 706 | 0.936 |
| Age 75+ | 529 | 0.954 |

- **The under-40 gap is significant** (XGBoost baseline: DeLong p = 1.6 × 10⁻⁵).
- **Sensitivity under 40 is 0.485,** compared with 0.838 for patients 40 and over.
- **Excluding ECGs labelled "normal plus a mild finding"** raises under-40 AUROC only to 0.882, so most of the gap is genuine.
- **The male–female difference is small,** and its confidence intervals overlap.
- **The age gap replicates externally:** on CPSC 2018, under-40 AUROC is 0.870 (n = 965) vs 0.918 for 40 and over.

## Interpretability

- **Heartbeat-aligned Grad-CAM:** attention concentrates on the QRS complex for conduction disturbances and hypertrophy, and on the ST segment and T wave for infarction and ST/T changes. This is consistent with clinical criteria.
- **Lead ablation:** no single lead is critical; the largest single-lead drop is 0.015 AUROC (V1, for conduction disturbances). Removing all chest leads costs up to 0.103 AUROC (MI).

## Known failure modes

- **Mild or non-specific findings are often missed:**
  - non-specific intraventricular conduction delay: 41% missed
  - incomplete right bundle branch block: 18% missed
  - non-diagnostic T-wave abnormalities: 18% missed
- **Clinically important misses:** anterior MI (29% missed) and inferior MI (19%).
- **ECGs labelled "normal plus a finding"** are missed 55% of the time.
- **False alarms** occur mostly in older patients labelled normal, some of whom have high QRS voltages.
- **Noise robustness:** performance is nearly unchanged on recordings flagged as noisy (AUROC 0.938 vs 0.944) and under strong synthetic noise (0.920 at SNR 0 dB).
- **ST changes at other hospitals:** fewer than half of ST depressions and elevations in CPSC 2018 are detected at the PTB-XL threshold.
- **Missing leads:** the 12-lead model fails when leads are missing (0.682 on lead I alone). Use a model retrained for the device's leads.

## Ethical considerations and caveats

- **Population:** the training data comes from one country and one era, and older patients predominate. External validation on Chinese hospital data shows good transfer for conduction disturbances but not for ST changes; other populations and devices remain untested.
- **Young patients:** missed abnormalities in young patients could delay care if the model were ever used for screening. A "normal" output for a patient under 40 should not be relied on.
- **Thresholds:** decision thresholds were tuned on PTB-XL. On CPSC 2018 the optimal threshold fell from 0.46 to 0.23, so any new setting needs its own threshold and calibration check.
