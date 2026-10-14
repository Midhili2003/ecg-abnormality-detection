"""Streamlit demo: 12-lead ECG abnormality detection with a 1D CNN ensemble and Grad-CAM explanations."""
import json
import pathlib

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from keras import layers
from scipy.signal import butter, filtfilt, resample_poly

ASSETS = pathlib.Path(__file__).parent / 'assets'
REPO_URL = 'https://github.com/Midhili2003/ecg-abnormality-detection'
CLASS_NAMES = {'NORM': 'Normal ECG', 'MI': 'Myocardial infarction', 'STTC': 'ST/T change',
               'CD': 'Conduction disturbance', 'HYP': 'Hypertrophy'}

st.set_page_config(page_title='ECG Abnormality Detection', page_icon='🫀', layout='wide')


# ---------- Model ----------

def conv_bn_relu(x, filters, kernel):
    x = layers.Conv1D(filters, kernel, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


def build_cnn(n_out, input_shape=(1000, 12)):
    """Same architecture as src/models.py: 5 conv blocks, global pooling, sigmoid outputs."""
    inp = keras.Input(input_shape)
    x = inp
    for filters in [32, 64, 128, 128, 256]:
        x = conv_bn_relu(x, filters, 7)
        x = layers.MaxPooling1D(2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    return keras.Model(inp, layers.Dense(n_out, activation='sigmoid')(x))


@st.cache_resource(show_spinner='Loading the 10 trained CNNs...')
def load_models():
    """Rebuild each CNN and load its saved weights; return Grad-CAM-ready models (block-4 features + output)."""
    cfg = json.loads((ASSETS / 'config.json').read_text())
    ensembles = {}
    for task, n_out in [('binary', 1), ('multi', len(cfg['classes']))]:
        ensembles[task] = []
        for s in range(5):
            m = build_cnn(n_out)
            w = np.load(ASSETS / f'cnn_{task}_s{s}.npz')
            m.set_weights([w[f'arr_{i}'] for i in range(len(w.files))])
            relu = [l for l in m.layers if isinstance(l, layers.ReLU)][3]          # conv block 4: 80 ms resolution
            ensembles[task].append(keras.Model(m.inputs, [relu.output, m.outputs[0]]))
    return cfg, ensembles


@st.cache_data
def load_examples():
    d = np.load(ASSETS / 'examples.npz', allow_pickle=False)
    info = pd.DataFrame({k: d[k] for k in ['ecg_id', 'category', 'superclass', 'y', 'age', 'sex']})
    info['label'] = [f"{c} · ECG {e} · {s}, {a}" for c, e, s, a in zip(info.category, info.ecg_id, info.sex, info.age)]
    return info, d['X'], d['Y']


def predict_and_explain(ensemble, x, out_idx):
    """Ensemble probabilities, plus Grad-CAM for one output averaged over the 5 seeds."""
    xt = tf.constant(x[None], tf.float32)
    probs, cams = [], []
    for gm in ensemble:
        with tf.GradientTape() as tape:
            feats, preds = gm(xt, training=False)
            score = preds[:, out_idx]
        grads = tape.gradient(score, feats)
        weights = tf.reduce_mean(grads, axis=1, keepdims=True)
        cams.append(tf.nn.relu(tf.reduce_sum(feats * weights, axis=-1)).numpy()[0])
        probs.append(preds.numpy()[0])
    cam = np.mean(cams, axis=0)
    cam = np.interp(np.arange(x.shape[0]), np.linspace(0, x.shape[0] - 1, cam.size), cam)
    return np.mean(probs, axis=0), cam / (cam.max() + 1e-8)


# ---------- Signal processing ----------

def bandpass(x, fs=100, low=0.5, high=40, order=3):
    b, a = butter(order, [low, high], btype='band', fs=fs)
    return filtfilt(b, a, x, axis=0)


def prepare_upload(df, fs, cfg):
    """Uploaded CSV (samples x 12 leads, millivolts) -> filtered, standardized (1000, 12) array, plus notes."""
    notes = []
    names = [str(c).strip().upper() for c in df.columns]
    leads = [l.upper() for l in cfg['leads']]
    if all(l in names for l in leads):
        x = df.iloc[:, [names.index(l) for l in leads]].to_numpy(float)
    elif df.shape[1] >= 12:
        x = df.iloc[:, :12].to_numpy(float)
        notes.append('Lead names not found in the header, so the first 12 columns were used in the order '
                     'I, II, III, aVR, aVL, aVF, V1–V6.')
    else:
        raise ValueError(f'The file has {df.shape[1]} columns; 12 leads are needed.')
    if np.isnan(x).any():
        raise ValueError('The file contains empty or non-numeric values.')
    if len(x) < cfg['seconds'] * fs:
        raise ValueError(f"The recording is {len(x) / fs:.1f} s long; at least {cfg['seconds']} s is needed.")
    x = x[:cfg['seconds'] * fs]
    if fs != cfg['fs']:
        x = resample_poly(x, cfg['fs'], fs, axis=0)
        notes.append(f"Resampled from {fs} Hz to {cfg['fs']} Hz.")
    x = bandpass(x, cfg['fs'])
    if np.median(x.std(axis=0)) > 20:
        x = x / 1000
        notes.append('Values looked like microvolts, so they were converted to millivolts.')
    mu, sd = np.array(cfg['mu']), np.array(cfg['sd'])
    return ((x - mu) / sd).astype(np.float32), notes


# ---------- Plot ----------

def plot_ecg(x_std, cam, cfg, title):
    """12 leads in two columns (limb | chest), in millivolts, with Grad-CAM attention shaded in red."""
    x = x_std * np.array(cfg['sd']) + np.array(cfg['mu'])
    t = np.arange(len(x)) / cfg['fs']
    fig, axes = plt.subplots(6, 2, figsize=(14, 8.5), sharex=True)
    for j, lead in enumerate(cfg['leads']):
        ax = axes[j % 6, j // 6]
        lo, hi = np.percentile(x[:, j], [0.5, 99.5])
        pad = 0.15 * (hi - lo + 1e-6)
        lo, hi = lo - pad, hi + pad
        ax.imshow(cam[None, :], aspect='auto', cmap='Reds', alpha=0.55, vmin=0, vmax=1,
                  extent=[0, t[-1], lo, hi], interpolation='bilinear')
        ax.plot(t, x[:, j], color='black', lw=0.7)
        ax.set_ylim(lo, hi)
        ax.set_yticks([])
        ax.text(0.005, 0.82, lead, transform=ax.transAxes, fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))
        for side in ['top', 'right', 'left']:
            ax.spines[side].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('Time (s)')
    fig.suptitle(title, fontsize=12, x=0.01, ha='left')
    fig.tight_layout()
    return fig


# ---------- Page ----------

cfg, models = load_models()
examples, X_ex, Y_ex = load_examples()

st.title('🫀 12-Lead ECG Abnormality Detection')
st.markdown(
    'A 1D CNN ensemble (5 seeds) trained on **PTB-XL** reads a 10-second, 12-lead ECG, estimates whether it is '
    'abnormal, scores 5 diagnostic classes, and shows **where in the recording** it looked (Grad-CAM). '
    f'Test AUROC {cfg["test_auroc_task_a"]:.3f}; external AUROC 0.920 on CPSC 2018. [Code and full evaluation]({REPO_URL})'
)
st.caption('Research and portfolio demo only. Not a medical device; do not use it for any decision about a patient.')

with st.sidebar:
    st.header('1. Choose an ECG')
    source = st.radio('Source', ['Example from the PTB-XL test set', 'Upload your own (CSV)'], label_visibility='collapsed')
    x, truth, notes = None, None, []
    if source.startswith('Example'):
        choice = st.selectbox('Example ECG', examples.label, help='Test-set ECGs never seen in training. '
                              'Includes the model\'s mistakes on purpose.')
        i = int(np.where(examples.label == choice)[0][0])
        x, truth = X_ex[i], examples.iloc[i]
    else:
        up = st.file_uploader('CSV: one column per lead, one row per sample, in millivolts', type='csv')
        fs = st.number_input('Sampling rate (Hz)', min_value=50, max_value=2000, value=500, step=50)
        with open(ASSETS / 'example_upload_100Hz.csv', 'rb') as f:
            st.download_button('Download an example file (100 Hz)', f, 'example_upload_100Hz.csv', 'text/csv')
        if up is not None:
            try:
                x, notes = prepare_upload(pd.read_csv(up), int(fs), cfg)
            except Exception as e:
                st.error(f'Could not read this file: {e}')

    st.header('2. Explain')
    targets = ['Abnormal (overall)'] + [f'{c}: {CLASS_NAMES[c]}' for c in cfg['classes']]
    target = st.selectbox('Show where the model looked for', targets)

    st.divider()
    st.markdown('**Preprocessing:** first 10 s, resampled to 100 Hz, 0.5–40 Hz bandpass filter, '
                'standardized with PTB-XL training statistics.')

if x is None:
    st.info('Upload a CSV in the sidebar, or switch back to the examples.')
    st.stop()
for n in notes:
    st.caption(f'ℹ️ {n}')

# Predictions (Task A with Grad-CAM for the chosen target; Task B probabilities)
if target.startswith('Abnormal'):
    p_bin, cam = predict_and_explain(models['binary'], x, 0)
    p_multi, _ = predict_and_explain(models['multi'], x, 0)
else:
    k = cfg['classes'].index(target.split(':')[0])
    p_bin, _ = predict_and_explain(models['binary'], x, 0)
    p_multi, cam = predict_and_explain(models['multi'], x, k)

raw = float(p_bin[0])
z = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / (1 - np.clip(raw, 1e-6, 1 - 1e-6)))
p_cal = 1 / (1 + np.exp(-(cfg['platt_a'] * z + cfg['platt_b'])))
abnormal = raw >= cfg['threshold_raw']

c1, c2, c3 = st.columns(3)
c1.metric('Model prediction', 'Abnormal' if abnormal else 'Normal')
c2.metric('Probability abnormal (calibrated)', f'{p_cal:.0%}',
          help=f"Platt-scaled ensemble output. The decision threshold corresponds to {cfg['threshold_calibrated']:.0%}.")
if truth is not None:
    label = 'Normal' if truth.y == 0 else 'Abnormal: ' + truth.superclass.replace('|', ', ')
    correct = (truth.y == 1) == abnormal
    c3.metric("Cardiologists' label", label, 'Model agrees' if correct else 'Model disagrees',
              delta_color='normal' if correct else 'inverse')

left, right = st.columns([2.6, 1.4])
with left:
    title = f"Grad-CAM for '{target}' (red = where the model looked most)"
    st.pyplot(plot_ecg(x, cam, cfg, title), clear_figure=True)
    st.caption('Grad-CAM works at 80 ms resolution and gives one map shared by all 12 leads, because the CNN mixes '
               'leads in its first layer. It shows when in the recording the evidence was, not which lead.')
with right:
    st.subheader('Diagnostic classes')
    table = pd.DataFrame({'Class': [CLASS_NAMES[c] for c in cfg['classes']], 'Probability': p_multi})
    if truth is not None:
        table['Label'] = ['✓' if v else '' for v in Y_ex[i]]
    st.dataframe(table, hide_index=True, width='stretch',
                 column_config={'Probability': st.column_config.ProgressColumn(format='percent', min_value=0, max_value=1,
                                                                               width='small')})
    st.caption('Each class is scored independently (an ECG can have several findings), so these do not sum to 1.')

with st.expander('About the model, its performance and its limits'):
    st.markdown(f"""
- **Data:** PTB-XL, 21,388 ECGs labelled by cardiologists; official patient-safe splits (test fold n = 2,158).
- **Model:** 1D CNN, 5 convolution blocks, 422K parameters; average of 5 independently trained seeds.
- **Performance (Normal vs Abnormal):** test AUROC 0.944 (95% CI 0.934–0.952), sensitivity 0.819, specificity 0.918;
  better than XGBoost on 88 handcrafted features (0.917, DeLong p < 0.001).
- **External validation:** AUROC 0.920 on CPSC 2018 (Chinese hospitals). Bundle branch blocks transfer well;
  ST depression and elevation are detected less than half the time there.
- **Known limits:** weaker under age 40 (sensitivity 0.49); often misses mild or non-specific findings;
  needs all 12 leads (a CNN retrained on lead I alone reaches 0.880).
- Full statistics, subgroup analysis, robustness tests and the model card: [GitHub]({REPO_URL}).
""")
