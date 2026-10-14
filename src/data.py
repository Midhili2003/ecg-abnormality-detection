"""Loading processed PTB-XL splits and building tf.data pipelines with augmentation."""
import numpy as np
import tensorflow as tf

DATA_DIR = '/content/drive/MyDrive/ecg-project/data_processed'
CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


def load_split(name, task='binary', data_dir=DATA_DIR):
    """Load X and labels for 'train', 'val' or 'test'. task: 'binary' or 'multi'."""
    X = np.load(f'{data_dir}/X_{name}.npy')
    y = np.load(f"{data_dir}/{'ybin' if task == 'binary' else 'Ymulti'}_{name}.npy")
    return X, y


def augment(x, y):
    """Random, label-preserving changes applied to one ECG of shape (1000, 12)."""
    x = x * tf.random.uniform([1, 12], 0.9, 1.1)                         # amplitude scaling per lead
    x = x + tf.random.normal(tf.shape(x), stddev=0.03)                   # Gaussian noise
    x = tf.roll(x, tf.random.uniform([], -50, 51, tf.int32), axis=0)     # time shift up to 0.5 s
    x = x * tf.cast(tf.random.uniform([1, 12]) > 0.1, x.dtype)           # drop about 10% of leads
    return x, y


def make_dataset(X, y, training, batch_size=64, seed=42):
    """Batched tf.data pipeline; shuffling and augmentation only when training=True."""
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(len(X), seed=seed).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
