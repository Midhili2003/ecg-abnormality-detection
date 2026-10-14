"""1D CNN and 1D ResNet architectures for 12-lead ECG classification."""
import keras
from keras import layers


def conv_bn_relu(x, filters, kernel, stride=1):
    x = layers.Conv1D(filters, kernel, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


def build_cnn(n_out, input_shape=(1000, 12)):
    """Plain 1D CNN: 5 conv blocks, each halving the time axis, then global pooling."""
    inp = keras.Input(input_shape)
    x = inp
    for filters in [32, 64, 128, 128, 256]:
        x = conv_bn_relu(x, filters, 7)
        x = layers.MaxPooling1D(2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_out, activation='sigmoid')(x)
    return keras.Model(inp, out, name='cnn1d')


def res_block(x, filters, stride):
    """Two convolutions plus a skip connection (projected when the shape changes)."""
    shortcut = x
    y = conv_bn_relu(x, filters, 7, stride)
    y = layers.Conv1D(filters, 7, padding='same', use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    return layers.ReLU()(layers.Add()([y, shortcut]))


def build_resnet(n_out, input_shape=(1000, 12)):
    """1D ResNet: a wide stem, then 7 residual blocks with growing width and shrinking time axis."""
    inp = keras.Input(input_shape)
    x = conv_bn_relu(inp, 64, 15, stride=2)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)
    for filters, stride in [(64, 1), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (256, 2)]:
        x = res_block(x, filters, stride)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_out, activation='sigmoid')(x)
    return keras.Model(inp, out, name='resnet1d')


BUILDERS = {'cnn': build_cnn, 'resnet': build_resnet}
