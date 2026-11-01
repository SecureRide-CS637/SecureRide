import tensorflow as tf
from tensorflow.keras import layers, regularizers

def attention_block(x):
    """
    Squeeze-and-Excitation channel attention (lightweight, MCU-friendly).
    Multiplies channel-wise attention weights across input sequence.
    """
    C = x.shape[-1]
    se = layers.GlobalAveragePooling1D()(x)          # (batch, C)
    se = layers.Dense(max(C // 4, 4), activation="relu")(se)
    se = layers.Dense(C, activation="sigmoid")(se)
    se = layers.Reshape((1, C))(se)
    return layers.multiply([x, se])

def residual_block(x, filters, kernel=3, dropout=0.3, l2=0.01):
    """
    Paper 3-conv residual structure:
    Conv1D -> BN -> ReLU -> Dropout
    Conv1D -> BN -> ReLU -> Dropout
    Conv1D -> BN
    Shortcut projection (if needed) + Add -> ReLU
    """
    shortcut = x
    y = layers.Conv1D(filters, kernel, padding="same",
                      kernel_regularizer=regularizers.l2(l2))(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Dropout(dropout)(y)

    y = layers.Conv1D(filters, kernel, padding="same",
                      kernel_regularizer=regularizers.l2(l2))(y)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Dropout(dropout)(y)

    y = layers.Conv1D(filters, kernel, padding="same",
                      kernel_regularizer=regularizers.l2(l2))(y)
    y = layers.BatchNormalization()(y)

    if shortcut.shape[-1] != filters:                 # project shortcut if needed
        shortcut = layers.Conv1D(filters, 1, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.add([y, shortcut])
    return layers.ReLU()(y)

def build_model(L_seq, hl, filters=32, num_classes=2, dropout=0.3, l2=0.01):
    """
    Full ResNet-style 1D CNN model builder.
    HL = number of residual blocks (HL1, HL2, HL3).
    """
    inp = layers.Input(shape=(L_seq, 3))
    # Stem convolution: expands 3 raw channels to `filters` before attention
    x = layers.Conv1D(filters, 3, padding="same",
                      kernel_regularizer=regularizers.l2(l2))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Attention block
    x = attention_block(x)

    # Residual blocks (HL1..HL3)
    for _ in range(hl):
        x = residual_block(x, filters, dropout=dropout, l2=l2)

    # Classifier head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inp, out, name=f"SecureRide_ResNet1D_L{L_seq}_HL{hl}")

def build_gru(L_seq, n_layers, units=32, num_classes=2, dropout=0.3, l2=0.01):
    """
    GRU baseline for CNN-vs-RNN comparison (Paper Fig. 5c/d).
    """
    inp = layers.Input(shape=(L_seq, 3))
    x = inp
    for i in range(n_layers):
        x = layers.GRU(units, return_sequences=(i < n_layers - 1),
                       dropout=dropout, kernel_regularizer=regularizers.l2(l2))(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name=f"SecureRide_GRU_L{L_seq}_HL{n_layers}")

if __name__ == "__main__":
    m = build_model(5, 1)
    m.summary()
    print("Model output shape:", m.output_shape)
