from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense
from tensorflow.keras.models import Model

def build_model():
    inputs = Input(shape=(8, 8, 2))
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for _ in range(5):
        x = res_block(x, 64)

    x = Flatten()(x)
    policy_head = Dense(64, activation='softmax')(x)
    value_head = Dense(1, activation='tanh')(x)

    return Model(inputs=inputs, outputs=[policy_head, value_head])

def res_block(x, filters):
    shortcut = x
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    return Activation('relu')(x)
