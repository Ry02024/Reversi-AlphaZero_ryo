from ResidualNetwork import DN_INPUT_SHAPE
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

RN_EPOCHS = 10  # 学習回数

def load_data():
    history_path = sorted(Path('./data').glob('*.history'))[-1]
    with history_path.open(mode='rb') as f:
        return pickle.load(f)

def train_network():
    training_data = load_data()
    xs, y_policies, y_values = zip(*training_data)

    a, b, c = DN_INPUT_SHAPE
    xs = np.array(xs)
    xs = xs.reshape(len(xs), c, a, b).transpose(0, 2, 3, 1)
    y_policies = np.array(y_policies)
    y_values = np.array(y_values)

    # データセットをトレーニングデータと検証データに分割
    xs_train, xs_val, y_policies_train, y_policies_val, y_values_train, y_values_val = train_test_split(
        xs, y_policies, y_values, test_size=0.2, random_state=42)

    model = load_model('./model/best.h5')
    model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam')

    def step_decay(epoch):
        x = 0.001
        if epoch >= 50: x = 0.0005
        if epoch >= 80: x = 0.00025
        return x
    lr_decay = LearningRateScheduler(step_decay)

    print_callback = LambdaCallback(
        on_epoch_begin=lambda epoch,logs:
                print('\rTrain {}/{}'.format(epoch + 1, RN_EPOCHS), end=''))

    training_history = model.fit(xs_train, [y_policies_train, y_values_train], batch_size=128, epochs=RN_EPOCHS,
                                 validation_data=(xs_val, [y_policies_val, y_values_val]),
                                 verbose=0, callbacks=[lr_decay, print_callback])
    print('save latest model')

    model.save('./model/latest.keras')

    K.clear_session()
    del model
    
    history = {
        'loss': training_history.history['loss'],
        'val_loss': training_history.history.get('val_loss', [])
    }
    print('save history')
    with open('./data/training_history.pkl', 'wb') as f:
        pickle.dump(history, f)

if __name__ == '__main__':
    train_network()
