import numpy as np
import pickle
from tensorflow.keras.optimizers import Adam
from ResidualNetwork import build_model

def load_data():
    with open('data/self_play_data.pkl', 'rb') as f:
        return pickle.load(f)

def train_model():
    model = build_model()
    model.compile(optimizer=Adam(), loss=['sparse_categorical_crossentropy', 'mse'])

    data = load_data()
    boards, actions, results = zip(*data)
    boards = np.array(boards)
    actions = np.array(actions)
    results = np.array(results)

    model.fit(boards, [actions, results], epochs=10)
    model.save_weights('model/latest.h5')

if __name__ == '__main__':
    train_model()
