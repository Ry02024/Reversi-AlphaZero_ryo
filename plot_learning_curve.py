import pickle
import matplotlib.pyplot as plt

def plot_learning_curve(history):
    loss = history['loss']
    epochs = range(1, len(loss) + 1)
    val_loss = history.get('val_loss', [])

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    if val_loss:
        plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def load_training_history():
    with open('./data/training_history.pkl', 'rb') as f:
        return pickle.load(f)

def plot_saved_learning_curve():
    history = load_training_history()
    plot_learning_curve(history)

if __name__ == '__main__':
    plot_saved_learning_curve()
