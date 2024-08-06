import joblib
import numpy as np
import matplotlib.pyplot as plt
import keras

class BaseModel():
    def __init__(self, modelName):
        self.modelName = modelName
        self.model = None
        self.maxLen = 0

    def train(self):
        pass

    def build_model(self):
        pass

    def test(self):
        pass

    def save_model(self, filename):
        self.model.save(filename)

    @classmethod
    def load_model(cls, filename):
        try:
            model = keras.models.load_model(filename+'.keras')
        except ValueError:
            return None
        return model

    def convert_to_one_hot(self, val, totalBits):
        return np.eye(totalBits)[val.reshape(-1)]

    def get_max_len(self):
        return 0

    def model_fit(self, X, Y):
        pass

    def tokenize(self, X):
        pass

    def draw_training_graphs(self, history):
        best_epoch = history.history['val_accuracy'].index(max(history.history['val_accuracy'])) + 1

        # Create a subplot with 1 row and 2 columns
        fig, axs = plt.subplots(1, 2, figsize=(16, 5))

        # Plot training and validation accuracy
        axs[0].plot(history.history['accuracy'], label='Training Accuracy', color='blue')
        axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
        axs[0].scatter(best_epoch - 1, history.history['val_accuracy'][best_epoch - 1], color='green', label=f'Best Epoch: {best_epoch}')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_title('Training and Validation Accuracy')
        axs[0].legend()


        # Plot training and validation loss
        axs[1].plot(history.history['loss'], label='Training Loss', color='blue')
        axs[1].plot(history.history['val_loss'], label='Validation Loss', color='red')
        axs[1].scatter(best_epoch - 1, history.history['val_loss'][best_epoch - 1], color='green',label=f'Best Epoch: {best_epoch}')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Loss')
        axs[1].set_title('Training and Validation Loss')
        axs[1].legend()

        plt.tight_layout()
        plt.show()