import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class Trainer_tensorflow:
    def __init__(self, model, train_dataset, test_dataset, lr, wd, epochs, device='GPU'):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.epochs = epochs
        self.device = device

        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # If weight decay (L2 regularization) is specified
        if wd > 0:
            for layer in self.model.layers:
                if hasattr(layer, 'kernel_regularizer'):
                    layer.kernel_regularizer = tf.keras.regularizers.l2(wd)

        self.train_loss_results = []
        self.train_accuracy_results = []

        self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    def train(self, save=False, plot=False):
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            total_samples = 0

            # Reset accuracy metric at start of each epoch
            self.train_acc_metric.reset_states()

            progress_bar = tqdm(self.train_dataset, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False)

            for step, (x_batch_train, y_batch_train) in enumerate(progress_bar):
                with tf.GradientTape() as tape:
                    logits = self.model(x_batch_train, training=True)
                    loss_value = self.loss_fn(y_batch_train, logits)

                grads = tape.gradient(loss_value, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                self.train_acc_metric.update_state(y_batch_train, logits)
                epoch_loss += loss_value.numpy() * x_batch_train.shape[0]
                total_samples += x_batch_train.shape[0]

                avg_loss = epoch_loss / total_samples
                avg_acc = self.train_acc_metric.result().numpy() * 100

                progress_bar.set_postfix({
                    'Batch Acc': f"{self.train_acc_metric.result().numpy()*100:.2f}%",
                    'Avg Acc': f"{avg_acc:.2f}%",
                    'Loss': f"{avg_loss:.4f}"
                })

            self.train_loss_results.append(avg_loss)
            self.train_accuracy_results.append(avg_acc)

        if save:
            self.model.save("Rihanatou_BANKOLE_model.h5")
        if plot:
            self.plot_training_history()

    def evaluate(self):
        test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        test_loss = tf.keras.metrics.Mean()

        for x_batch_test, y_batch_test in tqdm(self.test_dataset, desc="Evaluating", leave=False):
            logits = self.model(x_batch_test, training=False)
            loss_value = self.loss_fn(y_batch_test, logits)

            test_loss.update_state(loss_value)
            test_acc_metric.update_state(y_batch_test, logits)

        print(f"\nTest Accuracy: {test_acc_metric.result().numpy()*100:.2f}%  |  Test Loss: {test_loss.result().numpy():.4f}")
        return test_acc_metric.result().numpy()*100, test_loss.result().numpy()

    def plot_training_history(self):
        epochs_range = range(1, len(self.train_loss_results) + 1)

        fig, ax1 = plt.subplots(figsize=(8, 5))

        color_loss = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color_loss)
        ax1.plot(epochs_range, self.train_loss_results, color=color_loss, label='Loss')
        ax1.tick_params(axis='y', labelcolor=color_loss)

        ax2 = ax1.twinx()
        color_acc = 'tab:red'
        ax2.set_ylabel('Accuracy (%)', color=color_acc)
        ax2.plot(epochs_range, self.train_accuracy_results, color=color_acc, label='Accuracy')
        ax2.tick_params(axis='y', labelcolor=color_acc)

        plt.title('Training Loss and Accuracy')
        fig.tight_layout()
        plt.show()
