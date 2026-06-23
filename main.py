from keras import ops
from keras.datasets import mnist
from keras import optimizers

import os
import tensorflow as tf

from batch_generator import BatchGenerator
from naive_dense import NaiveDense
from naive_sequential import NaiveSequential

os.environ["KERAS_BACKEND"] = "tensorflow"
optimizer = optimizers.SGD(learning_rate=1e-3)


# Loading data from MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# We are creating a naive-implementation of a Keras model, instead of compiling one
model = NaiveSequential(
    [
        NaiveDense(input_size=28 * 28, output_size=512, activation=ops.relu),
        NaiveDense(input_size=512, output_size=10, activation=ops.softmax),
    ]
)
assert len(model.weights) == 4


def update_weights_manually(gradients, weights):
    learning_rate = 1e-3
    for g, w in zip(gradients, weights):
        w.assign(w - g * learning_rate)

# Instead of implementing manually as above, you would normally use an Optimizer instance from Keras
def update_weights(gradients, weights):
    optimizer.apply_gradients(zip(gradients, weights))

def one_training_step(model, images_batch, labels_batch):
    # The API through which you can use TensorFlow’s automatic differentiation capabilities
    # is the tf.GradientTape object. It’s a Python scope that will “record” the tensor operations
    # that run inside it, in the form of a computation graph (sometimes called a tape).
    with tf.GradientTape() as tape:
        predictions = model(images_batch)
        loss = ops.sparse_categorical_crossentropy(labels_batch, predictions)
        average_loss = ops.mean(loss)
    gradients = tape.gradient(average_loss, model.weights)
    update_weights_manually(gradients, model.weights)
    return average_loss

def accuracy(model, test_images, test_labels):
    predictions = model(test_images)
    predicted_labels = ops.argmax(predictions, axis=1)
    matches = predicted_labels == test_labels
    print(f"accuracy: {ops.mean(matches):.2f}")

def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter + 1}")
        batch_generator = BatchGenerator(images, labels)
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
            loss = one_training_step(model, images_batch, labels_batch)
            if batch_counter % 100 == 0:
                print(f"loss at batch {batch_counter + 1}: {loss:.2f}")
        accuracy(model, test_images, test_labels)

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

if __name__ == "__main__":
    fit(model, train_images, train_labels, epochs=5, batch_size=128)
    fit(model, train_images, train_labels, epochs=5, batch_size=128)