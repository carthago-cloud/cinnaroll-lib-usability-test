# Ref: https://keras.io/examples/vision/mnist_convnet/

import PIL
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)


def load_data(num_classes, limit):
    # Load the data and split it between train and test sets
    data_array = keras.datasets.mnist.load_data()

    if limit is not None:
        tmp = data_array

        data_array = [[None] * 2, [None] * 2]
        for j in range(2):
            for k in range(2):
                data_array[j][k] = tmp[j][k][:limit]

    all_data = {
        "test": {"X": data_array[1][0], "Y": data_array[1][1]},
        "train": {"X": data_array[0][0], "Y": data_array[0][1]},
    }

    print("\nDataset info:")

    for key, val in all_data.items():
        print(f"Number of samples in {key} set: {val['X'].shape[0]}")
        # the range of each pixel should be a float in [0, 1] interval
        val["X"] = val["X"].astype("float32") / 255
        # every image should have shape (28, 28, 1)
        val["X"] = np.expand_dims(val["X"], -1)
        # convert numerical labels to one-hot encodings
        val["Y"] = keras.utils.to_categorical(val["Y"], num_classes)

    return all_data


def construct_model(num_classes, input_shape):
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    # print model summary
    print("\nModel info:")
    model.summary()

    # compile model
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"],
    )

    return model


def train_model(model, all_data, batch_size, epochs):
    model.fit(
        all_data["train"]["X"],
        all_data["train"]["Y"],
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(all_data["test"]["X"], all_data["test"]["Y"]),
    )


def evaluate_model(model, data):
    score = model.evaluate(data["X"], data["Y"], verbose=0)
    print(f"Test loss: {score[0]:.3f}")
    print(f"Test accuracy: {score[1]:.3f}")


def preprocess_image(img_path):
    img = PIL.Image.open(img_path)
    img_processed = img.convert("L").resize(INPUT_SHAPE[:2])
    img_array = np.array(img_processed).reshape((1,) + INPUT_SHAPE)
    return img_array


def make_prediction(x):
    return int(np.argmax(x))


def main():
    # define the number of classes and expected input shape
    all_data = load_data(num_classes=NUM_CLASSES, limit=100)
    model = construct_model(num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE)

    # evaluate performance on the test set *before* training
    print("\nPerformance of a randomly initialised model:")
    evaluate_model(model=model, data=all_data["test"])

    # set training parameters and perform training
    print("\nPerform training...")
    train_model(model=model, all_data=all_data, batch_size=128, epochs=10)

    # evaluate performance on the test set *after* training
    print("\nPerformance of a trained model:")
    evaluate_model(model=model, data=all_data["test"])


if __name__ == "__main__":
    main()
