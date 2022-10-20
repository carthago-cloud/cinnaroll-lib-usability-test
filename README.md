 A simple convolutional network in Keras for the MNIST dataset

### Environment setup

Create a virtual environment as follows:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If this does not work and you are on M1 Mac you can try `conda` and `requirements_macos.txt` instead (you can follow instructions given [here](https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706)). I have tested this on Linux and macOS with `Python 3.10`.

### Model

To make sure that the model is working, execute a basic routine (creating, training, evaluating) by running:

```
python cnn_mnist_keras.py
```

The MNIST dataset can be accessed through the `tensorflow.keras.datasets` package. To make a prediction on an arbitrary image use the `preprocess_image` function. The output of that you can feed directly into the model.
