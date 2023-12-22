## CNN Interpretability: Investigating Feature Maps


VGG16 is a deep convolutional neural network developed by researchers to investigate the effect of the depth
of a network on its accuracy in a large-scale image recognition setting. TensorFlow and other deep learning
libraries have a pretrained version of VGG16 using the ImageNet data set.

1. Load the pretrained version of VGG16 using the tf.keras.applications API, and print its architecture.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions
from tensorflow.keras.preprocessing import image

"""### Task 1: Investigating Feature Maps"""

# Load pretrained VGG16 model
vgg16 = VGG16(weights='imagenet')
print(vgg16.summary())

"""2. VGG16 takes as input a 224 by 224 RGB image, and outputs the label


probabilities. To prepare an image for VGG16.

### Task 2: Predict the top-5 labels for an image

"""

def prepare_image(path):
    # VGG16 takes as input an image of size 224 by 224 pixels
    img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
    # convert the loaded image into a numpy array
    x = tf.keras.preprocessing.image.img_to_array(img)
    # put the loaded image in a batch of 1 image of shape 224x224x3
    x = np.expand_dims(x, axis=0)
    # convert the image in the batch from RGB to BGR,
    # then each color channel is zero-centered, without scaling.
    x = tf.keras.applications.vgg16.preprocess_input(x)
    return x

# Load an example image
image_path = '/content/bird.jpg'
img = prepare_image(image_path)

"""Predict the top-5 labels of the provided image.
To infer labels from logits, one can use **tf.keras.applications.vgg16.decode_predictions**
"""

predictions = vgg16.predict(img)
decoded_predictions = decode_predictions(predictions, top=5)[0]

print("Top-5 predictions:")
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}: {label} ({score:.2f})")

"""### Task 3: Show feature maps

Now let's show the feature maps of each convolutional layer in the form of a gray scale images. How does the
earlier layers compare to the later layers?
"""

def show_feature_maps(feature_maps):
    n_features = feature_maps.shape[3]
    sp = int(np.sqrt(n_features))
    if np.floor(sp) != sp:
        sp += 1
    plt.figure(figsize=(15, 15))
    for i in range(1, n_features + 1):
        plt.subplot(sp, sp, i)
        plt.imshow(feature_maps[0, :, :, i - 1], cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.show()

"""Truncate the model to a specific layer

"""

truncated_model = tf.keras.models.Model(
    inputs=vgg16.input,
    outputs=vgg16.get_layer('block1_conv2').output
)

# Get feature maps for the example image
feature_maps = truncated_model.predict(img)
show_feature_maps(feature_maps)

"""**How does the earlier layers compare to the later layers?**

By examining the feature maps for each layer, we observe how the network learns hierarchical representations, where earlier layers capture low-level features (e.g., edges and textures), and later layers capture high-level features and more abstract representations.

## 2. Transfer Learning: Recognizing Flower Types Using Pretrained VGG16

Transfer Learning is a machine learning method where a model initially trained for a task is reused as the starting point model to perform another task. In the following, our goal is to partially retrain the VGG16 model
to recognize flower types, using the “tf_flowers” data set. The approach consists of first freezing the weights
of the convolution layers (the early layers of VGG16), then, training the top dense layers only. Thus, the early
layers’ job is to infer a latent representation of the input images, and the backpropagation is performed only
on top dense layers, in a supervised learning fashion.
"""

import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

"""# Task 1: Load and preprocess the tf_flowers dataset"""

dataset_name = 'tf_flowers'
(train_ds, test_ds), info = tfds.load(
    name=dataset_name,
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True
)

num_classes = info.features['label'].num_classes
print(num_classes )

"""## 2. Preprocess images for VGG16 input

"""

def preprocess_for_vgg16(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image, label

train_ds = train_ds.map(preprocess_for_vgg16)
test_ds = test_ds.map(preprocess_for_vgg16)

"""## 3. Modify VGG16 architecture for transfer learning

"""

base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False

from tensorflow.keras import layers, models, optimizers

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')
])

model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

"""## 4. Train the model

"""

train_dataset = tf.data.Dataset.from_tensor_slices((np.stack(list(train_ds.map(lambda x, y: x))), np.stack(list(train_ds.map(lambda x, y: y)))))
test_dataset = tf.data.Dataset.from_tensor_slices((np.stack(list(test_ds.map(lambda x, y: x))), np.stack(list(test_ds.map(lambda x, y: y)))))

history = model.fit(train_dataset.batch(32), epochs=3)

# Task 3: Fine-tuning of the entire model
# Fine-tuning
base_model.trainable = True
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history_finetune = model.fit(train_dataset.batch(16), epochs=3)

test_loss, test_accuracy = model.evaluate(test_dataset.batch(32))
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')



