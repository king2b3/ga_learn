"""train.py
Developer: Bayley King
Date: 2-19-2022
Description: Tensorflow API
"""
################################## Imports ###################################
import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
##############################################################################

################################## Globals ###################################
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.random.set_seed(42)
##############################################################################

model = tf.keras.Sequential([
    tf.keras.layers.Dense(13, activation='relu'),
    tf.keras.layers.Dense(15, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss=tf.keras.losses.poisson,
    optimizer=tf.keras.optimizers.Adam(lr=0.05)
)


def train(x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train, epochs=50, verbose=0)

    predictions = model.predict(x_test)

    prediction_classes = [
        1 if prob > 0.5 else 0 for prob in np.ravel(predictions)
    ]

    print(confusion_matrix(y_test, prediction_classes))

    print(f'Accuracy: {accuracy_score(y_test, prediction_classes):.2f}')
    print(f'Precision: {precision_score(y_test, prediction_classes):.2f}')
    print(f'Recall: {recall_score(y_test, prediction_classes):.2f}')
