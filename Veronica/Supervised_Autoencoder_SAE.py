from keras import models,layers, regularizers, Input
import tensorflow as tf
import pandas as pd

#Reconstruction Loss
def reconstruction_loss(y_true,y_pred):
  loss=tf.keras.losses.MeanSquaredError()(y_true,y_pred)
  return loss
#Classification Loss
def classification_loss(y_true,y_pred):
  loss=tf.keras.losses.BinaryCrossentropy()
  return loss(y_true,y_pred)

def sae(input_shape,encoder_shape0,encoder_shape1,alpha):
    input = Input(shape=(input_shape,))


    encoder0=layers.Dense(encoder_shape0, activation='relu')(input)

    encoder1=layers.Dense(encoder_shape1, activation='relu', name='layer_reduced',
                            kernel_regularizer=regularizers.l2(0.00001))(encoder0)


    # decoded9 = layers.Dense(64, activation='linear')(encoded9)

    decoder= layers.Dense(input_shape, activation='linear', name='decoded_output')(encoder1)

    #
    classification = layers.Dense(1, activation='sigmoid', name='classification_output')(encoder1)

    output = decoder, classification
    model9 = models.Model(inputs=input, outputs=output)
    model9.compile(optimizer='adam',
                   loss={'decoded_output': reconstruction_loss, 'classification_output': classification_loss},
                   loss_weights={'decoded_output': alpha, 'classification_output': 1 - alpha},
                   metrics={'decoded_output': tf.keras.metrics.MeanSquaredError(),
                            'classification_output': tf.keras.metrics.CategoricalAccuracy()})

    return sae

def load_dataset(dataset_name):
    if dataset_name == 'dataset1':
        return pd.read_csv('../data/df_autismo_clr.csv')
    elif dataset_name == 'dataset2':
        return pd.read_csv('../data/df_completo_filtrato_tesi.csv')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")