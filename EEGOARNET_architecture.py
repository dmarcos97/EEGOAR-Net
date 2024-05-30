import os

from tensorflow.keras.layers import Activation, Input
from tensorflow.keras.layers import Dropout, BatchNormalization, SpatialDropout2D
from tensorflow.keras.layers import Conv2D, AveragePooling2D, DepthwiseConv2D
from tensorflow.keras.layers import UpSampling2D, Add, MaxPooling2D
from tensorflow.keras.constraints import max_norm
from tensorflow import keras
from tensorflow.math import multiply, log
from tensorflow import expand_dims, repeat, reshape,abs,slice
from tensorflow import cast, float32, complex64
from tensorflow.keras.losses import mse
from tensorflow.signal import fft
from tensorflow import transpose


def custom_mse(y_true,y_pred):
    fft_y_true = transpose(fft(cast(transpose(y_true,
                                                  perm=[3,0,2,1]),complex64)),
                           perm=[1,3,2,0])
    fft_y_pred = transpose(fft(cast(transpose(y_pred,
                                                  perm=[3,0,2,1]),complex64)),
                           perm=[1,3,2,0])
    mse_fft = cast(mse(fft_y_true,fft_y_pred),float32)
    return cast(mse(y_true,y_pred),float32) + mse_fft


def EEGOARNET(input_time=1000, fs=128, ncha=64, filters_per_branch=8,
                 scales_time=(500, 250, 125), dropout_rate=0.25,
                 activation='elu', learning_rate=0.001):
    # ============================= CALCULATIONS ============================= #
    input_samples = int(input_time * fs / 1000)
    scales_samples = [int(s * fs / 1000) for s in scales_time]

    # ================================ INPUT ================================= #
    input_layer_signal = Input((input_samples, ncha, 1))
    input_layer_mask = Input((ncha,1))
    # ========================== BLOCK 1: INCEPTION ========================== #
    b1_units = list()
    for i in range(len(scales_samples)):
        unit = Conv2D(filters=filters_per_branch,
                      kernel_size=(scales_samples[i], 1),
                      kernel_initializer='he_normal',
                      padding='same')(input_layer_signal)
        unit = BatchNormalization()(unit)
        unit = Activation(activation)(unit)

        unit = DepthwiseConv2D((1, 2*filters_per_branch),
                               padding='same',
                               use_bias=False,
                               depth_multiplier=2,
                               depthwise_constraint=max_norm(1.))(unit)
        unit = BatchNormalization()(unit)
        unit = Activation(activation)(unit)

        b1_units.append(unit)

    # Concatenation
    b1_out = keras.layers.concatenate(b1_units, axis=3)

    b1_out = MaxPooling2D((4, 2))(b1_out)

    # ========================== BLOCK 2: INCEPTION ========================== #
    b2_units = list()
    for i in range(len(scales_samples)):
        unit = Conv2D(filters=filters_per_branch,
                      kernel_size=(int(scales_samples[i] / 4), 1),
                      kernel_initializer='he_normal',
                      use_bias=False,
                      padding='same')(b1_out)
        unit = BatchNormalization()(unit)
        unit = Activation(activation)(unit)

        unit = DepthwiseConv2D((1, 2*filters_per_branch),
                               padding='same',
                               use_bias=False,
                               depth_multiplier=2,
                               depthwise_constraint=max_norm(1.))(unit)
        unit = BatchNormalization()(unit)
        unit = Activation(activation)(unit)

        b2_units.append(unit)

    # Concatenate + Average pooling
    b2_out = keras.layers.concatenate(b2_units, axis=3)
    b2_out = MaxPooling2D((2, 2))(b2_out)

    # ========================== BLOCK 3: ENCODING========================== #
    b3_u1 = Conv2D(filters=int(filters_per_branch*len(scales_samples)*4),
                   kernel_size=(3, 1),
                   kernel_initializer='he_normal',
                   use_bias=False,
                   padding='same')(b2_out)
    b3_u1 = BatchNormalization()(b3_u1)
    b3_u1 = Activation(activation)(b3_u1)
    b3_u1 = SpatialDropout2D(dropout_rate)(b3_u1)

    b3_u2 = Conv2D(filters=int(filters_per_branch*len(scales_samples)*4),
                   kernel_size=(3, 1),
                   kernel_initializer='he_normal',
                   use_bias=False,
                   padding='same')(b3_u1)
    b3_u2 = BatchNormalization()(b3_u2)
    b3_u2 = Activation(activation)(b3_u2)
    b3_u2 = SpatialDropout2D(dropout_rate)(b3_u2)

    b3_out = MaxPooling2D((2, 2))(b3_u2)

    # ============================ BLOCK 4: ENCODING  ======================== #
    b4_u1 = Conv2D(filters=int(filters_per_branch*len(scales_samples)*6),
                   kernel_size=(3, 1),
                   kernel_initializer='he_normal',
                   use_bias=False,
                   padding='same')(b3_out)
    b4_u1 = BatchNormalization()(b4_u1)
    b4_u1 = Activation(activation)(b4_u1)
    # b4_u1 = SpatialDropout2D(dropout_rate)(b4_u1)

    b4_u2 = Conv2D(filters=int(filters_per_branch*len(scales_samples)*6),
                   kernel_size=(3, 1),
                   kernel_initializer='he_normal',
                   use_bias=False,
                   padding='same')(b4_u1)
    b4_u2 = BatchNormalization()(b4_u2)
    b4_u2 = Activation(activation)(b4_u2)
    b4_u2 = SpatialDropout2D(dropout_rate)(b4_u2)
    # ============================ BLOCK 5: DECODING  ======================== #

    b5_u1 = Conv2D(filters=int(filters_per_branch * len(scales_samples) * 4),
                   kernel_size=(3, 3),
                   kernel_initializer='he_normal',
                   use_bias=False,
                   padding='same')(UpSampling2D(size = (2,2))(b4_u2))
    b5_u1 = BatchNormalization()(b5_u1)
    b5_u1 = Activation(activation)(b5_u1)

    b5_u1 = keras.layers.concatenate([b3_u2,b5_u1],axis=3)

    b5_u1 = Conv2D(filters=int(filters_per_branch * len(scales_samples) * 4),
                   kernel_size=(3, 3),
                   kernel_initializer='he_normal',
                   use_bias=False,
                   padding='same')(b5_u1)
    b5_u1 = BatchNormalization()(b5_u1)
    b5_u1 = Activation(activation)(b5_u1)

    b5_u2 = Conv2D(filters=int(filters_per_branch*len(scales_samples)*2),
                   kernel_size=(3, 3),
                   kernel_initializer='he_normal',
                   use_bias=False,
                   padding='same')(UpSampling2D(size=(2, 2))(b5_u1))
    b5_u2 = BatchNormalization()(b5_u2)
    b5_u2 = Activation(activation)(b5_u2)

    b5_u2 = keras.layers.concatenate([
        keras.layers.concatenate(b2_units, axis=3),b5_u2],axis=3)

    b5_u2 = Conv2D(filters=int(filters_per_branch*len(scales_samples)*2),
                   kernel_size=(3, 3),
                   kernel_initializer='he_normal',
                   use_bias=False,
                   padding='same')(b5_u2)
    b5_u2 = BatchNormalization()(b5_u2)
    b5_u2 = Activation(activation)(b5_u2)

    b5_u3 = Conv2D(filters=filters_per_branch*len(scales_samples)*2 ,
                   kernel_size=(3, 3),
                   kernel_initializer='he_normal',
                   use_bias=False,
                   padding='same')(UpSampling2D(size=(4, 2))(b5_u2))
    b5_u3 = BatchNormalization()(b5_u3)
    b5_u3 = Activation(activation)(b5_u3)

    b5_u3 = keras.layers.concatenate([
        keras.layers.concatenate(b1_units, axis=3), b5_u3], axis=3)

    b5_u3 = Conv2D(filters=filters_per_branch*len(scales_samples)*2,
                   kernel_size=(3, 3),
                   kernel_initializer='he_normal',
                   use_bias=False,
                   padding='same')(b5_u3)
    b5_u3 = BatchNormalization()(b5_u3)
    b5_u3 = Activation(activation)(b5_u3)

    b5_u3 = Conv2D(filters=1,
                   kernel_size=(1, 1),
                   kernel_initializer='he_normal',
                   use_bias=False,
                   padding='same')(b5_u3)
    b5_u3 = BatchNormalization()(b5_u3)
    b5_u3 = Activation('linear')(b5_u3)

    output = multiply(b5_u3 , expand_dims(cast(input_layer_mask,float32),axis=1))

    model = keras.models.Model(inputs=[input_layer_signal,input_layer_mask], outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9,
                                      beta_2=0.999, amsgrad=False)
    model.compile(loss=custom_mse, optimizer=optimizer,
                  metrics=['mse'])
    return model
if __name__ == "__main__":
    with open('modelsummary.txt', 'w') as f:
        model = EEGOARNET()
        model.summary(print_fn=lambda x: f.write(x + '\n'))

