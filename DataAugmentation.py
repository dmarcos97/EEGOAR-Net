import tensorflow as tf
import numpy as np
import random


def get_predefined_montages(n_cha, channels, type='uniform'):
    if type == 'vep':
        eight_ch = ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"]
        return np.isin(channels, eight_ch)
    else:
        eight_ch = ["F3", "F4", "C3", "Cz", "C4", "P3", "P4", "Oz"]
        sixteen_ch = ["F7", "Fz", "F8", "Pz", "PO3", "PO4","POz", "FPz"] + eight_ch
        twentyfour_ch = ["T7", "T8", "FC3", "FCz", "FC4", "CP3", "CPz", "CP4"] + sixteen_ch
        thirtytwo_ch = ["F5", "F6", "C5", "C6", "P5", "P6", "O1", "O2"] + twentyfour_ch
        fourty_ch = ["AF7", "AF8", "FT7", "FT8", "P7", "P8", "PO7", "PO8"] + thirtytwo_ch
        fourtyeight_ch = ["FC5", "FC6", "CP5", "CP6", "TP7", "TP8", "AF3", "AF4"] + fourty_ch
        fiftysix_ch = ["F1", "F2", "FC1", "FC2", "C1", "C2", "P1", "P2"] + fourtyeight_ch
        if n_cha == 8:
            return np.isin(channels, eight_ch)
        elif n_cha == 16:
            return np.isin(channels, sixteen_ch)
        elif n_cha == 24:
            return np.isin(channels, twentyfour_ch)
        elif n_cha == 32:
            return np.isin(channels, thirtytwo_ch)
        elif n_cha == 40:
            return np.isin(channels, fourty_ch)
        elif n_cha == 48:
            return np.isin(channels, fourtyeight_ch)
        elif n_cha == 56:
            return np.isin(channels, fiftysix_ch)
        else:
            return None

def data_augmentation(X, Y):
    eigh_ch = np.array([False, False, False, False, False, False, False, True, False,
                        False, False, True, False, False, False, False, False, False,
                        False, False, False, False, False, False, False, False, False,
                        True, False, True, False, True, False, False, False, False,
                        False, False, False, False, False, False, False, False, False,
                        False, False, True, False, False, False, True, False, False,
                        False, False, False, False, False, False, False, False, True,
                        False])
    sixteen_ch = np.array([True, False, False, False, False, True, False, True, False,
                           True, False, True, False, True, False, False, False, False,
                           False, False, False, False, False, False, False, False, False,
                           True, False, True, False, True, False, False, False, False,
                           False, False, False, False, False, False, False, False, False,
                           False, False, True, False, True, False, True, False, False,
                           False, False, True, False, True, False, False, False, True,
                           False])

    twentyfour_ch = np.array([True, False, False, False, False, True, False, True, False,
                              True, False, True, False, True, False, True, False, True,
                              False, True, False, False, False, False, False, True, False,
                              True, False, True, False, True, False, True, False, False,
                              False, False, False, True, False, True, False, True, False,
                              False, False, True, False, True, False, True, False, False,
                              False, False, True, False, True, False, False, False, True,
                              False])

    thirtytwo_ch = np.array([True, False, False, False, False, True, True, True, False,
                             True, False, True, True, True, False, True, False, True,
                             False, True, False, False, False, False, False, True, True,
                             True, False, True, False, True, True, True, False, False,
                             False, False, False, True, False, True, False, True, False,
                             False, True, True, False, True, False, True, True, False,
                             False, False, True, False, True, False, False, True, True,
                             True])

    fourty_ch = np.array([True, True, False, False, True, True, True, True, False,
                          True, False, True, True, True, False, True, False, True,
                          False, True, False, False, True, True, False, True, True,
                          True, False, True, False, True, True, True, False, False,
                          False, False, False, True, False, True, False, True, False,
                          True, True, True, False, True, False, True, True, True,
                          False, True, True, False, True, True, False, True, True,
                          True])

    fourtyeight_ch = np.array([True, True, True, True, True, True, True, True, False,
                               True, False, True, True, True, True, True, False, True,
                               False, True, True, False, True, True, False, True, True,
                               True, False, True, False, True, True, True, False, True,
                               True, False, True, True, False, True, False, True, True,
                               True, True, True, False, True, False, True, True, True,
                               False, True, True, False, True, True, False, True, True,
                               True])

    fiftysix_ch = np.array([True, True, True, True, True, True, True, True, True,
                            True, True, True, True, True, True, True, True, True,
                            True, True, True, False, True, True, False, True, True,
                            True, True, True, True, True, True, True, False, True,
                            True, False, True, True, False, True, False, True, True,
                            True, True, True, True, True, True, True, True, True,
                            False, True, True, False, True, True, False, True, True,
                            True])

    vep_ch = np.array([False, False, False, False, False, False, False, False, False,
                       True, False, False, False, False, False, False, False, False,
                       False, False, False, False, False, False, False, False, False,
                       False, False, True, False, False, False, False, False, False,
                       False, False, False, False, False, False, False, False, False,
                       False, False, True, False, True, False, True, False, False,
                       False, True, False, False, False, True, False, False, True,
                       False])

    signal, mask = X
    augmentations = dict()
    augmentations["eigh_ch"] = False
    augmentations["sixteen_ch"] = False
    augmentations["twentyfour_ch"] = False
    augmentations["thirtytwo_ch"] = False
    augmentations["fourty_ch"] = False
    augmentations["fourtyeight_ch"] = False
    augmentations["fiftysix_ch"] = False
    augmentations["vep_ch"] = False
    augmentations["no_augmentation"] = False
    augmentation = True

    if augmentation:
        selection = np.random.choice(["vep_ch", "eigh_ch", "sixteen_ch",
                                      "twentyfour_ch", "thirtytwo_ch",
                                      "fourty_ch", "fourtyeight_ch",
                                      "fiftysix_ch", "no_augmentation"
                                      ], p=None)
        augmentations[selection] = True

        if augmentations["eigh_ch"]:
            mask = eigh_ch
        elif augmentations["sixteen_ch"]:
            mask = sixteen_ch
        elif augmentations["twentyfour_ch"]:
            mask = twentyfour_ch
        elif augmentations["thirtytwo_ch"]:
            mask = thirtytwo_ch
        elif augmentations["fourty_ch"]:
            mask = fourty_ch
        elif augmentations["fourtyeight_ch"]:
            mask = fourtyeight_ch
        elif augmentations["fiftysix_ch"]:
            mask = fiftysix_ch
        elif augmentations["vep_ch"]:
            mask = vep_ch
    return (signal * mask[:, None], mask[:, None]), Y * mask[:, None]

BUFFER_SIZE = 1024
BATCH_SIZE = 32


# AUTO = tf.data.AUTOTUNE

def prepare_data(X, Y, is_train=True):
    def numpy_generator(X, Y):
        num_examples, num_samples, _, _ = X[0].shape
        if not is_train:
            random.seed(1)
        for i in range(num_examples):
            x, y = data_augmentation((X[0][i], X[1][i]), Y[i])
            yield x, y

    dataset = tf.data.Dataset.from_generator(
        lambda: numpy_generator(X, Y), output_types=((tf.float32, tf.bool), tf.float32),
        output_shapes=(
        (tf.TensorShape([X[0].shape[1], 64, 1]), tf.TensorShape([64, 1])), tf.TensorShape([X[0].shape[1], 64, 1])))

    dataset = dataset.batch(BATCH_SIZE).map(lambda x, y: (x, y), num_parallel_calls=None)

    return dataset.prefetch(BUFFER_SIZE)

if __name__=='__main__':
    signal = np.random.random((1500,64))
    mask = np.ones(64)
    a,b = data_augmentation([signal[:,:,None],mask],signal[:,:,None])