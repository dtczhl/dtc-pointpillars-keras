from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, MaxPool1D


def make_pillar_feature_net():
    D = 9
    N = 100
    C = 64

    x = Input(shape=(N, D), name='input')
    conv_1 = Conv1D(filters=C, kernel_size=1, strides=1, activation='relu', name='conv_1')(x)
    bn_1 = BatchNormalization(name='bn_1')(conv_1)
    pool_1 = MaxPool1D(pool_size=100, strides=1, name='output')(bn_1)

    return Model(inputs=x, outputs=pool_1)


if __name__ == '__main__':

    pillar_feature_net_model = make_pillar_feature_net()
    pillar_feature_net_model.summary()
