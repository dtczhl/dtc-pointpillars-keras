from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, MaxPool1D, Conv2D, Deconv2D, Concatenate, Dense

D = 9
N = 100
C = 64

def make_pillar_feature_net():
    x = Input(shape=(N, D), name='input')
    conv_1 = Conv1D(filters=C, kernel_size=1, strides=1, activation='relu', name='conv_1')(x)
    bn_1 = BatchNormalization(name='bn_1')(conv_1)
    pool_1 = MaxPool1D(pool_size=100, strides=1, name='output')(bn_1)

    return Model(inputs=x, outputs=pool_1)


def make_backbone(width, height, feature_C):
    x = Input(shape=(width, height, feature_C), name='input')
    conv_1 = Conv2D(C, 3, strides=2, padding='same', activation='relu', name='conv_1')(x)
    bn_conv_1 = BatchNormalization(name='bn_conv_1')(conv_1)
    conv_2 = Conv2D(2*C, 3, strides=2, padding='same', activation='relu', name='conv_2')(bn_conv_1)
    bn_conv_2 = BatchNormalization(name='bn_conv_2')(conv_2)
    conv_3 = Conv2D(4*C, 3, strides=2, padding='same', activation='relu', name='conv_3')(bn_conv_2)
    bn_conv_3 = BatchNormalization(name='bn_conv_3')(conv_3)

    deconv_1 = Deconv2D(2 * C, 3, strides=1, padding='same', activation='relu', name='deconv_1')(bn_conv_1)
    bn_deconv_1 = BatchNormalization(name='bn_deconv_1')(deconv_1)

    deconv_2 = Deconv2D(2 * C, 3, strides=2, padding='same', activation='relu', name='deconv_2')(bn_conv_2)
    bn_deconv_2 = BatchNormalization(name='bn_deconv_2')(deconv_2)

    deconv_3 = Deconv2D(2 * C, 3, strides=4, padding='same', activation='relu', name='deconv_3')(bn_conv_3)
    bn_deconv_3 = BatchNormalization(name='bn_deconv_3')(deconv_3)

    concat_1 = Concatenate(name='concat_1')([bn_deconv_1, bn_deconv_2, bn_deconv_3])

    return Model(inputs=x, outputs=concat_1)


if __name__ == '__main__':

    # pillar_feature_net_model = make_pillar_feature_net()
    # pillar_feature_net_model.summary()

    backbone_model = make_backbone(120, 80, 64)
    backbone_model.summary()


