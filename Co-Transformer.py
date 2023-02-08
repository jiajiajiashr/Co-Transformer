"""Co-Transformer"""

import tensorflow as tf
from crfnet.detr_models.transformer.utils import scaled_dot_product_attention
import math
#tf.keras.backend.set_floatx("float32")
#import pickle
import ipdb  # noqa: F401
import numpy as np
import keras
from keras_layer_normalization import LayerNormalization
from keras.layers import Lambda

# from __future__ import print_function
from keras import backend as K
from keras.engine.topology import Layer
from keras import backend as K
K.clear_session()

class Position_Embedding(keras.layers.Layer):

    def __init__(self, size=None, **kwargs):
        self.size = size  # 必须为偶数
        super(Position_Embedding, self).__init__(**kwargs)


        self.dim = 256


    def call(self, x):
        self.size = int(x.shape[-1])
        height = int(x.shape[1])
        width = int(x.shape[2])
        if self.size == 64:
            self.dim =32
        if self.size == 128:
            self.dim =64
        if self.size == 256:
            self.dim =128
        if self.size == 512:
            self.dim = 256
        if self.size == 1024:
            self.dim = 512
        if self.size == 2048:
            self.dim = 1024
        # c = int(fm_shape.shape[3])
        print('x:', x)

        y_embed = np.repeat(np.arange(1, height + 1), width).reshape(height, width)
        x_embed = np.full(shape=(height, width), fill_value=np.arange(1, width + 1))
        # d/2 entries for each dimension x and y

        div_term = np.arange(self.dim)
        div_term = 10000 ** (2 * (div_term // 2) / self.dim)
        pos_x = x_embed[:, :, None] / div_term
        pos_y = y_embed[:, :, None] / div_term

        pos_x_even = np.sin(pos_x[:, :, 0::2])
        pos_x_uneven = np.sin(pos_x[:, :, 1::2])

        pos_y_even = np.sin(pos_y[:, :, 0::2])
        pos_y_uneven = np.sin(pos_y[:, :, 1::2])

        pos_x = np.concatenate((pos_x_even, pos_x_uneven), axis=2)
        pos_y = np.concatenate((pos_y_even, pos_y_uneven), axis=2)

        positional_encodings = np.concatenate((pos_y, pos_x), axis=2)
        positional_encodings = np.expand_dims(positional_encodings, 0)
        positional_encodings = keras.backend.cast(positional_encodings, 'float32')


        return positional_encodings

    def compute_output_shape(self, input_shape):
        return input_shape





class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, dim_transformer,name ='MultiHeadAttention', **kwargs):
        super(MultiHeadAttention, self).__init__(name = name, **kwargs)

        # self.num_heads = num_heads
        self.dim_transformer = int(dim_transformer)
        self.pos = Position_Embedding(self.dim_transformer)

        # self.depth = dim_transformer // self.num_heads

        # self.wq = keras.layers.Dense(self.dim_transformer)
        # self.wk = keras.layers.Dense(self.dim_transformer)
        # self.wv = keras.layers.Dense(self.dim_transformer)
        #
        # self.dense = keras.layers.Dense(self.dim_transformer)


    def __call__(self, src1,src2,name,trainable=True):
        src1_positional_encodings = self.pos(src1)
        src2_positional_encodings = self.pos(src2)


        k_1 = keras.layers.add([src1, src1_positional_encodings])
        v_1 = keras.layers.add([src1, src1_positional_encodings])
        q_1 = keras.layers.add([src1, src1_positional_encodings])
        q_2 = keras.layers.add([src2, src2_positional_encodings])
        # v, k, q = args
        if q_2.shape[-1] == 64:
            dim = 8
        if q_2.shape[-1] == 128:
            dim = 16
        if q_2.shape[-1] == 256:
            dim = 32
        if q_2.shape[-1] == 512:
            dim = 64
        if q_2.shape[-1] == 1024:
            dim = 128
        if q_2.shape[-1] == 2048:
            dim = 256

        q2 = keras.layers.Dense(self.dim_transformer)(q_2)
        q1 = keras.layers.Dense(self.dim_transformer)(q_1)
        # print('_____________q:', q)
        k = keras.layers.Dense(self.dim_transformer)(k_1)

        v = keras.layers.Dense(self.dim_transformer)(v_1)


        if 'decoder' in name:
            q2 = keras.layers.Reshape((q2.shape[1].value * q2.shape[2].value, 8, dim))(q2)


            k = keras.layers.Reshape((k.shape[1].value * k.shape[2].value, 8, dim))(k)

            V_co = keras.layers.Reshape((v.shape[1].value * v.shape[2].value, 8, dim))(v)

            q_t = Lambda(Permute_dimensions)(q2)
            k_t = Lambda(Permute_dimensions_1)(k)
            v_co = Lambda(Permute_dimensions)(V_co)

            matmul_k = Lambda(Batch_D)([q_t, k_t])

            # scale matmul_qk
            dk = Lambda(Cast)(k_t)
            scaled_attention = Lambda(Divide)([matmul_k, Lambda(keras.backend.sqrt)(dk)])


        elif 'radar_encoder' in name:
        # else:
            q_s_2 = keras.layers.Reshape((q2.shape[1].value * q2.shape[2].value, 8, dim))(q2)

            k_co_1 = Lambda(lambda k: k[:, :, :, :4*dim])(k)
            k_co_2 = Lambda(lambda k: k[:, :, :, 4 * dim:])(k)



            k_conv_r = keras.layers.Conv2D(int(k_co_1.shape[-1]), (3, 3), padding='same', strides=(1, 1))(k_co_1)
            k_conv = keras.layers.Concatenate(axis=-1)([k_conv_r, k_co_2])
            k_s_2 = keras.layers.Reshape((k.shape[1].value * k.shape[2].value, -1, dim))(k_conv)



            V_co = keras.layers.Reshape((v.shape[1].value * v.shape[2].value, 8, dim))(v)

            q_p_2 = Lambda(Permute_dimensions)(q_s_2)
            k_p_2 = Lambda(Permute_dimensions_1)(k_s_2)
            v_co = Lambda(Permute_dimensions)(V_co)

            matmul_co = Lambda(Batch_D)([q_p_2, k_p_2])

            # scale matmul_qk

            dk_co = Lambda(Cast)(k_p_2)
            scaled_attention = Lambda(Divide)([matmul_co, Lambda(keras.backend.sqrt)(dk_co)])
        else:
            Q1 = keras.layers.Reshape((q1.shape[1].value * q1.shape[2].value, 8, dim))(q1)
            q_self_1 = Lambda(lambda Q1: Q1[:, :, :4, :])(Q1)
            q_self_2 = Lambda(lambda Q1: Q1[:, :, 4:, :])(Q1)
            q_s_1 = keras.layers.Add()([q_self_1,q_self_2])

            Q2 = keras.layers.Reshape((q2.shape[1].value * q2.shape[2].value,8,dim))(q2)
            q2_1 = Lambda(lambda Q2: Q2[:, :, :4, :])(Q2)
            q2_2 = Lambda(lambda Q2: Q2[:, :, 4:, :])(Q2)
            q_s_2 = keras.layers.Add()([q2_1,q2_2])

            k_s = Lambda(lambda k: k[:, :, :, :4*dim])(k)
            k_co_1 = Lambda(lambda k: k[:, :, :, 4*dim:6*dim])(k)
            k_co_2 = Lambda(lambda k: k[:, :, :, 6*dim:])(k)

            k_s_1 = keras.layers.Reshape((k.shape[1].value * k.shape[2].value,-1,dim))(k_s)
            k_conv_r = keras.layers.Conv2D(int(k_co_1.shape[-1]), (3, 3), padding='same', strides=(1, 1))(k_co_1)



            k_conv = keras.layers.Concatenate(axis=-1)([k_conv_r, k_co_2])
            k_s_2 = keras.layers.Reshape((k.shape[1].value * k.shape[2].value, -1, dim))(k_conv)


            V_co = keras.layers.Reshape((v.shape[1].value * v.shape[2].value,8,dim))(v)

            q_p_1 = Lambda(Permute_dimensions)(q_s_1)
            q_p_2 = Lambda(Permute_dimensions)(q_s_2)
            k_p_1 = Lambda(Permute_dimensions_1)(k_s_1)
            k_p_2 = Lambda(Permute_dimensions_1)(k_s_2)
            v_co = Lambda(Permute_dimensions)(V_co)

            matmul_self = Lambda(Batch_D)([q_p_1, k_p_1])
            matmul_coco = Lambda(Batch_D)([q_p_2, k_p_2])




            # scale matmul_qk
            dk_self = Lambda(Cast)(k_p_1)
            dk_co = Lambda(Cast)(k_p_2)
            scaled_attention_logits_self = Lambda(Divide)([matmul_self, Lambda(keras.backend.sqrt)(dk_self)])
            scaled_attention_logits_co = Lambda(Divide)([matmul_coco, Lambda(keras.backend.sqrt)(dk_co)])

            # scaled_attention = keras.layers.Concatenate(axis=1)([scaled_attention_logits_self, scaled_attention_logits_co])
            scaled_attention = keras.layers.Concatenate(axis=1)([scaled_attention_logits_co, scaled_attention_logits_self])



        # attention_weights = Lambda(SoftMax)(scaled_attention_logits)
        attention_weights = keras.layers.Softmax(axis=-1)(scaled_attention)

        # output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        output = Lambda(Batch_D)([attention_weights, v_co])

        concat_attention = Lambda(Permute_dimensions)(output)
        concat_attention_1 = keras.layers.Reshape((concat_attention.shape[1], self.dim_transformer))(concat_attention)
        # concat_attention = Lambda(
        #     lambda concat_attention: tf.reshape(concat_attention, (-1, concat_attention.shape[1], self.dim_transformer)))(
        #     concat_attention)
        concat_attention = keras.layers.Dense(self.dim_transformer)(concat_attention_1)

        return concat_attention









def Permute_dimensions(input):
    output = keras.backend.permute_dimensions(input, (0, 2, 1, 3))
    return output

def Divide(inputs):
    a, b = inputs
    output = a/b
    return output

def Cast(inputs):
    output = keras.backend.cast(inputs.shape[-2], dtype='float32')
    return output
def Permute_dimensions_1(input):
    output = keras.backend.permute_dimensions(input, (0, 2, 3, 1))
    return output

def Batch_D(inputs):
    a, b = inputs
    output = keras.backend.batch_dot(a, b)
    return output




class BiTransformerEncoder(keras.layers.Layer):
    def __init__(
        self, layers, dim_transformer, dim_feedforward=1024, dropout=0.1, name = 'radar_encoder', **kwargs
    ):
        super(BiTransformerEncoder, self).__init__(name = name, **kwargs)
#        self.num_layers = num_layers
        self.dim_transformer = int(dim_transformer)
#        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.layers = layers
        # self.enc_layers = BiTransformerEncoderLayer(
        #         self.dim_transformer, self.dim_feedforward, self.dropout
        #     )
        self.enc_layers = [
            BiTransformerEncoderLayer(
                self.dim_transformer, self.dim_feedforward, self.dropout
            )
            for _ in range(self.layers)
        ]




    def __call__(self,enc_output1, enc_output2,name = 'radar_encoder',trainable=True):
        for i in range(self.layers):
            enc_output2 = self.enc_layers[i](enc_output1, enc_output2,name = 'radar_encoder')

        return enc_output2








class BiTransformerEncoderLayer(keras.layers.Layer):
    def __init__(self, dim_transformer, dim_feedforward, dropout=0.1, name = 'BiTransformerEncoderLayer', **kwargs):
        super(BiTransformerEncoderLayer, self).__init__(name = name, **kwargs)
        self.dim_transformer = int(dim_transformer)
#        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout


        self.linear1 = keras.layers.Dense(self.dim_feedforward, activation="relu")
        self.   linear2 = keras.layers.Dense(self.dim_transformer)

        self.dropout1 = keras.layers.Dropout(self.dropout)
        self.dropout2 = keras.layers.Dropout(self.dropout)
        self.dropout3 = keras.layers.Dropout(self.dropout)

        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.attention = MultiHeadAttention(self.dim_transformer)
        self.pos = Position_Embedding(self.dim_transformer)




    def __call__(self, inp1, inp2, name,trainable=True):



        src1 = keras.layers.Reshape((inp1.shape[1].value * inp1.shape[2].value, inp1.shape[3].value))(inp1)

        R_output1  = self.attention(inp1, inp2,name = name)
        R_output = self.dropout1(R_output1)
        out = keras.layers.add([src1, R_output])

        out1 = self.norm1(out)
        ffn_1 = self.linear1(out1)
        ffn_2 = self.dropout2(ffn_1)
        ffn_output1 = self.linear2(ffn_2)

        ffn_output1 = self.dropout3(ffn_output1)
        out1 = keras.backend.cast(out1, dtype='float32')
        ffn_output1 = keras.backend.cast(ffn_output1, dtype=tf.float32)
        out2 = keras.layers.add([out1, ffn_output1])
        out3 = self.norm2(out2)

        x = keras.layers.Reshape((inp2.shape[1].value, inp2.shape[2].value, inp2.shape[3].value))(out3)

        return x




class BiTransformerEncoder_Decoder(keras.layers.Layer):
    def __init__(
        self, layers, dim_transformer, dim_feedforward=1024, dropout=0.1, name = 'image_encoder', **kwargs
    ):
        super(BiTransformerEncoder_Decoder, self).__init__(name = name, **kwargs)
#        self.num_layers = num_layers
        self.dim_transformer = int(dim_transformer)
#        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.layers = layers
        # self.enc_layers = BiTransformerEncoderLayer(
        #         self.dim_transformer, self.dim_feedforward, self.dropout
        #     )
        self.enc_layers = [
            BiTransformerEncoder_Decoder_Layer(
                self.dim_transformer, self.dim_feedforward, self.dropout,name = 'image_encoder'
            )
            for _ in range(self.layers)
        ]




    def __call__(self,enc_output1, enc_output2,name = 'image_encoder',trainable=True):
        for i in range(self.layers):
            enc_output2 = self.enc_layers[i](enc_output2,enc_output1, name = 'image_encoder')

        return enc_output2



class BiTransformerEncoder_Decoder_Layer(keras.layers.Layer):
    def __init__(self, dim_transformer, dim_feedforward, dropout=0.1, name = 'BiTransformerEncoder_Decoder_Layer', **kwargs):
        super(BiTransformerEncoder_Decoder_Layer, self).__init__(name = name, **kwargs)
        self.dim_transformer = int(dim_transformer)
#        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout


        self.linear1 = keras.layers.Dense(self.dim_feedforward, activation="relu")
        self.linear2 = keras.layers.Dense(self.dim_transformer)


        self.dropout1 = keras.layers.Dropout(self.dropout)
        self.dropout2 = keras.layers.Dropout(self.dropout)
        self.dropout3 = keras.layers.Dropout(self.dropout)

        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.attention = MultiHeadAttention(self.dim_transformer)
        self.pos = Position_Embedding(self.dim_transformer)




    def __call__(self, inp1, inp2, name,trainable=True):



        src1 = keras.layers.Reshape((inp1.shape[1].value * inp1.shape[2].value, inp1.shape[3].value))(inp1)

        R_output1  = self.attention(inp1, inp2,name = name)
        R_output = self.dropout1(R_output1)
        out = keras.layers.add([src1, R_output])

        out1 = self.norm1(out)
        ffn_1 = self.linear1(out1)
        ffn_2 = self.dropout2(ffn_1)
        ffn_output1 = self.linear2(ffn_2)

        ffn_output1 = self.dropout3(ffn_output1)
        out1 = keras.backend.cast(out1, dtype='float32')
        ffn_output1 = keras.backend.cast(ffn_output1, dtype=tf.float32)
        out2 = keras.layers.add([out1, ffn_output1])
        out3 = self.norm2(out2)

        x = keras.layers.Reshape((inp2.shape[1].value, inp2.shape[2].value, inp2.shape[3].value))(out3)

        return x





class BiTransformerDecoder(keras.layers.Layer):
    def  __init__(
        self, layers, dim_transformer,  dim_feedforward, dropout=0.1,name="BiTransformerDecoder", **kwargs
    ):
        super(BiTransformerDecoder, self).__init__(name = name, **kwargs)
        self.dim_transformer = int(dim_transformer)
#        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.layers = layers
        self.dec_layers = [
            BiTransformerDecoderLayer(
                self.dim_transformer, self.dim_feedforward, self.dropout
            )
            for _ in range(self.layers)
        ]



    def __call__(self, memory,dec_output,trainable=True):
        for i in range(self.layers):
            memory,dec_output  = self.dec_layers[i](
                memory,dec_output
            )

        return memory,dec_output





class BiTransformerDecoderLayer(keras.layers.Layer):
    def __init__(self, dim_transformer, dim_feedforward, dropout=0.1, name = 'BiTransformerDecoderLayer', **kwargs ):
        super(BiTransformerDecoderLayer, self).__init__(name = name, **kwargs)
        self.dim_transformer = int(dim_transformer)
#        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.attention = MultiHeadAttention(self.dim_transformer)


        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(
                    self.dim_feedforward, activation="relu"
                ),  # (batch_size, seq_len, dim_feedforward)
                keras.layers.Dropout(self.dropout),
                keras.layers.Dense(
                    self.dim_transformer
                ),  # (batch_size, seq_len, dim_transformer)
            ]
        )


        self.dropout1 = keras.layers.Dropout(self.dropout)
        self.dropout2 = keras.layers.Dropout(self.dropout)
        self.dropout3 = keras.layers.Dropout(self.dropout)
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.pos = Position_Embedding(self.dim_transformer)




    def __call__(self, inp1, inp2, trainable=True):

        tgt = keras.layers.Reshape(( inp2.shape[1].value * inp2.shape[2].value, -1))(inp2)





        C_output21 = self.attention(inp1, inp2,name='decoder')
        C_output2 = self.dropout2(C_output21)
        out= keras.layers.add([C_output2, tgt])
        C_output2= self.norm1(out)

        C_ffn_output = self.ffn(C_output2)
        C_ffn_output = self.dropout3(C_ffn_output)
        out5= keras.layers.add([C_ffn_output , C_output2])
        C_output2 = self.norm2(out5)
        y = keras.layers.Reshape(  (inp2.shape[1].value,inp2.shape[2].value, inp2.shape[3].value))(C_output2)


        return inp1,y









