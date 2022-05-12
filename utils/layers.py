import tensorflow as tf
from tensorflow.keras import layers


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(
            self,
            maxlen,
            vocab_size,
            embed_dim,
            item_embedding_trainable=True,
            embedding_weights=None
    ):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            trainable=item_embedding_trainable,
            weights=embedding_weights
        )
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, inputs):
        maxlen = tf.shape(inputs)[-1]
        positions = tf.range(start=maxlen - 1, limit=0, delta=-1)
        positions = self.pos_emb(positions)
        inputs = self.token_emb(inputs)
        return inputs + positions


class PositionEmbedding(layers.Layer):
    def __init__(
            self,
            maxlen,
            embed_dim,
    ):
        super(PositionEmbedding, self).__init__()
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, inputs):
        maxlen = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return positions


class RBFLayer(layers.Layer):
    def __init__(self, weights, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        # shape=(embedding_dimension, num_units)
        self.embedding_weights = tf.transpose(weights)
        self.gamma = gamma

    def call(self, inputs):
        l2 = tf.math.reduce_sum(
            tf.math.squared_difference(
                x=tf.expand_dims(inputs, axis=-1),
                y=self.embedding_weights
            ),
            axis=1
        )
        res = tf.math.exp(-self.gamma * l2)
        return res / tf.reduce_sum(res, axis=1, keepdims=True)
