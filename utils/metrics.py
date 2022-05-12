import tensorflow as tf


def mrr_top_at(at=100):
    @tf.function
    def mrr_top(y_true, y_pred):
        top_k = tf.math.top_k(y_pred, k=at)
        rr = top_k.indices
        idx = tf.range(start=1, limit=at + 1, delta=1)
        y_true = tf.reshape(y_true, [-1, 1])
        y_true = tf.cast(y_true, tf.int32)
        ranking = tf.where(tf.math.equal(rr, y_true), idx, 0)
        ranking = tf.reduce_sum(ranking, axis=-1)
        ranking = tf.where(ranking > 0, 1 / ranking, 0)
        ranking = tf.reduce_mean(ranking)
        return ranking

    return mrr_top
