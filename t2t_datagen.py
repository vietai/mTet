"""Train and evaluate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensor2tensor.bin import t2t_datagen
import problems
import tensorflow.compat.v1 as tf


if __name__ == '__main__':
  tf.app.run(t2t_datagen.main)