"""Decode."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensor2tensor.bin import t2t_decoder
from tensor2tensor.models import transformer
import problems
import tensorflow as tf
import decoding
from tensor2tensor.utils import registry

flags = tf.flags
FLAGS = flags.FLAGS

@registry.register_hparams
def transformer_tall9():
  hparams = transformer.transformer_big()
  hparams.hidden_size = 768
  hparams.filter_size = 3072
  hparams.num_hidden_layers = 9
  hparams.num_heads = 12
  return hparams


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  # tf.app.run(t2t_decoder.main)
  decoding.t2t_decoder(
      FLAGS.problem, 
      FLAGS.data_dir, 
      FLAGS.decode_from_file, 
      FLAGS.decode_to_file,
      FLAGS.checkpoint_path or FLAGS.output_dir)
