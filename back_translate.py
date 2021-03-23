"""Train and evaluate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensor2tensor.bin import t2t_decoder
from tensor2tensor.models import transformer

import decoding
import problems

registry = problems.registry

tf.flags.DEFINE_string(
    'from_problem', 
    'translate_vien_iwslt32k', 
    'Problem name for source to intermediate language translation.')
tf.flags.DEFINE_string(
    'to_problem', 
    'translate_envi_iwslt32k', 
    'Problem name for intermediate to source language translation.')
tf.flags.DEFINE_string(
    'from_data_dir', 
    'gs://vien-translation/data/translate_vien_iwslt32k', 
    'Data directory for source to intermediate language translation.')
tf.flags.DEFINE_string(
    'to_data_dir', 
    'gs://vien-translation/data/translate_envi_iwslt32k', 
    'Data directory for intermediate to source language translation.')
tf.flags.DEFINE_string(
    'from_ckpt', 
    'gs://vien-translation/checkpoints/translate_vien_iwslt32k_tiny/avg/', 
    'Pretrain checkpoint directory for source to intermediate language translation.')
tf.flags.DEFINE_string(
    'to_ckpt', 
    'gs://vien-translation/checkpoints/translate_envi_iwslt32k_tiny/avg/', 
    'Pretrain checkpoint directory for intermediate to source language translation.')
tf.flags.DEFINE_string(
    'paraphrase_from_file', 
    'test_input.vi', 
    'Input text file to paraphrase.')
tf.flags.DEFINE_string(
    'paraphrase_to_file', 
    'test_output.vi', 
    'Output text file to paraphrase.')
tf.flags.DEFINE_boolean(
    'backtranslate_interactively',
    False,
    'Whether to back-translate interactively.')

FLAGS = tf.flags.FLAGS
  


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)

  # Convert directory into checkpoints
  from_ckpt = FLAGS.from_ckpt
  to_ckpt = FLAGS.to_ckpt
  if tf.gfile.IsDirectory(FLAGS.from_ckpt):
    from_ckpt = tf.train.latest_checkpoint(FLAGS.from_ckpt)
  if tf.gfile.IsDirectory(FLAGS.to_ckpt):
    to_ckpt = tf.train.latest_checkpoint(FLAGS.to_ckpt)

  if FLAGS.backtranslate_interactively:
    decoding.backtranslate_interactively(
        FLAGS.from_problem, FLAGS.to_problem,
        FLAGS.from_data_dir, FLAGS.to_data_dir,
        FLAGS.from_ckpt, FLAGS.to_ckpt)
  else:
    # For back translation from file, we need a temporary file in the other language
    # before back-translating into the source language.
    tmp_file = os.path.join(
        '{}.tmp.txt'.format(FLAGS.paraphrase_from_file)
    )

    # Step 1: Translating from source language to the other language.
    decoding.t2t_decoder(FLAGS.from_problem, FLAGS.from_data_dir,
                         FLAGS.paraphrase_from_file, tmp_file,
                         from_ckpt)

    # Step 2: Translating from the other language (tmp_file) to source.
    decoding.t2t_decoder(FLAGS.to_problem, FLAGS.to_data_dir,
                         tmp_file, FLAGS.paraphrase_to_file,
                         to_ckpt)
