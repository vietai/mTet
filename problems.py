"""Translation datasets augmented with append and style-tagging."""

from __future__ import print_function
from __future__ import division

from tensor2tensor.data_generators import translate_envi
from tensor2tensor.utils import registry

# ----- Problem for MTet

@registry.register_problem
class TranslateVienMtet32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for MTet En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return [['', ('train.vi', 'train.en')]] if train else [['',('dev.vi', 'dev.en')]]


@registry.register_problem
class TranslateEnviMtet32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for MTet En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return [['', ('train.en', 'train.vi')]] if train else [['',('dev.en', 'dev.vi')]]
