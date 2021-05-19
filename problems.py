"""Translation datasets augmented with append and style-tagging."""

from __future__ import print_function
from __future__ import division

from tensor2tensor.data_generators import translate_envi
from tensor2tensor.utils import registry

# ----- Problem for augmented Data

@registry.register_problem
class TranslateClass11AppendtagVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    # This error will not be raised unless the t2t_datagen with this problem is called.
    raise NotImplementedError('Please specify append and style-tagging data augmented files for training.')


@registry.register_problem
class TranslateClass11AppendtagEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    # This error will not be raised unless the t2t_datagen with this problem is called.
    raise NotImplementedError('Please specify append and style-tagging data augmented files for training.')


# ----- Problem for original Data

@registry.register_problem
class TranslateClass11PureVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return ['', ('train.vi', 'train.en')]] if train else [['',('tst2012.vi', 'tst2012.en')]]


@registry.register_problem
class TranslateClass11PureEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return ['', ('train.en', 'train.vi')]] if train else [['',('tst2012.en', 'tst2012.vi')]]
