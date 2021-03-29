# SAT

## Styled Augmented Translation

<br>

### Introduction

By collecting high-quality data, we were able to train a model that outperforms Google Translate on 6 different domains of English-Vietnamese Translation. 

Visit our [blog post](https://ntkchinh.github.io/) for more details.

<br>

### Using the code
This code is build on top of [vietai/dab](https://github.com/vietai/dab):

To generate `tfrecords` from raw text files:

<prev>

    python t2t_datagen.py \
    --data_dir=$path_to_folder_contains_vocab_file \
    --tmp_dir=$path_to_folder_that_contains_training_data \
    --problem=$problem
</prev>

To train your model on the generated `tfrecords`

<prev>

    python t2t_trainer.py \
    --data_dir=$path_to_folder_contains_vocab_file_and_tf_records \
    --problem=$problem \
    --hparams_set=$hparams_set \
    --model=transformer \
    --output_dir=$path_to_folder_to_save_checkpoints
</prev>

To run inference on the trained model.

<prev>

    python t2t_decoder.py \
    --data_dir=$path_to_folde_contains_vocab_file_and_tf_records \
    --problem=$problem \
    --hparams_set=$hparams_set \
    --model=transformer \
    --output_dir=$path_to_folder_contains_checkpoints
</prev>

In [this colab](https://colab.research.google.com/drive/1iYjm2E_iMb5qHfrdR5iQF_jq-BwC-DFM?usp=sharing), we demonstrated how to run these three phases in the context of hosting data/model on Google Cloud Storage.

<br>

### Dataset

Our data contains roughly 3.3 million pairs of texts. After augmentation, the data is of size 25 million pairs of texts. A more detail breakdown of our data is shown in the table below.

<table align="center">
<thead>
<tr>
<th></th>
<th>Pure Data</th>
<th>Augmented Data</th>
</tr>
</thead>

<tbody>
<tr>
<td>Fictional Books</td>
<td>333 189</td>
<td>2 516 787</td>
</tr>

<tr>
<td>Legal Document</td>
<td>1 150 266</td>
<td>3 450 801</td>
</tr>

<tr>
<td>Medical Publication</td>
<td>5 861</td>
<td>27 588</td>
</tr>

<tr>
<td>Movies Subtitles</td>
<td>250 000</td>
<td>3 698 046</td>
</tr>

<tr>
<td>Software</td>
<td>79 912</td>
<td>239 745</td>
</tr>

<tr>
<td>TED Talk</td>
<td>352 652</td>
<td>4 983 294</td>
</tr>

<tr>
<td>Wikipedia</td>
<td>645 326</td>
<td>1 935 981</td>
</tr>

<tr>
<td>News</td>
<td>18 449</td>
<td>139 341</td>
</tr>

<tr>
<td>Religious</td>
<td>124 389</td>
<td>1 182 726</td>
</tr>


<tr>
<td>Education</td>
<td>337 252</td>
<td>7 635 723</td>
</tr>


<tr>
<td>Others</td>
<td>65 273</td>
<td>905 918</td>
</tr>

</table>

</br>

Data sources is described in more details [here](https://github.com/vietai/SAT/blob/main/scrape_sources.txt).

