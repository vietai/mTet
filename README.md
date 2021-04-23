# SAT

## Style Augmented Translation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/better-translation-for-vietnamese/machine-translation-on-iwslt2015-english-1)](https://paperswithcode.com/sota/machine-translation-on-iwslt2015-english-1?p=better-translation-for-vietnamese)


### Introduction

By collecting high-quality data, we were able to train a model that outperforms Google Translate on 6 different domains of English-Vietnamese Translation. 

**English to Vietnamese Translation (BLEU score)**

<img src="envi.png" alt="drawing" width="500"/>

**Vietnamese to English Translation (BLEU score)**

<img src="vien.png" alt="drawing" width="500"/>

Get data and model at [Google Cloud Storage](https://console.cloud.google.com/storage/browser/vietai_public/best_vi_translation)

Check out our [demo web app](https://demo.vietai.org/)

Visit our [blog post](https://blog.vietai.org/sat/) for more details.

<br>

### Using the code
This code is build on top of [vietai/dab](https://github.com/vietai/dab):

To prepare for training, generate `tfrecords` from raw text files:

<prev>

    python t2t_datagen.py \
    --data_dir=$path_to_folder_contains_vocab_file \
    --tmp_dir=$path_to_folder_that_contains_training_data \
    --problem=$problem
</prev>

To train a Transformer model on the generated `tfrecords`

<prev>

    python t2t_trainer.py \
    --data_dir=$path_to_folder_contains_vocab_file_and_tf_records \
    --problem=$problem \
    --hparams_set=$hparams_set \
    --model=transformer \
    --output_dir=$path_to_folder_to_save_checkpoints
</prev>

To run inference on the trained model:

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

Our data contains roughly 3.3 million pairs of texts. After augmentation, the data is of size 26.7 million pairs of texts. A more detail breakdown of our data is shown in the table below.

<table align="center">
<thead>
<tr>
<th></th>
<th>Pure</th>
<th>Augmented</th>
</tr>
</thead>

<tbody>
<tr>
<td>Fictional Books</td>
<td style="text-align:right;">333,189</td>
<td style="text-align:right;">2,516,787</td>
</tr>

<tr>
<td>Legal Document</td>
<td style="text-align:right;">1,150,266</td>
<td style="text-align:right;">3,450,801</td>
</tr>

<tr>
<td>Medical Publication</td>
<td style="text-align:right;">5,861</td>
<td style="text-align:right;">27,588</td>
</tr>

<tr>
<td>Movies Subtitles</td>
<td style="text-align:right;">250,000</td>
<td style="text-align:right;">3,698,046</td>
</tr>

<tr>
<td>Software</td>
<td style="text-align:right;">79,912</td>
<td style="text-align:right;">239,745</td>
</tr>

<tr>
<td>TED Talk</td>
<td style="text-align:right;">352,652</td>
<td style="text-align:right;">4,983,294</td>
</tr>

<tr>
<td>Wikipedia</td>
<td style="text-align:right;">645,326</td>
<td style="text-align:right;">1,935,981</td>
</tr>

<tr>
<td>News</td>
<td style="text-align:right;">18,449</td>
<td style="text-align:right;">139,341</td>
</tr>

<tr>
<td>Religious texts</td>
<td style="text-align:right;">124,389</td>
<td style="text-align:right;">1,182,726</td>
</tr>


<tr>
<td>Educational content</td>
<td style="text-align:right;">397,008</td>
<td style="text-align:right;">8,475,342</td>
</tr>


<tr>
<td>No tag</td>
<td style="text-align:right;">5,517</td>
<td style="text-align:right;">66,299</td>
</tr>

<tr>
<td>Total</td>
<td style="text-align:right;">3,362,569</td>
<td style="text-align:right;">26,715,950</td>
</tr>


</table>

</br>

Data sources is described in more details [here](https://github.com/vietai/SAT/blob/main/scrape_sources.txt).

