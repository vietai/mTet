# MTet: Multi-domain Translation for English-Vietnamese


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mtet-multi-domain-translation-for-english-and/machine-translation-on-iwslt2015-english-1)](https://paperswithcode.com/sota/machine-translation-on-iwslt2015-english-1?p=mtet-multi-domain-translation-for-english-and)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mtet-multi-domain-translation-for-english-and/on-phomt)](https://paperswithcode.com/sota/on-phomt?p=mtet-multi-domain-translation-for-english-and)

### Release
- [v2.0 (Current)](https://github.com/vietai/mTet)
- [v1.0](https://github.com/vietai/SAT/releases/tag/v1.0)

### Introduction

We are excited to introduce a new larger and better quality Machine Translation dataset, **MTet**, which stands for **M**ulti-domain **T**ranslation for **E**nglish and Vie**T**namese. In our new release, we extend our previous dataset ([v1.0](https://github.com/vietai/SAT/releases/tag/v1.0)) by adding more high-quality English-Vietnamese sentence pairs on various domains. In addition, we also show our new larger Transformer models can achieve state-of-the-art results on multiple test sets.

<!-- **English to Vietnamese Translation (BLEU score)**

<img src="envi.png" alt="drawing" width="500"/>

**Vietnamese to English Translation (BLEU score)**

<img src="vien.png" alt="drawing" width="500"/> -->

Get data and model at [Google Cloud Storage](https://console.cloud.google.com/storage/browser/vietai_public/best_vi_translation/v2)

Visit our  📄 [paper](https://arxiv.org/abs/2210.05610) or 📝 [blog post](https://research.vietai.org/mtet/) for more details.

<br>

### HuggingFace 🤗

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


model_name = "VietAI/envit5-translation"
tokenizer = AutoTokenizer.from_pretrained(model_name)  
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

inputs = [
    "vi: VietAI là tổ chức phi lợi nhuận với sứ mệnh ươm mầm tài năng về trí tuệ nhân tạo và xây dựng một cộng đồng các chuyên gia trong lĩnh vực trí tuệ nhân tạo đẳng cấp quốc tế tại Việt Nam.",
    "vi: Theo báo cáo mới nhất của Linkedin về danh sách việc làm triển vọng với mức lương hấp dẫn năm 2020, các chức danh công việc liên quan đến AI như Chuyên gia AI (Artificial Intelligence Specialist), Kỹ sư ML (Machine Learning Engineer) đều xếp thứ hạng cao.",
    "en: Our teams aspire to make discoveries that impact everyone, and core to our approach is sharing our research and tools to fuel progress in the field.",
    "en: We're on a journey to advance and democratize artificial intelligence through open source and open science."
    ]

outputs = model.generate(tokenizer(inputs, return_tensors="pt", padding=True).input_ids.to('cuda'), max_length=512)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

# ['en: VietAI is a non-profit organization with the mission of nurturing artificial intelligence talents and building an international - class community of artificial intelligence experts in Vietnam.',
#  'en: According to the latest LinkedIn report on the 2020 list of attractive and promising jobs, AI - related job titles such as AI Specialist, ML Engineer and ML Engineer all rank high.',
#  'vi: Nhóm chúng tôi khao khát tạo ra những khám phá có ảnh hưởng đến mọi người, và cốt lõi trong cách tiếp cận của chúng tôi là chia sẻ nghiên cứu và công cụ để thúc đẩy sự tiến bộ trong lĩnh vực này.',
#  'vi: Chúng ta đang trên hành trình tiến bộ và dân chủ hoá trí tuệ nhân tạo thông qua mã nguồn mở và khoa học mở.']

```

## Results

![image](https://user-images.githubusercontent.com/44376091/195998681-5860e443-2071-4048-8a2b-873dcee14a72.png)

## Citation
```
@misc{https://doi.org/10.48550/arxiv.2210.05610,
  doi = {10.48550/ARXIV.2210.05610},
  url = {https://arxiv.org/abs/2210.05610},
  author = {Ngo, Chinh and Trinh, Trieu H. and Phan, Long and Tran, Hieu and Dang, Tai and Nguyen, Hieu and Nguyen, Minh and Luong, Minh-Thang},
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {MTet: Multi-domain Translation for English and Vietnamese},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```



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
    --output_dir=$path_to_folder_contains_checkpoints \
    --checkpoint_path=$path_to_checkpoint
</prev>

In [this colab](https://colab.research.google.com/drive/1LH4wO7LcrklrUGwaMdLlXpJJcu2opGK6?usp=sharing), we demonstrated how to run these three phases in the context of hosting data/model on Google Cloud Storage.

<br>

### Dataset

Our data contains roughly 4.2 million pairs of texts, ranging across multiple different domains such as medical publications, religious texts, engineering articles, literature, news, and poems. A more detail breakdown of our data is shown in the table below.

<table align="center">
<thead>
<tr>
<th></th>
<th>v1</th>
<th>v2 (MTet)</th>
</tr>
</thead>

<tbody>
<tr>
<td>Fictional Books</td>
<td style="text-align:right;">333,189</td>
<td style="text-align:right;">473,306</td>
</tr>

<tr>
<td>Legal Document</td>
<td style="text-align:right;">1,150,266</td>
<td style="text-align:right;">1,134,813</td>
</tr>

<tr>
<td>Medical Publication</td>
<td style="text-align:right;">5,861</td>
<td style="text-align:right;">13,410</td>
</tr>

<tr>
<td>Movies Subtitles</td>
<td style="text-align:right;">250,000</td>
<td style="text-align:right;">721,174</td>
</tr>

<tr>
<td>Software</td>
<td style="text-align:right;">79,912</td>
<td style="text-align:right;">79,132</td>
</tr>

<tr>
<td>TED Talk</td>
<td style="text-align:right;">352,652</td>
<td style="text-align:right;">303,131</td>
</tr>

<tr>
<td>Wikipedia</td>
<td style="text-align:right;">645,326</td>
<td style="text-align:right;">1,094,248</td>
</tr>

<tr>
<td>News</td>
<td style="text-align:right;">18,449</td>
<td style="text-align:right;">18,389</td>
</tr>

<tr>
<td>Religious texts</td>
<td style="text-align:right;">124,389</td>
<td style="text-align:right;">48,927</td>
</tr>


<tr>
<td>Educational content</td>
<td style="text-align:right;">397,008</td>
<td style="text-align:right;">213,284</td>
</tr>


<tr>
<td>No tag</td>
<td style="text-align:right;">5,517</td>
<td style="text-align:right;">63,896</td>
</tr>

<tr>
<td>Total</td>
<td style="text-align:right;">3,362,569</td>
<td style="text-align:right;">4,163,710</td>
</tr>


</table>

</br>

Data sources is described in more details [here](https://github.com/vietai/SAT/blob/main/data_distribution.txt).

### Acknowledgment
We would like to thank Google for the support of Cloud credits and TPU quota!
