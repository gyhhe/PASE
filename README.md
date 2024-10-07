# Progressive Semantic Aggregation and Structured Cognitive Enhancement for Image-Text Matching

![Static Badge](https://img.shields.io/badge/Pytorch-EE4C2C)
![License: MIT](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)

The codes for our paper "Progressive Semantic Aggregation and Structured Cognitive Enhancement for Image-Text Matching(PASE)".
We referred to the implementations of [GPO](https://github.com/woodfrog/vse_infty), [ESL](https://github.com/CrossmodalGroup/ESL), and [SVSE](https://github.com/zengzhixian/SoftPool_SVSE) to build up our codes. We extend our gratitude to these awesome works.  

**Note**: The complete codebase will be made public upon acceptance of our paper.

## Introduction


![Overview](https://github.com/gyhhe/PASE/blob/main/model_overview.png)



## Performance

Our method achieves state-of-the-art results on standard benchmarks:

![tab1](https://github.com/gyhhe/PASE/blob/main/tab1.png)

![tab2](https://github.com/gyhhe/PASE/blob/main/tab2.png)


## Preparation

### Environments

We recommended the following dependencies.

- Python 3.9
- [PyTorch](http://pytorch.org/) 1.11
- transformers  4.36.0
- numpy 1.23.5
- nltk 3.7
- The specific required environment can be found [here](https://github.com/Image-Text-Matching/AAHR/blob/main/requirements.txt)


### Data

All data sets used in the experiment and the necessary external components are organized in the following manner:

```
data
├── coco
│   ├── precomp  # pre-computed BUTD region features for COCO, provided by SCAN
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── train_caps_coco_entity_triples.txt
│   │      ├── train_caps_coco_subjects_triples.txt
│   │      ├── train_caps_coco_relations_triples.txt
│   │      ├── train_caps_coco_objects_triples.txt
│   │      ├── ......
│   
├── f30k
│   ├── precomp  # pre-computed BUTD region features for Flickr30K, provided by SCAN
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── train_caps_f30k_entity_triples.txt
│   │      ├── train_caps_f30k_subjects_triples.txt
│   │      ├── train_caps_f30k_relations_triples.txt
│   │      ├── train_caps_f30k_objects_triples.txt
│   │      ├── ......
│   │
│   
└── vocab  # vocab files provided by SCAN (only used when the text backbone is BiGRU)

PASE
├── bert-base-uncased    # the pretrained checkpoint files for BERT-base
│   ├── config.json
│   ├── tokenizer_config.txt
│   ├── vocab.txt
│   ├── pytorch_model.bin
│   ├── ......
│  
└── ....

```

#### Data Sources:

- BUTD features: [SCAN (Kaggle)](https://www.kaggle.com/datasets/kuanghueilee/scan-features) or [Baidu Yun](https://pan.baidu.com/s/1Dmnf0q9J29m4-fyL7ubqdg?pwd=AAHR)

## Training

Train MSCOCO and Flickr30K from scratch:

```
bash  run_f30k.sh
```

```
bash  run_coco.sh
```

## Evaluation

Modify the corresponding parameters in eval.py to test the Flickr30K or MSCOCO data set:

```
python eval.py  --dataset f30k  --data_path "path/to/dataset"
```

```
python eval.py  --dataset coco --data_path "path/to/dataset"
```

##  Citation

