# Assignment 4: GoEmotions Pytorch

NOTE: Our Assignemnt is split into 2 repositories. This repository is for the Transfer Learning Experiments using Goemotions data. You can find the repository for Semi_Supervised Learning using GAN-BERT architecture at https://github.com/SamarthMM/ganbert-pytorch

# Abstract

Data has become ubiquitous in the modern day and age. However, the challenge lies in acquiring truthfully annotated high quality data. In this paper, we try to look into the challenge of limited labeled training data availability for NLP sentiment analysis tasks. We talk about the Sentiment Analysis task and it's broader usage in different fields and on varied datasets. We perform an extensive literature survey on the various model architectures used for emotion classification. We perform transfer learning by using a BERT base cased model on GoEmotions dataset for zero shot and one-shot fine tuning on twitter dataset. We then investigate the viability of using a GAN model as a semi supervised technique to leverage the presence of unlabeled data.

This code has been adapted from [monologg's implementation of GoEmotions](https://github.com/monologg/GoEmotions-pytorch). The original README has been copied over to README_1.md
Pytorch Implementation of [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) with [Huggingface Transformers](https://github.com/huggingface/transformers)


### Requirements

- torch==1.4.0
- transformers==2.11.0
- attrdict==2.0.1

### Hyperparameters

You can change the parameters from the json files in `config` directory.

| Parameter         |      |
| ----------------- | ---: |
| Learning rate     | 5e-5 |
| Warmup proportion |  0.1 |
| Epochs            |   10 |
| Max Seq Length    |   50 |
| Batch size        |   16 |

## How to Run

### Baseline (Twitter):
```bash
python3 run_goemotions.py --taxonomy twitter
```

### Experiment 1:
For zero shot learning:

```bash
#First train model on GoEmotions group taxonomy
python3 run_goemotions.py --taxonomy group
#Then evaluate on twitter data
python3 Zero_Shot_Prediction.py --taxonomy twitter_zeroshot
```

### Experiment 2:
For one shot learning:

```bash
#Change "train_file": <"labeled_200.tsv"|"labeled_8000.tsv"> in config/twitter_frozenberg.json to allow training with 200 vs 8000 examples respectively
python3 Retrain_Goemotions_classifier_layer.py --taxonomy twitter_frozenbert 
```

## Plotting Graphs:
```bash
python3 Results.py --out_dir ckpt/<taxonomy>/<checkpoint directory> --taxonomy <name_of_plots>

#FOr example, the below command creates plots for accuracy and macro f1 score using the runs saved in the checkpoint directory 'ckpt/twitter/bert-base-cased-goemotions-twitter'. The name of the plots will start with 'twitter_unfrozen'. --taxonomy option in this command is just lazy naming and does not refer to the taxonomy used to generate the checkpoint results!
python3 Results.py --out_dir ckpt/twitter/bert-base-cased-goemotions-twitter --taxonomy twitter_unfrozen
```

## Reference
- [monologg's implementation of GoEmotions](https://github.com/monologg/GoEmotions-pytorch)
- [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)
- [GoEmotions Github](https://github.com/google-research/google-research/tree/master/goemotions)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
