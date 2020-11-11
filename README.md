# OVSlotTagging
Code for EMNLP 2020 paper: [Adversarial Semantic Decoupling for Recognizing Open-Vocabulary Slots](https://www.aclweb.org/anthology/2020.emnlp-main.490.pdf).

## Requirements
```
python==3.7.0
pytorch==1.3.1
allennlp==0.9.0
```

## BERT Pretrained Model
The BERT pretrained models is available at [https://huggingface.co/models](https://huggingface.co/models). To run experiments with BERT, you should download the *bert-large-uncased* model to local file system and modify the path (`"pretrained_model": "/local/path/to/bert-large-uncased/"`) in the config file to your downloading path.

## Dataset
The datasets we use are publicly available at:
- [Snips](https://github.com/MiuLab/SlotGated-SLU/tree/master/data/snips)
- [MIT-restaurant](https://groups.csail.mit.edu/sls/downloads/restaurant/)

We provided dataset in `/data` dir, together with the statistics including:
- number of samples for each slot type
- list of intents where the slot type appears
- example with longest slot value (in number of tokens)
- number of tokens for longest slot value
- list of slot values
- number of slot value types

The original splits for MIT-restaurant only contain train set (7660 samples) and test set (1521 samples). We further split the original train set into train (6894 samples) and valid (766 samples) sets.

## Commands

To train the model, run:
```
python3 train.py --config_path ./config/SETTING.json --output_dir ./outputs/SETTING/
```
where `SETTING` should be the specific setting in `/config` dir, e.g. `rnn-snips`, `bert-mr` or your customized configs.

To test the trained model, run:
```
python3 test.py --output_dir ./outputs/SETTING/
```
where `./outputs/SETTING/` should be the `output_dir` param same as that in the training command.

## Citation
If you find the code useful, please consider citing our paper:
```
@inproceedings{yan-etal-2020-adversarial,
    title = "Adversarial Semantic Decoupling for Recognizing Open-Vocabulary Slots",
    author = "Yan, Yuanmeng  and
      He, Keqing  and
      Xu, Hong  and
      Liu, Sihong  and
      Meng, Fanyu  and
      Hu, Min  and
      Xu, Weiran",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.490",
    pages = "6070--6075",
}
```
