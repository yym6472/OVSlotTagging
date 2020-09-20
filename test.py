import os
import json
import tqdm
import argparse
from typing import Any, Union, Dict, Iterable, List, Optional, Tuple

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from models import SlotTaggingModel
from predictors import SlotTaggingPredictor
from dataset_readers import MultiFileDatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedBertIndexer

from allennlp.data import vocabulary
vocabulary.DEFAULT_OOV_TOKEN = "[UNK]"  # set for bert


def main(args):
    archive = load_archive(args.output_dir)
    predictor = Predictor.from_archive(archive=archive, predictor_name="slot_tagging_predictor")
    while True:
        try:
            input_text = input("Input sentence: ")
            if not input_text.strip():
                continue
            output_dict = predictor.predict({"tokens": input_text.split()})
            print(f"Predicted labels: {' '.join(output_dict['predict_labels'])}\n")
        except KeyboardInterrupt:
            print("Exited.")
            break


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--output_dir", type=str, default="./output/bert-large-atis/",
                            help="the directory that stores training output")
    args = arg_parser.parse_args()
    main(args)