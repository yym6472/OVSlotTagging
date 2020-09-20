from typing import Optional, Iterator, List, Dict

import os
import json
import numpy

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, ArrayField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token


@DatasetReader.register("multi_file")
class MultiFileDatasetReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 open_vocabulary_slots: List[str] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        # default open_vocabulary slots: for SNIPS dataset
        self.open_vocabulary_slots = open_vocabulary_slots or ["playlist", "entity_name", "poi",
            "restaurant_name", "geographic_poi", "album", "track", "object_name", "movie_name"]
    
    def text_to_instance(self, tokens: List[str], slots: List[str] = None) -> Instance:
        sentence_field = TextField([Token(token) for token in tokens], self.token_indexers)
        fields = {"sentence": sentence_field}
        if slots:
            slot_label_field = SequenceLabelField(labels=slots, sequence_field=sentence_field)
            fields["slot_labels"] = slot_label_field
            open_vocabulary_mask = [1 if any(
                slot.endswith(ov_slot) for ov_slot in self.open_vocabulary_slots) else 0
                for slot in slots]
            ov_slot_mask_field = ArrayField(numpy.array(open_vocabulary_mask, dtype=numpy.long),
                                            dtype=numpy.long)
            fields["ov_slot_mask"] = ov_slot_mask_field
            slot_mask = [1 if slot != "O" else 0 for slot in slots]
            slot_mask_field = ArrayField(numpy.array(slot_mask, dtype=numpy.long), dtype=numpy.long)
            fields["slot_mask"] = slot_mask_field
        return Instance(fields)
    
    def _read(self, file_path: str) -> Iterator[Instance]:
        token_file_path = os.path.join(file_path, "seq.in")
        label_file_path = os.path.join(file_path, "seq.out")
        with open(token_file_path, "r", encoding="utf-8") as f_token:
            token_lines = f_token.readlines()
        with open(label_file_path, "r", encoding="utf-8") as f_label:
            label_lines = f_label.readlines()
        assert len(token_lines) == len(label_lines)
        for token_line, label_line in zip(token_lines, label_lines):
            if not token_line.strip() or not label_line.strip():
                continue
            tokens: List[str] = token_line.strip().split(" ")
            labels: List[str] = label_line.strip().split(" ")
            if len(tokens) == 0 or len(labels) == 0:
                continue
            tokens = [token.strip() for token in tokens if token.strip()]
            labels = [label.strip() for label in labels if label.strip()]
            assert len(tokens) == len(labels)
            yield self.text_to_instance(tokens, labels)