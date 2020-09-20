from typing import List, Optional
from overrides import overrides

import torch

from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.span_based_f1_measure import SpanBasedF1Measure, TAGS_TO_SPANS_FUNCTION_TYPE
from allennlp.training.metrics.metric import Metric


class OVSpecSpanBasedF1Measure(SpanBasedF1Measure):
    """
    Modified SpanBasedF1Measure to add open-vocabulary-slots-speicified metrics.
    """
    def __init__(self,
                 vocabulary: Vocabulary,
                 tag_namespace: str = "tags",
                 ignore_classes: List[str] = None,
                 label_encoding: Optional[str] = "BIO",
                 tags_to_spans_function: Optional[TAGS_TO_SPANS_FUNCTION_TYPE] = None,
                 open_vocabulary_slots: List[str] = None) -> None:
        super(OVSpecSpanBasedF1Measure, self).__init__(vocabulary=vocabulary,
                                                       tag_namespace=tag_namespace,
                                                       ignore_classes=ignore_classes,
                                                       label_encoding=label_encoding,
                                                       tags_to_spans_function=tags_to_spans_function)
        self._open_vocabulary_slots = open_vocabulary_slots or []
    
    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A Dict per label containing following the span based metrics:
        precision : float
        recall : float
        f1-measure : float

        Additionally, an ``overall`` key is included, which provides the precision,
        recall and f1-measure for all spans.

        (*) Additionally, for open-vocabulary slots and normal (not open-vocabulary)
        slots, an ``ov`` key and a ``normal`` key are included respectively, which
        provide the precision, recall and f1-measure for all open-vocabulary spans
        and all normal spans, respectively.
        """
        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}
        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(self._true_positives[tag],
                                                                  self._false_positives[tag],
                                                                  self._false_negatives[tag])
            precision_key = "precision" + "-" + tag
            recall_key = "recall" + "-" + tag
            f1_key = "f1-measure" + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(sum(self._true_positives.values()),
                                                              sum(self._false_positives.values()),
                                                              sum(self._false_negatives.values()))
        all_metrics["precision-overall"] = precision
        all_metrics["recall-overall"] = recall
        all_metrics["f1-measure-overall"] = f1_measure

        # (*) Compute the precision, recall and f1 for all open-vocabulary spans jointly.
        precision, recall, f1_measure = self._compute_metrics(
            sum(map(lambda x: x[1], filter(lambda x: x[0] in self._open_vocabulary_slots, self._true_positives.items()))),
            sum(map(lambda x: x[1], filter(lambda x: x[0] in self._open_vocabulary_slots, self._false_positives.items()))),
            sum(map(lambda x: x[1], filter(lambda x: x[0] in self._open_vocabulary_slots, self._false_negatives.items()))))
        all_metrics["precision-ov"] = precision
        all_metrics["recall-ov"] = recall
        all_metrics["f1-measure-ov"] = f1_measure

        # (*) Compute the precision, recall and f1 for all normal not (open-vocabulary) spans jointly.
        precision, recall, f1_measure = self._compute_metrics(
            sum(map(lambda x: x[1], filter(lambda x: x[0] not in self._open_vocabulary_slots, self._true_positives.items()))),
            sum(map(lambda x: x[1], filter(lambda x: x[0] not in self._open_vocabulary_slots, self._false_positives.items()))),
            sum(map(lambda x: x[1], filter(lambda x: x[0] not in self._open_vocabulary_slots, self._false_negatives.items()))))
        all_metrics["precision-normal"] = precision
        all_metrics["recall-normal"] = recall
        all_metrics["f1-measure-normal"] = f1_measure

        if reset:
            self.reset()
        return all_metrics