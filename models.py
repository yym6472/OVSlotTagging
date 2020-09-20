from typing import List, Dict, Optional

import torch
import logging

from allennlp.common import Params
from allennlp.models import Model
from allennlp.modules import TimeDistributed
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.conditional_random_field import ConditionalRandomField, allowed_transitions
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from metrics import OVSpecSpanBasedF1Measure


logger = logging.getLogger(__name__)


@Model.register("slot_tagging")
class SlotTaggingModel(Model):
    
    def __init__(self, 
                 vocab: Vocabulary,
                 bert_embedder: Optional[PretrainedBertEmbedder] = None,
                 encoder: Optional[Seq2SeqEncoder] = None,
                 dropout: Optional[float] = None,
                 use_crf: bool = True,
                 add_random_noise: bool = False,
                 add_attack_noise: bool = False,
                 do_noise_normalization: bool = True,
                 noise_norm: Optional[float] = None,
                 noise_loss_prob: Optional[float] = None,
                 add_noise_for: str = "ov",
                 rnn_after_embeddings: bool = False,
                 open_vocabulary_slots: Optional[List[str]] = None,
                 metrics_for_each_slot_type: bool = False) -> None:
        """
        Params
        ------
        vocab: the allennlp Vocabulary object, will be automatically passed
        bert_embedder: the pretrained BERT embedder. If it is not None, the pretrained BERT
                embedding (parameter fixed) will be used as the embedding layer. Otherwise, a look-up
                embedding matrix will be initialized with the embedding size 1024. The default is None.
        encoder: the contextual encoder used after the embedding layer. If set to None, no contextual
                encoder will be used.
        dropout: the dropout rate, won't be set in all our experiments.
        use_crf: if set to True, CRF will be used at the end of the model (as output layer). Otherwise,
                a softmax layer (with cross-entropy loss) will be used.
        add_random_noise: whether to add random noise to slots. Can not be set simultaneously 
                with add_attack_noise. This setting is used as baseline in our experiments.
        add_attack_noise: whether to add adversarial attack noise to slots. Can not be set simultaneously
                with add_random_noise.
        do_noise_normalization: if set to True, the normalization will be applied to gradients w.r.t. 
                token embeddings. Otherwise, the gradients won't be normalized.
        noise_norm: the normalization norm (L2) applied to gradients.
        noise_loss_prob: the alpha hyperparameter to balance the loss from normal forward and adversarial
                forward. See the paper for more details. Should be set from 0 to 1.
        add_noise_for: if set to ov, the noise will only be applied to open-vocabulary slots. Otherwise,
                the noise will be applied to all slots (both open-vocabulary and normal slots).
        rnn_after_embeddings: if set to True, an additional BiLSTM layer will be applied after the embedding
                layer. Default is False.
        open_vocabulary_slots: the list of open-vocabulary slots. If not set, will be set to open-vocabulary
                slots of Snips dataset by default.
        metrics_for_each_slot_type: whether to log metrics for each slot type. Default is False.
        """
        super().__init__(vocab)

        if bert_embedder:
            self.use_bert = True
            self.bert_embedder = bert_embedder
        else:
            self.use_bert = False
            self.basic_embedder = BasicTextFieldEmbedder({
                "tokens": Embedding(vocab.get_vocab_size(namespace="tokens"), 1024)
            })
            self.rnn_after_embeddings = rnn_after_embeddings
            if rnn_after_embeddings:
                self.rnn = Seq2SeqEncoder.from_params(Params({     
                    "type": "lstm",
                    "input_size": 1024,
                    "hidden_size": 512,
                    "bidirectional": True,
                    "batch_first": True
                }))

        self.encoder = encoder

        if encoder:
            hidden2tag_in_dim = encoder.get_output_dim()
        else:
            hidden2tag_in_dim = bert_embedder.get_output_dim()
        self.hidden2tag = TimeDistributed(torch.nn.Linear(
            in_features=hidden2tag_in_dim,
            out_features=vocab.get_vocab_size("labels")))
        
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        
        self.use_crf = use_crf
        if use_crf:
            crf_constraints = allowed_transitions(
                constraint_type="BIO",
                labels=vocab.get_index_to_token_vocabulary("labels")
            )
            self.crf = ConditionalRandomField(
                num_tags=vocab.get_vocab_size("labels"),
                constraints=crf_constraints,
                include_start_end_transitions=True
            )
        
        # default open_vocabulary slots: for SNIPS dataset
        open_vocabulary_slots = open_vocabulary_slots or ["playlist", "entity_name", "poi",
            "restaurant_name", "geographic_poi", "album", "track", "object_name", "movie_name"]
        self.f1 = OVSpecSpanBasedF1Measure(vocab, 
                                     tag_namespace="labels",
                                     ignore_classes=[],
                                     label_encoding="BIO",
                                     open_vocabulary_slots=open_vocabulary_slots)
        
        self.add_random_noise = add_random_noise
        self.add_attack_noise = add_attack_noise
        assert not (add_random_noise and add_attack_noise), "both random and attack noise applied"
        if add_random_noise or add_attack_noise:
            self.do_noise_normalization = do_noise_normalization
            assert noise_norm is not None
            assert noise_loss_prob is not None and 0. <= noise_loss_prob <= 1.
            self.noise_norm = noise_norm
            self.noise_loss_prob = noise_loss_prob
            assert add_noise_for in ["ov", "all"]
            self.ov_noise_only = (add_noise_for == "ov")
        
        self.metrics_for_each_slot_type = metrics_for_each_slot_type

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                slot_labels: torch.Tensor = None,
                ov_slot_mask: torch.Tensor = None,
                slot_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Params
        ------
        sentence: a Dict contains tensors of token ids (in "tokens" key) or (If use BERT as embedding
                layer) BERT BPE ids, offsets, segment ids. This parameter is the output of
                TextField.as_tensors(), see ~allennlp.data.fields.text_field.TextField for details.
                Each field should have shape (batch_size, seq_length)
        slot_labels: slot label ids (in BIO format), of shape (batch_size, seq_length)
        ov_slot_mask: binary mask, 1 for tokens of open-vocabulary slots, 0 for otherwise (non-slot tokens
                 and tokens of normal slots). Of shape (batch_size, seq_length)
        slot_mask: binary mask, 1 for tokens of slots (all slots), 0 for non-slot tokens (i.e. the O tag).
                Of shape (batch_size, seq_length)
        
        Return a Dict (str -> torch.Tensor), which contains fields:
                mask - the mask matrix of ``sentence``, shape: (batch_size, seq_length)
                embeddings - the embedded tokens, shape: (batch_size, seq_length, embed_size)
                encoder_out - the output of contextual encoder, shape: (batch_size, seq_length, num_features)
                tag_logits - the output of tag projection layer, shape: (batch_size, seq_length, num_tags)
                predicted_tags - the output of CRF layer (use viterbi algorithm to obtain best paths),
                             shape: (batch_size, seq_length)
        """
        output = {}

        mask = get_text_field_mask(sentence)
        output["mask"] = mask
        
        if self.use_bert:
            embeddings = self.bert_embedder(sentence["bert"], sentence["bert-offsets"], sentence["bert-type-ids"])
            if self.dropout:
                embeddings = self.dropout(embeddings)
            output["embeddings"] = embeddings
        else:
            embeddings = self.basic_embedder(sentence)
            if self.dropout:
                embeddings = self.dropout(embeddings)
            output["embeddings"] = embeddings
            if self.rnn_after_embeddings:
                embeddings = self.rnn(embeddings, mask)
                if self.dropout:
                    embeddings = self.dropout(embeddings)
                output["rnn_out"] = embeddings
        
        if not self.training:  # when predict or evaluate, no need for adding noise
            output.update(self._inner_forward(embeddings, mask, slot_labels))
        elif not self.add_random_noise and not self.add_attack_noise:  # for baseline
            output.update(self._inner_forward(embeddings, mask, slot_labels))
        else:  # add random noise or attack noise for open-vocabulary slots
            if self.add_random_noise:  # add random noise
                unnormalized_noise = torch.randn(embeddings.shape).to(device=embeddings.device)
            else:  # add attack noise
                normal_loss = self._inner_forward(embeddings, mask, slot_labels)["loss"]
                embeddings.retain_grad()  # we need to get gradient w.r.t embeddings
                normal_loss.backward(retain_graph=True)
                unnormalized_noise = embeddings.grad.detach_()
                for p in self.parameters():
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()
            if self.do_noise_normalization:  # do normalization
                norm = unnormalized_noise.norm(p=2, dim=-1)
                normalized_noise = unnormalized_noise / (norm.unsqueeze(dim=-1) + 1e-10)  # add 1e-10 to avoid NaN
            else:  # no normalization
                normalized_noise = unnormalized_noise
            if self.ov_noise_only:  # add noise to open-vocabulary slots only
                ov_slot_noise = self.noise_norm * normalized_noise * ov_slot_mask.unsqueeze(dim=-1).float()
            else:  # add noise to all slots
                ov_slot_noise = self.noise_norm * normalized_noise * slot_mask.unsqueeze(dim=-1).float()
            output["ov_slot_noise"] = ov_slot_noise
            noise_embeddings = embeddings + ov_slot_noise  # semantics decoupling using noise
            normal_sample_loss = self._inner_forward(embeddings, mask, slot_labels)["loss"]  # normal forward
            noise_sample_loss = self._inner_forward(noise_embeddings, mask, slot_labels)["loss"]  # adversarial forward
            loss = normal_sample_loss * (1 - self.noise_loss_prob) + noise_sample_loss * self.noise_loss_prob
            output["loss"] = loss
        return output
    
    def _inner_forward(self,
                       embeddings: torch.Tensor,
                       mask: torch.Tensor,
                       slot_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward from **embedding space** to a loss or predicted-tags.
        """
        output = {}

        if self.encoder:
            encoder_out = self.encoder(embeddings, mask)
            if self.dropout:
                encoder_out = self.dropout(encoder_out)
            output["encoder_out"] = encoder_out
        else:
            encoder_out = embeddings
        
        tag_logits = self.hidden2tag(encoder_out)
        output["tag_logits"] = tag_logits

        if self.use_crf:
            best_paths = self.crf.viterbi_tags(tag_logits, mask)
            predicted_tags = [x for x, y in best_paths]  # get the tags and ignore the score
            predicted_score = [y for _, y in best_paths]
            output["predicted_tags"] = predicted_tags
            output["predicted_score"] = predicted_score
        else:
            output["predicted_tags"] = torch.argmax(tag_logits, dim=-1)  # pylint: disable=no-member
        
        if slot_labels is not None:
            if self.use_crf:
                log_likelihood = self.crf(tag_logits, slot_labels, mask)  # returns log-likelihood
                output["loss"] = -1.0 * log_likelihood  # add negative log-likelihood as loss
                
                # Represent viterbi tags as "class probabilities" that we can
                # feed into the metrics
                class_probabilities = tag_logits * 0.
                for i, instance_tags in enumerate(predicted_tags):
                    for j, tag_id in enumerate(instance_tags):
                        class_probabilities[i, j, tag_id] = 1
                self.f1(class_probabilities, slot_labels, mask.float())
            else:
                output["loss"] = sequence_cross_entropy_with_logits(tag_logits, slot_labels, mask)
                self.f1(tag_logits, slot_labels, mask.float())
        
        return output


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric = self.f1.get_metric(reset)

        results = {}

        if self.metrics_for_each_slot_type:
            results.update(metric)
        else:
            results.update({
                "precision": metric["precision-overall"],
                "precision-ov": metric["precision-ov"],
                "recall": metric["recall-overall"],
                "recall-ov": metric["recall-ov"],
                "f1": metric["f1-measure-overall"],
                "f1-ov": metric["f1-measure-ov"]
            })

        return results