"""
Command:
    python3 data/generate_statistics.py --include_split valid test \
                                        --output_filename label_statistics_valid_test.json \
                                        --data_path data/snips
    python3 data/generate_statistics.py --include_split valid test \
                                        --output_filename label_statistics_valid_test.json \
                                        --data_path data/mr-splited \
                                        --ignore_intent
"""

import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include_split", type=str, nargs="+", default=["valid", "test"],
                        help="the data split to include")
    parser.add_argument("--data_path", type=str, default="./data/snips")
    parser.add_argument("--output_filename", type=str, default="label_statistics.json",
                        help="the output filename")
    parser.add_argument("--ignore_intent", action="store_true", default=False,
                        help="whether to ignore intent statistics")
    return parser.parse_args()

def get_all_samples(args):
    """
    获取所有指定split的样本，返回（tokens, intent, slots）的三元组。
    """
    samples = []
    data_path = args.data_path
    for split in args.include_split:
        seq_in_file = os.path.join(data_path, split, "seq.in")
        seq_out_file = os.path.join(data_path, split, "seq.out")
        label_file = os.path.join(data_path, split, "label")
        with open(seq_in_file, "r") as f:
            seq_in_lines = f.readlines()
        with open(seq_out_file, "r") as f:
            seq_out_lines = f.readlines()
        if args.ignore_intent:
            label_lines = ["NO_INTENT\n" for _ in range(len(seq_in_lines))]
        else:
            with open(label_file, "r") as f:
                label_lines = f.readlines()
        assert len(seq_in_lines) == len(seq_out_lines) == len(label_lines)
        for seq_in_line, seq_out_line, label_line in zip(seq_in_lines, seq_out_lines, label_lines):
            tokens = seq_in_line.strip().split(" ")
            tokens = [token for token in tokens if token]
            intent = label_line.strip()
            slots = seq_out_line.strip().split(" ")
            slots = [slot for slot in slots if slot]
            assert len(tokens) == len(slots), tokens
            samples.append((tokens, intent, slots))
    return samples

def parse_bio(tags, tokens):
    """
    给定BIO标注列表，返回一个list，包含这句话中的slot label、长度、对应的slot value。
    """
    result = []
    token_count = 0
    slot_tokens = []
    last_tag = "O"
    current_label = None
    for tag, token in zip(tags, tokens):
        token = token.strip()
        if last_tag == "O":
            if tag != "O":
                assert tag[0] == "B" or tag[0] == "I"
                current_label = tag[2:]
                token_count = 1
                slot_tokens = [token]
        else:
            if tag == "O":
                result.append((current_label, token_count, slot_tokens))
                current_label = None
                token_count = 0
                slot_tokens = []
            elif tag[0] == "I":
                token_count += 1
                slot_tokens.append(token)
            else:
                result.append((current_label, token_count, slot_tokens))
                current_label = tag[2:]
                token_count = 1
                slot_tokens = [token]
        last_tag = tag
    if current_label is not None:
        result.append((current_label, token_count, slot_tokens))
    return result

def count_intents_and_slots(args, samples):
    """
    统计给定数据集中的intent和slots数目。
    """
    def generate_examples(tokens, intent, slots):
        return {
            "sentence": " ".join(tokens),
            "intent": intent,
            "slots": " ".join(slots)
        }
    counter = {
        "slots": {},
        "intents": {}
    }
    for tokens, intent, slots in samples:
        if intent not in counter["intents"]:
            counter["intents"][intent] = {
                "count": 1,
                "example": generate_examples(tokens, intent, slots)
            }
        else:
            counter["intents"][intent]["count"] += 1
        for slot_label, count, slot_tokens in parse_bio(slots, tokens):
            if slot_label not in counter["slots"]:
                counter["slots"][slot_label] = {
                    "count": 1,
                    "intent_list": [intent],
                    "example": generate_examples(tokens, intent, slots),
                    "max_token_count": count,
                    "slot_values": [" ".join(slot_tokens)],
                    "slot_value_count": 1
                }
            else:
                counter["slots"][slot_label]["count"] += 1
                if intent not in counter["slots"][slot_label]["intent_list"]:
                    counter["slots"][slot_label]["intent_list"].append(intent)
                if count > counter["slots"][slot_label]["max_token_count"]:
                    counter["slots"][slot_label]["max_token_count"] = count
                    counter["slots"][slot_label]["example"] = generate_examples(tokens, intent, slots)
                slot_value = " ".join(slot_tokens)
                if slot_value not in counter["slots"][slot_label]["slot_values"]:
                    counter["slots"][slot_label]["slot_values"].append(slot_value)
                    counter["slots"][slot_label]["slot_value_count"] += 1
    return counter

def main(args):
    all_samples = get_all_samples(args)
    counter = count_intents_and_slots(args, all_samples)
    output_file = os.path.join(args.data_path, args.output_filename)
    with open(output_file, "w") as f:
        json.dump(counter, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    main(args)