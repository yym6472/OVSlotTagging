import os
import random
random.seed(0)

seq_in_file = "../mr-origin/train/seq.in"
seq_out_file = "../mr-origin/train/seq.out"

with open(seq_in_file, "r") as f:
    seq_in_lines = f.readlines()
with open(seq_out_file, "r") as f:
    seq_out_lines = f.readlines()
assert len(seq_in_lines) == len(seq_out_lines) == 7660
samples = list(zip(seq_in_lines, seq_out_lines))
random.shuffle(samples)

with open("./train/seq.in", "w") as seq_in_f:
    with open("./train/seq.out", "w") as seq_out_f:
        for seq_in_line, seq_out_line in samples[:6894]:
            seq_in_f.write(seq_in_line)
            seq_out_f.write(seq_out_line)
with open("./valid/seq.in", "w") as seq_in_f:
    with open("./valid/seq.out", "w") as seq_out_f:
        for seq_in_line, seq_out_line in samples[6894:]:
            seq_in_f.write(seq_in_line)
            seq_out_f.write(seq_out_line)
