#!/bin/bash

# Usage: ./hw2_seq2seq.sh <data_directory> <output_filename>

python3 HW2.py "$1" "$2"

python3 bleu_eval.py "$1" "$2"
