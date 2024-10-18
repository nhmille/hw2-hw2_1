#!/bin/bash

# Usage: ./hw2_seq2seq.sh <data_directory> <output_filename>

python HW2.py "$1" "$2"

python bleu_eval.py "$1" "$2"