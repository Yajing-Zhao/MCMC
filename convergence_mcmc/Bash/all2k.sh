#!/bin/bash

sbatch -N1 -n1 -t 6:00:00 --mem=600M --wrap="time python ED_all2k.py"
