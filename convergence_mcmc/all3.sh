#!/bin/bash

sbatch -N1 -n1 -t 5:00:00 --mem=600M --wrap="time python ED_all3.py"
