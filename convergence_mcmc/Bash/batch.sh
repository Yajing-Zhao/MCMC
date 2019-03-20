#!/bin/bash

for FILE in *.py; do
  echo ${FILE}
  sbatch -N1 -n1 -t 3:00:00 --mem=200 --wrap="time python ${FILE}" 
  sleep 1 # pause to be kind to the scheduler
done
