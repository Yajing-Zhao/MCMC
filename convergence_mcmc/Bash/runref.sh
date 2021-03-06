#!/bin/bash

for FILE in ED_refed*; do
  echo ${FILE}
  sbatch -N1 -n1 -t 4:00:00 --mem=600 --wrap="time python ${FILE}" 
  sleep 1 # pause to be kind to the scheduler
done
