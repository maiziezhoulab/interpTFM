#!/bin/bash

# Define the list of numbers you want to loop over
for num in 11 10 9 8 7 6 5 4 3 2 1 0
do
    python train_probe.py "$num"
done
