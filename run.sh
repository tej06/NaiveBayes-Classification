#!/bin/bash

PROJECT_DIR=$HOME/Projects/NaiveBayes-Classification
source $PROJECT_DIR/py-ProbClassifier/bin/activate 
set -x
INPUT_FILE=$PROJECT_DIR/weatherAUS.csv
TARGET_LABEL="RainTomorrow"

python data_processing.py -i $INPUT_FILE
python training.py -l $TARGET_LABEL
python accuracy.py -l $TARGET_LABEL

echo "Finished"
