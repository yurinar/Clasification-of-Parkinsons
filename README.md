# Clasification of Parkinsons

Purpose : Classifying Parkinson's disease as normal.

Model : Neural network(3-layer, 50-Units)

Dataset : http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data


### Requirements
* Pytorch 1.4.0
* Numpy
* Pandas

### Usage
##### Training & Prediction
```
python training.py -h

optional arguments:
  -h, --help            show this help message and exit
  --epoch EPOCH, -e EPOCH
                        Number of sweeps over the dataset to train
  --gpu GPU, -g GPU     GPU ID (negative value indicates CPU)
  --dataset DATASET, -d DATASET
                        Directory of image files.
  --out OUT, -o OUT     Directory to output the result
  --seed SEED           Random seed
```
