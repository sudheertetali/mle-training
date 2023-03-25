import sklearn

from ingest_data import data_ingest
from score import score
from train import train

data_ingest()
train()
score()
