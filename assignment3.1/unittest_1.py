# import numpy as np
import pandas as pd


def test_shape():
    xTrain = pd.read_csv(
        "/home/sudheer/mle-training/assignment3.1/assign_sudheer/data/test/test.csv"
    )
    xTest = pd.read_csv(
        "/home/sudheer/mle-training/assignment3.1/assign_sudheer/data/train/train.csv"
    )
    print("Running Test")
    assert xTrain.shape[1] == xTest.shape[1]
    print("Test successful")


if __name__ == "__main__":
    test_shape()
