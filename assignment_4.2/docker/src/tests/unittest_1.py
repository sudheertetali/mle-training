# import numpy as np
import pandas as pd


def test_shape():
    xTrain = pd.read_csv(
        "../data/test/test.csv"
    )
    xTest = pd.read_csv(
        "../data/train/train.csv"
    )
    print("Running Test")
    assert xTrain.shape[1] == xTest.shape[1]
    print("Test successful")


if __name__ == "__main__":
    test_shape()
