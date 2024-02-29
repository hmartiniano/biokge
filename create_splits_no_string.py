#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


def data_split(examples, labels, train_frac, random_state=None):
    ''' https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    param data:       Data to be split
    param train_frac: Ratio of train set to whole dataset

    Randomly split dataset, based on these ratios:
        'train': train_frac
        'valid': (1-train_frac) / 2
        'test':  (1-train_frac) / 2

    Eg: passing train_frac=0.8 gives a 80% / 10% / 10% split
    '''

    assert train_frac >= 0 and train_frac <= 1, "Invalid training set fraction"

    X_train, X_tmp, Y_train, Y_tmp = train_test_split(
                                        examples, labels, train_size=train_frac, random_state=random_state)

    X_val, X_test, Y_val, Y_test  = train_test_split(
                                        X_tmp, Y_tmp, train_size=0.5, random_state=random_state)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def read_file(fname):
    return [line.strip() for line in open(fname)]


def analyze(df):
    print("object:", df["object"].nunique())
    print("subject:", df["subject"].nunique())
    print("predicate:", df["predicate"].nunique())
    print(df["predicate"].unique())


def main(out_dir="kg"):
    df = pd.read_csv("edges.csv.gz", index_col=0)
    print("initial dataframe:")
    print(df.shape)
    try: 
        predicates_to_exclude = read_file("exclude.txt")
        print("Excluding:")
        for predicate in predicates_to_exclude:
            print(predicate)
    except:
        predicates_to_exclude = []

    df = df[~df["predicate"].isin(predicates_to_exclude)]
    print("dataframe after excluding predicates:")
    print(df.shape)

    y = df["predicate"]
    X_train, X_val, X_test, Y_train, Y_val, Y_test = data_split(df, y, train_frac=0.6, random_state=42)
    for i in X_train, X_val, X_test:
        analyze(i)
    print("Train triples:", X_train.shape[0])
    print("Test triples:", X_test.shape[0])
    print("Val triples:", X_val.shape[0])

    # Save Training, Test and Validation sets
    X_train.to_csv(f"{out_dir}/kg_train.tsv", sep="\t", header=False, index=False)
    X_test.to_csv(f"{out_dir}/kg_test.tsv", sep="\t", header=False, index=False)
    X_val.to_csv(f"{out_dir}/kg_valid.tsv", sep="\t", header=False, index=False)

    # Create datasets for 10-fold CV predictition
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    X = pd.concat((X_train, X_test)).drop_duplicates()
    y = X["predicate"]
    for n, (train_index, test_index) in enumerate(kf.split(X, y=y)):
        print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        X_train.to_csv(f"{out_dir}/kg_train_fold{n}.tsv", sep="\t", header=False, index=False)
        X_test.to_csv(f"{out_dir}/kg_test_fold{n}.tsv", sep="\t", header=False, index=False)

    # Create full dataset
    df.to_csv(f"{out_dir}/kg_full.tsv", sep="\t", index=False, header=False)

if __name__ =='__main__':
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
