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


def analyze(df):
    print("Genes:", df.source.nunique())
    print("Diseases:", df.target.nunique())
    print("relationships:", df.label.nunique())
    print(df.label.unique())

df = pd.read_csv("edges.csv.gz")
df = df[df["label"].str.endswith("disease")]
gd = df.copy()
print("initial dataframe:")
print(df.shape)
"""
over2 = df.groupby(["source", "target"]).count()
print(over2.head())
print(over2.shape)
print(over2.describe())
over2 = over2[over2.label > 1].reset_index()
print("gene disease combinations over 2:")
print(over2.head())
df2 = df[df.target.isin(over2.target) & df.source.isin(over2.source)]
print("final dataframe:")
print(df2.nunique())
labels = df2["target"] + df2["source"]
"""
gene_counts = Counter(df.source)
disease_counts = Counter(df.target)
df = df[df.source.isin([g for g, c in gene_counts.items() if c > 2])]
df = df[df.target.isin([d for d, c in disease_counts.items() if c > 2])]
print("final dataframe:")
print(df.shape)
X_train, X_val, X_test, Y_train, Y_val, Y_test = data_split(df, df, train_frac=0.6, random_state=42)
for i in X_train, X_val, X_test:
    analyze(i)
    print(i.shape)
print(X_train.shape)
print(X_test.shape)
print(X_val.shape)
#genes = set([g for g in gene_counts if g in X_train.source.tolist() and g in X_test.source.tolist() and g in X_val.source.tolist()])
#diseases = set([d for d in disease_counts if d in X_train.target.tolist() and d in X_test.target.tolist() and d in X_val.target.tolist()])
X_train = X_train[X_train.source.isin(X_train.source) & X_train.source.isin(X_test.source) & X_train.source.isin(X_val.source)]
X_test = X_test[X_test.source.isin(X_train.source) & X_test.source.isin(X_test.source) & X_test.source.isin(X_val.source)]
X_val = X_val[X_val.source.isin(X_train.source) & X_val.source.isin(X_test.source) & X_val.source.isin(X_val.source)]
X_train = X_train[X_train.target.isin(X_train.target) & X_train.target.isin(X_test.target) & X_train.target.isin(X_val.target)]
X_test = X_test[X_test.target.isin(X_train.target) & X_test.target.isin(X_test.target) & X_test.target.isin(X_val.target)]
X_val = X_val[X_val.target.isin(X_train.target) & X_val.target.isin(X_test.target) & X_val.target.isin(X_val.target)]
for i in X_train, X_val, X_test:
    analyze(i)
    print(i.shape)
print(X_train.shape)
print(X_test.shape)
print(X_val.shape)
all_go = pd.concat((pd.read_csv("go.tsv", sep="\t"), pd.read_csv("goa.tsv", sep="\t"), pd.read_csv("goa_rna.tsv", sep="\t")))
all_go.to_csv("all_go.tsv", sep="\t", index=False, header=False)
X_train = pd.concat((X_train, all_go)) 
X_train.to_csv("go_h_train.tsv", sep="\t", header=False, index=False)
X_test.to_csv("go_h_test.tsv", sep="\t", header=False, index=False)
X_val.to_csv("go_h_valid.tsv", sep="\t", header=False, index=False)

# Create dataset for downstream tasks
X_val = pd.concat((X_val, all_go))
X_val.to_csv("go_h_valid_full.tsv", sep="\t", header=False, index=False)

# Create datasets for 10-fold CV predictition
kf = KFold(n_splits=10, random_state=42, shuffle=True)
for n, (train_index, test_index) in enumerate(kf.split(gd, y=gd[["source", "target"]].values)):
    print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
    X_train, X_test = gd.iloc[train_index, :], gd.iloc[test_index, :]
    X_train = pd.concat((X_train, all_go))
    X_train.to_csv(f"go_h_train_fold{n}.tsv", sep="\t", header=False, index=False)
    X_test = pd.concat((X_test, all_go))
    X_test.to_csv(f"go_h_test_fold{n}.tsv", sep="\t", header=False, index=False)

# Create full dataset with all G-D assocations
gd = pd.concat((gd, all_go))
gd.to_csv("go_h_full.tsv", sep="\t", index=False, header=False)
