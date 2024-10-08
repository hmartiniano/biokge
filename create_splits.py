#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pathlib
import argparse
from collections import Counter
import numpy as np
import pandas as pd
import networkx as nx
import obonet
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


def data_split(examples, labels, train_frac=0.8, random_state=None):
    ''' https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    param data:       Data to be split
    param train_frac: Ratio of train set to whole dataset

    Randomly split dataset, based on these ratios:
        'train': train_frac
        'valid': (1-train_frac) / 4
        'test':  (1-train_frac) / 4
        'holdout':  (1-train_frac) / 2

    Eg: passing train_frac=0.8 gives a 80% / 5% / 5% / 10% split
    '''

    assert train_frac >= 0 and train_frac <= 1, "Invalid training set fraction"

    X_train, X_tmp, Y_train, Y_tmp = train_test_split(
                                        examples, labels, train_size=train_frac, random_state=random_state)

    X_tmp, X_holdout, Y_tmp, Y_holdout  = train_test_split(
                                        X_tmp, Y_tmp, train_size=0.5, random_state=random_state)

    X_val, X_test, Y_val, Y_test  = train_test_split(
                                        X_tmp, Y_tmp, train_size=0.5, random_state=random_state)

    return X_train, X_val, X_test, X_holdout, Y_train, Y_val, Y_test, Y_holdout


def read_file(fname):
    return [line.strip() for line in open(fname)]


def analyze(df):
    print("object:", df["object"].nunique())
    print("subject:", df["subject"].nunique())
    print("predicate:", df["predicate"].nunique())
    print(df["predicate"].unique())


def clean_mondo(df):
    g = obonet.read_obo("mondo.obo")
    nodes = []
    for id_, data in g.nodes(data=True):
        data["id"] = id_
        nodes.append(data)
    corr = {}
    for node in nodes:
        for id_ in node.get("xref", []):
            if id_.startswith("HP:"):
                corr[node["id"]] = id_
    df = df[~df.subject.isin(corr.keys()) & ~df.object.isin(corr.keys())]
    return df


def get_parser():
    parser = argparse.ArgumentParser(
        prog="create_splits.py",
        description=(
            "create_splits.py: Split Kg into train, test, valudation and holdout sets."
        ),
    )
    parser.add_argument("-f", "--fname", help="Filename of CSV file with KG triples")
    parser.add_argument("-o", "--out_dir", 
        default="kg",
        help="Name of output directory.")
    parser.add_argument("-l", "--labels_to_predict", 
        default=[],
        nargs="+",
        help="Labels (predicates) to predict (comma separated)")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(f"Running with args: {args}")

    # Ensure args.out_dir exists
    p = pathlib.Path(args.out_dir)
    p.mkdir(parents=True, exist_ok=True)
    #
    raw_edges = pd.read_csv(args.fname, index_col=0)
    raw_edges = clean_mondo(raw_edges)
    print("initial dataframe:")
    print(raw_edges.shape)
    try: 
        predicates_to_exclude = read_file("exclude.txt")
        print("Excluding:")
        for predicate in predicates_to_exclude:
            print(predicate)
    except:
        predicates_to_exclude = []

    try: 
        predicates_to_include = read_file("include.txt")
        print("Including:")
        for predicate in predicates_to_include:
            print(predicate)
    except:
        predicates_to_include = []

    print("predicates to include:")
    print(predicates_to_include)
    print("predicates to exclude:")
    print(predicates_to_exclude)

    df = raw_edges[~raw_edges["predicate"].isin(predicates_to_exclude)]
    if predicates_to_include:
        df = df[df["predicate"].isin(predicates_to_include)]
    print("dataframe after excluding predicates:")
    print(df.shape)

    g = nx.from_pandas_edgelist(df, source='subject', target='object', edge_attr="predicate", create_using=nx.MultiDiGraph, edge_key="predicate")

    print("connected components:", Counter(map(len, nx.connected_components(g.to_undirected()))))
    lcc = max(nx.connected_components(g.to_undirected()), key=len)
    print("Size of lcc:", len(lcc))

    g = g.subgraph(lcc)

    edges = nx.to_pandas_edgelist(g, source='subject', target='object')[["subject", "predicate", "object"]]

    print("dataframe after excluding isolated nodes:")
    print(edges.shape)

    if args.labels_to_predict:
        other = edges[~edges["predicate"].isin(args.labels_to_predict)]
        df = edges[edges["predicate"].isin(args.labels_to_predict)]

        # to ensure that HPO terms are present in all splits
        y = df["predicate"].str.split("-").str.get(1)
    else:
        df = edges
        y = df["predicate"]

    X_train, X_val, X_test, X_holdout, Y_train, Y_val, Y_test, Y_holdout = data_split(df, y, train_frac=0.8, random_state=42)
    for i in X_train, X_val, X_test, X_holdout:
        analyze(i)
    print("Train triples:", X_train.shape[0])
    print("Test triples:", X_test.shape[0])
    print("Val triples:", X_val.shape[0])
    print("Hold out triples:", X_holdout.shape[0])

    if args.labels_to_predict:
        print(f"Combining {args.labels_to_predict} datasets with other triples")
        X_train = pd.concat((other, X_train)).sample(frac=1).reset_index(drop=True)
        print("Train triples:", X_train.shape[0])

    # Save Training, Test and Validation sets
    X_train.to_csv(f"{args.out_dir}/kg_train.tsv.gz", sep="\t", header=False, index=False)
    X_test.to_csv(f"{args.out_dir}/kg_test.tsv.gz", sep="\t", header=False, index=False)
    X_val.to_csv(f"{args.out_dir}/kg_valid.tsv.gz", sep="\t", header=False, index=False)
    X_holdout.to_csv(f"{args.out_dir}/kg_holdout.tsv.gz", sep="\t", header=False, index=False)

    # Create datasets for 10-fold CV prediction
    # 10 fold splits are done on the df dataframe 
    # KG data is added to the train fold if args.labels_to_predict is not empty
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    for n, (train_index, test_index) in enumerate(kf.split(df, y=y)):
        print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
        X_train_cv, X_test_cv = df.iloc[train_index, :], df.iloc[test_index, :]
        if args.labels_to_predict:
            X_train_cv_full = pd.concat((other, X_train_cv)).sample(frac=1).reset_index(drop=True)
        else:
            X_train_cv_full = X_train_cv
        X_train_cv_full.to_csv(f"{args.out_dir}/kg_train_fold{n}.tsv.gz", sep="\t", header=False, index=False)
        X_test_cv.to_csv(f"{args.out_dir}/kg_test_fold{n}.tsv.gz", sep="\t", header=False, index=False)

    # Create datasets for 10-fold CV prediction
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    X = pd.concat((X_train, X_test, X_val)).drop_duplicates()
    y = X["predicate"]
    for n, (train_index, test_index) in enumerate(kf.split(X, y=y)):
        print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
        X_train_cv, X_test_cv = X.iloc[train_index, :], X.iloc[test_index, :]
        if args.labels_to_predict:
            X_train_cv_full = pd.concat((other, X_train)).sample(frac=1).reset_index(drop=True)
        else:
            X_train_cv_full = X_train_cv
        X_train_cv.to_csv(f"{args.out_dir}/kg_train_fold{n}_noholdout.tsv.gz", sep="\t", header=False, index=False)
        X_test_cv.to_csv(f"{args.out_dir}/kg_test_fold{n}_noholdout.tsv.gz", sep="\t", header=False, index=False)

    # Create full dataset
    df.to_csv(f"{args.out_dir}/kg_full.tsv.gz", sep="\t", index=False, header=False)

    print("Normal Program Termination!")


if __name__ =='__main__':
    main()



