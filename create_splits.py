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


# Define a function to read lines from a file and strip any leading/trailing whitespace
def read_file(fname):
    with open(fname, 'r') as file:
        return [line.strip() for line in file]

# Analyze the unique values in the dataframe columns "object", "subject", and "predicate"
# Print the number of unique objects, subjects, and predicates
# Also print the unique predicates found in the dataframe
def analyze(df):
    print("Number of unique objects:", df["object"].nunique())
    print("Number of unique subjects:", df["subject"].nunique())
    print("Number of unique predicates:", df["predicate"].nunique())
    print("Unique predicates:", df["predicate"].unique())

# Clean the dataframe by removing rows where "subject" or "object" are in a list of Mondo Disease Ontology IDs
# that have a corresponding Human Phenotype Ontology (HPO) ID.
def clean_mondo(df):
    # Read the Mondo Disease Ontology from an OBO file into a networkx graph object
    g = obonet.read_obo("mondo.obo")

    # Iterate through the nodes in the graph and extract their IDs and data
    nodes = []
    for id_, data in g.nodes(data=True):
        data["id"] = id_
        nodes.append(data)

    # Create a dictionary mapping Mondo Disease Ontology IDs to their corresponding HPO IDs
    corr = {}
    for node in nodes:
        for id_ in node.get("xref", []):
            if id_.startswith("HP:"):
                corr[node["id"]] = id_

    # Remove rows from the dataframe where "subject" or "object" are in the list of Mondo Disease Ontology IDs
    # that have a corresponding HPO ID.
    df = df[~df.subject.isin(corr.keys()) & ~df.object.isin(corr.keys())]

    return df

# Define an argument parser to handle command-line arguments for this script
def get_parser():
    parser = argparse.ArgumentParser(
        prog="create_splits.py",
        description=(
            "Create_splits.py: Split knowledge graph (KG) into train, test, validation and holdout sets."
        ),
    )

    # Add a required argument for the filename of the CSV file containing KG triples
    parser.add_argument("-f", "--fname", help="Filename of CSV file with KG triples", required=True)

    # Add an optional argument for the name of the output directory (default is "kg")
    parser.add_argument("-o", "--out_dir",
        default="kg",
        help="Name of output directory.")

    # Add an optional argument for a list of labels (predicates) to predict
    parser.add_argument("-l", "--labels_to_predict",
        default=[],
        nargs="+",
        help="Labels (predicates) to predict")

    return parser


def main():
    # Get command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Print the command line arguments for debugging purposes
    print(f"Running with args: {args}")

    # Ensure that the output directory exists. If it doesn't, create it.
    p = pathlib.Path(args.out_dir)
    p.mkdir(parents=True, exist_ok=True)

    # Read in the raw edges from a CSV file and clean them by removing Mondo disease ontology terms
    raw_edges = pd.read_csv(args.fname, index_col=0)
    raw_edges = clean_mondo(raw_edges)

    # Print the shape of the dataframe before cleaning predicates
    print("initial dataframe:")
    print(raw_edges.shape)

    # Read in predicates to exclude from a file named "exclude.txt". If the file doesn't exist, set predicates_to_exclude to an empty list.
    try:
        predicates_to_exclude = read_file("exclude.txt")
        print("Excluding:")
        for predicate in predicates_to_exclude:
            print(predicate)
    except FileNotFoundError:
        predicates_to_exclude = []

    # Read in predicates to include from a file named "include.txt". If the file doesn't exist, set predicates_to_include to an empty list.
    try:
        predicates_to_include = read_file("include.txt")
        print("Including:")
        for predicate in predicates_to_include:
            print(predicate)
    except FileNotFoundError:
        predicates_to_include = []

    # Print the predicates to exclude and include
    print("predicates to include:")
    print(predicates_to_include)
    print("predicates to exclude:")
    print(predicates_to_exclude)

    # Remove excluded predicates from the dataframe. If there are included predicates, keep only those.
    df = raw_edges[~raw_edges["predicate"].isin(predicates_to_exclude)]
    if predicates_to_include:
        df = df[df["predicate"].isin(predicates_to_include)]
    # Print the shape of the dataframe after cleaning predicates
    print("dataframe after excluding predicates:")
    print(df.shape)

    # Create a directed graph from the edges in the dataframe
    g = nx.from_pandas_edgelist(df, source='subject', target='object', edge_attr="predicate", create_using=nx.MultiDiGraph, edge_key="predicate")

    # Print the number of connected components and the size of the largest connected component (LCC) in the graph.
    print("connected components:", Counter(map(len, nx.connected_components(g.to_undirected()))))
    lcc = max(nx.connected_components(g.to_undirected()), key=len)
    print("Size of lcc:", len(lcc))

    # Remove all nodes from the graph that are not part of the LCC
    g = g.subgraph(lcc)

    # Convert the subgraph back to a dataframe and drop the "predicate" column, which is now an attribute
    edges = nx.to_pandas_edgelist(g, source='subject', target='object')[["subject", "predicate", "object"]]

    # Print the shape of the dataframe after removing isolated nodes
    print("dataframe after excluding isolated nodes:")
    print(edges.shape)

    # Split the dataset into a training set and test set for prediction. If there are labels to predict, keep only those.
    if args.labels_to_predict:
        other = edges[~edges["predicate"].isin(args.labels_to_predict)]
        df = edges[edges["predicate"].isin(args.labels_to_predict)]

        # Extract the label from the predicate column (assuming it is in the format "label-id")
        y = df["predicate"].str.split("-").str.get(1)
    else:
        df = edges
        y = df["predicate"]

    # Split the dataset into training, validation, test and holdout sets
    X_train, X_val, X_test, X_holdout, Y_train, Y_val, Y_test, Y_holdout = data_split(df, y, train_frac=0.8, random_state=42)

    # Print statistics on each of the resulting splits (training, validation, test and holdout sets)
    for i in X_train, X_val, X_test, X_holdout:
        analyze(i)
    print("Train triples:", X_train.shape[0])
    print("Test triples:", X_test.shape[0])
    print("Val triples:", X_val.shape[0])
    print("Hold out triples:", X_holdout.shape[0])

    # If there are labels to predict, combine the training set with all other edges (not in the labels to predict)
    if args.labels_to_predict:
        print(f"Combining {args.labels_to_predict} datasets with other triples")
        X_train = pd.concat((other, X_train)).sample(frac=1).reset_index(drop=True)
    # Save Training, Test and Validation sets
    X_train.to_csv(f"{args.out_dir}/kg_train.tsv.gz", sep="\t", header=False, index=False)
    X_test.to_csv(f"{args.out_dir}/kg_test.tsv.gz", sep="\t", header=False, index=False)
    X_val.to_csv(f"{args.out_dir}/kg_valid.tsv.gz", sep="\t", header=False, index=False)
    X_holdout.to_csv(f"{args.out_dir}/kg_holdout.tsv.gz", sep="\t", header=False, index=False)

    # Create datasets for 10-fold cross-validation prediction.
    # The 10 fold splits are done on the df dataframe.
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    for n, (train_index, test_index) in enumerate(kf.split(df, y=y)):
        print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
        X_train_cv, X_test_cv = df.iloc[train_index, :], df.iloc[test_index, :]

        # If there are labels to predict, combine the training set with all other edges (not in the labels to predict)
        if args.labels_to_predict:
            X_train_cv_full = pd.concat((other, X_train_cv)).sample(frac=1).reset_index(drop=True)
        else:
            X_train_cv_full = X_train_cv

        # Save each fold's training and test sets
        X_train_cv_full.to_csv(f"{args.out_dir}/kg_train_fold{n}.tsv.gz", sep="\t", header=False, index=False)
        X_test_cv.to_csv(f"{args.out_dir}/kg_test_fold{n}.tsv.gz", sep="\t", header=False, index=False)

    # Create datasets for 10-fold cross-validation prediction without holdout set.
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    X = pd.concat((X_train, X_test, X_val)).drop_duplicates()
    y = X["predicate"]
    for n, (train_index, test_index) in enumerate(kf.split(X, y=y)):
        print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
        X_train_cv, X_test_cv = X.iloc[train_index, :], X.iloc[test_index, :]

        # If there are labels to predict, combine the training set with all other edges (not in the labels to predict)
        if args.labels_to_predict:
            X_train_cv_full = pd.concat((other, X_train)).sample(frac=1).reset_index(drop=True)
        else:
            X_train_cv_full = X_train_cv

        # Save each fold's training and test sets
        X_train_cv.to_csv(f"{args.out_dir}/kg_train_fold{n}_noholdout.tsv.gz", sep="\t", header=False, index=False)
        X_test_cv.to_csv(f"{args.out_dir}/kg_test_fold{n}_noholdout.tsv.gz", sep="\t", header=False, index=False)

    # Create full dataset
    df.to_csv(f"{args.out_dir}/kg_full.tsv.gz", sep="\t", index=False, header=False)

    print("Normal Program Termination!")


if __name__ =='__main__':
    main()



