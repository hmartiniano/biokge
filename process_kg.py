import sys
from collections import Counter
import pandas as pd
import networkx as nx

node_type = {"UNIPROTKB": "protein",
             "ENSEMBL": "disease",
             "RNACENTRAL": "ncrna", 
             "MONDO": "disease",
             }

def main(node_file, edge_file, output_file="edges.csv.gz"):
    nodes = pd.read_csv(
            node_file, 
            #usecols=["subject", "predicate", "object"], 
            sep="\t",
            index_col=0,
            low_memory=False)

    node_type = nodes["category"]

    edges = pd.read_csv(
            edge_file, 
            usecols=["subject", "predicate", "object"], 
            sep="\t",
            low_memory=False)
    edges["predicate"] = edges["subject"].map(node_type) + "-" + edges["predicate"] + "-" + edges["object"].map(node_type)
    print(edges.head())
    edges.to_csv(f"raw_{output_file}")

    predicates_to_exclude = [
    #        "biolink:Protein__biolink:interacts_with__biolink:Protein",
    ]

    grp = edges.groupby("predicate").count().iloc[:, 0]
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(grp)
    edges = edges[edges["predicate"].isin(grp[grp > 1000].index)]

    edges = edges[~edges["predicate"].isin(predicates_to_exclude)]
 
    edges.to_csv(f"{output_file}")

if __name__ == '__main__':
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])
