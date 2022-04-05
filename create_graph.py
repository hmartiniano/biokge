#!/usr/bin/env python
# coding: utf-8


import pathlib
import os
import sys
import argparse
from collections import OrderedDict

import sh
import pandas as pd

import obonet
import requests
import ensembl_rest
from pybiomart import Server





server = Server(host='http://www.ensembl.org')

dataset = (server.marts['ENSEMBL_MART_ENSEMBL']
                 .datasets['hsapiens_gene_ensembl'])


nodes = {
    "go": [],
    "gene": [],
}



# Import annotations from biomart


attribute_pairs = (
    ('ensembl_gene_id', 'go_id', "gene__has_annotation__go_id"),
)

biomart_data = []
for attr_pair in attribute_pairs:
    d = dataset.query(attributes=attr_pair[:2])
    print(d.columns)
    d["label"] = attr_pair[-1]
    d.columns = ["source", "target", "label"]
    d = d[["source", "label", "target"]]
    biomart_data.append(d)


print("GO annotations:")
for group in biomart_data:
    print(group.columns)
    print(group.shape, group.dropna().shape)



edges_from_biomart = pd.concat([
    biomart_data[0].dropna(),
#    biomart_data[1].dropna(),
])


# Using biomart we also produce two dataframes with mappings from gene symbol and gene entrez id to ensembl gene ID

if not pathlib.Path("entrez_to_ensembl.csv").exists():
    entrez_to_ensembl = (dataset.query(attributes=("entrezgene_id", "ensembl_gene_id"), use_attr_names=True)
                         .dropna()
                         .astype(int, errors="ignore")
                         .drop_duplicates()
                         .set_index("entrezgene_id"))

    entrez_to_ensembl.to_csv("entrez_to_ensembl.csv")
entrez_to_ensembl = pd.read_csv("entrez_to_ensembl.csv", index_col=0)


if not pathlib.Path("symbol_to_ensembl.csv").exists():
    symbol_to_ensembl = (dataset.query(attributes=("hgnc_symbol", "ensembl_gene_id"), use_attr_names=True)
                         .dropna()
                         .astype(int, errors="ignore")
                         .drop_duplicates()
                         .set_index("hgnc_symbol"))

    symbol_to_ensembl.to_csv("symbol_to_ensembl.csv")
symbol_to_ensembl = pd.read_csv("symbol_to_ensembl.csv", index_col=0)

if not pathlib.Path("rnacentral_to_ensembl.csv").exists():
    rnacentral_to_ensembl = (dataset.query(attributes=("rnacentral", "ensembl_gene_id"), use_attr_names=True)
                             .dropna()
                             .astype(int, errors="ignore")
                             .drop_duplicates()
                             .set_index("rnacentral"))

    rnacentral_to_ensembl.to_csv("rnacentral_to_ensembl.csv")
rnacentral_to_ensembl = pd.read_csv("rnacentral_to_ensembl.csv", index_col=0)


if not pathlib.Path("protein_to_gene.csv").exists():
    protein_to_ensembl = (dataset.query(attributes=("ensembl_peptide_id", "ensembl_gene_id"))
                         .dropna()
                         .astype(int, errors="ignore")
                         .drop_duplicates()
                         .set_index("Protein stable ID"))

    protein_to_ensembl.to_csv("protein_to_gene.csv")


if not pathlib.Path("protein_to_transcript.csv").exists():
    protein_to_transcript = (dataset.query(attributes=("ensembl_peptide_id", "ensembl_transcript_id"))
                         .dropna()
                         .astype(int, errors="ignore")
                         .drop_duplicates()
                         .set_index("Protein stable ID"))

    protein_to_ensembl.to_csv("protein_to_transcript.csv")
# ### Import GO data

url = "http://purl.obolibrary.org/obo/go.obo"
if not pathlib.Path("go.obo").exists():
    sh.wget(url)
go = obonet.read_obo("go.obo")
go_edges = []
for s, t, l in go.edges(keys=True):
    go_edges.append((s, go.nodes[s]['namespace'] + '__' + l + '__' + go.nodes[t]['namespace'], t))
go_to_namespace = {node: go.nodes[node]["namespace"] for node in go.nodes} 

go_edges = pd.DataFrame(go_edges, columns=["source", "label", "target"])
go_edges.to_csv("go.tsv", sep="\t", index=False)

"""
# ### import HPO data

url = "http://purl.obolibrary.org/obo/hp.obo"
hpo = obonet.read_obo(url)
hpo_edges = pd.DataFrame(hpo.edges, columns=["source", "target", "label"])
hpo_edges["label"] = "hpo__" + hpo_edges["label"] + "__hpo"
"""

# ### Import disgenet data


print("Disgenet:")
disgenet = pd.read_csv("all_gene_disease_associations.tsv.gz", sep="\t")
print(disgenet.head())


print(disgenet.geneId.drop_duplicates().shape)
print(disgenet.geneSymbol.drop_duplicates().shape)


disgenet2 = pd.merge(entrez_to_ensembl, disgenet.set_index("geneId"), left_index=True, right_index=True)


#cui_map = pd.read_csv("https://www.disgenet.org/static/disgenet_ap1/files/downloads/disease_mappings.tsv.gz", sep="\t")

#cui_map.head()

#cui_map.vocabulary.unique()


#gene_to_hpo = pd.merge(disgenet2.set_index("diseaseId"),
#                       cui_map[cui_map.vocabulary == "HPO"].set_index("diseaseId"),
#                       left_index=True, right_index=True)[["Gene stable ID", "code"]].dropna().drop_duplicates()
#gene_to_hpo.columns = ["source", "target"]
#gene_to_hpo["label"] = "gene__has_phenotype__hpo"
#gene_to_hpo = gene_to_hpo.reset_index(drop=True)


# ### Combine datasets into the final graph
gene_to_phenotype = disgenet2[["ensembl_gene_id", "diseaseType", "diseaseId"]].dropna().drop_duplicates()
gene_to_phenotype.columns = ["source", "label", "target"]
gene_to_phenotype["label"] = "gene__is_associated_to__" + gene_to_phenotype["label"]
gene_to_phenotype = gene_to_phenotype.reset_index(drop=True)



edges = pd.concat((edges_from_biomart, go_edges, gene_to_phenotype)).reset_index(drop=True)

print("Final KG:")
print(edges.shape)
print(edges.dropna().shape)
print(edges.columns)
print("Number of genes:", edges[edges.source.str.startswith("ENSG")].nunique())
print("Number of diseases:", edges[edges.target.str.startswith("C")].nunique())
print("relationships:", edges.label.unique())


edges.to_csv("edges.csv.gz", index=False)


url = "http://geneontology.org/gene-associations/goa_human.gaf.gz"
if not pathlib.Path("goa_human.gaf.gz").exists():
    sh.wget(url)
goa = pd.read_csv("goa_human.gaf.gz", sep="\t", comment="!", header=None, low_memory=False)
goa = goa[[2, 3, 4]]
goa.columns = ["source", "label", "target"]
goa = pd.merge(symbol_to_ensembl, goa.set_index("source"), left_index=True, right_index=True).dropna()
goa = goa.reset_index(drop=True)
goa.columns = ["source", "label", "target"]
goa = goa[~goa.label.str.startswith("NOT")]
goa["label"] = "gene__" + goa["label"].fillna("is_associated") + "__" + goa["target"].map(go_to_namespace)
goa.to_csv("goa.tsv", sep="\t", index=False)
print(goa.shape)

url = "http://geneontology.org/gene-associations/goa_human_rna.gaf.gz"
if not pathlib.Path("goa_human_rna.gaf.gz").exists():
    sh.wget(url)
goa_rna = pd.read_csv("goa_human_rna.gaf.gz", sep="\t", comment="!", header=None)
goa_rna = goa_rna[[2, 3, 4]]
goa_rna.columns = ["source", "label", "target"]
goa_rna["source"] = goa_rna["source"].str.split("_").str.get(0)
goa_rna = pd.merge(rnacentral_to_ensembl, goa_rna.set_index("source"), left_index=True, right_index=True).dropna()
goa_rna = goa_rna.reset_index(drop=True)
goa_rna.columns = ["source", "label", "target"]
goa_rna = goa_rna[~goa_rna.label.str.startswith("NOT")]
goa_rna["label"] = "gene__" + goa_rna["label"].fillna("is_associated") + "__" + goa_rna["target"].map(go_to_namespace)
goa_rna.to_csv("goa_rna.tsv", sep="\t", index=False)
print(goa_rna.shape)




