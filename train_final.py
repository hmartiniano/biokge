#!/usr/bin/env python3
import os
import sys
import glob
import pathlib

import pandas as pd
import numpy as np
import subprocess
import parse


template = "Test average {}: {:f}\n"

params = ["dataset", "method", "max_step", "hidden_dim", "neg_sample_size", "batch_size", "lr", "regularization_coef"]

metrics = ["MRR", "MR", "HITS@1", "HITS@3", "HITS@10"]
agg = {metric: ["mean", "std"] for metric in metrics}
agg.update({"fold": "count"})


def get_params(fname):
    """
    logs/TransE_l2_e1000_r512_n50_b5000_f1.log
    """
    fname = fname.split("/")[-1]
    p = parse.parse("{dataset}_m{method}_e{max_step}_r{hidden_dim}_n{neg_sample_size}_b{batch_size}_lr{lr}_rc{regularization_coef}_f{fold}.log", fname)
    return p.named


def get_results(fnames):
    data = []
    p = parse.compile(template)
    for fname in fnames:
        d = dict(p.findall(open(fname).read()))
        d.update(get_params(fname))
        data.append(d)
    data = pd.DataFrame(data)
    final = []
    for method in data["method"].unique():
        d = data[data.method == method].groupby(params, as_index=False).agg(agg).sort_values(by=("MRR", "mean"), ascending=False).head(1).to_dict(orient="records")[0]
        d = {k[0]: v for k, v in d.items() if k[1] != "std"}
        final.append(d)
    return final


os.environ["DGLBACKEND"] = "pytorch"


def parse_file(fname):
    with open(fname) as f:
        data = dict(list(parse.findall("Test average {}: {:f}\n", f.read())))
    assert "MRR" in data
    return data


def mkdir(dirname):
    p = pathlib.Path(dirname)
    p.mkdir(parents=True, exist_ok=True)


def run_test(dataset=None, method=None, max_step=None, hidden_dim=None, neg_sample_size=None, batch_size=None, regularization_coef=None, lr=None, **kwargs):
    all_data = []
    for fold in (0, ):
        out_file = f"valid_{dataset}_m{method}_e{max_step}_r{hidden_dim}_n{neg_sample_size}_b{batch_size}_lr{lr}_rc{regularization_coef}_f{fold}.log"
        out_dir = f"valid_{dataset}_m{method}_e{max_step}_r{hidden_dim}_n{neg_sample_size}_b{batch_size}_lr{lr}_rc{regularization_coef}_f{fold}"
        print(f"fold {fold}")
        try:
            data = parse_file(out_file)
            print("found file:", out_file)
        except:
            print("no output file or calculation didn't finish")
            data = None
        if data is None:
            command = f"DGLBACKEND=pytorch " + \
                      f"dglke_train --dataset {dataset} --data_path . " + \
                      f"--data_files ../{dataset}_train.tsv ../{dataset}_valid.tsv ../{dataset}_valid.tsv " + \
                      f"--format 'raw_udd_hrt' --model_name {method} --batch_size {batch_size} " + \
                      f"--neg_sample_size {neg_sample_size} --hidden_dim {hidden_dim} --gamma 12.0 --lr {lr} " + \
                      f"--max_step {max_step} --log_interval 500 --batch_size_eval 2048 -adv " + \
                      f"--regularization_coef {regularization_coef} --test --num_thread 32 " + \
                      f"--num_proc 1 --eval_interval {max_step} " + \
                      f"--neg_sample_size_eval 16 > \"{out_file}\""
            print(command)
            print(subprocess.check_output(command, shell=True))
            data = parse_file(out_file)
            mkdir(out_dir)
            subprocess.check_output(f"cp entities.tsv relations.tsv {out_dir}", shell=True)
        print(data)
        all_data.append(data)
    return all_data[0]

def run_valid(dataset=None, method=None, max_step=None, hidden_dim=None, neg_sample_size=None, batch_size=None, regularization_coef=None, lr=None, **kwargs):
    """ Evaluates all methods on validation dataset. """
    all_data = []
    for fold in (0, ):
        out_file = f"valid_{dataset}_m{method}_e{max_step}_r{hidden_dim}_n{neg_sample_size}_b{batch_size}_lr{lr}_rc{regularization_coef}_f{fold}.log"
        out_dir = f"valid_{dataset}_m{method}_e{max_step}_r{hidden_dim}_n{neg_sample_size}_b{batch_size}_lr{lr}_rc{regularization_coef}_f{fold}"
        print(f"fold {fold}")
        try:
            data = parse_file(out_file)
            print("found file:", out_file)
        except:
            print("no output file or calculation didn't finish")
            data = None
        if data is None:
            command = f"DGLBACKEND=pytorch " + \
                      f"dglke_train --dataset {dataset} --data_path . " + \
                      f"--data_files ../{dataset}_train.tsv ../{dataset}_valid.tsv ../{dataset}_valid.tsv " + \
                      f"--format 'raw_udd_hrt' --model_name {method} --batch_size {batch_size} " + \
                      f"--neg_sample_size {neg_sample_size} --hidden_dim {hidden_dim} --gamma 12.0 --lr {lr} " + \
                      f"--max_step {max_step} --log_interval 500 --batch_size_eval 2048 -adv " + \
                      f"--regularization_coef {regularization_coef} --test --num_thread 32 " + \
                      f"--num_proc 1 --eval_interval {max_step} " + \
                      f"--neg_sample_size_eval 16 > \"{out_file}\""
            print(command)
            print(subprocess.check_output(command, shell=True))
            data = parse_file(out_file)
            mkdir(out_dir)
            subprocess.check_output(f"cp entities.tsv relations.tsv {out_dir}", shell=True)
        print(data)
        all_data.append(data)
    return all_data[0]


def run_full(dataset=None, method=None, max_step=None, hidden_dim=None, neg_sample_size=None, batch_size=None, regularization_coef=None, lr=None, **kwargs):
    """ Produces final embeddings. """
    for fold in (0, ):
        out_file = f"full_{dataset}_m{method}_e{max_step}_r{hidden_dim}_n{neg_sample_size}_b{batch_size}_lr{lr}_rc{regularization_coef}_f{fold}.log"
        out_dir = f"full_{dataset}_m{method}_e{max_step}_r{hidden_dim}_n{neg_sample_size}_b{batch_size}_lr{lr}_rc{regularization_coef}_f{fold}"
        print(f"fold {fold}")
        try:
            data = parse_file(out_file)
            print("found file:", out_file)
        except:
            print("no output file or calculation didn't finish")
            data = None
        if data is None:
            command = f"DGLBACKEND=pytorch " + \
                      f"dglke_train --dataset {dataset} --data_path . " + \
                      f"--data_files ../{dataset}_full.tsv ../{dataset}_valid.tsv ../{dataset}_valid.tsv " + \
                      f"--format 'raw_udd_hrt' --model_name {method} --batch_size {batch_size} " + \
                      f"--neg_sample_size {neg_sample_size} --hidden_dim {hidden_dim} --gamma 12.0 --lr {lr} " + \
                      f"--max_step {max_step} --log_interval 500 --batch_size_eval 2048 -adv " + \
                      f"--regularization_coef {regularization_coef} --num_thread 32 " + \
                      f"--num_proc 1 --eval_interval {max_step} " + \
                      f"--neg_sample_size_eval 16 > \"{out_file}\""
            print(command)
            print(subprocess.check_output(command, shell=True))
            mkdir(out_dir)
            subprocess.check_output(f"cp entities.tsv relations.tsv {out_dir}", shell=True)
        print(data)
        all_data.append(data)


def run_final(dataset=None, method=None, max_step=None, hidden_dim=None, neg_sample_size=None, batch_size=None, regularization_coef=None, lr=None, **kwargs):
    """ Produces embeddings for downstream methods. """
    for fold in (0, ):
        out_file = f"final_{dataset}_m{method}_e{max_step}_r{hidden_dim}_n{neg_sample_size}_b{batch_size}_lr{lr}_rc{regularization_coef}_f{fold}.log"
        out_dir = f"final_{dataset}_m{method}_e{max_step}_r{hidden_dim}_n{neg_sample_size}_b{batch_size}_lr{lr}_rc{regularization_coef}_f{fold}"
        print(f"fold {fold}")
        try:
            data = parse_file(out_file)
            print("found file:", out_file)
        except:
            print("no output file or calculation didn't finish")
            data = None
        if data is None:
            command = f"DGLBACKEND=pytorch " + \
                      f"dglke_train --dataset {dataset} --data_path . " + \
                      f"--data_files ../{dataset}_valid_full.tsv ../{dataset}_valid.tsv ../{dataset}_valid.tsv " + \
                      f"--format 'raw_udd_hrt' --model_name {method} --batch_size {batch_size} " + \
                      f"--neg_sample_size {neg_sample_size} --hidden_dim {hidden_dim} --gamma 12.0 --lr {lr} " + \
                      f"--max_step {max_step} --log_interval 500 --batch_size_eval 2048 -adv " + \
                      f"--regularization_coef {regularization_coef} --num_thread 32 " + \
                      f"--num_proc 1 --eval_interval {max_step} " + \
                      f"--neg_sample_size_eval 16 > \"{out_file}\""
            print(command)
            print(subprocess.check_output(command, shell=True))
            mkdir(out_dir)
            subprocess.check_output(f"cp entities.tsv relations.tsv {out_dir}", shell=True)

def run_folds(dataset=None, method=None, max_step=None, hidden_dim=None, neg_sample_size=None, batch_size=None, regularization_coef=None, lr=None, **kwargs):
    """ Produces embeddings for downstream methods. """
    for fold in range(10):
        out_file = f"fold{fold}_{dataset}_m{method}_e{max_step}_r{hidden_dim}_n{neg_sample_size}_b{batch_size}_lr{lr}_rc{regularization_coef}_f{fold}.log"
        out_dir = f"fold{fold}_{dataset}_m{method}_e{max_step}_r{hidden_dim}_n{neg_sample_size}_b{batch_size}_lr{lr}_rc{regularization_coef}_f{fold}"
        print(f"fold {fold}")
        try:
            data = parse_file(out_file)
            print("found file:", out_file)
        except:
            print("no output file or calculation didn't finish")
            data = None
        if data is None:
            command = f"DGLBACKEND=pytorch " + \
                      f"dglke_train --dataset {dataset} --data_path . " + \
                      f"--data_files ../{dataset}_train_fold{fold}.tsv ../{dataset}_test_{fold}.tsv ../{dataset}_test_{fold}.tsv " + \
                      f"--format 'raw_udd_hrt' --model_name {method} --batch_size {batch_size} " + \
                      f"--neg_sample_size {neg_sample_size} --hidden_dim {hidden_dim} --gamma 12.0 --lr {lr} " + \
                      f"--max_step {max_step} --log_interval 500 --batch_size_eval 2048 -adv " + \
                      f"--regularization_coef {regularization_coef} --num_thread 32 " + \
                      f"--num_proc 1 --eval_interval {max_step} " + \
                      f"--neg_sample_size_eval 16 > \"{out_file}\""
            print(command)
            print(subprocess.check_output(command, shell=True))
            mkdir(out_dir)
            subprocess.check_output(f"cp entities.tsv relations.tsv {out_dir}", shell=True)


def run(param):
    d = run_test(dataset, method, max_step, hidden_dim, neg_sample_size, batch_size, regularization_coef, lr)
    for metric in d:
        print(metric, d[metric])
    return d["MRR"]


if __name__ == '__main__':
    from operator import itemgetter
    d = get_results(glob.glob("embeddings/logs/*.log"))
    print(d)
    d = list(sorted(d, key=itemgetter("method")))
    print("Best results for each metod:")
    print(d)
    best = list(sorted(d, key=itemgetter("MRR")))[-1]
    print("Best:", best)
    run_folds(**best)
    run_final(**best)
    run_full(**best)
    print("All others")
    for i in d:
        print(i)
        print(sys.stdout.flush())
        print(run_valid(**i))

