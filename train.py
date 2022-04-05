#!/usr/bin/env python3
import os
import subprocess
import numpy as np
import parse
import optuna
from optuna.trial import TrialState
import joblib

os.environ["DGLBACKEND"] = "pytorch"


def parse_file(fname):
    with open(fname) as f:
        data = dict(list(parse.findall("Test average {}: {:f}\n", f.read())))
    assert "MRR" in data
    return data


def run_experiment(dataset, method, max_step, hidden_dim, neg_sample_size, batch_size, regularization_coef, lr):
    all_data = []
    for fold in (0, ):
        out_file = f"logs/{dataset}_m{method}_e{max_step}_r{hidden_dim}_n{neg_sample_size}_b{batch_size}_lr{lr}_rc{regularization_coef}_f{fold}.log"
        print(f"fold {fold}")
        try:
            data = parse_file(out_file)
            print("found file")
        except:
            print("no output file or calculation didn't finish")
            data = None
        if data is None:
            command = f"DGLBACKEND=pytorch " + \
                      f"dglke_train --dataset {dataset} --data_path . " + \
                      f"--data_files {dataset}_train.tsv {dataset}_valid.tsv {dataset}_test.tsv " + \
                      f"--format 'raw_udd_hrt' --model_name {method} --batch_size {batch_size} " + \
                      f"--neg_sample_size {neg_sample_size} --hidden_dim {hidden_dim} --gamma 12.0 --lr {lr} " + \
                      f"--max_step {max_step} --log_interval 500 --batch_size_eval 2048 -adv " + \
                      f"--regularization_coef {regularization_coef} --test --num_thread 32 " + \
                      f"--num_proc 1 --no_save_emb --eval_interval {max_step} " + \
                      f"--neg_sample_size_eval 16 > \"{out_file}\""
            print(command)
            print(subprocess.check_output(command, shell=True))
            data = parse_file(out_file)
        print(data)
        all_data.append(data)
    return all_data[0]


class Objective:

    def __init__(self, dataset, method):
        self.dataset = dataset
        self.method = method

    def __call__(self, trial):
        max_step = trial.suggest_categorical("max_step", [1000, 2000, 5000, 10000, 20000, 50000])
        hidden_dim = trial.suggest_categorical("hidden_dim", [100, 200, 300, 400, 500])
        neg_sample_size = trial.suggest_categorical("neg_sample_size", list(range(100, 1100, 100)))
        #batch_size = trial.suggest_int("batch_size", list(range(1000, 11000, 1000)))
        dataset = self.dataset
        method = self.method
        #method = trial.suggest_categorical("method", ["TransE_l2"])
        batch_size = trial.suggest_categorical("batch_size", [1000, 2000, 5000, 10000])
        regularization_coef = trial.suggest_categorical("regularization_coef", [1e-5, 1e-6, 1e-7, 1e-8, 1e-9])
        lr = trial.suggest_categorical("lr", [0.1, 0.01])
        d = run_experiment(dataset, method, max_step, hidden_dim, neg_sample_size, batch_size, regularization_coef, lr)
        for metric in d:
            trial.set_user_attr(metric, d[metric])
        return d["MRR"]


#run_experiment("go", "ComplEx", "1000", "128", "16", "32", 0.1)
n_trials = 20

for method in ["ComplEx", "DistMult", "TransE_l1", "TransE_l2"]:
    for dataset in ["go_h"]:
        print(dataset, method)
        study_name = f"{dataset}_{method}.study"
        study = optuna.create_study(direction="maximize", study_name=study_name, storage=f'sqlite:///{dataset}_{method}.db', load_if_exists=True)
        if study.user_attrs.get("dataset", None) is None:
            study.set_user_attr("dataset", dataset)
            study.set_user_attr("method", method)
        objective = Objective(dataset, method)
        try:
            n_complete_trials = len([trial for trial in study.get_trials() if trial.state == TrialState.COMPLETE])
        except:
            n_complete_trials = 0
        print(f"Study {study_name} has {n_complete_trials} complete trials.")
        if n_complete_trials < n_trials:
            study.optimize(objective, n_trials=n_trials)
        joblib.dump(study, study_name)
        df = study.trials_dataframe()
        print(df.head())

