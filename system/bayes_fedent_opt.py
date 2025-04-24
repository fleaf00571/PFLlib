# bayes_fedent_opt.py
import os, json, subprocess, time, h5py, csv, argparse
import numpy as np
import optuna
from datetime import datetime
from pathlib import Path

# ---------- 路径设置 ----------
REPO = Path(__file__).resolve().parent
SYS  = REPO 
RESULT_DIR = REPO.parent / "results"
CONFIG_DIR = REPO / "auto_configs"; CONFIG_DIR.mkdir(exist_ok=True)
LOG_DIR    = REPO / "logs"; LOG_DIR.mkdir(exist_ok=True)
HIST_CSV   = REPO / "history.csv"
MAIN_PY    = SYS / "main.py"

# ---------- 工具函数 ----------
def compute_score(acc_hist, c_v, last_n=30):
    last = np.asarray(acc_hist[-last_n:], dtype=np.float32)
    mean, std = last.mean(), last.std()
    cv = std / mean if mean > 1e-8 else 1.0
    return float(mean - c_v * cv)

def load_h5(h5_path):
    with h5py.File(h5_path, 'r') as f:
        return f['rs_test_acc'][:]

def write_hist(row):
    newfile = not HIST_CSV.exists()
    with open(HIST_CSV, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if newfile: w.writeheader()
        w.writerow(row)

def done(json_name):
    if not HIST_CSV.exists():
        return False
    with open(HIST_CSV) as f:
        for r in csv.DictReader(f):
            if r["json_config"] == json_name and r["trained"] == "yes":
                return True
    return False

def gen_json(params, json_name):
    stage_info = [{
        "end_round": st["gr"],
        "H_min": round(st["H_min"], 2),
        "alpha_vp": round(st["alpha_vp"], 2),
        "beta": round(st["beta"], 2),
        "buffer_size": st["buffer_size"]
    } for st in (params["stage1"], params["stage2"], params["stage3"])]
    cfg = {"stage_info": stage_info}
    path = CONFIG_DIR / json_name
    with open(path, 'w') as f: 
        json.dump(cfg, f, indent=2)
    return path

def run_cmd(cmd):                # 阻塞执行
    t0 = time.time()
    ret = subprocess.run(cmd, shell=True)
    return ret.returncode, time.time()-t0

# ---------- Optuna 目标函数 ----------
def objective(trial, args):
    # 硬编码细调起始点
    fine_tune_after = 40
    fine = trial.number >= fine_tune_after

    steps = {
        'coarse': {
            'stage1': {'gr': 20, 'H_min': 0.2, 'alpha_vp': 0.2, 'beta': 0.2, 'buffer_size': 10},
            'stage2': {'gr': 20, 'H_min': 0.2, 'alpha_vp': 0.4, 'beta': 0.4, 'buffer_size': 10},
            'stage3': {'gr': 20, 'H_min': 0.3, 'alpha_vp': 0.4, 'beta': 0.4, 'buffer_size': 10},
        },
        'fine': {
            'stage1': {'gr': 5, 'H_min': 0.02, 'alpha_vp': 0.02, 'beta': 0.02, 'buffer_size': 2},
            'stage2': {'gr': 5, 'H_min': 0.02, 'alpha_vp': 0.03, 'beta': 0.02, 'buffer_size': 2},
            'stage3': {'gr': 5, 'H_min': 0.02, 'alpha_vp': 0.03, 'beta': 0.05, 'buffer_size': 2},
        }
    }
    # coarse
    # ranges = {
    #     'stage1': {'gr': (50, 200), 'H_min': (0.0, 1.0), 'alpha_vp': (0.0, 1.0), 'beta': (0.0, 1.0), 'buffer_size': (0, 40)},
    #     'stage2': {'gr': (100, 450), 'H_min': (0.0, 1.0), 'alpha_vp': (0.0, 2.0), 'beta': (0.0, 2.0), 'buffer_size': (0, 40)},
    #     'stage3': {'H_min': (0.0, 1.5), 'alpha_vp': (0.0, 2.0), 'beta': (0.0, 2.0), 'buffer_size': (0, 40)},
    # }

    # fine
        # fine
    ranges = {
        'stage1': {'gr': (110, 130), 'H_min': (0.6, 0.8), 'alpha_vp': (0.8, 1.0), 'beta': (0.0, 0.2), 'buffer_size': (20, 30)},
        'stage2': {'gr': (300, 320), 'H_min': (0.4, 0.6), 'alpha_vp': (1.2, 1.5), 'beta': (1.8, 2.0), 'buffer_size': (15, 25)},
        'stage3': {'gr': (500, 500), 'H_min': (0.6, 0.8), 'alpha_vp': (1.5, 1.8), 'beta': (0.0, 0.0), 'buffer_size': (15, 25)},
    }

    step_type = 'fine' if fine else 'coarse'

    params = {}
    for stage_idx in range(1, 4):
        stage_key = f"stage{stage_idx}"
        step_sizes = steps[step_type][stage_key]
        range_vals = ranges[stage_key]

        # Use the range values directly from the ranges dictionary for all stages
        if stage_idx == 3:
            # For stage 3, we use a fixed value of 500 for gr
            gr_value = 500
        else:
            # For all other stages, we use the range specified in the ranges dictionary
            gr_value = trial.suggest_int(f"gr_{stage_idx}", range_vals['gr'][0], range_vals['gr'][1], step=step_sizes['gr'])

        params[stage_key] = {
            "gr": gr_value,
            "H_min": trial.suggest_float(f"H_min_{stage_idx}", range_vals['H_min'][0], range_vals['H_min'][1], step=step_sizes['H_min']),
            "alpha_vp": trial.suggest_float(f"alpha_vp_{stage_idx}", range_vals['alpha_vp'][0], range_vals['alpha_vp'][1], step=step_sizes['alpha_vp']),
            "beta": trial.suggest_float(f"beta_{stage_idx}", range_vals['beta'][0], range_vals['beta'][1], step=step_sizes['beta']),
            "buffer_size": trial.suggest_int(f"buffer_size_{stage_idx}", range_vals['buffer_size'][0], range_vals['buffer_size'][1], step=step_sizes['buffer_size'])
        }

    json_name = f"{args.tag}_{trial.number:04d}.json"
    json_path = gen_json(params, json_name)

    if done(json_name):
        with open(HIST_CSV) as f:
            for r in csv.DictReader(f):
                if r["json_config"] == json_name:
                    return float(r["score"])

    go_tag = f"{args.tag}_{trial.number:04d}"
    cmd = (f"cd {SYS} && python {MAIN_PY.name} --dataset {args.dataset} --model CNN "
           f"--algorithm FedEntPlusSmooth --dynamic_config_path {json_path} "
           f"--num_clients 100 --join_ratio 0.1 --global_rounds 500 --local_epochs 5 "
           f"--local_learning_rate 0.01 --learning_rate_decay True --learning_rate_decay_gamma 0.98 "
           f"--batch_size 64 --momentum 0.9 --eval_gap 1 -go {go_tag} > {LOG_DIR/(go_tag+'.log')} 2>&1")

    rc, _ = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError("main.py failed")

    h5 = list(RESULT_DIR.glob(f"{args.dataset}_FedEntPlusSmooth*{go_tag}*.h5"))[-1]
    acc_hist = load_h5(h5)
    score = compute_score(acc_hist, args.cv, last_n=30)

    # --- 写历史 ---
    write_hist({
        "time"       : datetime.now().isoformat(timespec='seconds'),
        "json_config": json_name,
        "trained"    : "yes",
        "params"     : json.dumps(params),
        "c_v"        : args.cv,
        "score"      : score
    })

    return score

# ---------- 主入口 ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag",      required=True)
    ap.add_argument("--dataset",  default="Cifar10")
    ap.add_argument("--n_trials", type=int, default=30)
    ap.add_argument("--study",    default="fedent_opt.db")
    ap.add_argument("--cv",       type=float, default=0.1,
                    help="用户指定的准确率 CV 惩罚系数")
    args = ap.parse_args()

    storage = f"sqlite:///{args.study}"
    study = optuna.create_study(direction="maximize",
                                study_name="FedEntSmooth_BO",
                                storage=storage,
                                load_if_exists=True)
    study.optimize(lambda t: objective(t, args), n_trials=args.n_trials)

    print("Best score:", study.best_value)
    print("Best params:", study.best_params)
