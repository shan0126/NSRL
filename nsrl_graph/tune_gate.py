import optuna
import subprocess
import os
import numpy as np

def objective(trial):
    num_units_gate = trial.suggest_categorical("num_units_gate", [64, 128, 256])
    num_layers_gate = trial.suggest_categorical("num_layers_gate", [2, 3, 4])
    lr_gate = trial.suggest_categorical("lr_gate", [1e-3, 5e-4, 2e-4, 1e-4, 5e-5])
    batch_size_gate = trial.suggest_categorical("batch_size_gate", [64, 128, 256])

    hyperpara_name = f"Neu{num_units_gate}_Lay{num_layers_gate}_Lr{lr_gate}_Bs{batch_size_gate}"
    
    print(f"\n[INFO] Running trial {trial.number} for seed=0,1,2,3,4")

    cmd = [
        "python", "PPO_discrete_main.py",
        "--max_train_steps", "400000", # 400000
        "--algorithm_name", "ppo",
        "--env_name", "sudoku4",
        "--num_units_gate", str(num_units_gate),
        "--num_layers_gate", str(num_layers_gate),
        "--lr_gate", str(lr_gate),
        "--batch_size_gate", str(batch_size_gate)
    ]
    
    subprocess.run(cmd)
    seeds = [0, 1, 2, 3, 4]
    acc_list = []

    for seed in seeds:
        csv_path = f"csvs_sl/sudoku4/ppo/{hyperpara_name}/{seed}.csv"
        try:
            with open(csv_path, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last_line = lines[-1]
                    acc = float(last_line.strip().split(",")[1])
                    acc_list.append(acc)
                else:
                    acc_list.append(0.0)
        except Exception as e:
            print(f"[WARN] Could not read accuracy for seed {seed}: {e}")
            print(f"no folder {csv_path}")
            exit(0)
            acc_list.append(0.0)

    mean_acc = np.mean(acc_list)
    print(f"[RESULT] Trial {trial.number} mean acc = {mean_acc:.4f} from {acc_list}")
    return mean_acc


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40, n_jobs=8)

    print("\n? Best trial:")
    print(f"Value: {study.best_trial.value:.4f}")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")