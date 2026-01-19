import optuna
import subprocess
import os
import numpy as np

def objective(trial):
    alpha_r = trial.suggest_categorical("alpha_r", [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])

    hyperpara_name = f"alphar{alpha_r}"
    
    print(f"\n[INFO] Running trial {trial.number} for seed=0,1,2,3,4")

    cmd = [
        "python", "PPO_discrete_main.py",
        "--max_train_steps", "400000", # 400000
        "--algorithm_name", "ppo",
        "--env_name", "sudoku4",
        "--num_units_gate", "64",
        "--num_layers_gate", "3",
        "--lr_gate", "0.0005",
        "--batch_size_gate", "64",
        "--alpha_r", str(alpha_r)
    ]
    
    subprocess.run(cmd)
    seeds = [0, 1, 2, 3, 4]
    acc_list = []

    for seed in seeds:
        csv_path = f"csvs/sudoku4/ppo/{hyperpara_name}/{seed}.csv"
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
    study.optimize(objective, n_trials=12, n_jobs=4)

    print("\n? Best trial:")
    print(f"Value: {study.best_trial.value:.4f}")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")