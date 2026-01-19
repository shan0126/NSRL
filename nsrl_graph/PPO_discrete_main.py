import torch
import numpy as np
import gym
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_discrete import PPO_discrete

from torch.distributions import kl_divergence

from envs.sudoku_env import SudokuEnv
from envs.queens_env import QueensEnv
from envs.coloringG_env import ColoringG
import argparse

import os

import csv

import sys
sys.path.insert(0, './pypsdd')

from GatingFunction import DenseGatingFunction
from compute_mpe import CircuitMPE

import random


def evaluate_policy(args, env, agent, state_norm):
    times = 10
    evaluate_reward = 0
    win_count = 0
    for _ in range(times):
        s = env.reset().flatten()
        if args.use_state_norm:  # During the evaluating,update=False
            s = state_norm(s, update=False)
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            s_, r, done, _ = env.step(a)
            s_ = s_.flatten()
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward
        if episode_reward > args.win_rew:
            win_count += 1

    return evaluate_reward / times, win_count / times
    


def main(args, env_name, number, seed):

    np.random.seed(seed)
    torch.manual_seed(seed)

    if env_name == 'sudoku2':
        env = SudokuEnv(2)
        env_evaluate = SudokuEnv(2)
        args.state_dim = env.reset().flatten().shape[0]
        args.action_dim = env.max_action_index
        args.max_episode_steps = env._max_episode_steps
        
        args.win_rew = 1.25  # upper than 1.25 then must win
        
        vtree_dir = 'utils/SDDCircuits/sudoku2/b1/sudoku2.vtree'
        sdd_dir = 'utils/SDDCircuits/sudoku2/b1/sudoku2.sdd'
        
        size = 2
        
        print("env={}".format(env_name))
        print("state_dim={}".format(args.state_dim))
        print("action_dim={}".format(args.action_dim))
        print("max_episode_steps={}".format(args.max_episode_steps))
    
        train_data = np.zeros((100000, args.state_dim))
        train_label = np.zeros((100000, size * size * size))
        current_ptr = 0
        current_size = 0
        
    elif env_name == 'sudoku3':
        env = SudokuEnv(3)
        env_evaluate = SudokuEnv(3)
        args.state_dim = env.reset().flatten().shape[0]
        args.action_dim = env.max_action_index
        args.max_episode_steps = env._max_episode_steps
        
        args.win_rew = 1.55
        
        vtree_dir = 'utils/SDDCircuits/sudoku3/b1/sudoku3.vtree'
        sdd_dir = 'utils/SDDCircuits/sudoku3/b1/sudoku3.sdd'
        
        size = 3
        
        print("env={}".format(env_name))
        print("state_dim={}".format(args.state_dim))
        print("action_dim={}".format(args.action_dim))
        print("max_episode_steps={}".format(args.max_episode_steps))
    
        train_data = np.zeros((100000, args.state_dim))
        train_label = np.zeros((100000, size * size * size))
        current_ptr = 0
        current_size = 0
        
    elif env_name == 'sudoku4':
        env = SudokuEnv(4)
        env_evaluate = SudokuEnv(4)
        args.state_dim = env.reset().flatten().shape[0]
        args.action_dim = env.max_action_index
        args.max_episode_steps = env._max_episode_steps
        
        args.win_rew = 2.05
        
        vtree_dir = 'utils/SDDCircuits/sudoku4/b1/sudoku4.vtree'
        sdd_dir = 'utils/SDDCircuits/sudoku4/b1/sudoku4.sdd'
        
        size = 4
        
        print("env={}".format(env_name))
        print("state_dim={}".format(args.state_dim))
        print("action_dim={}".format(args.action_dim))
        print("max_episode_steps={}".format(args.max_episode_steps))
    
        train_data = np.zeros((100000, args.state_dim))
        train_label = np.zeros((100000, size * size * size))
        current_ptr = 0
        current_size = 0
        
    elif env_name == 'sudoku5':
        env = SudokuEnv(5)
        env_evaluate = SudokuEnv(5)
        args.state_dim = env.reset().flatten().shape[0]
        args.action_dim = env.max_action_index
        args.max_episode_steps = env._max_episode_steps
        
        args.win_rew = 2.65
        
        vtree_dir = 'utils/SDDCircuits/sudoku5/b1/sudoku5.vtree'
        sdd_dir = 'utils/SDDCircuits/sudoku5/b1/sudoku5.sdd'
        
        size = 5
        
        print("env={}".format(env_name))
        print("state_dim={}".format(args.state_dim))
        print("action_dim={}".format(args.action_dim))
        print("max_episode_steps={}".format(args.max_episode_steps))
    
        train_data = np.zeros((100000, args.state_dim))
        train_label = np.zeros((100000, size * size * size))
        current_ptr = 0
        current_size = 0
    
    elif env_name == '4queens':
        env = QueensEnv(4)
        env_evaluate = QueensEnv(4)
        args.state_dim = env.reset().flatten().shape[0]
        args.action_dim = env.max_action_index
        args.max_episode_steps = env._max_episode_steps
        
        args.win_rew = 0.9
        
        size = 4
        
        print("env={}".format(env_name))
        print("state_dim={}".format(args.state_dim))
        print("action_dim={}".format(args.action_dim))
        print("max_episode_steps={}".format(args.max_episode_steps))
    
        train_data = np.zeros((100000, args.state_dim))
        train_label = np.zeros((100000, size * size))
        current_ptr = 0
        current_size = 0
        
        vtree_dir = 'utils/SDDCircuits/nqueens4/nqueens4.vtree'
        sdd_dir = 'utils/SDDCircuits/nqueens4/nqueens4.sdd'
        
    elif env_name == '6queens':
        env = QueensEnv(6)
        env_evaluate = QueensEnv(6)
        args.state_dim = env.reset().flatten().shape[0]
        args.action_dim = env.max_action_index
        args.max_episode_steps = env._max_episode_steps
        
        args.win_rew = 0.9
        
        size = 6
        
        print("env={}".format(env_name))
        print("state_dim={}".format(args.state_dim))
        print("action_dim={}".format(args.action_dim))
        print("max_episode_steps={}".format(args.max_episode_steps))
    
        train_data = np.zeros((100000, args.state_dim))
        train_label = np.zeros((100000, size * size))
        current_ptr = 0
        current_size = 0
        
        vtree_dir = 'utils/SDDCircuits/nqueens6/nqueens6.vtree'
        sdd_dir = 'utils/SDDCircuits/nqueens6/nqueens6.sdd'
        
    elif env_name == '8queens':
        env = QueensEnv(8)
        env_evaluate = QueensEnv(8)
        args.state_dim = env.reset().flatten().shape[0]
        args.action_dim = env.max_action_index
        args.max_episode_steps = env._max_episode_steps
        
        args.win_rew = 0.9
        
        size = 8
        
        print("env={}".format(env_name))
        print("state_dim={}".format(args.state_dim))
        print("action_dim={}".format(args.action_dim))
        print("max_episode_steps={}".format(args.max_episode_steps))
    
        train_data = np.zeros((100000, args.state_dim))
        train_label = np.zeros((100000, size * size))
        current_ptr = 0
        current_size = 0
        
        vtree_dir = 'utils/SDDCircuits/nqueens8/nqueens8.vtree'
        sdd_dir = 'utils/SDDCircuits/nqueens8/nqueens8.sdd'
        
    elif env_name == '10queens':
        env = QueensEnv(10)
        env_evaluate = QueensEnv(10)
        args.state_dim = env.reset().flatten().shape[0]
        args.action_dim = env.max_action_index
        args.max_episode_steps = env._max_episode_steps
        
        args.win_rew = 0.9
        
        size = 10
        
        print("env={}".format(env_name))
        print("state_dim={}".format(args.state_dim))
        print("action_dim={}".format(args.action_dim))
        print("max_episode_steps={}".format(args.max_episode_steps))
    
        train_data = np.zeros((100000, args.state_dim))
        train_label = np.zeros((100000, size * size))
        current_ptr = 0
        current_size = 0
        
        vtree_dir = 'utils/SDDCircuits/nqueens10/nqueens10.vtree'
        sdd_dir = 'utils/SDDCircuits/nqueens10/nqueens10.sdd'
        
    elif env_name == 'g1':
        env = ColoringG(1)
        env_evaluate = ColoringG(1)
        args.state_dim = env.reset().flatten().shape[0]
        args.action_dim = env.max_action_index
        args.max_episode_steps = env._max_episode_steps
        
        args.win_rew = 0.1
        
        print("env={}".format(env_name))
        print("state_dim={}".format(args.state_dim))
        print("action_dim={}".format(args.action_dim))
        print("max_episode_steps={}".format(args.max_episode_steps))
    
        train_data = np.zeros((100000, args.state_dim))
        train_label = np.zeros((100000, args.action_dim))
        current_ptr = 0
        current_size = 0
        
        vtree_dir = 'utils/SDDCircuits/coloringG1/coloringG1.vtree'
        sdd_dir = 'utils/SDDCircuits/coloringG1/coloringG1.sdd'
        
    elif env_name == 'g2':
        env = ColoringG(2)
        env_evaluate = ColoringG(2)
        args.state_dim = env.reset().flatten().shape[0]
        args.action_dim = env.max_action_index
        args.max_episode_steps = env._max_episode_steps
        
        args.win_rew = 0.1
        
        print("env={}".format(env_name))
        print("state_dim={}".format(args.state_dim))
        print("action_dim={}".format(args.action_dim))
        print("max_episode_steps={}".format(args.max_episode_steps))
    
        train_data = np.zeros((100000, args.state_dim))
        train_label = np.zeros((100000, args.action_dim))
        current_ptr = 0
        current_size = 0
        
        vtree_dir = 'utils/SDDCircuits/coloringG2/coloringG2.vtree'
        sdd_dir = 'utils/SDDCircuits/coloringG2/coloringG2.sdd'
        
    elif env_name == 'g3':
        env = ColoringG(3)
        env_evaluate = ColoringG(3)
        args.state_dim = env.reset().flatten().shape[0]
        args.action_dim = env.max_action_index
        args.max_episode_steps = env._max_episode_steps
        
        args.win_rew = 0.1
        
        print("env={}".format(env_name))
        print("state_dim={}".format(args.state_dim))
        print("action_dim={}".format(args.action_dim))
        print("max_episode_steps={}".format(args.max_episode_steps))
    
        train_data = np.zeros((100000, args.state_dim))
        train_label = np.zeros((100000, args.action_dim))
        current_ptr = 0
        current_size = 0
        
        vtree_dir = 'utils/SDDCircuits/coloringG3/coloringG3.vtree'
        sdd_dir = 'utils/SDDCircuits/coloringG3/coloringG3.sdd'
        
    elif env_name == 'g4':
        env = ColoringG(4)
        env_evaluate = ColoringG(4)
        args.state_dim = env.reset().flatten().shape[0]
        args.action_dim = env.max_action_index
        args.max_episode_steps = env._max_episode_steps
        
        args.win_rew = 0.1
        
        print("env={}".format(env_name))
        print("state_dim={}".format(args.state_dim))
        print("action_dim={}".format(args.action_dim))
        print("max_episode_steps={}".format(args.max_episode_steps))
    
        train_data = np.zeros((100000, args.state_dim))
        train_label = np.zeros((100000, args.action_dim))
        current_ptr = 0
        current_size = 0
        
        vtree_dir = 'utils/SDDCircuits/coloringG4/coloringG4.vtree'
        sdd_dir = 'utils/SDDCircuits/coloringG4/coloringG4.sdd'
    
        
        
        
    else:

        env = gym.make(env_name)
        env_evaluate = gym.make(env_name)  # When evaluating the policy, we need to rebuild an environment
        # Set random seed
        env.seed(seed)
        env.action_space.seed(seed)
        env_evaluate.seed(seed)
        env_evaluate.action_space.seed(seed)
        
        args.state_dim = env.observation_space.shape[0]
        args.action_dim = env.action_space.n
        args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
            
        
    

    
    
    
    cmpe = CircuitMPE(vtree_dir, sdd_dir)
    
    device = torch.device("cpu")
    
    gate = DenseGatingFunction(cmpe.beta, gate_layers=[args.state_dim]+[args.num_units_gate]*args.num_layers_gate, num_reps=args.num_reps_gate).to(device)
    
    optimizer = torch.optim.Adam(list(gate.parameters()), lr=args.lr_gate)
    
    
    epsilon = args.epsilon_init
    epsilon_min = args.epsilon_min
    epsilon_start_decay = args.max_train_steps * args.epsilon_start_decay
    epsilon_decay = (args.epsilon_init - args.epsilon_min) / (args.max_train_steps * args.epsilon_decay_steps)
    

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    agent = PPO_discrete(args)
    
    # hyperpara_name
    hyperpara_name = 'Neu'+str(args.num_units_gate)+'_Lay'+str(args.num_layers_gate)+'_Lr'+str(args.lr_gate)+'_Bs'+str(args.batch_size_gate)
    hyperpara_name_rl = 'alpr'+str(args.alpha_r) + '_isMarkov'+str(args.is_Markov)
    
    csv_file = 'csvs/'+ args.env_name + '/' + args.algorithm_name + '/' + hyperpara_name_rl + '/' + str(seed) +'.csv'
    csv_sl_file = 'csvs_sl/'+ args.env_name + '/' + args.algorithm_name + '/' + hyperpara_name + '/' + str(seed) +'.csv'
        
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    os.makedirs(os.path.dirname(csv_sl_file), exist_ok=True)
        
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Total Steps", "Evaluate Reward", "Epsilon", "Violation", "Win_rate"])
            
    with open(csv_sl_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode_idx", "correct_rate", "buffer_size"])

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    
    seen_samples = set()
    episode_idx = 0
    updated_acc = 0.0
    episode_violation = 0
    
    
    
    while total_steps < args.max_train_steps:
        s = env.reset().flatten()
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        episode_violation_not_count = True
        
        action_shield = None
        
        
        # Evaluate the policy every 'evaluate_freq' steps
        if episode_idx % args.evaluate_freq_epi == 0:
            # evaluate RL performance
            evaluate_num += 1
            evaluate_reward, win_rate = evaluate_policy(args, env_evaluate, agent, state_norm)
            evaluate_rewards.append(evaluate_reward)
            # print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
            # Save the rewards
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([total_steps, evaluate_reward, 1.0, episode_violation/episode_idx if episode_idx>0 else 0, win_rate])
                
            if episode_idx > 0:
                # evaluate SPL
                gate.eval()
                index_id = min(current_size, 100000)
                X = torch.tensor(train_data[:index_id]).float().to(device)
                y = torch.tensor(train_label[:index_id]).float().to(device)
            
                thetas = gate(X)
                cmpe.set_params(thetas, log_space=True)
            
                mpe = cmpe.get_mpe_inst(X.shape[0])
                preds = (mpe > 0).long()
            
                # print(y.shape)  # [n, m]
                # print(preds.shape) # [n, m]
                
                y_binary = (y > 0.5).long()
                valid_mask = (y != 0.5)
                
                valid_preds = preds[valid_mask]
                valid_labels = y_binary[valid_mask]
                correct = (valid_preds == valid_labels).sum().item()
                total = valid_mask.sum().item()
                
                accuracy = correct / total if total > 0 else 0.0
                updated_acc = accuracy
                
                with open(csv_sl_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([episode_idx, accuracy, current_size])
                
                # exit(0)
                
        episode_idx += 1
        
        current_label = [0.5 for _ in range(args.action_dim)]
        last_label = [0.5 for _ in range(args.action_dim)]
        performa_last = [0.5 for _ in range(args.action_dim)]

        while not done:
        
            X = torch.tensor(s).float().unsqueeze(0).to(device)
            thetas = gate(X)
            cmpe.set_params(thetas, log_space=True)
            
            mpe = cmpe.get_mpe_inst(X.shape[0])
            action_shield = (mpe > 0).long()
                
            episode_steps += 1
            a, a_logprob = agent.choose_action(s, action_shield, 1.0)  # Action and the corresponding log probability
            # print(env.state*5)
            # print(a)
            # print(env.decode_action(a))
            s_, r, done, performa = env.step(a)
            s_ = s_.flatten()
            # print(performa)
            # exit(0)
            
            if total_steps > epsilon_start_decay:
            
                epsilon = epsilon - epsilon_decay if epsilon - epsilon_decay > epsilon_min else epsilon_min
            
            if performa is not None:
                current_label[a] = performa
                key = (tuple(s.tolist()), tuple(current_label))
                if key not in seen_samples:
                    train_data[current_ptr] = s
                    train_label[current_ptr] = current_label
                    
                    seen_samples.add(key)
                
                    current_ptr = (current_ptr + 1) % 100000
                    current_size = current_size + 1
            else:
                if episode_violation_not_count:
                    episode_violation += 1
                    episode_violation_not_count = False
                    # print(action_shield)
                    # print(a)
                    # print(a_logprob)
                    # print(s)
                    # print(env.state)
                    # index_list = [i for i, v in enumerate(action_shield[0].tolist()) if v == 1]
                    # for A in [1, 5, 6, 11, 12, 16, 18, 22, 26]:
                    #     print(f'{A}: {env.decode_action(A)}')
                    
                    # exit(0)
                

            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False
                    
                
            if done:
                current_label = [0.5 for _ in range(args.action_dim)] 
                episode_violation_not_count = True

            replay_buffer.store(s, a, a_logprob, r, s_, dw, done, None if performa_last is None else last_label)
            s = s_
            total_steps += 1
            performa_last = performa
            last_label = current_label
            

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps, gate, cmpe, True if updated_acc >= args.th_gate else False)
                replay_buffer.count = 0
                
            if total_steps % args.update_gate_freq == 0 and current_ptr >= args.batch_size_gate:
                
                for i in range(args.update_gate_freq):
                    gate.train()
                    gate.zero_grad()
                
                    batch_size = min(args.batch_size_gate, current_size)
                    indices = np.random.choice(min(100000, current_size), size=batch_size, replace=False)
                
                    states = train_data[indices]
                    labels = train_label[indices]
                    X = torch.tensor(np.array(states)).float().to(device)
                    y = torch.tensor(np.array(labels)).float().to(device)
                
                    thetas = gate(X)
                
                    cmpe.set_params(thetas, log_space=True)
                
                    cross_entropy = cmpe.cross_entropy_mar(y, log_space=True)
                
                    loss = 1.0 * cross_entropy.mean()
                
                    loss.backward()
                    optimizer.step()
        # exit(0)

            


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(2e5), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    
    # gating function
    parser.add_argument("--num_units_gate", type=int, default=128, help="Number of units in each layer of the gating function")
    parser.add_argument("--num_layers_gate", type=int, default=3, help="Number of layers in the gating function")
    parser.add_argument("--num_reps_gate", type=int, default=1, help="Number of components in the mixture")
    parser.add_argument("--lr_gate", type=float, default=2e-4, help="Learning rate of the gating function")
    parser.add_argument("--update_gate_freq", type=int, default=1000, help="The freq to update gate")
    parser.add_argument("--batch_size_gate", type=int, default=128, help="Batch size for gate")
    
    parser.add_argument("--env_name", type=str, default="sudoku2", help="Name of the environment")
    parser.add_argument("--evaluate_freq_epi", type=float, default=50, help="Evaluate the policy every 'evaluate_freq' steps")
    
    # algorithm setting
    parser.add_argument("--algorithm_name", type=str, default="DDQN", help="--")
    
    
    parser.add_argument("--alpha_r", type=float, default=0.001, help="--")
    parser.add_argument("--alpha_kl", type=float, default=0.001, help="--")
    parser.add_argument("--is_Markov", type=bool, default=True, help="--")
    parser.add_argument("--th_gate", type=float, default=0.7, help="--")
    
    
    parser.add_argument("--epsilon_init", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_min", type=float, default=0.1, help="Minimum epsilon")
    parser.add_argument("--epsilon_start_decay", type=float, default=0.1, help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--epsilon_decay_steps", type=float, default=0.2, help="How many steps before the epsilon decays to the minimum")

    args = parser.parse_args()
    
    
    print(args)

    for seed in [0, 1, 2, 3, 4]:
        main(args, env_name=args.env_name, number=1, seed=seed)

    
