import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from traj_generator import load_dataset
import time
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from lstm_models import EncoderRNN, AttnDecoderRNN, Seq2Seq, Attention
from sklearn.preprocessing import MinMaxScaler
import pickle
import joblib


def generate_normalizer(A, B, joint_num):
    def format_data(data):
        sample_list = []
        for sample in range(len(data)):
            for timestep in range(len(data[sample])):
                sample_list.append(data[sample][timestep])
        sample_list.append(np.array([1.747999, 0.7, 0.7893]))
        return np.stack(sample_list, axis=0)
    A_stacked = format_data(A)
    B_stacked = format_data(B)
    C_stacked = np.concatenate((A_stacked, B_stacked), axis=0)
    scaler = MinMaxScaler()
    scaler.fit(C_stacked)
    scaler.transform(C_stacked)
    scaler_filename = f"normalizer_scaler_joint_{joint_num}.save"
    joblib.dump(scaler, scaler_filename)


def strip_1(data):
    # strips ending 1
    total_number = 0
    for sample in range(len(data)):
        for timestep in range(len(data[sample])):
            data[sample][timestep] = data[sample][timestep][:-1]
            total_number += 3
    return data


def combine_expert_learner(expert, learner):
    combined = []
    for i in range(len(expert)):
        expert_seq = np.stack(expert[i], axis=0)
        learner_seq = np.stack(learner[i], axis=0)
        combined.append([expert_seq, learner_seq])
    return combined
    # with open("combined.pkl", "wb") as filehandle:
    #     pickle.dump(combined, filehandle)


def process_data(expert_name, learner_name, joint_num):
    A = pickle.load(open(expert_name, "rb"))
    B = pickle.load(open(learner_name, "rb"))
    A, B = strip_1(A), strip_1(B)
    generate_normalizer(A, B, joint_num)
    combined_dataset = combine_expert_learner(A, B)
    split_train_val(combined_dataset, joint_num)


def visualize_sample(expert_subgoals, true_learner_subgoals):
    expert_subgoals = np.transpose(expert_subgoals)
    true_learner_subgoals = np.transpose(true_learner_subgoals)
    ax = plt.axes(projection='3d')
    # ax.plot3D(expert_subgoals[0], expert_subgoals[1], expert_subgoals[2], 'gray')
    ax.scatter3D(true_learner_subgoals[0], true_learner_subgoals[1], true_learner_subgoals[2], c='green')
    ax.scatter3D(expert_subgoals[0], expert_subgoals[1], expert_subgoals[2], c='blue')
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 2])
    ax.set_zlim([0, 2])
    plt.show()


def custom_collate(data):
    expert_traj, learner_traj = data[0][0], data[0][1]
    return expert_traj, learner_traj


def split_train_val(combined_dataset, joint_num):
    print(f'length of dataset: {len(combined_dataset)}')
    train = combined_dataset[:6000]
    val = combined_dataset[6000:]
    with open(f"train_joint_{joint_num}.pkl", "wb") as filehandle:
        pickle.dump(train, filehandle)
    with open(f"val_joint_{joint_num}.pkl", "wb") as filehandle:
        pickle.dump(val, filehandle)




if __name__ == '__main__':
    process_data('trainset_expert_j5.data', 'trainset_learner_j5.data', 5)

    # goal_lists = pickle.load(open("goallist.data", "rb"))
    # # print(goal_lists)
    # a = pickle.load(open("trainset_expert_cut.data", "rb"))
    # b = pickle.load(open("trainset_learner_cut.data", "rb"))

    # print(c.shape)
    # combine_expert_learner(a, b)
    # generate_normalizer(a, b)
    # a_mod = preprocess(a)

    # combined = pickle.load(open("combined.pkl", "rb"))
    # split_train_val(combined)
    # print(len(combined))
    # scaler = joblib.load('normalizer_scaler.save')
    # sample_num = 2
    # train_loader = torch.utils.data.DataLoader(combined, batch_size=1, shuffle=False, collate_fn=custom_collate)
    # expert, learner = next(iter(train_loader))
    # print(learner)

    # first_input = torch.tensor([[1.747999, 0.7, 0.7893]])
    # print(first_input.shape)


    # visualize_sample(combined[sample_num][0], combined[sample_num][1])
    # print(combined[sample_num][0])
    # print(combined[sample_num][1])



    # for i in range(len(a)):
    #     print(len(a[i]))

    # a = [[['A1', 'A1'], ['A2', 'A2']], [['a1', 'a1'], ['a2', 'a2']]]
    # b = [[['B1', 'B1'], ['B2', 'B2']], [['b1', 'b1'], ['b2', 'b2']]]
    # c = list(zip(a, b))
    # print(c)



