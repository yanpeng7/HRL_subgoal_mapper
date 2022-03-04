import argparse as ap
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
from lstm_models import EncoderRNN, AttnDecoderRNN
from subgoal_lstm import preprocess_data, SubgoalLSTM
import pickle
import joblib

if __name__ == '__main__':
    parser = ap.ArgumentParser(description="use this to add more arguments")

    # model
    parser.add_argument("--encoder_input_dim", dest="encoder_input_dim", type=int, default=3)
    parser.add_argument("--encoder_hidden_dim", dest="encoder_hidden_dim", type=int, default=200)
    parser.add_argument("--encoder_dropout", dest="encoder_dropout", type=float, default=0.0)
    parser.add_argument("--decoder_hidden_dim", dest="decoder_hidden_dim", type=int, default=200)
    parser.add_argument("--decoder_output_dim", dest="decoder_output_dim", type=int, default=3)
    parser.add_argument("--decoder_dropout", dest="decoder_dropout", type=float, default=0.1)
    parser.add_argument("--max_pred_len", dest="max_pred_len", type=int, default=10)
    parser.add_argument("--final_goal_range", dest="final_goal_range", type=float, default=0.1)

    # training
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=1)
    parser.add_argument("--lr", dest="lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--scheduler_gamma", dest="scheduler_gamma", type=float, default=0.1, help="learning rate")
    parser.add_argument("--gradient_clip", dest="gradient_clip", type=float, default=5,
                        help="gradient_clip_value")
    parser.add_argument("--gradient_clip_value", dest="gradient_clip_value", type=float, default=5,
                        help="gradient_clip_value")
    parser.add_argument("--teacher_forcing_ratio", dest="teacher_forcing_ratio", type=float, default=0.75)
    parser.add_argument("--print_freq", dest="print_freq", type=int, default=200)

    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=10)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tr_set, val_set = preprocess_data()
    #

    tr_set = pickle.load(open("train_joint_5.pkl", "rb"))
    val_set = pickle.load(open("val_joint_5.pkl", "rb"))

    scaler = joblib.load('normalizer_scaler_joint_5.save')

    model = SubgoalLSTM(tr_set, val_set, scaler, args, device)
    model.train()

    # sample_number = 20
    # expert_traj = torch.tensor(val_set[sample_number][0])
    # true_learner_traj = torch.tensor(val_set[sample_number][1])
    #
    # model.load_from_pretrained('seq2seq.pt')
    # prediction, loss = model.predict(expert_traj, true_learner_traj, visualize=True)
    # print(loss)




