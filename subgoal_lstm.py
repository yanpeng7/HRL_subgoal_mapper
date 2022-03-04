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

def custom_collate(data):
    expert_traj, learner_traj = data[0][0], data[0][1]
    learner_traj = np.concatenate(([[1.747999, 0.7, 0.7893]], learner_traj))
    return expert_traj, learner_traj

class SubgoalLSTM:
    def __init__(self, train_set, validation_set, normalizer, args, device):
        self.args = args
        self.device = device
        self.train_set = train_set
        self.validation_set = validation_set
        self.normalizer = normalizer
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.args.batch_size,
                                                        shuffle=True, collate_fn=custom_collate)
        self.val_loader = torch.utils.data.DataLoader(self.validation_set, batch_size=self.args.batch_size,
                                                      shuffle=True, collate_fn=custom_collate)
        self.criterion = nn.MSELoss(reduction='mean')
        attn = Attention(args, device)
        encoder = EncoderRNN(args, device)
        decoder = AttnDecoderRNN(attn, args, device)
        self.model = Seq2Seq(encoder, decoder, normalizer, args, device).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5, 10], gamma=self.args.scheduler_gamma)

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        bundled_batch_loss = 0
        epoch_seq_len_loss = 0
        bundled_batch_seq_len_loss = 0

        for i, (src_unnormalized, trg_unnormalized) in enumerate(tqdm(self.train_loader)):
            src = self.normalizer.transform(src_unnormalized)
            trg = self.normalizer.transform(trg_unnormalized)
            src = torch.tensor(src).float().to(self.device)
            trg = torch.tensor(trg).float().to(self.device)

            self.optimizer.zero_grad()
            output, seq_len_diff = self.model(src, trg, self.args.teacher_forcing_ratio)

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output = output.squeeze(1)

            trg = trg[1:]
            output = output[1:]
            loss = self.criterion(output, trg)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip_value)

            self.optimizer.step()

            output_denormalized = self.normalizer.inverse_transform(output.detach().cpu().numpy())
            denormalized_loss = self.calc_avg_distance(trg_unnormalized[1:, :], output_denormalized)

            epoch_loss += denormalized_loss
            bundled_batch_loss += denormalized_loss
            epoch_seq_len_loss += seq_len_diff
            bundled_batch_seq_len_loss += seq_len_diff
            # epoch_loss += loss.item()
            # bundled_batch_loss += loss.item()

            if i % self.args.print_freq == 0 and i != 0:
                mse_loss = bundled_batch_loss / self.args.print_freq
                average_bundled_batch_seq_len_loss = bundled_batch_seq_len_loss / self.args.print_freq
                tqdm.write(f'epoch: {epoch}, iter: {i}, train coordinate loss: {mse_loss:.5f}, average seq len diff:'
                           f'{average_bundled_batch_seq_len_loss}')
                bundled_batch_loss = 0
                bundled_batch_seq_len_loss = 0

        return epoch_loss / len(self.val_loader)

    def validation_epoch(self, epoch):
        self.model.eval()
        epoch_loss = 0
        bundled_batch_loss = 0
        epoch_seq_len_loss = 0
        bundled_batch_seq_len_loss = 0

        with torch.no_grad():
            for i, (src_unnormalized, trg_unnormalized) in enumerate(self.val_loader):
                src = self.normalizer.transform(src_unnormalized)
                trg = self.normalizer.transform(trg_unnormalized)
                src = torch.tensor(src).float().to(self.device)
                trg = torch.tensor(trg).float().to(self.device)

                output, seq_len_diff = self.model(src, trg, 0)  # turn off teacher forcing
                # trg = [trg len, batch size]
                # output = [trg len, batch size, output dim]
                output = output.squeeze(1)

                # trg = [(trg len - 1) * batch size]
                # output = [(trg len - 1) * batch size, output dim]
                trg = trg[1:]
                output = output[1:]
                loss = self.criterion(output, trg)

                output_denormalized = self.normalizer.inverse_transform(output.detach().cpu().numpy())
                denormalized_loss = self.calc_avg_distance(trg_unnormalized[1:, :], output_denormalized)

                epoch_loss += denormalized_loss
                bundled_batch_loss += denormalized_loss
                epoch_seq_len_loss += seq_len_diff
                bundled_batch_seq_len_loss += seq_len_diff

                # epoch_loss += loss.item()
                # bundled_batch_loss += loss.item()

                if i % self.args.print_freq == 0 and i != 0:
                    mse_loss = bundled_batch_loss / self.args.print_freq
                    average_bundled_batch_seq_len_loss = bundled_batch_seq_len_loss / self.args.print_freq
                    tqdm.write(
                        f'epoch: {epoch}, iter: {i}, validation coordinate loss: {mse_loss:.5f}, average seq len diff:'
                        f'{average_bundled_batch_seq_len_loss}')
                    bundled_batch_loss = 0
                    bundled_batch_seq_len_loss = 0

        return epoch_loss / len(self.val_loader)

    def train(self):
        val_loss = self.validation_epoch(0)
        for epoch in range(1, self.args.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validation_epoch(epoch)
        torch.save(self.model.state_dict(), 'seq2seq_joint_5.pt')

    def load_from_pretrained(self, pt_path):
        self.model.load_state_dict(torch.load(pt_path))

    @staticmethod
    def visualize_traj(expert_subgoals, true_learner_subgoals, gen_learner_subgoals):
        expert_subgoals = np.concatenate(([[1.747999, 0.7, 0.7893]], expert_subgoals), axis=0)
        true_learner_subgoals = np.concatenate(([[1.747999, 0.7, 0.7893]], true_learner_subgoals), axis=0)
        ax = plt.axes(projection='3d')
        # ax.plot3D(expert_subgoals[0], expert_subgoals[1], expert_subgoals[2], 'gray')
        ax.scatter3D(true_learner_subgoals[:, 0], true_learner_subgoals[:, 1], true_learner_subgoals[:, 2], c='green')
        ax.scatter3D(gen_learner_subgoals[:, 0], gen_learner_subgoals[:, 1], gen_learner_subgoals[:, 2], c='red')
        ax.scatter3D(expert_subgoals[:, 0], expert_subgoals[:, 1], expert_subgoals[:, 2], c='blue')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])
        plt.show()

    # seq_len * 3
    @staticmethod
    def calc_avg_distance(true, pred):
        point_wise_err = np.linalg.norm((true-pred), axis=1)
        # print(point_wise_err)
        return np.mean(point_wise_err)

    # need learner trejectory so we can use its first subgoal, no other subgaols used as teach_forcing_ratio=0
    # expect seq_len * 3
    def predict(self, expert_trajectory, learner_trajectory, visualize=False):
        with torch.no_grad():
            learner_trajectory = np.concatenate(([[1.747999, 0.7, 0.7893]], learner_trajectory))
            expert_trajectory_norm = self.normalizer.transform(expert_trajectory)
            learner_trajectory_norm = self.normalizer.transform(learner_trajectory)
            expert_trajectory_in = torch.tensor(expert_trajectory_norm).float().to(self.device)
            learner_trajectory_in = torch.tensor(learner_trajectory_norm).float().to(self.device)
            pred, seq_len_diff = self.model(expert_trajectory_in, learner_trajectory_in, 0)
            pred = pred.squeeze(1)
            pred = pred[1:]
            # loss = self.criterion(learner_trajectory_in, pred).cpu().numpy()
            pred = pred.cpu().numpy()
            denormed_pred = self.normalizer.inverse_transform(pred)
            loss = self.calc_avg_distance(learner_trajectory[1:,:], denormed_pred)
        # print(expert_trajectory_norm)
        # print('------')
        # print(learner_trajectory_norm)
        # print('------')
        # print(pred)
        if visualize is True:
            self.visualize_traj(expert_trajectory_norm, learner_trajectory_norm, pred)
        return pred, loss


def preprocess_data():
    train, val = load_dataset()
    tr_data, tr_label = train[0], train[1]
    val_data, val_label = val[0], val[1]
    tr_set = np.stack((tr_data, tr_label), axis=1)
    val_set = np.stack((val_data, val_label), axis=1)
    return tr_set, val_set


if __name__ == '__main__':
    # val_set = pickle.load(open("real_val_757.pkl", "rb"))
    # scaler = joblib.load('normalizer_scaler.save')
    #
    # loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=custom_collate)
    # expert, learner = next(iter(loader))
    # learner = scaler.transform(learner)
    # print(expert)
    # print(learner)
    pass
