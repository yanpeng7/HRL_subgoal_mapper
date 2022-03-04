import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def getEquidistantPoints(p1, p2, parts):
    dist = np.linalg.norm(p2 - p1)
    x_list = np.linspace(p1[0], p2[0], parts)
    y_list = np.linspace(p1[1], p2[1], parts)
    z_list = np.linspace(p1[2], p2[2], parts)
    z_list_modified = z_list.copy()
    for i in range(len(z_list)):
        curr_point = np.array([x_list[i], y_list[i], z_list[i]])
        curr_dist = np.linalg.norm(curr_point - p1)
        z_list_modified[i] += 0.2 * np.sin((2*np.pi)/dist*curr_dist)

    return np.asarray([x_list, y_list, z_list]), np.asarray([x_list, y_list, z_list_modified])


def gen_traj(init_pt, num_subgoals, edge_limit=1):
    end_goal = np.random.uniform(0.2, edge_limit, 3)
    expert_subgoals, learner_subgoals = getEquidistantPoints(init_pt, end_goal, num_subgoals)
    return expert_subgoals, learner_subgoals


def traj_dataset(num_samples, init_pt, num_subgoals, edge_limit=1):
    traj_data_list = []
    traj_label_list = []
    for i in range(num_samples):
        data, label = gen_traj(init_pt, num_subgoals, edge_limit=edge_limit)
        traj_data_list.append(data)
        traj_label_list.append(label)
    traj_data_list = np.stack(traj_data_list, axis=0)
    traj_label_list = np.stack(traj_label_list, axis=0)
    bundled = np.array([traj_data_list, traj_label_list])
    np.save('train_10000.npy', bundled)
    print(bundled.shape)


def visualize_traj(expert_subgoals, learner_subgoals):
    ax = plt.axes(projection='3d')
    ax.plot3D(expert_subgoals[0], expert_subgoals[1], expert_subgoals[2], 'gray')
    ax.scatter3D(learner_subgoals[0], learner_subgoals[1], learner_subgoals[2], cmap='Greens');
    ax.scatter3D(expert_subgoals[0], expert_subgoals[1], expert_subgoals[2], cmap='Reds');
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    plt.show()


def load_dataset():
    train_set = np.load('./train_10000.npy')
    val_set = np.load('./val_1000.npy')
    return train_set, val_set


if __name__ == "__main__":
    # expert_goals, learner_goals = gen_traj(np.array([0, 0, 0]), num_subgoals=8)
    traj_dataset(10000, [0,0,0], 10, edge_limit=1)
    # train, val = load_dataset()
    # print(train.shape)
    # visualize_traj(train[0, 2], train[1, 2])
    # visualize_traj(expert_goals, learner_goals)
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # plt.show()
