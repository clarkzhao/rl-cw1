import numpy as np
import matplotlib.pyplot as plt
import sys

def get_Mean_Variance(A):
    frames = A[:,1]
    episode = A.shape[0]
    rewards = A[:,2]
    total_frame = frames[-1]
    pre_frames = np.append(0,frames)[:episode]
    frame_per_action = frames - pre_frames
    mean = 0.
    # print np.sum(rewards)
    mean = np.sum(frame_per_action * rewards)*1./total_frame
    variance = np.sum(frame_per_action*(rewards-mean)*(rewards-mean).T)*1./total_frame
    return mean,variance
def get_mean_variance_simple(A):
    mean = np.mean(A[:,2])
    variance = np.var(A[:,2])
    return mean, variance

def get_total_reward(A):
    return A[-1,-1]

def plot_results(A):
    kernel = np.array([0.5] * 1)
    kernel = kernel / np.sum(kernel)

    fig = plt.figure(figsize=(16, 10))
    plt.plot(np.convolve(A, kernel, mode='same'), '-')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylabel('Avearage Reward')
    ax.set_xlabel('Epoch')
    fig.savefig('result_q_4_05.pdf')

def plot_multi_res(A,B):
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(A, label='3-line')
    ax.plot(B, label='4-line')
    ax.set_ylabel('Avearage Reward')
    ax.set_xlabel('Epoch')
    ax.legend(loc=0)
    fig.savefig('result_3_6.pdf')


def getReward(file_name):
    with open(file_name,'rb') as f_handle:
        results = np.loadtxt(f_handle, delimiter=",",
            dtype='int32')
    pre = 1
    index = [0]
    # Split the data for each episode and
    # store the index of the new episode in array index
    for i in range(results.shape[0]):
        ep = results[i][0]
        if ep != pre:
            index.append(i)
            pre = ep
    index.append(results.shape[0])
    # means = []
    # variances = []
    total_rewards = []
    for i in range(100):
        print i
        result = results[index[i]:index[i+1]]
        reward, _ = get_Mean_Variance(result)
        total_rewards.append(reward)
    return total_rewards

total_rewards1 = getReward('q3_6_1_result.csv')
total_rewards2 = getReward('q3_6_result.csv')

# mean = np.mean(total_rewards)
# variance = np.var(total_rewards)

# plot_results(total_rewards)
plot_multi_res(total_rewards1,total_rewards2)
# print mean, variance
