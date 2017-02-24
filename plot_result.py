import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import interpolate
from scipy import ndimage

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
        result = results[index[i]:index[i+1]]
        reward, var = get_Mean_Variance(result)
        total_rewards.append(reward)
    return total_rewards

def generate_smooth(data):
    y = data
    x = np.linspace(1,100,len(y))

    x_sm = np.array(x)
    y_sm = np.array(y)

    # resample to lots more points - needed for the smoothed curves
    x_smooth = np.linspace(x_sm.min(), x_sm.max(), 500)

    # spline - always goes through all the data points x/y
    y_spline = interpolate.spline(x, y, x_smooth)

    spl = interpolate.UnivariateSpline(x, y)

    sigma = 2
    x_g1d = ndimage.gaussian_filter1d(x_sm, sigma)
    y_g1d = ndimage.gaussian_filter1d(y_sm, sigma)
    return x_g1d, y_g1d

def plot_2_smooth(A,B,file_name):
    x1,y1 = generate_smooth(A)
    x2,y2 = generate_smooth(B)
    x_sm = np.array(np.linspace(1,100,len(y2)))
    y1_sm = np.array(A)
    y2_sm = np.array(B)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1,1,1)

    ax.plot(x_sm, y1_sm, 'green', linewidth=1,ls='--',color='g',label = 'raw data: Baseline')
    ax.plot(x_sm, y2_sm, linewidth=1,ls='--',label = 'raw data: Q-learn')
    ax.plot(x1,y1, 'magenta', linewidth=1.2,ls='solid', color='r',label = 'smoothed data:Baseline')
    ax.plot(x2,y2, 'magenta', linewidth=1.2, ls='solid', label='smoothed data: Q-learn')
    ax.set_ylabel('Average Reward')
    ax.set_xlabel('Epoch')
    ax.set_yscale('log')
    ax.grid('on')
    lgd = ax.legend(loc=2, bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)
    plt.show()
    fig.savefig(file_name,box_extra_artists=(lgd,), bbox_inches='tight')

def plot_with_smooth(data,file_name):
    y = data
    x = np.linspace(1 ,100,len(y))

    # convert both to arrays
    x_sm = np.array(x)
    y_sm = np.array(y)

    # resample to lots more points - needed for the smoothed curves
    x_smooth = np.linspace(x_sm.min(), x_sm.max(), 500)

    # spline - always goes through all the data points x/y
    y_spline = interpolate.spline(x, y, x_smooth)

    spl = interpolate.UnivariateSpline(x, y)

    sigma = 2
    x_g1d = ndimage.gaussian_filter1d(x_sm, sigma)
    y_g1d = ndimage.gaussian_filter1d(y_sm, sigma)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1,1,1)

    ax.plot(x_sm, y_sm, 'green', linewidth=1,ls='--',color='g',label = 'raw data')
    # plt.plot(x_smooth, y_spline, 'red', linewidth=1)
    # plt.plot(x_smooth, spl(x_smooth), 'yellow', linewidth=1)
    ax.plot(x_g1d,y_g1d, 'magenta', linewidth=1.2,ls='solid', color='r',label = 'smoothed data')
    ax.set_ylabel('Average Reward')
    ax.set_xlabel('Epoch')
    # ax.set_xscale('log')
    ax.grid('on')
    lgd = ax.legend(loc=2, bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)
    plt.show()
    fig.savefig(file_name,box_extra_artists=(lgd,), bbox_inches='tight')

def plot_3_smooth(A,B,C,file_name):
    x1,y1 = generate_smooth(A)
    x2,y2 = generate_smooth(B)
    x3,y3 = generate_smooth(C)

    x_sm = np.array(np.linspace(1,100,len(y2)))
    y1_sm = np.array(A)
    y2_sm = np.array(B)
    y3_sm = np.array(C)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1,1,1)

    ax.plot(x_sm, y1_sm, linewidth=1,ls=':',label = 'Raw data: LA_4')
    ax.plot(x_sm, y2_sm, linewidth=1,ls=':',label = 'Raw data: LA_6')
    ax.plot(x_sm, y3_sm, linewidth=1,ls=':',label = 'Raw data: LA_8')

    ax.plot(x1,y1, linewidth=1.2, ls='solid', label = 'smoothed data: LA_4')
    ax.plot(x2,y2, linewidth=1.2, ls='solid', label='smoothed data: LA_6')
    ax.plot(x3,y3, linewidth=1.2, ls='solid', label='smoothed data: LA_8')
    ax.set_ylabel('Average Reward')
    ax.set_xlabel('Epoch')
#    ax.set_yscale('log')
    ax.grid('on')
    lgd = ax.legend(loc=2, bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)
    plt.show()
    fig.savefig(file_name,box_extra_artists=(lgd,), bbox_inches='tight')

# data = getReward('q344_0050109.csv')
# data = getReward('total_reward_results.csv')
# data2 = getReward('q346_0050109.csv')
# data3 = getReward('q348_0050109.csv')
data = getReward('q_table_less_0050011.csv')

# mean = np.mean(total_rewards)
# variance = np.var(total_rewards)

#plot_results(data)
plot_with_smooth(data, 'q_table_less_0050011.pdf')
print("Mean:{0}, Var:{1}").format(np.mean(data),np.var(data))
#plot_2_smooth(data2,data,'compare_R_Q.pdf')
#plot_3_smooth(data,data2,data3, 'compare_LA.pdf')
# print mean, variance
