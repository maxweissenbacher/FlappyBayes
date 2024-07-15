import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    with open('experiment_2mins_boreward/rewards.txt', 'rb') as f:
        rewards = np.loadtxt(f)

    overall_mean = np.mean(rewards)
    
    stds = []
    diffs = []
    rms = []
    
    for n in range(1, 5000):
        
        means = []
        for i in range(0, len(rewards), n):
            m = np.mean(rewards[i: i+n])
            means.append(m)
            
        means= np.asarray(means)
        
        stds.append(np.std(means))
        diffs.append(np.mean(np.abs(overall_mean - means)))
        rms.append(np.linalg.norm(overall_mean - means))
            
    
    plt.plot(stds, label='stds')
    plt.plot(rms, label='absolute rms')
    plt.plot(np.abs(rms/overall_mean), label='relative rms')
    plt.plot(diffs, label='absolute mean abs error')
    plt.plot(np.abs(diffs/overall_mean), label='relative mean abs error')
    plt.legend()
    plt.ylim(0,1)
    plt.show()
    
    plt.plot(rewards)
    plt.title('Rewards')
    plt.show()
    
    