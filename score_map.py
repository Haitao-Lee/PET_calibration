# conding = utf-8
import matplotlib.pyplot as plt
import numpy as np


def score_map(alpha, theta):
    map = np.zeros([256, 256])
    for n in range(0, 256):
        for m in range(129, 256):
            if (n-128)/(m-128) <= np.tan(theta) - 1e-7:
                map[n, m] = (alpha*(m-128) - (1-alpha)*(n-128))/((n-128)**2+(m-128)**2)
    tmp = 0.5
    map[map > tmp*map.mean()] = map[map > tmp*map.mean()]*map[map > tmp*map.mean()]/map.mean()/tmp
    map[map > 0] = np.log(map[map > 0])
    map[map < 0] = 0
    return map


if __name__ == '__main__':
    alphas = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1])
    thetas = np.pi*np.array([1/4, 1/3, 5/12, 1/2])
    tmp = [r'$\frac{\pi}{4}$', r'$\frac{\pi}{3}$', r'$\frac{5\pi}{12}$', r'$\frac{\pi}{2}$']
    score_maps = []
    for theta in thetas:
        alpha_maps = []
        for alpha in alphas:
            alpha_maps.append(score_map(alpha, theta))
        score_maps.append(alpha_maps)
    delta = 24
    l = 128-delta
    u = 128+delta
    for n in range(thetas.shape[0]):
        for m in range(alphas.shape[0]):
            plt.matshow(score_maps[n][m][l:u, l:u], cmap='afmhot')
            # plt.title('%s = %s,    %s = %s' % (r'$\alpha$', str(alphas[m]), r'$\theta$', tmp[n]), fontsize=40)
            plt.title('%s = %s' % (r'$\alpha$', str(alphas[m])), fontsize=40)
            plt.colorbar()
            plt.plot(delta, delta, 'b*', markersize=20)
            plt.show()
    
                
