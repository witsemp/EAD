import matplotlib.pyplot as plt
import numpy as np
def normal_dist_density_function(std_dev, mean, x):
    f = (1 / (std_dev * np.sqrt(np.pi))) * np.exp((-(x - mean) ** 2) / (2 * std_dev))
    return f
def task1_2():
    args = np.linspace(-5, 5, 100)
    means = [-2, 0, 3, 4]
    std_devs = [1, 2, 3, 4]
    f_vals = [[normal_dist_density_function(std_devs[i], means[i], x) for x in args] for i in range(4)]
    fig, ax = plt.subplots()
    ax.set_title('Rozkład Gaussa', fontsize=16)
    ax.plot(args, f_vals[0], 'or')
    ax.plot(args, f_vals[1], 'x--b')
    ax.plot(args, f_vals[2], ':k')
    ax.plot(args, f_vals[3], '*-.')
    ax.legend(['u = ' + str(means[i]) + ' dev = ' + str(std_devs[i]) for i in range(4)], loc='upper left')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(b=True)
    plt.show()
if __name__ == '__main__':
    task1_2()
