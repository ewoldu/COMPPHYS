import numpy as np
import matplotlib.pyplot as plt
import h5py
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense

def local_energy(lattice, J, i, j):
    '''
    Calculates the energy of a specific point (i, j) on the lattice
    by adding up its interaction with each of its nearest neighbors
    according to E = -J * sum(s_ij*s_neighbor).
    '''
    #Every term of the sum has s_ij in it, so we can factor it out
    return -1 * J * lattice[i,j] * neighbors_sum(lattice, i, j)

def total_energy(lattice, J):
    '''
    Calculates the total energy of the lattice by counting all of the
    interactions while making sure that no interactions are double counted.
    Do so by calculating the local energy of every other lattice point
    '''
    energy_sum = 0
    for i in range(lattice.shape[0]):
        for j in range(lattice.shape[1]):
            if (i + j) % 2 == 0:
                energy_sum += local_energy(lattice, J, i, j)
    return energy_sum

def energy(num):
    lattice = np.empty(3,3)
    for r in range(3):
        for c in range(3):
            lattice[r][c] = (num // 2**(M*r + c)) % 2
    return total_energy(lattice, 1)

X = np.shuffle(np.array(np.arange(2**9)))
n_train = int(.9 * len(X))
n_test = len(X) - n_train
energies = energy(X)
avg = (np.max(energies) - np.min(energies))/2
y = energies > avg
y = 1*y
X = np.array(X)[np.newaxis].T
X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=n_test, random_state=1)

model = Sequential()

    

