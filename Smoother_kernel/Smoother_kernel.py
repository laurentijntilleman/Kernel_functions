'''
Created on Fri Aug 28 2015
Last update: Mon Aug 28 2015
@author: Michiel Stock
michielfmstock@gmail.com
Version: Python 3.5
Implementation of a smooting kernel:
    $$
    K(x, x') = \alpha + (1-\alpha)\delta(x, x')
    $$
This kernel can be used as an uninformative task-kernel: it only smooths
the tasks such that their output is more similar.
The gram matrix is a weighted sum of an identity matrix and an all-one matrix.
It is easy to show that for ridge regression, the Gram matrix is just a weighted
'''

import numpy as np

# diagonal weight hat matrix
diag_weight = lambda N, a, l : a / (a + l)

# ones weight hat matrix
ones_weight = lambda N, a, l : l * (a - 1) / (a + l) / (N * (a - 1) - a - l)

def generate_smoothing_Gram(n_instances, alpha):
    '''
    Generates the Gram matrix of a smoothing matrix.
    Inputs:
        - n_instances
        - alpha
    Outputs:
        - Gram matrix
    '''
    return (1.0 - alpha) * np.ones((n_instances, n_instances)) +\
            alpha * np.eye(n_instances)


def generate_smoothing_hat(n_instances, alpha, reg):
    '''
    Generates the hat matrix of a smoothing matrix.
    Inputs:
        - n_instances
        - alpha
        - regularization parameter
    Outputs:
        - hat matrix
    '''
    return diag_weight(n_instances, alpha, reg) * np.eye(n_instances) +\
            ones_weight(n_instances, alpha, reg) * np.ones((n_instances, n_instances))

def smooth_predictions(Y, alpha, reg):
    '''
    Smooths the predictions using the hat matrix of a smoothing kernel
    '''
    n_instances, n_tasks = Y.shape
    diag = diag_weight(n_tasks, alpha, reg)
    ones = ones_weight(n_tasks, alpha, reg)
    return Y * diag + np.sum(Y, axis=1, keepdims=True) * ones

def generate_smoothing_leverages(n_instances, alpha, reg):
    '''
    Generates the leverages of a smoothing matrix. Usefull for LOOCV
    Inputs:
        - n_instances
        - alpha
        - regularization parameter
    Outputs:
        - leverages
    '''
    return (diag_weight(n_instances, alpha, reg) +\
            ones_weight(n_instances, alpha, reg)) * np.ones(n_instances)

if __name__ == '__main__':
    n_instances = 100
    alpha = 0.5
    reg = 2.0


    gram = generate_smoothing_Gram(n_instances, alpha)
    print(gram)

    hat = generate_smoothing_hat(n_instances, alpha, reg)
    print(hat)
    print('Check if hat matrix is correct:')
    print(np.allclose(hat, np.linalg.inv(gram + np.eye(n_instances) * reg).dot(gram)))
    print('Check if predictions are correct:')
    Y = np.random.rand(50, n_instances)
    print(np.allclose(Y.dot(hat), smooth_predictions(Y, alpha, reg)))
