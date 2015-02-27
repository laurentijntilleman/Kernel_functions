"""
Created on Sun Dec 28 2014
Last update: Sun Dec 28 2014

@author: Michiel Stock
michielfmstock@gmail.com

Implementation of some useful machine learning algorithms needed
for the efficient inference project:
    - methods for multi-task learning, tailored for large data sets
    - methods to extract features from protein sequences
    - methods to calculate (approximations of the) diffusion kernel
"""


import random as rd
import numpy as np
from scipy import sparse
from math import factorial
from scipy.sparse.linalg import eigsh
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import SGDClassifier
from Bio import pairwise2
from Bio.SubsMat.MatrixInfo import blosum45


Z_1 = {'A': 0.07, 'V' : -2.69, 'L' : -4.19, 'I' : -4.44, 'P' : -1.22, 'F' : -4.92, 'W': -4.75, 'M':-2.49, 'K': 2.84, 'R' : 2.88, 'H': 2.41, 'G' :2.23, 'S': 1.96, 'T' : 0.92, 'C' : 0.71, 'Y' : -1.39, 'N' : 3.22, 'Q' : 2.18, 'D' : 3.64, 'E' : 3.08}

Z_2 = {'A': -1.73, 'V' : -2.53, 'L' : -1.03, 'I' : -1.68, 'P' : 0.88, 'F' : 1.3, 'W': 3.65, 'M':-.27, 'K': 1.41, 'R' : 2.52, 'H': 1.74, 'G' :-5.36, 'S': -1.63, 'T' : -2.09, 'C' : 0.71, 'Y' : 2.32, 'N' : 0.01, 'Q' : 0.53, 'D' : 1.13, 'E' : 0.39}

Z_3 = {'A': 0.09, 'V' : -1.29, 'L' : -0.98, 'I' : -1.03, 'P' : 2.23, 'F' : 0.45, 'W': 0.85, 'M':-.41, 'K': -3.14, 'R' : -3.44, 'H': 1.11, 'G' :0.30, 'S': 0.57, 'T' : -1.4, 'C' :4.13 , 'Y' : 0.01, 'N' : 0.84, 'Q' : -1.14, 'D' : 2.36, 'E' : -0.07}


#########################################################
##########              Methods                ##########
#########################################################

'''
def myPLS(X, Y, n_components = 2, regularization = 1.0):
    """
    Implements a 'smart' version of Partial Least Squares when the number of instances i much larger than the dimensionality of the in-and outputs
    Input: X, Y and the number of latent features to be calculated
    Output: W and C.T, such that Y = X * W * C.T
    """
    XTY = X.T.dot(Y)
    E, W = eigsh(XTY.dot(XTY.T) , n_components)
    E, C = eigsh(XTY.T.dot(XTY) + regularization * np.eye(XTY.shape[1]) , n_components)
    del XTY
    return W, C.T

'''

def myPLS(X, Y, n_components = 2):
    """
    Implements a 'smart' version of Partial Least Squares when the number of instances i much larger than the dimensionality of the in-and outputs
    Input: X, Y and the number of latent features to be calculated
    Output: W and C.T, such that Y = X * W * C.T
    """
    K = n_components #for ease of use
    N, Mx = X.shape
    N, My = Y.shape
    C = np.zeros((My, K)) #projection matrix of Y
    U = np.zeros((Mx, K)) #projection matrix of X
    P = np.zeros((Mx, K)) #loadings of X
    Xi = X + 0
    for i in range(K):
        print 'Training for component %s of %s' %(i+1, K)
        XiY = Xi.T.dot(Y)
        XTX = Xi.T.dot(Xi)
        ui = eigsh(XiY.dot(XiY.T),1)[1] #calculate first eigenvector of this thing
        scalingfactor = (ui.T.dot(XTX).dot(ui))
        pi = XTX.dot(ui)/scalingfactor
        ci = XiY.T.dot(ui)/scalingfactor
        Xi -= Xi.dot(ui).dot(pi.T)
        U[:, i] = ui.reshape(-1)
        C[:, i] = ci.reshape(-1)
        P[:, i] = pi.reshape(-1)
    del XiY
    del XTX
    return U.dot(np.linalg.inv(P.T.dot(U))), C.T

def Multi_SVM(X, Y, Reg = 1):
    """
    Trains a number of SVC's for each of the tasks
    """
    if len(X) < 100000:
        model  = LinearSVC(C = 0.1, dual = False )
    else:
        model = SGDClassifier(alpha = 0.1)
    W = np.zeros((X.shape[1], Y.shape[1]))
    for t in range(Y.shape[1]):
        if Y[:,t].var() > 0:
            model.fit(X, Y[:,t])
            W[:,t] = model.coef_
        print 'Traing SVC %s of %s' %(t+1, Y.shape[1])
    return W

def FDA_relabeling(Y):
    '''
    Relabels a matrix with labels y in {+1, -1} or {+1, 0} to a matrix
    which can beused for FDA, i.e. + => N/N_+, - => -N/N_-
    '''
    N, T = Y.shape
    for t in xrange(T):
        Npos = np.sum(Y[:,t] > 0 )*1.0
        Nneg = N - Npos
        Y[Y[:,t] > 0, t] = N/Npos
        Y[Y[:,t] <= 0, t] = -N/Nneg
    return Y

def instance_AUC(Y, X, W):
    '''
    Calculates the AUC over the ROWS

    P = X * W
    '''
    N, T = Y.shape
    AUC = 0.0
    Counts = 0
    for i in range(N):
        if Y[i].var() > 0:
            AUC += auc(Y[i], X[i].dot(W))
            Counts += 1
    return AUC/Counts

def Multi_Ridge(X, Y, lamdb = 1.0):
    '''
    Ridge regression in primal space for multiple outputs
    '''
    N, P = X.shape
    return np.linalg.inv(X.T.dot(X) + lamdb*np.eye(P)).dot((Y.T.dot(X)).T)


def Multi_Ridge_recoded(X, Y, lamdb = 1.0, C1 = 1, C2 = 1):
    '''
    Ridge regression in primal space for multiple outputs (classification)
    Recodes such that negative instances get a weight of -1 (absolute weight can be given)
    '''
    N, P = X.shape
    B = np.linalg.inv(X.T.dot(X) + lamdb*np.eye(P)).dot(X.T)
    return B.dot(Y)*(C1 + C2) - C2*np.ones(Y.shape)


#########################################################
##########           Methods Sparse               ##########
#########################################################

'''
def myPLS(X, Y, n_components = 2, regularization = 1.0):
    """
    Implements a 'smart' version of Partial Least Squares when the number of instances i much larger than the dimensionality of the in-and outputs
    Input: X, Y and the number of latent features to be calculated
    Output: W and C.T, such that Y = X * W * C.T
    """
    XTY = X.T.dot(Y)
    E, W = eigsh(XTY.dot(XTY.T) , n_components)
    E, C = eigsh(XTY.T.dot(XTY) + regularization * np.eye(XTY.shape[1]) , n_components)
    del XTY
    return W, C.T

'''


def myPLS_sparse(X, Y, n_components = 2):
    """
    Implements a 'smart' version of Partial Least Squares when the number of instances i much larger than the dimensionality of the in-and outputs
    Input: X, Y and the number of latent features to be calculated
    Output: W and C.T, such that Y = X * W * C.T
    """
    Y = Y.tocsc()
    K = n_components #for ease of use
    N, Mx = X.shape
    N, My = Y.shape
    C = np.zeros((My, K)) #projection matrix of Y
    U = np.zeros((Mx, K)) #projection matrix of X
    P = np.zeros((Mx, K)) #loadings of X
    Xi = X + 0
    for i in range(K):
        print 'Training for component %s of %s' %(i+1, K)
        XiY = (Y.T.dot(Xi)).T #for if Y is sparse
        XTX = Xi.T.dot(Xi)
        ui = eigsh(XiY.dot(XiY.T),1)[1] #calculate first eigenvector of this thing
        scalingfactor = (ui.T.dot(XTX).dot(ui))
        pi = XTX.dot(ui)/scalingfactor
        ci = XiY.T.dot(ui)/scalingfactor
        Xi -= Xi.dot(ui).dot(pi.T)
        U[:, i] = ui.reshape(-1)
        C[:, i] = ci.reshape(-1)
        P[:, i] = pi.reshape(-1)
    del XiY
    del XTX
    return U.dot(np.linalg.inv(P.T.dot(U))), C.T

def Multi_SVM_sparse(X, Y, Reg = 1):
    """
    Trains a number of SVC's for each of the tasks
    """
    Y = Y.tocsc()
    if len(X) < 100000:
        model  = LinearSVC(C = 0.1, dual = False )
    else:
        model = SGDClassifier(alpha = 0.1)
    W = np.zeros((X.shape[1], Y.shape[1]))
    for t in range(Y.shape[1]):
        y = np.array(Y[:,t].todense()).ravel()
        if y.var() > 0:
            model.fit(X, y)
            W[:,t] = model.coef_
        print 'Traing SVC %s of %s' %(t+1, Y.shape[1])
    return W


def instance_AUC_sparse(Y, X, W):
    '''
    Calculates the AUC over the ROWS

    P = X * W
    '''
    Y = Y.tocsr()
    N, T = Y.shape
    AUC = 0.0
    Counts = 0
    for i in range(N):
        y = np.array(Y[i].todense()).ravel()
        if y.var() > 0:
            AUC += auc(y, X[i].dot(W))
            Counts += 1
    return AUC/Counts

def Multi_Ridge_sparse(X, Y, lamdb = 1.0):
    '''
    Ridge regression in primal space for multiple outputs
    '''
    N, P = X.shape
    Y = Y.tocsc()
    return np.linalg.inv(X.T.dot(X) + lamdb*np.eye(P)).dot((Y.T.dot(X)).T)



#########################################################
##########         Protein features            ##########
#########################################################

def local_alignment_score(seq1, seq2, substitution_matrix = blosum45,
        print_alignment = False):
    '''
    Returns the score obtained by locally aligning seq1 and seq2,
    standardly uses blosum45 (distantly related proteins)
    optionally prints the alignment
    '''
    gap_open = -10
    gap_extend = -0.5
    alignment = pairwise2.align.localds(seq1, seq2, substitution_matrix,
            gap_open, gap_extend)[0]
    aln1, aln2, score, begin, end = alignment
    if print_alignment:
        print aln1
        print aln2
    return score


def calc_AC( seq, lag, Za, Zb = None ):
    if Zb is None:
        Zb = Za
    AC = 0
    n = len(seq)
    for i in range( n - lag ):
        if Za.has_key(seq[ i ]) and Zb.has_key(seq[ i + lag ]):
            AC += ( Za[ seq[ i ] ] * Zb[ seq[ i + lag ] ] )/( n - lag )
    return AC


def ProteinFeatures(sequence, lagRange = range(1, 25), discriptors = [Z_1, Z_2, Z_3]):
    """
    Calculates features of protein sequences based on lagged correlation
    of physicochemical properties of the amino acids
    """
    k = len(lagRange)
    p = len(discriptors)
    n_features = (p + (p*(p-1))/2)*k
    x = np.zeros(n_features)
    ind = 0
    for lag in lagRange:
        for Z in discriptors:
           x[ind] =  calc_AC(sequence, lag, Z)
           ind += 1
        for i in range(p-1):
            for j in range(i+1, p):
                x[ind] =  calc_AC(sequence, lag, discriptors[i], discriptors[j])
                ind += 1
    return x


def NormalizedSpectrumKernel(seq1, seq2, k = 3):
    """
    Makes the normalized spectrum kernel with k-mer length of k for
    the sequences seq1 and seq2
    """
    seq1 = seq1.upper()
    seq2 = seq2.upper()
    value = 0.
    substr1 = [seq1[i:i+k] for i in range(0, len(seq1) - k + 1)]
    substr2 = [seq2[i:i+k] for i in range(0, len(seq2) - k + 1)]
    for s in (set(substr1) & set(substr2)):
        value += min([substr1.count(s), substr2.count(s)])
    return value

#########################################################
##########      Kernel approximations          ##########
#########################################################

def Nystrom(kernel, instances, n_components):
    """
    For a list of instances and a kernel function generate an
    approximate decomposition of dimension n_components
    """
    n = len(instances)
    m = n_components
    backbones = [rd.choice(instances) for i in range(m)]
    W = np.zeros((m,m))
    for i in range(m):
        for j in range(i, m):
            inst_i = backbones[i]
            inst_j = backbones[j]
            W[i,j] = kernel(inst_i, inst_j)
            W[j,i] = W[i,j]
    E = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            inst_i = instances[i]
            inst_j = backbones[j]
            E[i,j] = kernel(inst_i, inst_j)
    eval, evec = np.linalg.eig(W)
    eval[eval < 0 ] = 0
    return (E.dot(evec).dot(np.diag(eval**0.5))).real

#########################################################
##########               Graphs                ##########
#########################################################

def TaylorApproximatedDiffusion(i, j, L, beta = 1.0, order = 8):
    """
    Generates a approximation of diffusion kernel between
    nodes i and j, with laplacian L and leak-rate of beta
    """
    L = L.tocsr()
    if i == j:
        k = 1.0
    else:
        k = 0.0
    k += beta*L[i,j]
    lj = L[j].T
    li = L[i]
    for ord in range(2, order + 1):
        li = li.dot(L)
        k += (1.0/factorial(ord))*(beta**ord)*li.dot(lj)[0,0]
    return k

def DecomposedDiffusionKernel(L, beta = None, n_components = 100):
    '''
    Calculates an decomposed approximation of the diffusion kernel for a graph.
    Standardly uses the first 100 eigenvalues and eigenvectors for the decomposition.
    If no beta is provided the eigenvalues will be normalized such that the largest eigenvalue equals to one
    '''
    E, V = eigsh(L, n_components)
    if beta:
        return V * np.exp( beta * 0.5 * E)
    else:
        E /= E.max()
        return V * np.exp( 0.5 * E)

def makeLaplacian(l1, l2):
    """
    Generates the Laplacian matrix for a series of links l1, l2
    such that l1[i] - l2[i]
    """
    if len(l1) != len(l2):
        print 'l1 and l2 have to be of equal length!'
        raise IndexError
    n_edges = len(l1)
    n_nodes = max((max(l1), max(l2))) + 1 #count from 0!
    L = sparse.coo_matrix(([1]*n_edges, (l1, l2)), shape = (n_nodes,n_nodes))
    d = np.array(L.sum(0))[0]
    L = L.tocsr()
    L = L + (L.T).tocsr()
    #L.setdiag(-d)
    return L, d
