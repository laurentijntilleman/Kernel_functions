# -*- coding: utf-8 -*-
"""
Created on: Fri Oct 30 2015
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

An implementation of the gap weighted subsequence kernel
"""

import numpy as np
import numba

def dyn_gap_weighted_subseq(seq1, seq2, DPS, DP, kernel_values, ss_length,
                            gap_weight):
    """
    Performas the calculation for the gap weigthed subsequence kernels
    """
    length1 = len(seq1)
    length2 = len(seq2)
    for i in xrange(length1):
        for j in xrange(length2):
            if seq1[i] == seq2[j]:
                DPS[i, j] = gap_weight**2
    for l in xrange(1, ss_length):
        for i in xrange(length1):
            for j in xrange(length2):
                DP[i+1, j+1] = DPS[i, j] + gap_weight * DP[i, j+1] +\
                    gap_weight * DP[i+1, j] - gap_weight** 2* DP[i, j]
                if seq1[i] == seq2[j]:
                   DPS[i, j] = gap_weight**2*DP[i, j]
                   kernel_values[l] = kernel_values[l] + DPS[i, j]
    return kernel_values

def gap_weighted_subsequence(sequence1, sequence2, ss_length, gap_weight,
                             alphabet=None, char_weigths=None, full=False):
    """
    Computes the gap weigthed subsequnce kernel betweern two sequences
    
    Inputs : 
        - sequence1 : first sequence (str)
        - sequence2 : second sequence (str)
        - k : length of the substrings (int)
        - gap_weight : weight of the length of the subsequence (float)
        - alphabet : optinally give the alphabet, otherwise deduced from 
                    sequences (str or list of char)
        - char_weights : optionally give weights of char-char similarity (dict)
        - full : return also result for lower length substrings at low 
                    computational cost
    
    Ouputs: either the kernel value or an array of kernel values for different
            lengths
    """
    # initialize the alphabet
    if alphabet is None:
        alphabet = list(set(sequence1) + set(sequence2))
    alphabet_dict = {char : i for i, char in enumerate(alphabet)}
    
    # sequences to integers
    length1 = len(sequence1)
    length2 = len(sequence2)
    seq1_array = np.array(map(lambda c : alphabet_dict[c], sequence1))
    seq2_array = np.array(map(lambda c : alphabet_dict[c], sequence2))
    
    # make dynamic prog matrices
    DPS = np.zeros((length1, length2))
    DP = np.zeros((length1 + 1, length2 + 1))    
    kernel_values = np.zeros(ss_length)
    
    # calculate the kernel values
    if char_weigths is None:
        kernel_values = dyn_gap_weighted_subseq(seq1_array, seq2_array, DPS, DP,
                                    kernel_values, ss_length, gap_weight)
                                    
    if full:
        return kernel_values
    else: return kernel_values[-1]

    
if __name__ == '__main__':
    
    #string1 = np.array(list('logaritm'))
    #string2 = np.array(list('algorithm'))
    
    string1 = np.random.randint(0, 100, size=100)
    string2 = np.random.randint(0, 100, size=123)
    
    l1 = len(string1)
    l2 = len(string2)
    
    print dyn_gap_weighted_subseq(string1, string2, np.zeros((l1, l2)),np.zeros((l1+1, l2+1)), np.zeros(5),5, 0.6)