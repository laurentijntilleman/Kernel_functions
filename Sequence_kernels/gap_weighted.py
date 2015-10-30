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

@numba.jit
def dyn_gap_weighted_subseq(seq1, seq2, DPS, DP, kernels, ss_length, gap_weight):
    length_1 = len(seq1)
    length_2 = len(seq2)
    for i in range(length_1):
        for j in range(length_2):
            if seq1[i] == seq2[j]:
                DPS[i, j] = gap_weight**2
    for l in range(1, ss_length):
        for i in range(length_1):
            for j in range(length_2):
                DP[i+1, j+1] = DPS[i, j] + gap_weight*DP[i, j+1] +\
                    gap_weight*DP[i+1, j] - gap_weight**2*DP[i, j]
                if seq1[i] == seq2[j]:
                   DPS[i, j] = gap_weight**2*DP[i, j]
                   kernels[l] = kernels[l] + DPS[i, j]
    return kernels
    
if __name__ == '__main__':
    
    string1 = np.array(list('logaritm'))
    string2 = np.array(list('algorithm'))
    
    l1 = len(string1)
    l2 = len(string2)
    
    print dyn_gap_weighted_subseq(string1, string2, np.zeros((l1, l2)),np.zeros((l1+1, l2+1)), np.zeros(5),5, 0.6)