# -*- coding: utf-8 -*-
"""
Created on: Fri Oct 30 2015
Last update: Tue Nov 10 2015
@author: Michiel Stock
michielfmstock@gmail.com
An implementation of the gap weighted subsequence kernel,
makes use of numba's Just In Time compilation. On the benchmark this is
about 250 times faster than the naive implementation
Python 3.5 version
"""

import numpy as np
import numba
import functools

@numba.jit(nopython=True)
def dyn_gap_weighted_subseq(seq1, seq2, DPS, DP, kernel_values, ss_length,
                            gap_weight):
    """
    Performas the calculation for the gap weigthed subsequence kernels
    """
    length1 = len(seq1)
    length2 = len(seq2)
    for i in range(length1):
        for j in range(length2):
            if seq1[i] == seq2[j]:
                DPS[i, j] = gap_weight**2
    for l in range(1, ss_length):
        for i in range(length1):
            for j in range(length2):
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
        - ss_length : length of the substrings (int)
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
        alphabet = list(set(sequence1) | set(sequence2))
    alphabet_dict = {char : i for i, char in enumerate(alphabet)}

    # sequences to integers
    length1 = len(sequence1)
    length2 = len(sequence2)
    seq1_array = np.array(list(map(lambda c : alphabet_dict[c], sequence1)))
    seq2_array = np.array(list(map(lambda c : alphabet_dict[c], sequence2)))

    # make dynamic prog matrices
    DPS = np.zeros((length1, length2))
    DP = np.zeros((length1 + 1, length2 + 1)) 
    kernel_values = np.zeros(ss_length)
    
    # calculate the kernel values
    if char_weigths is None:
        kernel_values = dyn_gap_weighted_subseq(seq1_array, seq2_array, DPS,
                                    DP, kernel_values, ss_length, gap_weight)
    else:
        pass# complete the implementation for character weighted kernel
                   
    if full:
        return kernel_values
    else: return kernel_values[-1]
    
def gap_weighted_subsequence_Gram(sequences, ss_length, gap_weight,
                            test_sequences=None, alphabet=None,
                            char_weigths=None, full=False):
    """
    Computes the Gram matrix using the gap weighted subsequences kernel
    Either do this for a set of training sequences or, optionally return 
    a set of test sequences to generate a non-square Gram matrix (rows=train)
    
    Inputs : 
        - sequences : a list of sequences (list of strings)
        - ss_length : length of the substrings (int)
        - gap_weight : weight of the length of the subsequence (float)
        - test_sequences : optionally, give list of sequences to calculate
                    the kernel values of the test set to
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
        alphabet = functools.reduce(lambda x, y : x | y, list(map(set, sequences)), set([]))
    alphabet_dict = {char : i for i, char in enumerate(alphabet)}

    # sequences to integers
    char_to_int = lambda char : alphabet_dict[char]
    sequences_array = [np.array(list(map(char_to_int, seq))) for seq in sequences]
    if test_sequences is not None:
        sequences_array_test = [np.array(list(map(char_to_int, seq))) for seq\
                                in test_sequences]
    
    kernel_values = np.zeros(ss_length)
    # if no test sequences
    if test_sequences is None:
        if not full:
            gram = np.zeros((len(sequences), len(sequences)))
        else:
            gram = np.zeros((len(sequences), len(sequences), ss_length))
        for i in range(len(sequences)):
            for j in range(i, len(sequences)):
                length1 = len(sequences[i])
                length2 = len(sequences[j])
                DPS = np.zeros((length1, length2))
                DP = np.zeros((length1 + 1, length2 + 1)) 
                kernel_values[:] = 0.0
                if char_weigths is None:
                        kernel_values = dyn_gap_weighted_subseq(sequences_array[i],
                                    sequences_array[j], DPS, DP, kernel_values,
                                    ss_length, gap_weight)
                else:
                    pass  # fill for weighted
                if full:
                    gram[i,j,:] = kernel_values
                    gram[j,i,:] = kernel_values
                else:
                    gram[i,j] = kernel_values[-1]
                    gram[j,i] = kernel_values[-1]
    else:  # if with respect to test set
        if not full:
            gram = np.zeros((len(sequences), len(test_sequences)))
        else:
            gram = np.zeros((len(sequences), len(test_sequences), ss_length))
        for i in range(len(sequences)):
            for j in range((len(test_sequences))):
                length1 = len(sequences[i])
                length2 = len(test_sequences[j])
                DPS = np.zeros((length1, length2))
                DP = np.zeros((length1 + 1, length2 + 1))
                kernel_values[:] = 0.0
                if char_weigths is None:
                    kernel_values = dyn_gap_weighted_subseq(sequences_array[i],
                                    sequences_array_test[j], DPS, DP, kernel_values,
                                    ss_length, gap_weight)
                else:
                    pass  # fill for weighted
                if full:
                    gram[i,j,:] = kernel_values
                else:
                    gram[i,j] = kernel_values[-1]
    return gram

if __name__ == '__main__':
    
    #string1 = np.array(list('logaritm'))
    #string2 = np.array(list('algorithm'))
    
    string1 = 'MFKKRGRQTVLIAAVLAFFTASSPLLARTQGEPTQVQQKLAALEKQSGGRLGVALINTADRSQILYRGDERFAMCSTSKTMVAAAVLKQSETQHDILQQKMVIKKADLTNWNPVTEKYVDKEMTLAELSAATLQYSDNTAMNKLLEHLGGTSNVTAFARSIGDTTFRLDRKEPELNTAIPGDERDTTCPLAMAKSLHKLTLGDALAGAQRAQLVEWLKGNTTGGQSIRAGLPEGWVVGDKTGAGDYGTTNDIAVIWPEDRAPLILVTYFTQPQQDAKGRKDILAAAAKIVTEGL'
    string2 = 'MRNRGFGRRELLVAMAMLVSVTGCARHASGARPASTTLPAGADLADRFAELERRYDARLGVYVPATGTTAAIEYRADERFAFCSTFKAPLVAAVLHQNPLTHLDKLITYTSDDIRSISPVAQQHVQTGMTIGQLCDAAIRYSDGTAANLLLADLGGPGGGTAAFTGYLRSLGDTVSRLDAEEPELNRDPPGDERDTTTPHAIALVLQQLVLGNALPPDKRALLTDWMARNTTGAKRIRAGFPADWKVIDKTGTGDYGRANDIAVVWSPTGVPYVVAVMSDRAGGGYDAEPREALLAEAATCVAGVLA'
    
    alphabet = set(string1) | set(string2)

    
    print(gap_weighted_subsequence(string1, string2, ss_length=3, gap_weight=0.5,
                                   full=True))
