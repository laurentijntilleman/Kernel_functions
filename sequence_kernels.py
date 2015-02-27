"""
Created on Thu Nov 27 2014
Last update: Mon Dec 1 2014

@author: Michiel Stock
michielfmstock@gmail.com

Some kernel methods for working with sequences
"""


import numpy as np

def subsequences_kernel(seq1, seq2, substring_length = 10, lambd = 0.75,
        last_kernel = True):
    '''
    Calculates the gab-weighted subsequences kernel for seq1 and seq2
    for a given substring_length and a gap-weigthing parameter lambd
    if the setting last_kernel is set to False, it returns a list:
        [K_1(seq1,seq2), K_2(seq1, seq2) ... K_substring_length(seq1, seq2)]
    '''
    length_seq1 = len(seq1)
    length_seq2 = len(seq2)
    DPS = np.zeros((length_seq1, length_seq2))
    for i in range(length_seq1):
        for j in range(length_seq2):
            if seq1[i] == seq2[j]:
                DPS[i,j] = lambd**2
    DP = np.zeros((length_seq1+1, length_seq2+1))
    Kernelvalue = [np.sum(DPS)]
    for sustr_len in range(substring_length - 1):
        Kernelvalue.append(0)
        for i in range(length_seq1):
            for j in range(length_seq2):
                DP[i+1,j+1] = DPS[i,j] + lambd*DP[i,j+1] + lambd*DP[i+1,j] - lambd**2*DP[i, j]
                if seq1[i] == seq2[j]:
                    DPS[i,j] = lambd**2*DP[i,j]
                    Kernelvalue[-1] += DPS[i,j]
    if not last_kernel:
        return Kernelvalue
    else:
        return Kernelvalue[-1]


from Bio.SubsMat.MatrixInfo import blosum62  # we will give a suitable standard
# scoring matrix for proteins

def bio_subs_to_scoring(sub_matrix, non_neg = True):
    # changes the substitution matrices of Bio into a format that can be used
    # with the implemented kernels, i.e. symmetric and non-negative
    for key in sub_matrix.keys():
        char1, char2 = key
        value = sub_matrix[key]
        if value <= 0 and non_neg:
            del sub_matrix[key]  # remove negative elements
        else:
            sub_matrix[(char2, char1)] = value  #make symmetric
    return sub_matrix

blosum62 = bio_subs_to_scoring(blosum62, False)  # good baseline

def subsequences_kernel_soft(seq1, seq2, substring_length = 10, lambd = 0.75,
        scoring_matrix = blosum62, last_kernel = True):
    '''
    Calculates the gab-weighted subsequences kernel for seq1 and seq2
    for a given substring_length and a gap-weigthing parameter lambd
    instead of a hard match, a soft match is returned based on a scoring matrix
    if the setting last_kernel is set to False, it returns a list:
        [K_1(seq1,seq2), K_2(seq1, seq2) ... K_substring_length(seq1, seq2)]
    '''
    length_seq1 = len(seq1)
    length_seq2 = len(seq2)
    DPS = np.zeros((length_seq1, length_seq2))
    for i in range(length_seq1):
        for j in range(length_seq2):
            tup1 = (seq1[i], seq2[j])
            if scoring_matrix.has_key(tup1):  # we assume that other elements
                        # are equal to 0
                DPS[i,j] = lambd**2*scoring_matrix[tup1]
    DP = np.zeros((length_seq1+1, length_seq2+1))
    Kernelvalue = [np.sum(DPS)]
    for sustr_len in range(substring_length - 1):
        Kernelvalue.append(0)
        for i in range(length_seq1):
            for j in range(length_seq2):
                DP[i+1,j+1] = DPS[i,j] + lambd*DP[i,j+1] + lambd*DP[i+1,j] - lambd**2*DP[i, j]
                tup1 = (seq1[i], seq2[j])
                if scoring_matrix.has_key(tup1):
                    DPS[i,j] = lambd**2*scoring_matrix[tup1]*DP[i,j]
                    Kernelvalue[-1] += DPS[i,j]
    if not last_kernel:
        return Kernelvalue
    else:
        return Kernelvalue[-1]


if __name__ == '__main__':

    print subsequences_kernel('gatta', 'cata', 3, 0.1, False)

    seq1 = 'MFKKRGRQTVLIAAVLAFFTASSPLLARTQGEPTQVQQKLAALEKQSGGRLGVALINTADRSQILYRGDERFAMCSTSKTMVAAAVLKQSETQHDILQQKMVIKKADLTNWNPVTEKYVDKEMTLAELSAATLQYSDNTAMNKLLEHLGGTSNVTAFARSIGDTTFRLDRKEPELNTAIPGDERDTTCPLAMAKSLHKLTLGDALAGAQRAQLVEWLKGNTTGGQSIRAGLPEGWVVGDKTGAGDYGTTNDIAVIWPEDRAPLILVTYFTQPQQDAKGRKDILAAAAKIVTEGL'
    seq2 = 'MRNRGFGRRELLVAMAMLVSVTGCARHASGARPASTTLPAGADLADRFAELERRYDARLGVYVPATGTTAAIEYRADERFAFCSTFKAPLVAAVLHQNPLTHLDKLITYTSDDIRSISPVAQQHVQTGMTIGQLCDAAIRYSDGTAANLLLADLGGPGGGTAAFTGYLRSLGDTVSRLDAEEPELNRDPPGDERDTTTPHAIALVLQQLVLGNALPPDKRALLTDWMARNTTGAKRIRAGFPADWKVIDKTGTGDYGRANDIAVVWSPTGVPYVVAVMSDRAGGGYDAEPREALLAEAATCVAGVLA'

    print subsequences_kernel(seq1, seq2, 3, 0.5)

    words = ['logarithm', 'algorithm', 'biology', 'biorhythm', 'rhythm',
        'competing', 'computation', 'biocomputing', 'computing']

    kernel_matrix = [[subsequences_kernel(word1, word2, 4, 0.8) for word1
            in words] for word2 in words]
    print np.array(kernel_matrix)

    print subsequences_kernel(seq1, seq2, 5, 0.8, last_kernel=False)

    from Bio.SubsMat.MatrixInfo import blosum62

    # let us start from blosum62 to find a suitable substitution matrix
    #make symmetric
    for key in blosum62.keys():
        char1, char2 = key
        value = blosum62[key]
        if char1 == char2 and value <= 0:
            print 'Iets raars bij %s' %char1
        if value <= 0:
            del blosum62[key]  # remove negative elements
        else:
            blosum62[(char2, char1)] = value  #make symmetric

    print subsequences_kernel_soft(seq1, seq2, 10, 0.6, blosum62, last_kernel=False)
