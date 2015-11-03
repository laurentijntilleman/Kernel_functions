# -*- coding: utf-8 -*-
"""
Created on: Tue Nov 2 2015
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Example of the use of the gap weighted subsequence kernel. 
Here, a kernel PCA is performed on a set of glycosylases
"""

from Bio import SeqIO
from gap_weighted import gap_weighted_subsequence_Gram
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA

decomposer = KernelPCA(n_components=2, kernel='precomputed')

sequences = []
for item in SeqIO.parse("glycosylases.fasta", "fasta"):
    sequences.append(str(item.seq))

ss_lengths = [3, 5, 10]
gap_weights = [0.2, 0.5, 0.9]

fig, axes = plt.subplots(nrows=3, ncols=3)

for gap_w, axes_h in zip(gap_weights, axes):
    wss_kernel_matrices = gap_weighted_subsequence_Gram(sequences, ss_length=10,
                                gap_weight=gap_w, full=True)
    for ax, ss_len in zip(axes_h, ss_lengths):
        X = decomposer.fit_transform(wss_kernel_matrices[:, :, ss_len-1])
        ax.scatter(X[:,0], X[:,1])
        ax.set_title('wss kernel, gap weight={0}\nss length = {1}'.format(gap_w,
                     ss_len))

fig.show()