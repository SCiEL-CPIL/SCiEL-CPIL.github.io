<<<<<<< SEARCH
"""Quartet-consensus phylogenetic reconstruction with NJ fallback and deadline guard."""
import math
import time
import numpy as np

DEADLINE_SECONDS = 260.0
SAFETY_MARGIN = 10.0


def _elapsed(start):
    return time.time() - start


def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor distances using NumPy."""
    n = len(labels)
    L = len(sequences[labels[0]])
    # Map chars to ints: A=0, C=1, G=2, T=3
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_array = np.zeros((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        for j in range(L):
            seq_array[i, j] = char_map.get(seq[j], 0)
    
    # Compute Hamming distances vectorized per pair
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diffs = np.count_nonzero(seq_array[i] != seq_array[j])
            p = diffs / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix
=======
"""Phylogenetic reconstruction with NJ + bootstrap voting and deadline guard."""
import math
import time
import numpy as np
from collections import Counter

DEADLINE_SECONDS = 260.0
SAFETY_MARGIN = 15.0


def _elapsed(start):
    return time.time() - start


def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor distances using NumPy vectorization."""
    n = len(labels)
    L = len(sequences[labels[0]])
    # Build integer array using numpy for speed
    seq_array = np.empty((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        codes = np.frombuffer(seq.encode(), dtype=np.uint8)
        # Map: A=65, C=67, G=71, T=84
        seq_array[i] = np.where(codes == 65, 0, np.where(codes == 67, 1, np.where(codes == 71, 2, 3)))
    
    # Compute pairwise Hamming distances vectorized
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        diffs = np.count_nonzero(seq_array[i] != seq_array[i+1:], axis=1)
        p = diffs / L
        p_clipped = np.clip(p, 0, 0.7499)
        d_vals = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
        d_vals[p >= 0.7499] = 3.0
        dist_matrix[i, i+1:] = d_vals
        dist_matrix[i+1:, i] = d_vals
    return dist_matrix
>>>>>>> REPLACE

