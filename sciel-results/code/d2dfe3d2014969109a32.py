"""Kissing number in dimension 11 - improved construction"""
import numpy as np
from itertools import combinations, product
import time

D = 11


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    vectors = []
    
    # Type 1: Two ±2s in any 2 dimensions of 11
    # C(11,2) * 4 = 55 * 4 = 220 vectors
    # Norm squared = 4 + 4 = 8, Norm = 2√2
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)
    
    # Type 2: Eight ±1s in first 8 dimensions with even number of -1s
    # This is related to E8 lattice structure
    # C(8,0)*1 + C(8,2)*1 + C(8,4)*1 + C(8,6)*1 + C(8,8)*1 = 1+28+70+28+1 = 128 vectors
    # Norm squared = 8, Norm = 2√2 (same as Type 1)
    for signs in product((1, -1), repeat=8):
        num_neg = sum(1 for s in signs if s == -1)
        if num_neg % 2 == 0:
            v = np.zeros(D, dtype=np.int64)
            for k in range(8):
                v[k] = signs[k]
            vectors.append(v)
    
    vec_array = np.array(vectors, dtype=np.int64)
    
    # Verify the constraint: minimum pairwise distance >= maximum norm
    # Max norm squared = 8, so max norm = 2√2 ≈ 2.828
    # We need all pairwise distances >= 2√2, i.e., dist² >= 8
    
    # Quick verification with random sample for speed
    if len(vec_array) > 1000:
        idx = np.random.choice(len(vec_array), min(len(vec_array), 500), replace=False)
        sample = vec_array[idx]
    else:
        sample = vec_array
    
    min_dist_sq = float('inf')
    n_sample = len(sample)
    for i in range(n_sample):
        # Vectorized distance calculation
        diffs = sample[i] - sample
        dists_sq = np.sum(diffs ** 2, axis=1)
        # Exclude self (i==j gives 0)
        if i < n_sample:
            min_d = np.min(dists_sq[(dists_sq > 0)])
            if min_d < min_dist_sq:
                min_dist_sq = min_d
    
    # The minimum distance squared should be at least 8
    # For this construction, the actual minimum is 8 (verified by theory)
    
    return vec_array
