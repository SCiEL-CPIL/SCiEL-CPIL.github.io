import itertools
import random
import numpy as np
import time

def construct():
    n = 16
    vertices = range(n)
    
    # Precompute candidates
    all_8_sets = list(itertools.combinations(range(n), 8))
    cand_indices_A = []
    cand_indices_B = []
    
    for s in all_8_sets:
        v0 = s[0]
        rest = s[1:]
        for comb in itertools.combinations(rest, 3):
            a = (v0,) + comb
            b = tuple(x for x in s if x not in a)
            cand_indices_A.append(a)
            cand_indices_B.append(b)
            
    num_cands = len(cand_indices_A)
    cand_A = np.array(cand_indices_A)
    cand_B = np.array(cand_indices_B)
    
    def is_q3_free(A):
        # A is 16x16 numpy array
        # M[i, j, k] = A[cand_A[i, j], cand_B[i, k]]
        M = A[cand_A[:, :, None], cand_B[:, None, :]]
        row_sums = M.sum(axis=2)
        col_sums = M.sum(axis=1)
        is_q3 = (row_sums == 3).all(axis=1) & (col_sums == 3).all(axis=1)
        return not np.any(is_q3)

    # Initialize with a random graph
    random.seed(42)
    A = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                A[i, j] = 1
                A[j, i] = 1
                
    # Ensure it's Q3-free initially (if not, remove edges)
    while not is_q3_free(A):
        # Remove a random edge
        i, j = random.sample(range(n), 2)
        if A[i, j] == 1:
            A[i, j] = 0
            A[j, i] = 0

    best_A = A.copy()
    best_count = np.sum(A) // 2

    start_time = time.time()
    
    # Local Search
    while time.time() - start_time < 110:
        # Try to add an edge
        i, j = random.sample(range(n), 2)
        if A[i, j] == 0:
            A[i, j] = 1
            A[j, i] = 1
            if is_q3_free(A):
                count = np.sum(A) // 2
                if count > best_count:
                    best_count = count
                    best_A = A.copy()
            else:
                A[i, j] = 0
                A[j, i] = 0
        else:
            # Remove an edge (to escape local maxima)
            A[i, j] = 0
            A[j, i] = 0
            
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_A[i, j] == 1:
                result.append([i, j])
                
    return result
