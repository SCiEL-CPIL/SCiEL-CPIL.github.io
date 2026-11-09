import itertools
import numpy as np
import time
import random

def generate_vectors_for_norm(N, d=11):
    vecs = []
    if N == 3:
        # Three 1s
        for idxs in itertools.combinations(range(d), 3):
            for signs in itertools.product([1, -1], repeat=3):
                v = np.zeros(d, dtype=np.int64)
                for idx, s in zip(idxs, signs):
                    v[idx] = s
                vecs.append(v)
    elif N == 4:
        # Four 1s
        for idxs in itertools.combinations(range(d), 4):
            for signs in itertools.product([1, -1], repeat=4):
                v = np.zeros(d, dtype=np.int64)
                for idx, s in zip(idxs, signs):
                    v[idx] = s
                vecs.append(v)
        # One 2, two 1s
        for idx2 in range(d):
            for idx1s in itertools.combinations([i for i in range(d) if i != idx2], 2):
                for s2 in [1, -1]:
                    for s1s in itertools.product([1, -1], repeat=2):
                        v = np.zeros(d, dtype=np.int64)
                        v[idx2] = 2 * s2
                        for idx, s in zip(idx1s, s1s):
                            v[idx] = s
                        vecs.append(v)
    else:
        # Fallback or other N
        pass
    return np.array(vecs)

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580 # Safety margin
    
    best_points = np.array([np.ones(11), -1 * np.ones(11)]).astype(np.int64)
    best_k = 22
    
    # Try N=3
    V3 = generate_vectors_for_norm(3)
    T3 = 1 # u.v <= 1
    
    # Try N=4
    V4 = generate_vectors_for_norm(4)
    T4 = 1 # u.v <= 1
    
    def find_subset(V, T, time_limit):
        n = len(V)
        if n == 0: return np.array([]).reshape(0, 11)
        
        # Precompute dot products
        # V is (n, 11)
        # M = V @ V.T
        M = V @ V.T
        
        best_set = []
        best_size = 0
        
        while time.time() < time_limit:
            # Random permutation
            perm = np.random.permutation(n)
            current_set = []
            current_indices = []
            
            for i in perm:
                # Check compatibility with current_set
                if not current_set:
                    current_set.append(i)
                else:
                    # Check dot product with all in current_set
                    # M[i, current_set] <= T
                    if np.all(M[i, current_set] <= T):
                        current_set.append(i)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set
                
        return V[best_set]

    # Run search for N=3
    t_limit = min(deadline, time.time() + 100)
    if time.time() < t_limit:
        pts3 = find_subset(V3, T3, t_limit)
        if len(pts3) > best_k:
            best_k = len(pts3)
            best_points = pts3
            
    # Run search for N=4
    t_limit = min(deadline, time.time() + 200)
    if time.time() < t_limit:
        pts4 = find_subset(V4, T4, t_limit)
        if len(pts4) > best_k:
            best_k = len(pts4)
            best_points = pts4
            
    # Final check
    # Verify best_points
    if len(best_points) >= 2:
        norms = np.sum(best_points**2, axis=1)
        max_norm_sq = np.max(norms)
        
        min_dist_sq = float('inf')
        k = len(best_points)
        for i in range(k):
            for j in range(i+1, k):
                diff = best_points[i] - best_points[j]
                d_sq = np.sum(diff**2)
                if d_sq < min_dist_sq:
                    min_dist_sq = d_sq
                    
        if min_dist_sq > max_norm_sq:
            return best_points
        else:
            # Fallback to seed
            return np.array([np.ones(11), -1 * np.ones(11)]).astype(np.int64)
    else:
        return np.array([np.ones(11), -1 * np.ones(11)]).astype(np.int64)
