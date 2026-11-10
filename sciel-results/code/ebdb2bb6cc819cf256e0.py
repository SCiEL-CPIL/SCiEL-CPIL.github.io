import time
import numpy as np
from math import gcd, log2
import random

def construct():
    deadline = time.time() + 55
    
    N = 512
    odds = [i for i in range(1, N + 1) if i % 2 == 1]
    num_odds = len(odds)
    
    # Precompute C
    C_mat = np.zeros((num_odds, num_odds), dtype=np.int64)
    for i in range(num_odds):
        for j in range(num_odds):
            g = gcd(odds[i], odds[j])
            l = (odds[i] // g) * odds[j]
            if l <= N:
                C_mat[i, j] = int(log2(N / l))
            else:
                C_mat[i, j] = -1
                
    def get_score(M):
        if not M:
            return 0
        M_list = list(M)
        # Vectorized min?
        # C_mat[np.ix_(M_list, M_list)]
        sub_C = C_mat[np.ix_(M_list, M_list)]
        min_vals = sub_C.min(axis=1)
        if np.any(min_vals < 0):
            return -1
        return int(np.sum(min_vals) + len(M_list))

    best_global_score = 0
    best_global_M = {0}
    
    # Run multiple restarts
    num_restarts = 50
    for _ in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Random start
        # Pick a random valid clique? Or just start with {0}
        current_M = {0}
        current_score = get_score(current_M)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            best_M = current_M
            best_score = current_score
            
            # Add
            for i in range(num_odds):
                if i in current_M:
                    continue
                new_M = current_M | {i}
                new_score = get_score(new_M)
                if new_score > best_score:
                    best_score = new_score
                    best_M = new_M
                    improved = True
                    
            # Remove
            for i in list(current_M):
                if len(current_M) == 1:
                    continue
                new_M = current_M - {i}
                new_score = get_score(new_M)
                if new_score > best_score:
                    best_score = new_score
                    best_M = new_M
                    improved = True
                    
            # Swap
            if not improved:
                # Randomized swap
                if current_M:
                    i = random.choice(list(current_M))
                    # Try a few random j
                    for _ in range(10):
                        j = random.randint(0, num_odds - 1)
                        if j in current_M:
                            continue
                        new_M = (current_M - {i}) | {j}
                        new_score = get_score(new_M)
                        if new_score > best_score:
                            best_score = new_score
                            best_M = new_M
                            improved = True
                            break
                            
            current_M = best_M
            current_score = best_score
            
        if current_score > best_global_score:
            best_global_score = current_score
            best_global_M = current_M

    # Construct result from best_global_M
    result = []
    M_set = best_global_M
    if M_set:
        M_list = list(M_set)
        sub_C = C_mat[np.ix_(M_list, M_list)]
        min_vals = sub_C.min(axis=1)
        for idx, i in enumerate(M_list):
            K = min_vals[idx]
            if K >= 0:
                m_val = odds[i]
                for k in range(K + 1):
                    val = m_val * (2 ** k)
                    if val <= N:
                        result.append(val)
    
    if not result:
        result = [1]
        
    return result
