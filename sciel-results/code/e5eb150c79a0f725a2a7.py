import numpy as np
import time
import random

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_time = time.time() + 140.0  # Safety margin
    
    # Target tensor T[i, j, k] = 1 for all i, j, k
    # We want U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r to equal 1.
    
    # Try ranks from lower bound up to 41
    # Lower bound for 2x3x7 is roughly 28, but let's start from 35 as a reasonable guess for integer solutions
    # Schoolbook is 42.
    
    best_result = None
    
    # Precompute index mappings for speed
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    for rank in range(35, 42):
        if time.time() > target_time:
            break
            
        # Try a few random restarts for this rank
        max_restarts = 50
        for _ in range(max_restarts):
            if time.time() > target_time:
                break
                
            # Initialize with random integers in {-1, 0, 1} or {0, 1}
            # Using {0, 1} is a good start for "0-1" decompositions
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)
            
            # Random initialization
            for r in range(rank):
                # Sparse initialization often works better
                # Assign random 1s
                for idx in range(n * m):
                    if random.random() < 0.3:
                        U[idx, r] = 1
                for idx in range(m * p):
                    if random.random() < 0.3:
                        V[idx, r] = 1
                for idx in range(p * n):
                    if random.random() < 0.3:
                        W[idx, r] = 1
            
            # Local Search / Hill Climbing
            # We want to minimize the sum of squared errors
            # Error = sum_{i,j,k} ( sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1 )^2
            
            current_error = float('inf')
            # Calculate initial error
            # Efficient calculation:
            # Let M1[i,j,r] = U[i,j,r]
            # Let M2[j,k,r] = V[j,k,r]
            # Let M3[k,i,r] = W[k,i,r]
            # Term for (i,j,k) is dot product of U[i,j,:], V[j,k,:], W[k,i,:]
            
            # To speed up error calculation, we can use tensor contraction
            # But for small dimensions, direct loop is fine.
            
            def calc_error(U, V, W):
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        u_vec = U[i * m + j]
                        for k in range(p):
                            v_vec = V[j * p + k]
                            w_vec = W[k * n + i]
                            # Sum of element-wise product
                            s = np.dot(u_vec, v_vec * w_vec)
                            err += (s - 1.0) ** 2
                return err

            current_error = calc_error(U, V, W)
            
            # If we found a solution
            if current_error < 1e-9:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
            # Local search: flip random entries
            # We can also try adding 1 or -1
            stuck_count = 0
            for step in range(500):
                if time.time() > target_time:
                    break
                if stuck_count > 100:
                    break
                    
                # Pick a random matrix and entry
                mat_choice = random.randint(0, 2)
                if mat_choice == 0:
                    idx = random.randint(0, n * m - 1)
                    r = random.randint(0, rank - 1)
                    old_val = U[idx, r]
                    # Try changing to old_val + 1 or old_val - 1 or 0
                    new_val = old_val + random.choice([-1, 1])
                    if new_val < -2 or new_val > 2:
                        continue
                    U[idx, r] = new_val
                    new_error = calc_error(U, V, W)
                    if new_error < current_error:
                        current_error = new_error
                        stuck_count = 0
                        if current_error < 1e-9:
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    else:
                        U[idx, r] = old_val
                        stuck_count += 1
                elif mat_choice == 1:
                    idx = random.randint(0, m * p - 1)
                    r = random.randint(0, rank - 1)
                    old_val = V[idx, r]
                    new_val = old_val + random.choice([-1, 1])
                    if new_val < -2 or new_val > 2:
                        continue
                    V[idx, r] = new_val
                    new_error = calc_error(U, V, W)
                    if new_error < current_error:
                        current_error = new_error
                        stuck_count = 0
                        if current_error < 1e-9:
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    else:
                        V[idx, r] = old_val
                        stuck_count += 1
                else:
                    idx = random.randint(0, p * n - 1)
                    r = random.randint(0, rank - 1)
                    old_val = W[idx, r]
                    new_val = old_val + random.choice([-1, 1])
                    if new_val < -2 or new_val > 2:
                        continue
                    W[idx, r] = new_val
                    new_error = calc_error(U, V, W)
                    if new_error < current_error:
                        current_error = new_error
                        stuck_count = 0
                        if current_error < 1e-9:
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    else:
                        W[idx, r] = old_val
                        stuck_count += 1

    # If no better rank found, return schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
