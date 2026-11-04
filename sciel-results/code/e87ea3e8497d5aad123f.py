import numpy as np
import time
import random

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 48
    target_rank = 48
    
    # If we can't find a better one quickly, return schoolbook
    def schoolbook():
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

    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin

    # Try to find a rank 48 decomposition
    # We use a randomized approach:
    # 1. Pick random U, V with small integer entries.
    # 2. Solve for W.
    # 3. Check if W has small entries (potential for rounding).
    
    best_result = None
    best_rank = 96
    
    # Try a few random initializations
    for _ in range(1000):
        if time.time() > deadline:
            break
            
        rank = target_rank
        # Initialize U, V with random integers in {-1, 0, 1}
        U_rand = np.random.randint(-1, 2, size=(n*m, rank))
        V_rand = np.random.randint(-1, 2, size=(m*p, rank))
        
        # Solve for W
        # For each (i, k), we have m equations:
        # sum_c (U[i,j,c] * V[j,k,c]) * W[k,i,c] = 1
        # Let A_ik be a matrix of size m x rank, where A_ik[j, c] = U[i,j,c] * V[j,k,c]
        # We want A_ik * w_ik = 1 (vector of ones)
        # w_ik is a vector of size rank.
        
        W_sol = np.zeros((p*n, rank))
        valid = True
        
        for i in range(n):
            for k in range(p):
                # Construct A_ik
                A_ik = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        A_ik[j, c] = U_rand[i*m + j, c] * V_rand[j*p + k, c]
                
                # Solve A_ik * w = 1
                # Use least squares
                try:
                    w, residuals, rank_s, sv = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                except:
                    valid = False
                    break
                
                # Check if the solution is "small"
                # If max abs value is small, it might be roundable
                if np.max(np.abs(w)) > 10:
                    valid = False
                    break
                    
                W_sol[k*n + i, :] = w
        
        if not valid:
            continue
            
        # Round W to nearest integer
        W_int = np.round(W_sol).astype(int)
        
        # Verify the solution with integer entries
        # Check sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U_rand[i*m + j, c] * V_rand[j*p + k, c] * W_int[k*n + i, c]
                    if s != 1:
                        error += 1
                        break
                if error > 0:
                    break
            if error > 0:
                break
                
        if error == 0:
            # Found a valid integer solution
            best_rank = rank
            best_result = {
                "U": U_rand.tolist(),
                "V": V_rand.tolist(),
                "W": W_int.tolist()
            }
            break

    if best_result is not None:
        return best_result
    else:
        return schoolbook()
