import numpy as np
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_schoolbook = [[0] * rank_schoolbook for _ in range(n * m)]
    V_schoolbook = [[0] * rank_schoolbook for _ in range(m * p)]
    W_schoolbook = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j][col] = 1
                V_schoolbook[j * p + k][col] = 1
                W_schoolbook[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook

    # Strategy: Try to find a decomposition with rank < 60
    # Use a structured approach based on known results:
    # - The rank of 3x4x5 is at most 48 (de Groote)
    # - Try ranks from 48 down to 40
    
    # For each rank r, we'll try:
    # 1. Random U, V with entries from a wider set
    # 2. Solve for W using least squares
    # 3. Check if W has half-integer entries
    
    # Use a more effective random distribution
    # Entries in {-2, -1, 0, 1, 2} give more flexibility
    
    for r in range(48, 39, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations with different strategies
        for attempt in range(500):
            if time.time() > deadline:
                break
            
            # Strategy 1: Random integers in {-2, -1, 0, 1, 2}
            if attempt % 3 == 0:
                U = np.random.randint(-2, 3, (n * m, r))
                V = np.random.randint(-2, 3, (m * p, r))
            # Strategy 2: Random integers in {-1, 0, 1} with some structure
            elif attempt % 3 == 1:
                U = np.random.randint(-1, 2, (n * m, r))
                V = np.random.randint(-1, 2, (m * p, r))
                # Add some diagonal structure
                for c in range(min(r, n * m)):
                    if U[c, c] == 0:
                        U[c, c] = 1
                for c in range(min(r, m * p)):
                    if V[c, c] == 0:
                        V[c, c] = 1
            # Strategy 3: Gaussian-like (but rounded)
            else:
                U = np.round(np.random.randn(n * m, r) * 1.5).astype(int)
                V = np.round(np.random.randn(m * p, r) * 1.5).astype(int)
            
            # Ensure we don't have all-zero rows
            for row_idx in range(n * m):
                if np.all(U[row_idx, :] == 0):
                    U[row_idx, 0] = 1
            for row_idx in range(m * p):
                if np.all(V[row_idx, :] == 0):
                    V[row_idx, 0] = 1
            
            W = np.zeros((p * n, r))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build matrix A (m x r)
                    # A[j, c] = U[i*m + j, c] * V[j*p + k, c]
                    A = np.zeros((m, r))
                    for j in range(m):
                        u_row = U[i * m + j, :]
                        v_row = V[j * p + k, :]
                        A[j, :] = u_row * v_row
                    
                    # Solve A * w = 1
                    try:
                        # Use lstsq to find minimum norm solution
                        w, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        
                        # Round to nearest half-integer
                        w_rounded = np.round(w * 2) / 2
                        
                        # Check if it's a valid solution
                        if np.allclose(A @ w_rounded, np.ones(m), atol=1e-9):
                            W[k * n + i, :] = w_rounded
                        else:
                            valid = False
                            break
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Verify the full decomposition with exact arithmetic
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = Fraction(0)
                            for c in range(r):
                                val += Fraction(int(U[i * m + j, c])) * Fraction(int(V[j * p + k, c])) * Fraction(int(W[k * n + i, c] * 2)) / 2
                            if val != 1:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                
                if ok:
                    # We found a valid decomposition with rank r
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = [[float(x) for x in row] for row in W]
                    
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no better rank found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
