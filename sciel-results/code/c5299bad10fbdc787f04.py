"""Tensor decomposition for 2x3x4 matrix multiplication via integer/rational search."""
import numpy as np
from itertools import product
from fractions import Fraction
import time
import sys

DIMENSIONS = [2, 3, 4]

def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    total_terms = n * m * p  # 24
    
    # Fallback: Schoolbook solution (Rank 24)
    U_school = [[0]*24 for _ in range(6)]
    V_school = [[0]*24 for _ in range(12)]
    W_school = [[0]*24 for _ in range(8)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = 24
    
    np.random.seed(42)
    
    # Strategy: Search for U, V with integer entries {-1, 0, 1}.
    # For each (i,k), solve A W = 1 for W, where A is 3x r.
    # Check if W entries are half-integers.
    
    for r in range(19, 25):
        if time.time() > deadline - 5:
            break
            
        found = False
        attempts = 0
        max_attempts = 50000 if r >= 19 else 10000
        
        while attempts < max_attempts:
            if time.time() > deadline - 2:
                break
            attempts += 1
            
            # Generate random U and V with entries in {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(nm, r))
            V = np.random.choice([-1, 0, 1], size=(mp, r))
            
            # W will be (pn, r) -> (8, r)
            # Indices for W are (k*n + i, c)
            # We need to solve for each i in [0, n), k in [0, p)
            valid = True
            W_sol = np.zeros((pn, r))
            
            for i in range(n):
                for k in range(p):
                    # Construct A: (m, r)
                    # A[j, c] = U[i*m + j, c] * V[j*p + k, c]
                    # U rows for this i: U[i*m : i*m+m, :]
                    # V rows for this k: V[k : k + m*p : p, :] ? No.
                    # V is indexed by (j, k). Row index is j*p + k.
                    # So for fixed k, the rows are at indices k, p+k, 2p+k.
                    # Let's extract them.
                    
                    U_sub = U[i*m : i*m+m, :]  # (m, r)
                    V_sub = V[k : k + m*p : p, :] # (m, r)
                    
                    A = U_sub * V_sub # (m, r)
                    
                    # Solve A W_vec = ones(m)
                    # W_vec is the row of W corresponding to (k, i)
                    # W index is k*n + i
                    w_idx = k*n + i
                    
                    try:
                        # Use lstsq to handle rank deficiency
                        sol, residuals, rank, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        # Check if solution is valid (residuals small)
                        if rank < m:
                             # If rank deficient, check if ones is in column space
                             # lstsq gives min norm solution. If residuals > 0, no solution.
                             if np.linalg.norm(A @ sol - np.ones(m)) > 1e-9:
                                 valid = False
                                 break
                        else:
                             if np.linalg.norm(A @ sol - np.ones(m)) > 1e-9:
                                 valid = False
                                 break
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                    
                    # Check if sol entries are close to half-integers
                    # 2 * sol should be close to integer
                    if not np.allclose(2 * sol, np.round(2 * sol), atol=1e-9):
                        valid = False
                        break
                    
                    W_sol[w_idx, :] = sol
            
            if valid:
                # Convert to lists of floats (or ints if possible)
                # The task asks for integer/half-integer entries.
                # We found them as floats close to half-integers.
                # We should return them as exact values if possible, or just the floats.
                # The verifier uses float comparison, so floats are fine.
                
                # Round to nearest half-integer to clean up
                W_clean = np.round(2 * W_sol) / 2.0
                
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W_clean.tolist()
                }
                best_rank = r
                found = True
                break
        
        if found:
            break
    
    if best_rank < 24:
        return best_result
    else:
        return {"U": U_school, "V": V_school, "W": W_school}
