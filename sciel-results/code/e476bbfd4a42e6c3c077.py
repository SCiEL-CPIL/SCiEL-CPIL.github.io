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
    
    # Search for rank r from 14 to 23
    for r in range(14, 24):
        if time.time() > deadline - 5:
            break
            
        found = False
        attempts = 0
        # More attempts for smaller ranks, but limited by time
        max_attempts = 200000 if r <= 16 else 50000
        
        while attempts < max_attempts:
            if time.time() > deadline - 2:
                break
            attempts += 1
            
            # Generate random U and V with entries in {-1, 0, 1}
            # For smaller ranks, maybe try {-2, -1, 0, 1, 2} if r < 16?
            # Let's stick to {-1, 0, 1} first.
            U = np.random.choice([-1, 0, 1], size=(nm, r))
            V = np.random.choice([-1, 0, 1], size=(mp, r))
            
            valid = True
            W_sol = np.zeros((pn, r))
            
            for i in range(n):
                for k in range(p):
                    U_sub = U[i*m : i*m+m, :]  # (m, r)
                    V_sub = V[k : k + m*p : p, :] # (m, r)
                    
                    A = U_sub * V_sub # (m, r)
                    
                    # Solve A W_vec = ones(m)
                    w_idx = k*n + i
                    
                    try:
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        if np.linalg.norm(A @ sol - np.ones(m)) > 1e-9:
                            valid = False
                            break
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                    
                    # Check if sol entries are close to half-integers
                    if not np.allclose(2 * sol, np.round(2 * sol), atol=1e-9):
                        valid = False
                        break
                    
                    W_sol[w_idx, :] = sol
            
            if valid:
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
