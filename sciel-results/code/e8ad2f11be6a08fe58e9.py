"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def _is_half_integer(x, tol=1e-9):
    # Check if x is close to an integer or half-integer
    x2 = 2 * x
    nearest = round(x2)
    if abs(x2 - nearest) < tol:
        return True
    return False

def _verify_half_integers(mat, tol=1e-9):
    for row in mat:
        for val in row:
            if not _is_half_integer(val, tol):
                return False
    return True

def _try_find_decomposition(n, m, p, rank, max_trials=2000):
    nm = n * m
    mp = m * p
    pn = p * n
    num_eqs = n * m * p
    num_unk = rank * pn
    
    for trial in range(max_trials):
        # Random U, V with entries in {-1, 0, 1}
        U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(nm)]
        V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(mp)]
        
        # Build A and b
        A = np.zeros((num_eqs, num_unk), dtype=float)
        b = np.zeros(num_eqs, dtype=float)
        b[:] = 1.0
        
        # Fill A
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    e = i * m * p + j * p + k
                    for r in range(rank):
                        u = r * pn + k * n + i
                        A[e, u] = U[i * m + j][r] * V[j * p + k][r]
        
        try:
            w, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
            
        W_mat = [[0] * rank for _ in range(pn)]
        for r in range(rank):
            for idx in range(pn):
                W_mat[idx][r] = w[r * pn + idx]
                
        if _verify_half_integers(W_mat, tol=1e-7):
            U_frac = [[Fraction(int(U[i][r]) if abs(U[i][r] - round(U[i][r])) < 1e-9 else 0) for r in range(rank)] for i in range(nm)]
            V_frac = [[Fraction(int(V[i][r]) if abs(V[i][r] - round(V[i][r])) < 1e-9 else 0) for r in range(rank)] for i in range(mp)]
            W_frac = [[Fraction(int(round(W_mat[i][r]*2))/2) for r in range(rank)] for i in range(pn)]
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for r in range(rank):
                            s += U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r]
                        if s != 1:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if valid:
                return U, V, W_mat, rank
                
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135.0
    
    lower_bound = max(n*m, m*p, p*n)
    
    # Try ranks from lower_bound to 35
    for rank in range(lower_bound, 36):
        if time.time() > deadline:
            break
            
        max_trials = 500 if rank < 30 else 200
        result = _try_find_decomposition(n, m, p, rank, max_trials)
        if result is not None:
            U, V, W, r = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
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
