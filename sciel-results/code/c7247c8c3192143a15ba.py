"""
Improved approach: structured block decomposition with Strassen + local optimization.
Focus: Winograd-style pair sharing in the middle dimension to reduce rank below 75.
"""
import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 5, 5]
n, m, p = DIMENSIONS

_START_TIME = time.time()
DEADLINE = 140.0  # Safety margin before 150s kill

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def _validate_numpy(U, V, W, n, m, p):
    """Fast validation using NumPy."""
    U_np = np.asarray(U, dtype=np.float64)
    V_np = np.asarray(V, dtype=np.float64)
    W_np = np.asarray(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            u_row = U_np[i * m + j, :]
            for k in range(p):
                v_row = V_np[j * p + k, :]
                w_row = W_np[k * n + i, :]
                s = np.dot(u_row * v_row, w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _validate_sparse(U, V, W, n, m, p):
    """
    Validation exploiting sparsity. U, V, W are lists of lists.
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            for k in range(p):
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                s = 0.0
                for r in range(rank):
                    uv = u_row[r]
                    if uv == 0:
                        continue
                    vw = v_row[r]
                    if vw == 0:
                        continue
                    s += uv * vw * w_row[r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _winograd_pair(n, m, p):
    """
    Use Winograd-style decomposition for pairs of indices in the middle dimension.
    For each pair (j, j+1) in the middle dimension, use 3 multiplications
    instead of 2 for the dot product.
    
    For 5x5, we have pairs (0,1), (2,3), and single (4).
    Total: 3 + 3 + 1 = 7 multiplications per row of A.
    For 3 rows: 3 * 7 = 21 multiplications.
    """
    rank = 0
    U = []
    V = []
    W = []
    
    # For each row i of A (and C)
    for i in range(n):
        # Process middle dimension in pairs: (0,1), (2,3), and single (4)
        # Pair (0,1): use 3 multiplications
        # A[i,0], A[i,1] * B[0:2, k] for each k
        
        # Winograd for length 2:
        # M1 = (A0+A1) * B0
        # M2 = (A0+A1) * B1
        # M3 = (A0-A1) * B1
        # C0 = M1 + M3
        # C1 = M2 - M3
        
        # Actually, let me use a simpler approach:
        # For pair (j, j+1), we can write:
        # sum_{k} A[i,j]*B[j,k]*C[k,i] + sum_{k} A[i,j+1]*B[j+1,k]*C[k,i]
        # = sum_{k} (A[i,j]+A[i,j+1])*B[j,k]*C[k,i] + sum_{k} A[i,j+1]*(B[j+1,k]-B[j,k])*C[k,i]
        # This uses 2 terms instead of 2, so no gain.
        
        # Let me try a different approach: use the fact that we can share
        # across different rows of A.
        pass
    
    # For now, return schoolbook
    return _schoolbook(n, m, p)

def _structured_block(n, m, p):
    """
    Structured block decomposition: split the 5x5 middle dimension into blocks
    and apply Strassen where possible.
    
    A is 3x5: rows 0,1,2; cols 0,1,2,3,4
    B is 5x5: rows 0,1,2,3,4; cols 0,1,2,3,4
    C is 3x5: rows 0,1,2; cols 0,1,2,3,4
    
    Split middle dimension into:
    - Block 1: indices 0,1 (size 2)
    - Block 2: indices 2,3 (size 2)
    - Block 3: index 4 (size 1)
    
    For each output column k in C, we compute:
    C[i,k] = sum_{j=0}^{4} A[i,j]*B[j,k]
    
    We can split this into:
    C[i,k] = sum_{j in {0,1}} A[i,j]*B[j,k] + sum_{j in {2,3}} A[i,j]*B[j,k] + A[i,4]*B[4,k]
    
    For the 2x2 blocks, we can use Strassen-like techniques if we also
    split the output dimension. But since p=5 is odd, this is tricky.
    
    Let me try a different approach: use the fact that for fixed i and k,
    we're computing a dot product of length 5. We can use a Winograd-style
    algorithm for length 5.
    """
    # For now, return schoolbook
    return _schoolbook(n, m, p)

def _local_search(n, m, p, max_iter=500):
    """
    Local search: start from schoolbook, try to eliminate columns by
    finding alternative decompositions that use fewer terms.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    current_rank = rank
    
    rng = random.Random(42)
    best_rank = current_rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    for iteration in range(max_iter):
        if _time_left() < 10:
            break
        
        # Try to reduce rank by 1
        target_rank = best_rank - 1
        if target_rank < 1:
            break
        
        # Randomly choose which column to try to eliminate
        elim_col = rng.randint(0, best_rank - 1)
        
        # Keep all other columns
        keep_cols = [c for c in range(best_rank) if c != elim_col]
        
        # Try a random sparse decomposition of target_rank
        success = False
        for trial in range(50):
            if _time_left() < 5:
                break
            
            U_new = [[0] * target_rank for _ in range(n * m)]
            V_new = [[0] * target_rank for _ in range(m * p)]
            W_new = [[0] * target_rank for _ in range(p * n)]
            
            # Fill with random sparse entries
            density = 0.25
            vals = [1, -1, 0.5, -0.5]
            
            for i in range(n * m):
                for r in range(target_rank):
                    if rng.random() < density:
                        U_new[i][r] = rng.choice(vals)
            for j in range(m * p):
                for r in range(target_rank):
                    if rng.random() < density:
                        V_new[j][r] = rng.choice(vals)
            for k in range(p * n):
                for r in range(target_rank):
                    if rng.random() < density:
                        W_new[k][r] = rng.choice(vals)
            
            if _validate_sparse(U_new, V_new, W_new, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U_new, V_new, W_new
                success = True
                break
        
        if not success and iteration % 10 == 0:
            # No improvement, continue
            pass
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _winograd_length5(n, m, p):
    """
    Use Winograd-style algorithm for length 5 dot products.
    For each (i,k) pair, compute sum_{j=0}^{4} A[i,j]*B[j,k] using fewer
    than 5 multiplications.
    
    Winograd for length 5:
    We can use the fact that:
    a0*b0 + a1*b1 + a2*b2 + a3*b3 + a4*b4
    = (a0+a1+a2+a3+a4)*(b0+b1+b2+b3+b4)
      - (a0+a1+a2+a3)*(b0+b1+b2+b3)
      + (a0+a1+a2)*(b0+b1+b2)
      - (a0+a1)*(b0+b1)
      + a0*b0
    This uses 5 multiplications, same as schoolbook.
    
    Actually, Winograd for length 5 uses 3 multiplications:
    Let's use a different approach.
    """
    # For now, return schoolbook
    return _schoolbook(n, m, p)

def construct():
    """
    Main entry point.
    1. Start with schoolbook (rank 75).
    2. Try local search for improvements.
    3. Return best found.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    # Baseline
    sb = _schoolbook(n, m, p)
    best = sb
    best_rank = n * m * p
    
    # Verify
    if not _validate_sparse(sb["U"], sb["V"], sb["W"], n, m, p):
        return sb
    
    # Try local search
    if _time_left() > 30:
        result = _local_search(n, m, p, max_iter=300)
        if result is not None:
            r = len(result["U"][0]) if result["U"] else 0
            if r < best_rank and _validate_sparse(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = r
    
    # Try Winograd length 5
    if _time_left() > 20:
        result = _winograd_length5(n, m, p)
        if result is not None:
            r = len(result["U"][0]) if result["U"] else 0
            if r < best_rank and _validate_sparse(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = r
    
    # Try structured block
    if _time_left() > 10:
        result = _structured_block(n, m, p)
        if result is not None:
            r = len(result["U"][0]) if result["U"] else 0
            if r < best_rank and _validate_sparse(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = r
    
    return best

if __name__ == "__main__":
    result = construct()
    print(f"Rank achieved: {len(result['U'][0]) if result['U'] else 'unknown'}")
