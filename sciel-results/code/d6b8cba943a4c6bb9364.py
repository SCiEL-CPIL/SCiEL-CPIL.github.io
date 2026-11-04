"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    Uses a hybrid approach:
    1. Start with schoolbook decomposition (rank 40).
    2. Attempt to reduce rank via structured low-rank approximations and exact verification.
    3. Use integer/half-integer constraints where possible.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Target tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # In the decomposition: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 for all valid i,j,k (this is the matrix multiplication tensor)
    
    # Schoolbook: rank = n*m*p = 2*4*5 = 40
    rank_schoolbook = n * m * p
    
    def schoolbook_decomposition():
        U = np.zeros((n*m, rank_schoolbook))
        V = np.zeros((m*p, rank_schoolbook))
        W = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook_decomposition()
    
    def verify(U, V, W, tol=1e-9):
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find lower rank decompositions using structured approaches
    # For matrix multiplication, known lower bounds and constructions exist.
    # For 2x4 * 4x5, the rank is at least max(2*4, 4*5, 2*5) = 20 (from dimension arguments)
    # But the actual rank is likely higher.
    
    best_rank = rank_schoolbook
    best_decomp = (U.copy(), V.copy(), W.copy())
    
    # Try a few candidate ranks
    candidate_ranks = [20, 22, 24, 25, 26, 28, 30, 32, 35, 38, 40]
    
    for target_rank in candidate_ranks:
        if target_rank >= best_rank:
            continue
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            U_cand = np.random.randn(n*m, target_rank)
            V_cand = np.random.randn(m*p, target_rank)
            W_cand = np.random.randn(p*n, target_rank)
            
            U_cand /= np.sqrt(target_rank)
            V_cand /= np.sqrt(target_rank)
            W_cand /= np.sqrt(target_rank)
            
            def tensor_residual(U_c, V_c, W_c):
                n, m, p = DIMENSIONS
                rank = U_c.shape[1]
                max_res = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(rank):
                                val += U_c[i*m+j, r] * V_c[j*p+k, r] * W_c[k*n+i, r]
                            max_res = max(max_res, abs(val - 1.0))
                return max_res
            
            res = tensor_residual(U_cand, V_cand, W_cand)
            if res < 1e-6:
                found = True
                break
            
            for iter in range(20):
                if time.time() > deadline - 10:
                    break
                res = tensor_residual(U_cand, V_cand, W_cand)
                if res < 1e-8:
                    found = True
                    break
                break
        
        if found:
            U_snap = np.round(U_cand * 2) / 2
            V_snap = np.round(V_cand * 2) / 2
            W_snap = np.round(W_cand * 2) / 2
            
            if verify(U_snap, V_snap, W_snap, tol=1e-6):
                best_rank = target_rank
                best_decomp = (U_snap, V_snap, W_snap)
            elif verify(U_cand, V_cand, W_cand, tol=1e-6):
                best_rank = target_rank
                best_decomp = (U_cand, V_cand, W_cand)
    
    # If no better decomposition found, return schoolbook
    if best_rank == rank_schoolbook:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U_best, V_best, W_best = best_decomp
        return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
