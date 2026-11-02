"""Rank-42 exact tensor decomposition for 2x4 @ 4x6 matrix multiplication.

Strategy: Block-partition into six 2x2 @ 2x2 multiplications, each computed
via Strassen's algorithm (rank 7). Total rank = 6 * 7 = 42.

Also attempts a time-bounded optimization search for lower-rank decompositions.

Partition:
  A (2x4) = [AL | AR], each 2x2
  B (4x6) = [B1 | B2 | B3], each 4x2
  Cj = A @ Bj = AL @ BjL + AR @ BjR,  j = 0,1,2
  Each Bj (4x2) = [BjL; BjR], each 2x2

  Six 2x2@2x2 products, each via Strassen => rank 42.
"""
import time
import numpy as np
import random

DIMENSIONS = [2, 4, 6]

def _strassen_factors():
    """Return (U1, V1, W1) for 2x2x2 Strassen, shapes (4,7),(4,7),(4,7).
    U1 row: i*2+j, V1 row: j*2+k, W1 row: k*2+i (trace convention)."""
    U1 = np.zeros((4, 7), dtype=float)
    V1 = np.zeros((4, 7), dtype=float)
    W1 = np.zeros((4, 7), dtype=float)

    # M1 = (A00+A11)(B00+B11)
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1  # C00 += M1
    W1[3, 0] = 1  # C11 += M1

    # M2 = (A10+A11)B00
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[1, 1] = 1  # C10 += M2
    W1[3, 1] = -1 # C11 -= M2

    # M3 = A00(B01-B11)
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[2, 2] = 1  # C01 += M3
    W1[3, 2] = 1  # C11 += M3

    # M4 = A11(B10-B00)
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1  # C00 += M4
    W1[1, 3] = 1  # C10 += M4

    # M5 = (A00+A01)B11
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1 # C00 -= M5
    W1[2, 4] = 1  # C01 += M5

    # M6 = (A10-A00)(B00+B01)
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1  # C11 += M6

    # M7 = (A01-A11)(B10+B11)
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1  # C00 += M7

    return U1, V1, W1


def _build_rank42():
    """Build rank-42 decomposition via 6 Strassen blocks."""
    n, m, p = 2, 4, 6
    U1, V1, W1 = _strassen_factors()
    total_rank = 42

    U = np.zeros((n * m, total_rank), dtype=float)  # (8, 42)
    V = np.zeros((m * p, total_rank), dtype=float)  # (24, 42)
    W = np.zeros((p * n, total_rank), dtype=float)  # (12, 42)

    # Six blocks: for j_block in 0..2 (which 4x2 block of B),
    #             for side in 0..1 (AL or AR, i.e., which 2x2 of A)
    # Block index b = j_block * 2 + side
    # Each block uses Strassen rank-7, so columns b*7 .. b*7+6

    for j_block in range(3):
        for side in range(2):
            b = j_block * 2 + side
            col_offset = b * 7

            # U contribution: A[i, a] where a = side * 2 + j_local
            # i in 0..1, j_local in 0..1 (within the 2x2 sub-block)
            # U1[i*2 + j_local, s] gives the coefficient
            for i in range(2):
                for j_local in range(2):
                    a = side * 2 + j_local  # column in A (0..3)
                    for s in range(7):
                        coeff = U1[i * 2 + j_local, s]
                        if coeff != 0:
                            U[i * m + a, col_offset + s] += coeff

            # V contribution: B[a, bcol] where a = side * 2 + j_local, bcol = j_block * 2 + k_local
            # V1[j_local * 2 + k_local, s]
            for j_local in range(2):
                a = side * 2 + j_local
                for k_local in range(2):
                    bcol = j_block * 2 + k_local
                    for s in range(7):
                        coeff = V1[j_local * 2 + k_local, s]
                        if coeff != 0:
                            V[a * p + bcol, col_offset + s] += coeff

            # W contribution: C[i, bcol] where i in 0..1, bcol = j_block * 2 + k_local
            # W1[k_local * 2 + i, s]
            for i in range(2):
                for k_local in range(2):
                    bcol = j_block * 2 + k_local
                    for s in range(7):
                        coeff = W1[k_local * 2 + i, s]
                        if coeff != 0:
                            W[bcol * n + i, col_offset + s] += coeff

    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] == 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(rank):
                    val += U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _to_output(U, V, W):
    """Convert numpy arrays to lists of ints where possible."""
    U_out = [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in U]
    V_out = [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in V]
    W_out = [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in W]
    return U_out, V_out, W_out


def _try_optimize_rank42(U, V, W, n, m, p, time_budget):
    """Attempt to reduce rank below 42 via optimization. Returns improved (U,V,W) or None."""
    start = time.time()
    deadline = start + time_budget
    rank = U.shape[1]
    
    # Try to find a lower-rank decomposition by optimizing a rank-r tensor
    # We'll try ranks from 41 down to 30, spending time on each
    # Use a simple approach: try to zero out entries while maintaining correctness
    
    # First, let's try a direct optimization approach
    # The tensor T[i,j,k] = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] must equal 1
    
    # Try rank 41: attempt to find a 41-rank decomposition
    for target_rank in range(41, 29, -1):
        if time.time() > deadline:
            break
        
        # Initialize from rank-42 by zeroing out some columns
        # and trying to recover via optimization
        U_t = np.zeros((n * m, target_rank), dtype=float)
        V_t = np.zeros((m * p, target_rank), dtype=float)
        W_t = np.zeros((p * n, target_rank), dtype=float)
        
        # Copy first target_rank columns from rank-42
        U_t[:, :target_rank] = U[:, :target_rank]
        V_t[:, :target_rank] = V[:, :target_rank]
        W_t[:, :target_rank] = W[:, :target_rank]
        
        # Simple iterative refinement: try to adjust entries to satisfy constraints
        # This is a non-convex problem, so we use a simple heuristic
        # Try random perturbations and accept if they reduce error
        best_error = _compute_error(U_t, V_t, W_t, n, m, p)
        
        # If the truncated version already works (unlikely but possible)
        if best_error < 1e-9:
            return U_t, V_t, W_t
        
        # Try a few random restarts with small perturbations
        n_restarts = 5
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            
            U_r = U_t + 0.01 * np.random.randn(*U_t.shape)
            V_r = V_t + 0.01 * np.random.randn(*V_t.shape)
            W_r = W_t + 0.01 * np.random.randn(*W_t.shape)
            
            # Gradient-free optimization: try to minimize error
            for step in range(50):
                if time.time() > deadline:
                    break
                
                err = _compute_error(U_r, V_r, W_r, n, m, p)
                if err < best_error:
                    best_error = err
                    if err < 1e-9:
                        return U_r, V_r, W_r
                
                # Simple coordinate descent on a few random entries
                for _ in range(3):
                    mat = random.choice([U_r, V_r, W_r])
                    idx = (random.randint(0, mat.shape[0]-1), random.randint(0, mat.shape[1]-1))
                    old_val = mat[idx]
                    for delta in [0.01, -0.01, 0.1, -0.1]:
                        mat[idx] = old_val + delta
                        new_err = _compute_error(U_r, V_r, W_r, n, m, p)
                        if new_err < best_error:
                            best_error = new_err
                        else:
                            mat[idx] = old_val
                            break
                    else:
                        mat[idx] = old_val
    
    return None


def _compute_error(U, V, W, n, m, p):
    """Compute max |sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] - 1| over all i,j,k."""
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(rank):
                    val += U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    U, V, W = _build_rank42()

    # Verify correctness of rank-42 baseline
    if not _verify(U, V, W, n, m, p):
        # Fallback: schoolbook rank 48
        schoolbook_rank = n * m * p
        U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
        V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
        W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_fb[ii * m + jj][col] = 1
                    V_fb[jj * p + kk][col] = 1
                    W_fb[kk * n + ii][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    # Try to find a lower-rank decomposition
    elapsed = time.time() - start_time
    time_budget = min(80.0, deadline - time.time() - 5.0)  # Leave 5s safety margin
    
    if time_budget > 10.0:
        result = _try_optimize_rank42(U, V, W, n, m, p, time_budget)
        if result is not None:
            U_opt, V_opt, W_opt = result
            if _verify(U_opt, V_opt, W_opt, n, m, p):
                U_out, V_out, W_out = _to_output(U_opt, V_opt, W_opt)
                return {"U": U_out, "V": V_out, "W": W_out}

    # Return rank-42 baseline
    U_out, V_out, W_out = _to_output(U, V, W)
    return {"U": U_out, "V": V_out, "W": W_out}
