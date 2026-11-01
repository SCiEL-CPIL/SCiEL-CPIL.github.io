"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import sys

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
from numpy.linalg import svd, norm
import random
from fractions import Fraction

def _build_rank14():
    """Return the verified rank-14 Strassen-based decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # C indices (W): C[0]=C00, C[1]=C10, C[2]=C01, C[3]=C11, C[4]=C02, C[5]=C12, C[6]=C03, C[7]=C13
    
    # Left block (cols 0,1) - Strassen
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Right block (cols 2,3) - Strassen
    # N1 = (A00+A11)(B02+B13)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # N2 = (A10+A11)B02
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # N3 = A00(B03-B13)
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    # N4 = A11(B12-B02)
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    # N5 = (A00+A01)B13
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # N6 = (A10-A00)(B02+B03)
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    # N7 = (A01-A11)(B12+B13)
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, tol=1e-10):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n, m, p = 2, 2, 4
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    r = U.shape[1]
    
    if U.shape != (n * m, r):
        return False
    if V.shape != (m * p, r):
        return False
    if W.shape != (p * n, r):
        return False
    
    max_err = 0.0
    for i in range(n):
        for j in range(p):
            for k in range(m):
                val = 0.0
                for rr in range(r):
                    val += U[i * m + k, rr] * V[k * p + j, rr] * W[j * n + i, rr]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    
    return max_err < tol


def _residual_matrix(U, V, W):
    """Compute the residual tensor as a flattened vector.
    Residual[i,k,j] = sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r] - 1
    Returns vector of length n*m*p = 16.
    """
    n, m, p = 2, 2, 4
    r = U.shape[1]
    res = np.zeros(n * m * p)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                val = 0.0
                for rr in range(r):
                    val += U[i * m + k, rr] * V[k * p + j, rr] * W[j * n + i, rr]
                res[i * m * p + k * p + j] = val - 1.0
    return res


def _search_rank13(deadline):
    """
    Search for a rank-13 decomposition using alternating least squares
    with structured initialization based on the rank-14 Strassen solution.
    
    Strategy: 
    1. Initialize from rank-14 by merging two terms.
    2. Use ALS to minimize the residual.
    3. Multiple random restarts with different merge strategies.
    """
    n, m, p = 2, 2, 4
    target_rank = 13
    np.random.seed(42)
    
    # Precompute index mappings
    # A index: i*m+k where i in {0,1}, k in {0,1}
    # B index: k*p+j where k in {0,1}, j in {0,1,2,3}
    # W index: j*n+i where j in {0,1,2,3}, i in {0,1}
    
    base = _build_rank14()
    U14 = np.array(base['U'], dtype=float)
    V14 = np.array(base['V'], dtype=float)
    W14 = np.array(base['W'], dtype=float)
    
    best_result = None
    best_residual = np.inf
    
    # Try multiple initialization strategies
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        
        if restart % 50 == 0 and restart > 0:
            np.random.seed(42 + restart)
        
        # Strategy: start with rank-14, merge two terms into one
        # Pick two terms to merge (prefer one from left block, one from right)
        left_terms = list(range(7))
        right_terms = list(range(7, 14))
        
        if restart < 100:
            # Merge one left term with one right term
            li = random.choice(left_terms)
            ri = random.choice(right_terms)
        else:
            # Merge two terms from same block
            block = random.choice([left_terms, right_terms])
            li, ri = random.sample(block, 2)
        
        # Create rank-13 initialization: remove terms li and ri, add a merged term
        # The merged term gets the sum of the two terms (scaled)
        U13 = np.zeros((n * m, target_rank))
        V13 = np.zeros((m * p, target_rank))
        W13 = np.zeros((p * n, target_rank))
        
        # Copy all terms except li and ri
        kept = [t for t in range(14) if t != li and t != ri]
        for new_idx, old_idx in enumerate(kept):
            U13[:, new_idx] = U14[:, old_idx]
            V13[:, new_idx] = V14[:, old_idx]
            W13[:, new_idx] = W14[:, old_idx]
        
        # Add merged term at the last position
        merge_idx = target_rank - 1
        U13[:, merge_idx] = 0.5 * (U14[:, li] + U14[:, ri])
        V13[:, merge_idx] = 0.5 * (V14[:, li] + V14[:, ri])
        W13[:, merge_idx] = 0.5 * (W14[:, li] + W14[:, ri])
        
        # Add some random perturbation
        U13 += 0.01 * np.random.randn(*U13.shape)
        V13 += 0.01 * np.random.randn(*V13.shape)
        W13 += 0.01 * np.random.randn(*W13.shape)
        
        # ALS iterations
        for it in range(200):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each column r, U[:,r] = argmin_U sum_{i,k,j} (target - sum_{r' != r} ...) * V[k*p+j,r] * W[j*n+i,r]
            # This is a linear least squares problem
            for r in range(target_rank):
                # Compute the contribution of all other terms
                if target_rank > 1:
                    other_U = np.delete(U13, r, axis=1)
                    other_V = np.delete(V13, r, axis=1)
                    other_W = np.delete(W13, r, axis=1)
                    other_contrib = np.zeros(n * m * p)
                    for i in range(n):
                        for k in range(m):
                            for j in range(p):
                                other_contrib[i * m * p + k * p + j] = np.dot(
                                    other_U[i * m + k, :],
                                    other_V[k * p + j, :] * other_W[j * n + i, :]
                                )
                else:
                    other_contrib = np.zeros(n * m * p)
                
                target = np.ones(n * m * p) - other_contrib
                
                # Build the linear system: U[i*m+k, r] = sum_j target[i*m*p+k*p+j] * V[k*p+j,r] * W[j*n+i,r]
                # This is: U_vec = M @ (V_col * W_col) where V_col and W_col are the r-th columns
                # Actually: for each (i,k), U[i*m+k,r] = sum_j target[i*m*p+k*p+j] * V[k*p+j,r] * W[j*n+i,r]
                
                vr = V13[:, r]
                wr = W13[:, r]
                
                for i in range(n):
                    for k in range(m):
                        # U[i*m+k, r] = sum_j target[i*m*p + k*p + j] * V[k*p+j, r] * W[j*n+i, r]
                        U13[i * m + k, r] = sum(
                            target[i * m * p + k * p + j] * vr[k * p + j] * wr[j * n + i]
                            for j in range(p)
                        )
            
            # Update V similarly
            for r in range(target_rank):
                if target_rank > 1:
                    other_U = np.delete(U13, r, axis=1)
                    other_V = np.delete(V13, r, axis=1)
                    other_W = np.delete(W13, r, axis=1)
                    other_contrib = np.zeros(n * m * p)
                    for i in range(n):
                        for k in range(m):
                            for j in range(p):
                                other_contrib[i * m * p + k * p + j] = np.dot(
                                    other_U[i * m + k, :],
                                    other_V[k * p + j, :] * other_W[j * n + i, :]
                                )
                else:
                    other_contrib = np.zeros(n * m * p)
                
                target = np.ones(n * m * p) - other_contrib
                
                ur = U13[:, r]
                wr = W13[:, r]
                
                for k in range(m):
                    for j in range(p):
                        # V[k*p+j, r] = sum_i target[i*m*p + k*p + j] * U[i*m+k, r] * W[j*n+i, r]
                        V13[k * p + j, r] = sum(
                            target[i * m * p + k * p + j] * ur[i * m + k] * wr[j * n + i]
                            for i in range(n)
                        )
            
            # Update W similarly
            for r in range(target_rank):
                if target_rank > 1:
                    other_U = np.delete(U13, r, axis=1)
                    other_V = np.delete(V13, r, axis=1)
                    other_W = np.delete(W13, r, axis=1)
                    other_contrib = np.zeros(n * m * p)
                    for i in range(n):
                        for k in range(m):
                            for j in range(p):
                                other_contrib[i * m * p + k * p + j] = np.dot(
                                    other_U[i * m + k, :],
                                    other_V[k * p + j, :] * other_W[j * n + i, :]
                                )
                else:
                    other_contrib = np.zeros(n * m * p)
                
                target = np.ones(n * m * p) - other_contrib
                
                ur = U13[:, r]
                vr = V13[:, r]
                
                for j in range(p):
                    for i in range(n):
                        # W[j*n+i, r] = sum_k target[i*m*p + k*p + j] * U[i*m+k, r] * V[k*p+j, r]
                        W13[j * n + i, r] = sum(
                            target[i * m * p + k * p + j] * ur[i * m + k] * vr[k * p + j]
                            for k in range(m)
                        )
        
        # Check the final residual
        res = _residual_matrix(U13, V13, W13)
        res_norm = np.linalg.norm(res)
        
        if res_norm < best_residual:
            best_residual = res_norm
            if res_norm < 1e-8:
                # Found a valid rank-13 decomposition!
                U13 = np.round(U13, 10)
                V13 = np.round(V13, 10)
                W13 = np.round(W13, 10)
                # Convert to list of lists
                U_list = [[float(x) for x in row] for row in U13]
                V_list = [[float(x) for x in row] for row in V13]
                W_list = [[float(x) for x in row] for row in W13]
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                return best_result
    
    return best_result


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # First, try to search for a lower rank
    result = _search_rank13(deadline)
    
    if result is not None:
        # Verify the found decomposition
        if _verify_decomposition(result['U'], result['V'], result['W']):
            return result
    
    # Fall back to the known rank-14 decomposition
    return _build_rank14()
# EVOLVE-BLOCK-END
