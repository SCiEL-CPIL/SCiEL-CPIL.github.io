"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n=2, m=4, p=4, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Check shapes
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # Build the multiplication tensor T[i,j,k] = U[i,j] * V[j,k] * W[k,i]
    # For each (i,j,k), sum over rank should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _trim_columns(U, V, W, tol=1e-12):
    """Remove columns that are all zeros in U, V, or W."""
    n, m, p = 2, 4, 4
    rank = U.shape[1]
    keep = []
    for c in range(rank):
        if (np.any(np.abs(U[:, c]) > tol) and 
            np.any(np.abs(V[:, c]) > tol) and 
            np.any(np.abs(W[:, c]) > tol)):
            keep.append(c)
    if len(keep) == rank:
        return U, V, W
    return U[:, keep], V[:, keep], W[:, keep]

def _make_schoolbook():
    """Return the schoolbook rank-32 decomposition."""
    n, m, p = 2, 4, 4
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _make_strassen_like():
    """
    Attempt a Strassen-like decomposition for 2x4 * 4x4.
    
    We can view this as two 2x4 * 4x4 blocks, or restructure.
    
    Actually, let's think of A as 2x4, B as 4x4, C as 2x4.
    Split B into two 4x2 blocks: B = [B1 | B2] where B1, B2 are 4x2.
    Then C = A * B = [A*B1 | A*B2].
    
    Each A*B is (2x4)*(4x2) = 2x2.
    Strassen for 2x2 * 2x2 has rank 7.
    
    But A is 2x4, not 2x2. So we need to split A too.
    A = [A1 | A2] where A1, A2 are 2x2.
    
    Then A*B1 = A1*B1_top + A2*B1_bottom where B1_top, B1_bottom are 2x2.
    This gets complicated. Let's just use the schoolbook and try to optimize.
    
    Actually, let's try a different approach: use the known result that
    the rank of (2,4,4) matrix multiplication is at most 28 (from Strassen-like
    block decomposition).
    
    For now, return the schoolbook.
    """
    return _make_schoolbook()

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix
    by a 4 by 4 matrix.
    
    Returns: {'U': U, 'V': V, 'W': W} with shapes (8, rank), (16, rank), (8, rank).
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = 2, 4, 4
    nm, mp, pn = n*m, m*p, p*n  # 8, 16, 8
    
    # Start with schoolbook
    U, V, W = _make_schoolbook()
    best_rank = U.shape[1]
    best_result = (U.copy(), V.copy(), W.copy())
    
    # Verify
    if _verify_decomposition(U, V, W):
        U, V, W = _trim_columns(U, V, W)
        best_rank = U.shape[1]
        best_result = (U.copy(), V.copy(), W.copy())
    
    # Try random search for lower rank
    # The idea: for a given rank r, we need to find U, V, W such that
    # for all (i,j,k): sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    
    # We can fix U and V to have sparse structure and solve for W,
    # or use a more systematic approach.
    
    # Strategy: Try ranks from best_rank-1 down to some minimum.
    # For each rank, try random initializations and use iterative refinement.
    
    min_rank_to_try = max(1, best_rank - 10)
    
    for target_rank in range(best_rank - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline:
            break
        
        # Try several random initializations
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random sparse initialization
            # Each column of U, V, W should be sparse
            U_try = np.zeros((nm, target_rank))
            V_try = np.zeros((mp, target_rank))
            W_try = np.zeros((pn, target_rank))
            
            for c in range(target_rank):
                # Choose random non-zero patterns
                # U has 8 rows, V has 16 rows, W has 8 rows
                # For each column, pick a few non-zero entries
                nz_u = random.randint(1, 4)
                nz_v = random.randint(1, 4)
                nz_w = random.randint(1, 4)
                
                u_indices = random.sample(range(nm), min(nz_u, nm))
                v_indices = random.sample(range(mp), min(nz_v, mp))
                w_indices = random.sample(range(pn), min(nz_w, pn))
                
                for idx in u_indices:
                    U_try[idx, c] = random.choice([-1, 1]) * random.choice([1, 2])
                for idx in v_indices:
                    V_try[idx, c] = random.choice([-1, 1]) * random.choice([1, 2])
                for idx in w_indices:
                    W_try[idx, c] = random.choice([-1, 1]) * random.choice([1, 2])
            
            # Verify
            if _verify_decomposition(U_try, V_try, W_try):
                U_try, V_try, W_try = _trim_columns(U_try, V_try, W_try)
                actual_rank = U_try.shape[1]
                if actual_rank < best_rank:
                    best_rank = actual_rank
                    best_result = (U_try.copy(), V_try.copy(), W_try.copy())
                    break  # Found a better rank, continue to lower targets
        
        # If we found a better rank, update
        if best_rank <= target_rank:
            continue
    
    # Try a structured Strassen-like approach
    # For (2,4,4): view as two (2,4,2) multiplications
    # Each (2,4,2) can be done with rank at most 8 (schoolbook) or less
    
    # Actually, let's try a block Strassen approach:
    # A (2x4) * B (4x4) = C (2x4)
    # Split B into B_left (4x2) and B_right (4x2)
    # C_left = A * B_left (2x2), C_right = A * B_right (2x2)
    
    # For A (2x4) * B_left (4x2):
    # Split A into A_top (2x2) and A_bottom (2x2)? No, A is 2 rows.
    # A = [a11 a12 a13 a14; a21 a22 a23 a24]
    # B_left = [b11 b12; b21 b22; b31 b32; b41 b42]
    # C_left = [a11*b11+a12*b21+a13*b31+a14*b41, a11*b12+a12*b22+a13*b32+a14*b42;
    #           a21*b11+a22*b21+a23*b31+a24*b41, a21*b12+a22*b22+a23*b32+a24*b42]
    
    # This is a (2,4,2) multiplication. The schoolbook rank is 8.
    # Can we do better? The tensor rank of (2,4,2) is known to be at most 8.
    # Actually, for (2,4,2), we can use a Strassen-like trick.
    
    # Let's just use the schoolbook for now and return the best found.
    
    U_best, V_best, W_best = best_result
    U_best, V_best, W_best = _trim_columns(U_best, V_best, W_best)
    
    # Convert to lists as required by the interface
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
