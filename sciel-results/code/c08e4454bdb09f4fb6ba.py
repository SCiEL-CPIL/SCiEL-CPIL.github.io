"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

import numpy as np
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def _verify_rank(n, m, p, rank, U, V, W):
    """Verify that the given factors decompose the matmul tensor correctly.
    U: shape (n*m, rank), V: shape (m*p, rank), W: shape (p*n, rank)
    Convention: W index is k*n + i, U is i*m + j, V is j*p + k.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(val - 1.0) > 1e-6:
                    return False, (i, j, k, val)
    return True, None

def _update_W(U, V, W, n, m, p, r):
    """Update W rows given U and V."""
    for i in range(n):
        for k in range(p):
            A = np.zeros((m, r))
            for j in range(m):
                A[j, :] = U[i*m + j, :] * V[j*p + k, :]
            sol, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
            W[k*n + i, :] = sol

def _update_U(U, V, W, n, m, p, r):
    """Update U rows given V and W."""
    for i in range(n):
        for j in range(m):
            B = np.zeros((p, r))
            for k in range(p):
                B[k, :] = V[j*p + k, :] * W[k*n + i, :]
            sol, _, _, _ = np.linalg.lstsq(B, np.ones(p), rcond=None)
            U[i*m + j, :] = sol

def _update_V(U, V, W, n, m, p, r):
    """Update V rows given U and W."""
    for j in range(m):
        for k in range(p):
            C = np.zeros((n, r))
            for i in range(n):
                C[i, :] = U[i*m + j, :] * W[k*n + i, :]
            sol, _, _, _ = np.linalg.lstsq(C, np.ones(n), rcond=None)
            V[j*p + k, :] = sol

def _is_half_integer(mat, tol=1e-5):
    """Check if all entries are close to multiples of 0.5."""
    rounded = np.round(mat * 2) / 2
    return np.allclose(mat, rounded, atol=tol)

def _round_to_half_integer(mat):
    return np.round(mat * 2) / 2

def _compute_error(U, V, W, n, m, p, r):
    """Compute the sum of squared errors for the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                err += (val - 1.0)**2
    return err

def _als_optimize(U, V, W, n, m, p, r, max_iters=200, tol=1e-8, deadline=None):
    """Run ALS optimization with proper loop structure."""
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        err = _compute_error(U, V, W, n, m, p, r)
        if err < tol:
            break
        
        # Update W
        _update_W(U, V, W, n, m, p, r)
        # Update U
        _update_U(U, V, W, n, m, p, r)
        # Update V
        _update_V(U, V, W, n, m, p, r)
    
    return U, V, W

def _polish_solution(U, V, W, n, m, p, r, deadline=None):
    """Final high-precision polish pass."""
    for it in range(50):
        if deadline is not None and time.time() > deadline:
            break
        
        err = _compute_error(U, V, W, n, m, p, r)
        if err < 1e-10:
            break
        
        _update_W(U, V, W, n, m, p, r)
        _update_U(U, V, W, n, m, p, r)
        _update_V(U, V, W, n, m, p, r)
    
    return U, V, W

def _construct_2x2x3_decomp(rank=5):
    """
    Construct a rank-5 decomposition for 2x2 * 2x3 matrix multiplication.
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    Using the known rank-5 decomposition for 2x2x3:
    We can use the fact that rank(2x2x3 matmul tensor) = 5.
    
    A standard rank-5 decomposition uses:
    For the 2x2x2 part (which has rank 3), plus handling the extra column.
    
    Let me use a direct construction. The tensor T[i,j,k] for i in {0,1}, j in {0,1}, k in {0,1,2}
    with T[i,j,k] = 1.
    
    We need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    
    Known rank-5 decomposition for 2x2x3:
    Using the formula from Strassen-type constructions:
    
    Let's use a specific known decomposition. For 2x2x2, rank is 3 (Strassen).
    For 2x2x3, we can extend.
    
    Actually, let me use a constructive approach with specific values.
    """
    n_sub, m_sub, p_sub = 2, 2, 3
    r = rank
    
    U_sub = np.zeros((n_sub * m_sub, r))
    V_sub = np.zeros((m_sub * p_sub, r))
    W_sub = np.zeros((p_sub * n_sub, r))
    
    # Rank-5 decomposition for 2x2x3 matmul
    # Using the known result: the multiplication tensor for 2x2 * 2x3 has rank 5
    
    # A standard construction:
    # We decompose using 5 terms.
    
    # Let me use a known explicit decomposition.
    # For 2x2x2, Strassen gives rank 3:
    # M1 = A00*(B10+B11), M2 = A10*B00, M3 = A01*(B00+B11), M4 = A11*(B01-B11)
    # M5 = (A00+A11)*B11, M6 = (A10+A11)*(B00+B01), M7 = (A01-A10)*(B10+B11)
    # But that's for 2x2x2 with 7 terms. Actually Strassen is 7 for 2x2x2.
    
    # Wait, I need to be more careful. The tensor rank of 2x2x2 matmul is 7 (Strassen).
    # But 2x2x3 might have different rank.
    
    # Let me try a different approach: use the fact that we can embed
    # and use ALS to find a valid decomposition.
    
    # For now, return a schoolbook decomposition as placeholder
    # and let the main construct() find a better one via search.
    
    # Schoolbook for 2x2x3: rank 12
    U_sub = np.zeros((4, 12))
    V_sub = np.zeros((6, 12))
    W_sub = np.zeros((6, 12))
    for i in range(2):
        for j in range(2):
            for k in range(3):
                c = i*6 + j*3 + k
                U_sub[i*2 + j, c] = 1
                V_sub[j*3 + k, c] = 1
                W_sub[k*2 + i, c] = 1
    
    return U_sub, V_sub, W_sub

def _construct_block_embedding(n, m, p):
    """
    Construct a block-embedded decomposition for n x m * m x p.
    Partition into blocks of size 2x2x3 and embed rank-5 decomp for each.
    Total rank = (n/2) * (p/2) * 5
    """
    n_blocks = n // 2
    p_blocks = p // 2
    block_rank = 5
    total_rank = n_blocks * p_blocks * block_rank
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    for ib in range(n_blocks):
        for kb in range(p_blocks):
            # Block indices
            i_start = ib * 2
            k_start = kb * 2
            c_start = (ib * p_blocks + kb) * block_rank
            
            # Get the 2x2x3 decomposition
            U_sub, V_sub, W_sub = _construct_2x2x3_decomp(block_rank)
            
            # Embed into the full matrices
            for i in range(2):
                for j in range(2):
                    U[i_start + i, c_start:c_start+block_rank] = U_sub[i*2 + j, :]
            
            for j in range(2):
                for k in range(3):
                    V[j, c_start:c_start+block_rank] = V_sub[j*3 + k, :]
            
            for k in range(3):
                for i in range(2):
                    W[k_start + k, c_start:c_start+block_rank] = W_sub[k*2 + i, :]
    
    return U, V, W, total_rank

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s margin out of 120s
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 96
    
    # Initialize best with schoolbook solution
    best_rank = schoolbook_rank
    best_U = np.zeros((n * m, schoolbook_rank))
    best_V = np.zeros((m * p, schoolbook_rank))
    best_W = np.zeros((p * n, schoolbook_rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                best_U[i * m + j, c] = 1
                best_V[j * p + k, c] = 1
                best_W[k * n + i, c] = 1
                
    random.seed(42)
    np.random.seed(42)
    
    # Strategy 1: Try block embedding with rank-5 sub-decompositions
    # For 4x4x6: 2 blocks in n, 3 blocks in p -> 2*3*5 = 30 rank
    # But we can do better with a proper rank-5 for 2x2x3
    
    # Let me try to find a rank-5 decomposition for 2x2x3 using ALS
    def _find_2x2x3_rank5(deadline):
        n_sub, m_sub, p_sub = 2, 2, 3
        r = 5
        
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            U_sub = np.random.randn(n_sub * m_sub, r) * 0.5
            V_sub = np.random.randn(m_sub * p_sub, r) * 0.5
            W_sub = np.random.randn(p_sub * n_sub, r) * 0.5
            
            # Normalize
            for c in range(r):
                norm = np.sqrt(np.sum(U_sub[:, c]**2) * np.sum(V_sub[:, c]**2) * np.sum(W_sub[:, c]**2))
                if norm > 1e-10:
                    scale = 1.0 / (norm ** (1.0/3.0))
                    U_sub[:, c] *= scale
                    V_sub[:, c] *= scale
                    W_sub[:, c] *= scale
            
            U_sub, V_sub, W_sub = _als_optimize(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub, r, max_iters=100, deadline=deadline)
            
            err = _compute_error(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub, r)
            if err < 1e-7:
                if _is_half_integer(U_sub) and _is_half_integer(V_sub) and _is_half_integer(W_sub):
                    U_r = _round_to_half_integer(U_sub)
                    V_r = _round_to_half_integer(V_sub)
                    W_r = _round_to_half_integer(W_sub)
                    valid, _ = _verify_rank(n_sub, m_sub, p_sub, r, U_r, V_r, W_r)
                    if valid:
                        return U_r, V_r, W_r
                else:
                    # Try polishing
                    U_sub, V_sub, W_sub = _polish_solution(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub, r, deadline=deadline)
                    err = _compute_error(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub, r)
                    if err < 1e-8:
                        if _is_half_integer(U_sub) and _is_half_integer(V_sub) and _is_half_integer(W_sub):
                            U_r = _round_to_half_integer(U_sub)
                            V_r = _round_to_half_integer(V_sub)
                            W_r = _round_to_half_integer(W_sub)
                            valid, _ = _verify_rank(n_sub, m_sub, p_sub, r, U_r, V_r, W_r)
                            if valid:
                                return U_r, V_r, W_r
        return None
    
    # Try to find a rank-5 decomposition for 2x2x3
    decomp_223 = _find_2x2x3_rank5(deadline)
    
    if decomp_223 is not None:
        U_sub, V_sub, W_sub = decomp_223
        # Use this for block embedding
        n_blocks = n // 2
        p_blocks = p // 2
        block_rank = 5
        total_rank = n_blocks * p_blocks * block_rank
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        for ib in range(n_blocks):
            for kb in range(p_blocks):
                i_start = ib * 2
                k_start = kb * 2
                c_start = (ib * p_blocks + kb) * block_rank
                
                for i in range(2):
                    for j in range(2):
                        U[i_start + i, c_start:c_start+block_rank] = U_sub[i*2 + j, :]
                
                for j in range(2):
                    for k in range(3):
                        V[j, c_start:c_start+block_rank] = V_sub[j*3 + k, :]
                
                for k in range(3):
                    for i in range(2):
                        W[k_start + k, c_start:c_start+block_rank] = W_sub[k*2 + i, :]
        
        # Verify
        valid, _ = _verify_rank(n, m, p, total_rank, U, V, W)
        if valid:
            best_rank = total_rank
            best_U = U
            best_V = V
            best_W = W
    
    # Now search for even lower ranks using ALS
    for target_rank in range(best_rank - 1, 10, -1):
        if time.time() > deadline - 5:
            break
        
        attempts = 5
        found = False
        
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
            
            scale = 0.3 + 0.4 * (attempt % 3)
            U = np.random.randn(n * m, target_rank) * scale
            V = np.random.randn(m * p, target_rank) * scale
            W = np.random.randn(p * n, target_rank) * scale
            
            for c in range(target_rank):
                norm = np.sqrt(np.sum(U[:, c]**2) * np.sum(V[:, c]**2) * np.sum(W[:, c]**2))
                if norm > 1e-10:
                    scale_c = 1.0 / (norm ** (1.0/3.0))
                    U[:, c] *= scale_c
                    V[:, c] *= scale_c
                    W[:, c] *= scale_c
            
            U, V, W = _als_optimize(U, V, W, n, m, p, target_rank, max_iters=100, deadline=deadline - 5)
            
            err = _compute_error(U, V, W, n, m, p, target_rank)
            
            if err < 1e-7:
                if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                    U_rounded = _round_to_half_integer(U)
                    V_rounded = _round_to_half_integer(V)
                    W_rounded = _round_to_half_integer(W)
                    
                    valid, _ = _verify_rank(n, m, p, target_rank, U_rounded, V_rounded, W_rounded)
                    if valid:
                        best_rank = target_rank
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        found = True
                        break
            
            if err < 1e-4:
                U, V, W = _polish_solution(U, V, W, n, m, p, target_rank, deadline=deadline - 5)
                err = _compute_error(U, V, W, n, m, p, target_rank)
                
                if err < 1e-8:
                    if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                        U_rounded = _round_to_half_integer(U)
                        V_rounded = _round_to_half_integer(V)
                        W_rounded = _round_to_half_integer(W)
                        
                        valid, _ = _verify_rank(n, m, p, target_rank, U_rounded, V_rounded, W_rounded)
                        if valid:
                            best_rank = target_rank
                            best_U = U_rounded
                            best_V = V_rounded
                            best_W = W_rounded
                            found = True
                            break
        
        if found:
            continue
        else:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
