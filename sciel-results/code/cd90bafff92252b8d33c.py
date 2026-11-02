"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def _verify_rank(n, m, p, rank, U, V, W):
    """Verify that the given factors decompose the matmul tensor correctly."""
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

def _compute_error_fast(n, m, p, U, V, W):
    """Fast vectorized error computation."""
    err = 0.0
    for i in range(n):
        for k in range(p):
            w_row = W[k * n + i, :]
            for j in range(m):
                u_row = U[i * m + j, :]
                v_row = V[j * p + k, :]
                val = np.dot(u_row, v_row * w_row)
                err += (val - 1.0) ** 2
    return err

def _try_round_and_verify(n, m, p, U, V, W, rank):
    """Try various rounding strategies and verify."""
    strategies = [
        ("half", lambda x: np.round(x * 2) / 2),
        ("int", lambda x: np.round(x)),
        ("half_up", lambda x: np.ceil(x * 2) / 2),
        ("half_down", lambda x: np.floor(x * 2) / 2),
    ]
    for name, fn in strategies:
        U_r = fn(U)
        V_r = fn(V)
        W_r = fn(W)
        valid, _ = _verify_rank(n, m, p, rank, U_r, V_r, W_r)
        if valid:
            return True, U_r, V_r, W_r
    
    # Try mixed: round U to int, V to half, W to half
    U_r = np.round(U)
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    valid, _ = _verify_rank(n, m, p, rank, U_r, V_r, W_r)
    if valid:
        return True, U_r, V_r, W_r
    
    # Try: U half, V int, W half
    U_r = np.round(U * 2) / 2
    V_r = np.round(V)
    W_r = np.round(W * 2) / 2
    valid, _ = _verify_rank(n, m, p, rank, U_r, V_r, W_r)
    if valid:
        return True, U_r, V_r, W_r
    
    return False, None, None, None

def _build_rank6_2x4x3(n2, m2, p2):
    """
    Build a rank-6 decomposition for 2x4 * 4x3 matrix multiplication.
    C[i,k] = sum_j A[i,j] * B[j,k]  where i in {0,1}, j in {0,1,2,3}, k in {0,1,2}
    
    We need: sum_c U[i*4+j, c] * V[j*3+k, c] * W[k*2+i, c] = 1 for all i,j,k
    with rank = 6.
    
    Strategy: Use the structure that each output C[i,k] depends on 4 products.
    With rank 6, we can try to find a structured solution.
    
    Let's try: for each (i,k) pair, we have 4 j-values contributing.
    Total constraints: 2*4*3 = 24 equations, 3*6*... unknowns.
    
    Actually, let's just use ALS to find it, but with good initialization.
    """
    rank = 6
    U = np.zeros((n2 * m2, rank))
    V = np.zeros((m2 * p2, rank))
    W = np.zeros((p2 * n2, rank))
    
    # Initialize with random values
    np.random.seed(42)
    U[:] = np.random.randn(n2 * m2, rank) * 0.5
    V[:] = np.random.randn(m2 * p2, rank) * 0.5
    W[:] = np.random.randn(p2 * n2, rank) * 0.5
    
    # ALS iterations
    for it in range(500):
        # Update W
        for i in range(n2):
            for k in range(p2):
                A = np.zeros((m2, rank))
                for j in range(m2):
                    A[j, :] = U[i * m2 + j, :] * V[j * p2 + k, :]
                sol, _, _, _ = np.linalg.lstsq(A, np.ones(m2), rcond=None)
                W[k * n2 + i, :] = sol
        
        # Update U
        for i in range(n2):
            for j in range(m2):
                B = np.zeros((p2, rank))
                for k in range(p2):
                    B[k, :] = V[j * p2 + k, :] * W[k * n2 + i, :]
                sol, _, _, _ = np.linalg.lstsq(B, np.ones(p2), rcond=None)
                U[i * m2 + j, :] = sol
        
        # Update V
        for j in range(m2):
            for k in range(p2):
                C = np.zeros((n2, rank))
                for i in range(n2):
                    C[i, :] = U[i * m2 + j, :] * W[k * n2 + i, :]
                sol, _, _, _ = np.linalg.lstsq(C, np.ones(n2), rcond=None)
                V[j * p2 + k, :] = sol
        
        # Check convergence
        err = 0.0
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    val = np.dot(U[i*m2 + j, :], V[j*p2 + k, :] * W[k*n2 + i, :])
                    err += (val - 1.0) ** 2
        if err < 1e-12:
            break
    
    return U, V, W, err

def _build_rank12_4x4x6():
    """
    Build a rank-12 decomposition for 4x4 * 4x6 by combining two rank-6 
    decompositions for 2x4 * 4x3 blocks.
    
    Split rows into two groups of 2, and columns into two groups of 3.
    Block (0,0): rows 0-1, cols 0-2 -> 2x4*4x3, rank 6
    Block (1,1): rows 2-3, cols 3-5 -> 2x4*4x3, rank 6
    Total rank: 12
    """
    n, m, p = 4, 4, 6
    rank = 12
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Block 0: i in {0,1}, k in {0,1,2}, rank terms 0-5
    U0, V0, W0, err0 = _build_rank6_2x4x3(2, 4, 3)
    for i in range(2):
        for j in range(4):
            U[i * m + j, 0:6] = U0[i * 4 + j, :]
    for j in range(4):
        for k in range(3):
            V[j * p + k, 0:6] = V0[j * 3 + k, :]
    for k in range(3):
        for i in range(2):
            W[k * n + i, 0:6] = W0[k * 2 + i, :]
    
    # Block 1: i in {2,3}, k in {3,4,5}, rank terms 6-11
    U1, V1, W1, err1 = _build_rank6_2x4x3(2, 4, 3)
    for i in range(2, 4):
        for j in range(4):
            U[i * m + j, 6:12] = U1[(i-2) * 4 + j, :]
    for j in range(4):
        for k in range(3, 6):
            V[j * p + k, 6:12] = V1[j * 3 + (k-3), :]
    for k in range(3, 6):
        for i in range(2, 4):
            W[k * n + i, 6:12] = W1[(k-3) * 2 + (i-2), :]
    
    # Verify
    valid, info = _verify_rank(n, m, p, rank, U, V, W)
    if valid:
        return True, U, V, W, rank
    
    # Try rounding
    ok, Ur, Vr, Wr = _try_round_and_verify(n, m, p, U, V, W, rank)
    if ok:
        return True, Ur, Vr, Wr, rank
    
    return False, None, None, None, rank

def _als_step(n, m, p, U, V, W, rank):
    """One ALS iteration."""
    for i in range(n):
        for k in range(p):
            A = np.zeros((m, rank))
            for j in range(m):
                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
            sol, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
            W[k * n + i, :] = sol
    
    for i in range(n):
        for j in range(m):
            B = np.zeros((p, rank))
            for k in range(p):
                B[k, :] = V[j * p + k, :] * W[k * n + i, :]
            sol, _, _, _ = np.linalg.lstsq(B, np.ones(p), rcond=None)
            U[i * m + j, :] = sol
    
    for j in range(m):
        for k in range(p):
            C = np.zeros((n, rank))
            for i in range(n):
                C[i, :] = U[i * m + j, :] * W[k * n + i, :]
            sol, _, _, _ = np.linalg.lstsq(C, np.ones(n), rcond=None)
            V[j * p + k, :] = sol

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 6
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
    
    # Strategy 1: Try the structured rank-12 construction
    if time.time() < deadline - 20:
        success, U12, V12, W12, r12 = _build_rank12_4x4x6()
        if success and r12 < best_rank:
            best_rank = r12
            best_U = U12
            best_V = V12
            best_W = W12
    
    # Strategy 2: Try ALS from various initializations for lower ranks
    target_ranks = [12, 14, 16, 18, 20]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 15:
            break
        if target_rank >= best_rank:
            continue
        
        max_attempts = 10
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            if attempt % 3 == 0:
                # Block-structured init
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                n_half = n // 2
                rank_half = target_rank // 2
                for block in range(2):
                    i_start = block * n_half
                    r_start = block * rank_half
                    r_end = r_start + rank_half
                    for i in range(i_start, i_start + n_half):
                        for j in range(m):
                            U[i * m + j, r_start:r_end] = np.random.randn(rank_half) * 0.5
                    for j in range(m):
                        for k in range(p):
                            V[j * p + k, r_start:r_end] = np.random.randn(rank_half) * 0.5
                    for k in range(p):
                        for i in range(i_start, i_start + n_half):
                            W[k * n + i, r_start:r_end] = np.random.randn(rank_half) * 0.5
            elif attempt % 3 == 1:
                U = np.random.randn(n * m, target_rank) * 0.5
                V = np.random.randn(m * p, target_rank) * 0.5
                W = np.random.randn(p * n, target_rank) * 0.5
            else:
                U = np.random.randn(n * m, target_rank) * 0.3
                V = np.random.randn(m * p, target_rank) * 0.3
                W = np.random.randn(p * n, target_rank) * 0.3
                for c in range(target_rank):
                    U[np.random.random(n * m) < 0.3, c] = 0
                    V[np.random.random(m * p) < 0.3, c] = 0
                    W[np.random.random(p * n) < 0.3, c] = 0
            
            prev_err = float('inf')
            for it in range(200):
                if time.time() > deadline - 15:
                    break
                
                err = _compute_error_fast(n, m, p, U, V, W)
                
                if err < 1e-10:
                    ok, Ur, Vr, Wr = _try_round_and_verify(n, m, p, U, V, W, target_rank)
                    if ok:
                        best_rank = target_rank
                        best_U = Ur
                        best_V = Vr
                        best_W = Wr
                        found = True
                        break
                
                if abs(prev_err - err) < 1e-14:
                    ok, Ur, Vr, Wr = _try_round_and_verify(n, m, p, U, V, W, target_rank)
                    if ok:
                        best_rank = target_rank
                        best_U = Ur
                        best_V = Vr
                        best_W = Wr
                        found = True
                        break
                    break
                
                prev_err = err
                _als_step(n, m, p, U, V, W, target_rank)
            
            if found:
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
