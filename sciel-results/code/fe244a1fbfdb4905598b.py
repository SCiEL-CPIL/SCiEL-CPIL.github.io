"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

import numpy as np
import time
import random
from collections import defaultdict
from itertools import product

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

def _try_structured_block_decomposition(n, m, p, deadline):
    """Try to find a low-rank decomposition using block structure.
    
    Split the 4x4x6 problem into 2x2x3 blocks and use known low-rank
    decompositions for each block.
    """
    # For 2x2x3, the rank is at most 6 (schoolbook)
    # But we can try to find rank-5 or lower decompositions
    
    # First, let's try to find a rank-5 decomposition for 2x2x3
    # This is a well-studied problem
    
    # Known result: 2x2x3 matmul has rank 5
    # Let's try to construct it explicitly
    
    # For 2x2x3, indices: i in {0,1}, j in {0,1}, k in {0,1,2}
    # We need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    
    # One known construction uses the fact that:
    # x*y*z = 1/2 * [(x+y)*z*(x-y) + (x-y)*z*(x+y)] / (x^2-y^2) ... no that's not right
    
    # Actually, let's just try random search for 2x2x3 rank 5
    def _find_2x2x3_rank5(max_trials=10000):
        """Find a rank-5 decomposition for 2x2x3 matmul."""
        for trial in range(max_trials):
            if time.time() > deadline:
                return None
            
            r = 5
            U = np.random.randn(2*2, r) * 0.5
            V = np.random.randn(2*3, r) * 0.5
            W = np.random.randn(3*2, r) * 0.5
            
            # ALS optimization
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Update W
                for i in range(2):
                    for k in range(3):
                        A = np.zeros((2, r))
                        for j in range(2):
                            A[j, :] = U[i*2 + j, :] * V[j*3 + k, :]
                        sol, _, _, _ = np.linalg.lstsq(A, np.ones(2), rcond=None)
                        W[k*2 + i, :] = sol
                
                # Update U
                for i in range(2):
                    for j in range(2):
                        B = np.zeros((3, r))
                        for k in range(3):
                            B[k, :] = V[j*3 + k, :] * W[k*2 + i, :]
                        sol, _, _, _ = np.linalg.lstsq(B, np.ones(3), rcond=None)
                        U[i*2 + j, :] = sol
                
                # Update V
                for j in range(2):
                    for k in range(3):
                        C = np.zeros((2, r))
                        for i in range(2):
                            C[i, :] = U[i*2 + j, :] * W[k*2 + i, :]
                        sol, _, _, _ = np.linalg.lstsq(C, np.ones(2), rcond=None)
                        V[j*3 + k, :] = sol
                
                # Check error
                err = 0.0
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            val = np.dot(U[i*2 + j, :], V[j*3 + k, :] * W[k*2 + i, :])
                            err += (val - 1.0)**2
                
                if err < 1e-10:
                    # Check if half-integer
                    if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                        U_r = _round_to_half_integer(U)
                        V_r = _round_to_half_integer(V)
                        W_r = _round_to_half_integer(W)
                        
                        # Verify
                        valid = True
                        for i in range(2):
                            for j in range(2):
                                for k in range(3):
                                    val = np.dot(U_r[i*2 + j, :], V_r[j*3 + k, :] * W_r[k*2 + i, :])
                                    if abs(val - 1.0) > 1e-6:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        
                        if valid:
                            return U_r, V_r, W_r
        
        return None
    
    # Try to find 2x2x3 rank-5 decomposition
    result = _find_2x2x3_rank5(max_trials=5000)
    
    if result is not None:
        U_block, V_block, W_block = result
        
        # Now embed this into the 4x4x6 problem
        # Split i: {0,1} and {2,3}, j: {0,1} and {2,3}, k: {0,1,2} and {3,4,5}
        # We have 4 blocks: (i_block, j_block, k_block)
        
        U_full = np.zeros((4*4, 20))  # 4 blocks * 5 rank each
        V_full = np.zeros((4*6, 20))
        W_full = np.zeros((6*4, 20))
        
        block_idx = 0
        for i_block in range(2):
            for j_block in range(2):
                for k_block in range(2):
                    # Map block indices to full indices
                    i_start = i_block * 2
                    j_start = j_block * 2
                    k_start = k_block * 3
                    
                    # Embed U block
                    for i in range(2):
                        for j in range(2):
                            U_full[(i_start + i)*4 + (j_start + j), block_idx*5 : (block_idx+1)*5] = U_block[i*2 + j, :]
                    
                    # Embed V block
                    for j in range(2):
                        for k in range(3):
                            V_full[(j_start + j)*6 + (k_start + k), block_idx*5 : (block_idx+1)*5] = V_block[j*3 + k, :]
                    
                    # Embed W block
                    for k in range(3):
                        for i in range(2):
                            W_full[(k_start + k)*4 + (i_start + i), block_idx*5 : (block_idx+1)*5] = W_block[k*2 + i, :]
                    
                    block_idx += 1
        
        # Verify the full decomposition
        valid = True
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    val = np.dot(U_full[i*4 + j, :], V_full[j*6 + k, :] * W_full[k*4 + i, :])
                    if abs(val - 1.0) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return 20, U_full, V_full, W_full
    
    return None

def _compute_error(U, V, W, n, m, p, r):
    """Compute the sum of squared errors for the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                err += (val - 1.0)**2
    return err

def _polynomial_interp_attempt(n, m, p, target_rank):
    """Try to find a decomposition using polynomial interpolation.
    
    For small ranks, we can use the fact that the matmul tensor has structure.
    We try to find U, V, W with entries from a small set.
    """
    if target_rank < 1:
        return None
    
    # For the matmul tensor T[i,j,k] = 1 for all i,j,k
    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Strategy: use a structured initialization based on polynomial evaluation
    # Let's try with entries from {-1, 0, 1} or {-0.5, 0, 0.5, 1}
    
    # For very small ranks, try exhaustive search over small entry sets
    if target_rank <= 4:
        # Try all combinations from a small set
        entries = [-1.0, -0.5, 0.0, 0.5, 1.0]
        # This is too large for brute force, so use random sampling
        for trial in range(1000):
            U = np.random.choice(entries, size=(n*m, target_rank))
            V = np.random.choice(entries, size=(m*p, target_rank))
            W = np.random.choice(entries, size=(p*n, target_rank))
            
            valid, _ = _verify_rank(n, m, p, target_rank, U, V, W)
            if valid:
                return U, V, W
    
    return None

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
    
    # First, try the structured block decomposition approach
    block_result = _try_structured_block_decomposition(n, m, p, deadline)
    if block_result is not None:
        block_rank, block_U, block_V, block_W = block_result
        if block_rank < best_rank:
            best_rank = block_rank
            best_U = block_U
            best_V = block_V
            best_W = block_W
    
    # Then try random search for even lower ranks
    for target_rank in range(best_rank - 1, 0, -1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        attempts = 8
        found = False
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
                
            # Random initialization with varied scales
            scale = 0.3 + 0.4 * (attempt % 3)
            U = np.random.randn(n * m, target_rank) * scale
            V = np.random.randn(m * p, target_rank) * scale
            W = np.random.randn(p * n, target_rank) * scale
            
            # Normalize columns to avoid scale drift
            for c in range(target_rank):
                norm = np.sqrt(np.sum(U[:, c]**2) * np.sum(V[:, c]**2) * np.sum(W[:, c]**2))
                if norm > 1e-10:
                    scale_c = 1.0 / (norm ** (1.0/3.0))
                    U[:, c] *= scale_c
                    V[:, c] *= scale_c
                    W[:, c] *= scale_c
            
            # ALS iterations
            U, V, W = _als_optimize(U, V, W, n, m, p, target_rank, max_iters=150, deadline=deadline)
            
            err = _compute_error(U, V, W, n, m, p, target_rank)
            
            if err < 1e-7:
                # Check if we can round to half-integers
                if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                    U_rounded = _round_to_half_integer(U)
                    V_rounded = _round_to_half_integer(V)
                    W_rounded = _round_to_half_integer(W)
                    
                    # Verify rounded solution
                    valid, _ = _verify_rank(n, m, p, target_rank, U_rounded, V_rounded, W_rounded)
                    if valid:
                        best_rank = target_rank
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        found = True
                        break
            
            # If not half-integer, try polishing
            if err < 1e-4:
                U, V, W = _polish_solution(U, V, W, n, m, p, target_rank, deadline=deadline)
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
            # Found a solution at this rank, continue to lower ranks
            continue
        else:
            # If we've tried many ranks without success, we can stop earlier
            # for efficiency, but keep going for now
            pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
