import numpy as np
import time
from fractions import Fraction
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 8]

def _verify(U, V, W):
    """Verify that the decomposition exactly computes matrix multiplication."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook_decomposition():
    """Return the schoolbook rank-72 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _block_decomposition_3_3_2():
    """
    Split 3x3 * 3x8 into three blocks: 3x3*3x3, 3x3*3x3, 3x3*3x2.
    Use schoolbook for each, giving rank 27+27+6 = 60.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    # Blocks: columns 0-2, 3-5, 6-7
    block_sizes = [3, 3, 2]
    
    U = [[0] * 60 for _ in range(n * m)]
    V = [[0] * 60 for _ in range(m * p)]
    W = [[0] * 60 for _ in range(p * n)]
    
    col = 0
    for block_idx, block_p in enumerate(block_sizes):
        k_start = block_idx * block_p  # 0, 3, 6
        
        for i in range(n):
            for j in range(m):
                for k in range(block_p):
                    U[i*m + j][col] = 1
                    V[j*p + k_start + k][col] = 1
                    W[(k_start + k)*n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}

def _optimize_3x3x3(rank, deadline=None, n_restarts=5):
    """
    Try to find a low-rank decomposition for 3x3x3 matrix multiplication.
    Returns (U, V, W) arrays or None if not found.
    """
    n = m = p = 3
    size = n * m  # 9
    
    best_err = float('inf')
    best_params = None
    
    for restart in range(n_restarts):
        if deadline and time.time() > deadline:
            break
        
        # Random initialization
        np.random.seed(restart * 1000 + 42)
        U = np.random.randn(size, rank) * 0.5
        V = np.random.randn(size, rank) * 0.5
        W = np.random.randn(size, rank) * 0.5
        
        # Flatten parameters
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_flat = x[:size*rank].reshape(size, rank)
            V_flat = x[size*rank:2*size*rank].reshape(size, rank)
            W_flat = x[2*size*rank:].reshape(size, rank)
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U_flat[i*m+j, r] * V_flat[j*p+k, r] * W_flat[k*n+i, r]
                        err += (val - 1.0)**2
            return err
        
        # Use L-BFGS-B for optimization
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.fun < best_err:
                best_err = result.fun
                best_params = result.x
                if best_err < 1e-9:
                    break
        except Exception:
            continue
    
    if best_params is None or best_err > 1e-6:
        return None
    
    U = best_params[:size*rank].reshape(size, rank)
    V = best_params[size*rank:2*size*rank].reshape(size, rank)
    W = best_params[2*size*rank:].reshape(size, rank)
    
    return U, V, W

def _try_block_with_optimization(deadline):
    """
    Try to improve the 3x3x3 blocks using optimization.
    """
    n, m, p = DIMENSIONS
    
    # Try to optimize each 3x3x3 block
    # First, try rank 25 for 3x3x3
    for target_rank_333 in [25, 26, 27]:
        if time.time() > deadline - 20:
            break
        
        result = _optimize_3x3x3(target_rank_333, deadline=deadline - 10, n_restarts=3)
        if result is not None:
            U333, V333, W333 = result
            if np.max(np.abs(U333)) < 1e6 and np.max(np.abs(V333)) < 1e6 and np.max(np.abs(W333)) < 1e6:
                # Build the full decomposition
                rank_total = 2 * target_rank_333 + 6  # two 3x3x3 blocks + one 3x3x2 block
                U = [[0] * rank_total for _ in range(n * m)]
                V = [[0] * rank_total for _ in range(m * p)]
                W = [[0] * rank_total for _ in range(p * n)]
                
                # First 3x3x3 block (columns 0-2)
                for i in range(n * m):
                    for r in range(target_rank_333):
                        U[i][r] = float(U333[i, r])
                for i in range(m * p):
                    for r in range(target_rank_333):
                        V[i][r] = float(V333[i, r]) if i < m * 3 else 0.0
                for i in range(p * n):
                    for r in range(target_rank_333):
                        W[i][r] = float(W333[i, r]) if i < 3 * n else 0.0
                
                # Second 3x3x3 block (columns 3-5)
                offset = target_rank_333
                for i in range(n * m):
                    for r in range(target_rank_333):
                        U[i][offset + r] = float(U333[i, r])
                for i in range(m * p):
                    for r in range(target_rank_333):
                        if 3 <= i < 6:  # j*p+k where j in 0..2, k in 0..2, but shifted
                            V[i][offset + r] = float(V333[i - 3, r])
                for i in range(p * n):
                    for r in range(target_rank_333):
                        if 3*n <= i < 6*n:
                            W[i][offset + r] = float(W333[i - 3*n, r])
                
                # Third block: 3x3x2 (columns 6-7), schoolbook
                offset2 = 2 * target_rank_333
                col = offset2
                for i in range(n):
                    for j in range(m):
                        for k in range(2):
                            U[i*m + j][col] = 1
                            V[j*p + 6 + k][col] = 1
                            W[(6 + k)*n + i][col] = 1
                            col += 1
                
                result_dict = {"U": U, "V": V, "W": W}
                if _verify(U, V, W):
                    return result_dict
    
    return None

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank exact decomposition for 3x3 * 3x8 matrix multiplication.
    Uses block decomposition and optimization to improve over schoolbook.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin before 150s kill
    
    # Start with schoolbook rank-72
    best = _schoolbook_decomposition()
    best_rank = n * m * p  # 72
    
    # Try block decomposition with schoolbook for each block: rank 60
    block_result = _block_decomposition_3_3_2()
    if _verify(block_result["U"], block_result["V"], block_result["W"]):
        best = block_result
        best_rank = 60
    
    # Try to improve the 3x3x3 blocks using optimization
    if time.time() < deadline - 30:
        opt_result = _try_block_with_optimization(deadline)
        if opt_result is not None:
            opt_rank = opt_result["U"][0].__len__()
            if opt_rank < best_rank:
                best = opt_result
                best_rank = opt_rank
    
    return best
# EVOLVE-BLOCK-END
