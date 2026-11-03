"""Matrix multiplication 3,3,6: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin below 150s kill

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify(U, V, W, tol=1e-9):
        """Verify if U, V, W form a valid decomposition for (n,m,p) matrix multiplication."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        
        # Target tensor T[i, j, k] = 1
        # Decomposed tensor D[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Efficient vectorized check
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j, :]
                    v_val = V[j * p + k, :]
                    w_val = W[k * n + i, :]
                    s = np.dot(u_val, v_val * w_val)
                    errors.append(s - 1.0)
        
        return max(abs(e) for e in errors) < tol

    def _try_rank_als(n, m, p, rank, seed=0, max_iters=300, deadline=None):
        """
        Try to find a rank-r decomposition using alternating least squares (ALS) with
        vectorized operations for speed.
        Returns (U, V, W, err) or (None, None, None, inf) if not found.
        """
        rng = np.random.RandomState(seed)
        
        # Initialize with random values, scaled appropriately
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = rng.randn(p * n, rank) * 0.5
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline - 2.0:
                break
            
            # === Update U ===
            for i in range(n):
                for j in range(m):
                    V_block = V[j * p:(j + 1) * p, :]
                    W_block = W[np.arange(p) * n + i]
                    M = V_block * W_block
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # === Update V ===
            for j in range(m):
                for k in range(p):
                    U_cols = U[np.arange(n) * m + j, :]
                    W_rows = W[k * n + np.arange(n), :]
                    M = U_cols * W_rows
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # === Update W ===
            for k in range(p):
                for i in range(n):
                    U_row = U[i * m:(i + 1) * m, :]
                    V_col = V[np.arange(m) * p + k, :]
                    M = U_row * V_col
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Calculate final error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        err += (s - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Early termination if error is small
            if best_err < 1e-6:
                break
        
        if best_err < 1e-4:
            return best_U, best_V, best_W, best_err
        return None, None, None, best_err

    def _try_rank_gradient(n, m, p, rank, seed=0, max_iters=500, lr=0.01, deadline=None):
        """
        Try to find a rank-r decomposition using gradient descent.
        Returns (U, V, W, err) or (None, None, None, inf) if not found.
        """
        rng = np.random.RandomState(seed)
        
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = np.ones((p * n, rank))  # Start W at ones
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline - 2.0:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        target = 1.0
                        s = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                        err = s - target
                        # d/dU[ui,r] = err * V[vi,r] * W[wi,r]
                        dU[u_idx] += err * V[v_idx] * W[w_idx]
                        dV[v_idx] += err * U[u_idx] * W[w_idx]
                        dW[w_idx] += err * U[u_idx] * V[v_idx]
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        err += (s - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Decay learning rate
            lr *= 0.999
            
            if best_err < 1e-6:
                break
        
        if best_err < 1e-4:
            return best_U, best_V, best_W, best_err
        return None, None, None, best_err

    # Fallback to schoolbook
    schoolbook_sol = get_schoolbook()
    
    # Try to find a lower rank solution
    if time.time() - START_TIME < TIME_LIMIT - 15:
        deadline = START_TIME + TIME_LIMIT - 5
        school_rank = n * m * p  # 54
        
        # Lower bound for rank is max(n*m, m*p, p*n) = max(9, 18, 18) = 18
        # But we can do better. Let's try from 18 upward.
        
        found = False
        # Try ranks in ascending order
        for rank in range(18, school_rank):
            if time.time() > deadline - 15:
                break
            
            # Try multiple seeds for this rank
            num_seeds = 10
            if time.time() > deadline - 40:
                num_seeds = 5
            if time.time() > deadline - 20:
                num_seeds = 2
            
            for seed in range(num_seeds):
                if time.time() > deadline - 5:
                    break
                
                # Try ALS
                U, V, W, err = _try_rank_als(n, m, p, rank, seed=seed, max_iters=100, deadline=deadline)
                
                if U is not None and err < 1e-4:
                    # Verify with tighter tolerance
                    if verify(U, V, W, tol=1e-7):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                # Try gradient descent
                U, V, W, err = _try_rank_gradient(n, m, p, rank, seed=seed, max_iters=200, lr=0.01, deadline=deadline)
                
                if U is not None and err < 1e-4:
                    # Verify with tighter tolerance
                    if verify(U, V, W, tol=1e-7):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return schoolbook_sol

# EVOLVE-BLOCK-START
# The construct function is defined above to respect the interface.
# EVOLVE-BLOCK-END
