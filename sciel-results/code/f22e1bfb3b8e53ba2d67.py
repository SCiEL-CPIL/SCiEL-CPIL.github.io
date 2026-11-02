import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin before 150s kill
    
    # Precompute the tensor structure
    # For each (i, j) pair in U x V, we have a corresponding k in W
    # i = a*m + b (a in [0,n), b in [0,m))
    # j = c*p + d (c in [0,m), d in [0,p))
    # k = d*n + a
    # The target is 1 if b == c, else 0
    
    # Build a list of (i, j, k, target) tuples
    tensor_entries = []
    for i in range(nm):
        a, b = divmod(i, m)
        for j in range(mp):
            c, d = divmod(j, p)
            k = d * n + a
            target = 1.0 if b == c else 0.0
            tensor_entries.append((i, j, k, target))
    
    n_entries = len(tensor_entries)
    
    def compute_residual(U, V, W):
        """Compute the exact residual for a given decomposition."""
        res = 0.0
        for i, j, k, target in tensor_entries:
            val = np.dot(U[i, :], V[j, :] * W[k, :])
            res += (val - target) ** 2
        return res
    
    def verify(U, V, W, tol=1e-8):
        """Verify if decomposition is exact."""
        for i, j, k, target in tensor_entries:
            val = np.dot(U[i, :], V[j, :] * W[k, :])
            if abs(val - target) > tol:
                return False
        return True
    
    def try_optimize(R, n_restarts=10, max_iter=300):
        """Try to find rank R decomposition using optimization with better initialization."""
        best_res = np.inf
        best_params = None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
                
            # Better initialization: start from schoolbook and randomly drop columns,
            # or use structured random initialization
            if restart == 0:
                # Start from schoolbook and try to compress
                if R >= rank_sb:
                    U = U_sb.copy()[:, :R]
                    V = V_sb.copy()[:, :R]
                    W = W_sb.copy()[:, :R]
                else:
                    # Take first R columns of schoolbook
                    U = U_sb.copy()[:, :R]
                    V = V_sb.copy()[:, :R]
                    W = W_sb.copy()[:, :R]
            elif restart == 1:
                # Random initialization with larger scale
                U = np.random.randn(nm, R)
                V = np.random.randn(mp, R)
                W = np.random.randn(pn, R)
            else:
                # Mixed: some schoolbook columns + some random
                n_sb_cols = min(R // 2, rank_sb)
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                W = np.zeros((pn, R))
                U[:, :n_sb_cols] = U_sb[:, :n_sb_cols]
                V[:, :n_sb_cols] = V_sb[:, :n_sb_cols]
                W[:, :n_sb_cols] = W_sb[:, :n_sb_cols]
                U[:, n_sb_cols:] = np.random.randn(nm, R - n_sb_cols) * 0.1
                V[:, n_sb_cols:] = np.random.randn(mp, R - n_sb_cols) * 0.1
                W[:, n_sb_cols:] = np.random.randn(pn, R - n_sb_cols) * 0.1
            
            # Adaptive gradient descent
            lr = 0.01
            prev_res = np.inf
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                    
                # Compute gradients
                err_U = np.zeros((nm, R))
                err_V = np.zeros((mp, R))
                err_W = np.zeros((pn, R))
                
                for i, j, k, target in tensor_entries:
                    uv = U[i, :] * V[j, :]
                    val = np.dot(uv, W[k, :])
                    err = val - target
                    
                    err_U[i, :] += err * (V[j, :] * W[k, :])
                    err_V[j, :] += err * (U[i, :] * W[k, :])
                    err_W[k, :] += err * (U[i, :] * V[j, :])
                
                # Update with momentum-like behavior
                U -= lr * err_U
                V -= lr * err_V
                W -= lr * err_W
                
                res = compute_residual(U, V, W)
                if res < best_res:
                    best_res = res
                    best_params = (U.copy(), V.copy(), W.copy())
                
                if res < 1e-14:
                    break
                
                # Adaptive learning rate
                if res > prev_res * 1.05:
                    lr *= 0.5
                elif it > 10 and res < prev_res * 0.99:
                    lr = min(lr * 1.1, 0.05)
                prev_res = res
                
                # Early stopping if stuck
                if it > 50 and abs(res - prev_res) < 1e-15:
                    break
        
        return best_res, best_params
    
    # Try to find lower rank
    # Theoretical lower bound for 3x4x7 matrix multiplication
    # Strassen showed that rank < n*m*p is possible for large matrices
    # For small matrices, the exact minimum rank is often unknown
    # Let's try from 83 down
    
    min_rank = 50
    for R in range(rank_sb - 1, min_rank - 1, -1):
        if time.time() > deadline - 15:
            break
        
        res, params = try_optimize(R, n_restarts=8, max_iter=200)
        
        if res < 1e-10 and params is not None:
            U, V, W = params
            if verify(U, V, W, tol=1e-8):
                best_rank = R
                best_U, best_V, best_W = U, V, W
                break
    
    # Return the best result found
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
