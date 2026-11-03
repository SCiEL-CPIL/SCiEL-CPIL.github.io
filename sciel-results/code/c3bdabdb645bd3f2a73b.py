"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = make_schoolbook()
    best_rank = n * m * p

    # Precompute index arrays for efficient verification
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx

    def verify(U, V, W, r):
        """Check if U, V, W form a valid tensor decomposition. Returns max error."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        u_vals = Ua[u_idx, :]  # (n*m*p, r)
        v_vals = Va[v_idx, :]  # (n*m*p, r)
        w_vals = Wa[w_idx, :]  # (n*m*p, r)
        
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        
        return np.max(np.abs(result - 1.0))

    def refine(U, V, W, r, max_iter=200):
        """Vectorized gradient descent refinement. Returns refined arrays."""
        Ua = np.array(U, dtype=float).copy()
        Va = np.array(V, dtype=float).copy()
        Wa = np.array(W, dtype=float).copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            u_vals = Ua[u_idx, :]
            v_vals = Va[v_idx, :]
            w_vals = Wa[w_idx, :]
            
            prod = u_vals * v_vals * w_vals
            result = prod.sum(axis=1)
            error = result - 1.0
            err_norm = np.linalg.norm(error)
            
            if err_norm < 1e-10:
                break
            
            # Reshape to 4D: (n, m, p, r)
            err_4d = error.reshape(n, m, p, 1)
            u_4d = u_vals.reshape(n, m, p, r)
            v_4d = v_vals.reshape(n, m, p, r)
            w_4d = w_vals.reshape(n, m, p, r)
            
            # Vectorized gradient computations
            # grad_U[i*m+j, c] = sum_k 2*err[i,j,k]*V[j*p+k,c]*W[k*n+i,c]
            grad_U_4d = 2 * err_4d * v_4d * w_4d
            grad_U = np.sum(grad_U_4d, axis=2).reshape(n*m, r)
            
            # grad_V[j*p+k, c] = sum_i 2*err[i,j,k]*U[i*m+j,c]*W[k*n+i,c]
            grad_V_4d = 2 * err_4d * u_4d * w_4d
            # Sum over i axis (axis=0) to get (m, p, r), then reshape to (m*p, r)
            grad_V = np.sum(grad_V_4d, axis=0).reshape(m*p, r)
            
            # grad_W[k*n+i, c] = sum_j 2*err[i,j,k]*U[i*m+j,c]*V[j*p+k,c]
            grad_W_4d = 2 * err_4d * u_4d * v_4d
            # Sum over j axis (axis=1) to get (n, p, r), then transpose to (p, n, r) and reshape
            grad_W = np.sum(grad_W_4d, axis=1).transpose(1, 0, 2).reshape(p*n, r)
            
            # Adaptive learning rate with backtracking
            lr = 0.1 / (1.0 + it * 0.1)
            
            Ua -= lr * grad_U
            Va -= lr * grad_V
            Wa -= lr * grad_W
        
        return Ua, Va, Wa

    def try_rank(r, n_trials=10, max_iter=100):
        """Try to find a rank-r decomposition with random restarts."""
        for trial in range(n_trials):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(trial * 104729 + r * 31 + 7)
            
            # Random initialization with various scales
            scale = [0.5, 1.0, 2.0, 0.1, 5.0][trial % 5]
            U = np.random.randn(n * m, r) * scale
            V = np.random.randn(m * p, r) * scale
            W = np.random.randn(p * n, r) * scale
            
            U_ref, V_ref, W_ref = refine(U, V, W, r, max_iter=max_iter)
            
            err = verify(U_ref, V_ref, W_ref, r)
            if err < 1e-8:
                return {
                    "U": U_ref.tolist(),
                    "V": V_ref.tolist(),
                    "W": W_ref.tolist()
                }
        
        return None

    # Search for lower rank decompositions
    # Lower bound for 4x4x7: at least max(n*m, m*p, p*n) = max(16, 28, 28) = 28
    # Realistic lower bound is higher; try from ~50 upward
    search_ranks = list(range(50, min(best_rank, 112)))
    
    for r in search_ranks:
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, n_trials=8, max_iter=80)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    if time.time() > deadline:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
