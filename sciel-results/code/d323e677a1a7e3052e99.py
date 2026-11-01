import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    N = n * m
    M = m * p
    P = p * n
    
    def validate(U, V, W, tol=1e-9):
        """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m + j, :]
                    v_val = V[j*p + k, :]
                    w_val = W[k*n + i, :]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def schoolbook_decomposition():
        """Return the standard rank-45 schoolbook decomposition"""
        rank = n * m * p
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        
        return U, V, W
    
    def random_init(rank, seed=None):
        """Initialize random factor matrices"""
        rng = np.random.RandomState(seed)
        U = rng.randn(N, rank) * 0.1
        V = rng.randn(M, rank) * 0.1
        W = rng.randn(P, rank) * 0.1
        return U, V, W
    
    def fast_gradient_descent(U, V, W, rank, max_iter=3000, lr=0.01, seed=None):
        """Fast gradient descent using vectorized operations"""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Precompute index arrays for all (i,j,k) triples
        triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        idx_u = np.array([i*m + j for i, j, k in triples])
        idx_v = np.array([j*p + k for i, j, k in triples])
        idx_w = np.array([k*n + i for i, j, k in triples])
        
        # Build mapping: for each row in U, which (i,j,k) triples use it
        u_to_triples = {}
        v_to_triples = {}
        w_to_triples = {}
        for t, (i, j, k) in enumerate(triples):
            u_idx = i*m + j
            v_idx = j*p + k
            w_idx = k*n + i
            u_to_triples.setdefault(u_idx, []).append(t)
            v_to_triples.setdefault(v_idx, []).append(t)
            w_to_triples.setdefault(w_idx, []).append(t)
        
        for it in range(max_iter):
            # Compute s_{i,j,k} for all triples
            s = np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :], axis=1)
            err = 1.0 - s  # shape: (n*m*p,)
            
            # Compute gradients using vectorized ops
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            # For each U row, accumulate contributions from all triples using that row
            for u_idx in range(N):
                if u_idx not in u_to_triples:
                    continue
                t_list = u_to_triples[u_idx]
                # err[t_list] has shape (len(t_list),)
                # V[idx_v[t_list], :] has shape (len(t_list), rank)
                # W[idx_w[t_list], :] has shape (len(t_list), rank)
                # grad_U[u_idx, :] = -2 * sum_t err[t] * V[idx_v[t], :] * W[idx_w[t], :]
                contrib = np.sum(err[t_list, None] * V[idx_v[t_list], :] * W[idx_w[t_list], :], axis=0)
                grad_U[u_idx, :] = -2.0 * contrib
            
            for v_idx in range(M):
                if v_idx not in v_to_triples:
                    continue
                t_list = v_to_triples[v_idx]
                contrib = np.sum(err[t_list, None] * U[idx_u[t_list], :] * W[idx_w[t_list], :], axis=0)
                grad_V[v_idx, :] = -2.0 * contrib
            
            for w_idx in range(P):
                if w_idx not in w_to_triples:
                    continue
                t_list = w_to_triples[w_idx]
                contrib = np.sum(err[t_list, None] * U[idx_u[t_list], :] * V[idx_v[t_list], :], axis=0)
                grad_W[w_idx, :] = -2.0 * contrib
            
            # Compute loss
            loss = np.sum(err**2)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            if loss < 1e-10:
                break
            
            # Check time
            if time.time() > deadline - 25:
                break
        
        return U, V, W, loss
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = schoolbook_decomposition()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known lower bound is max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # Try ranks from 15 up to 44
    
    for rank in range(15, min(44, n*m*p)):
        if time.time() > deadline - 35:
            break
        
        # Try multiple random initializations
        max_attempts = 2
        for attempt in range(max_attempts):
            if time.time() > deadline - 25:
                break
            
            seed = hash((rank, attempt)) % (2**32)
            U_init, V_init, W_init = random_init(rank, seed=seed)
            
            # Run fast gradient descent
            U_opt, V_opt, W_opt, loss = fast_gradient_descent(
                U_init, V_init, W_init, 
                rank, max_iter=2000, lr=0.005
            )
            
            # Validate
            if validate(U_opt, V_opt, W_opt, tol=1e-6):
                best_U, best_V, best_W = U_opt, V_opt, W_opt
                best_rank = rank
                break  # Found valid decomposition at this rank
        
        # If we found a valid decomposition at this rank, we can stop
        if best_rank == rank:
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
