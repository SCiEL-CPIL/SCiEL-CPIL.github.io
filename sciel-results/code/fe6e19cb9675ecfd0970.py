import numpy as np
import time
import random

TIME_LIMIT = 140  # safety margin from 150s

def _verify(U, V, W):
    """Check if U,V,W exactly decompose the 3x4x5 matmul tensor."""
    n, m, p = 3, 4, 5
    rank = U.shape[1]
    
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Compute tensor
    T = np.einsum('ur,vr,wr->uvw', U, V, W)
    
    # Expected non-zero entries
    valid_u = []
    valid_v = []
    valid_w = []
    for i in range(n):
        for l in range(m):
            for k in range(p):
                valid_u.append(i * m + l)
                valid_v.append(l * p + k)
                valid_w.append(k * n + i)
    
    valid_u = np.array(valid_u)
    valid_v = np.array(valid_v)
    valid_w = np.array(valid_w)
    
    # Check non-zero entries
    if not np.allclose(T[valid_u, valid_v, valid_w], 1.0, atol=1e-9):
        return False
    
    # Check zero entries
    mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    mask[valid_u, valid_v, valid_w] = True
    
    invalid = ~mask
    if np.any(np.abs(T[invalid]) > 1e-9):
        return False
    
    return True

def _seed_solution():
    """Return the seed 56-rank solution as U,V,W numpy arrays."""
    n, m, p = 3, 4, 5
    U_parts = []
    V_parts = []
    W_parts = []
    
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        return U, V, W
    
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        return U, V, W
    
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        return U, V, W
    
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        return U, V, W
    
    def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        for r in range(U_blk.shape[1]):
            U_parts.append(np.zeros(n * m))
            for i, idx in enumerate(u_idx):
                U_parts[-1][idx] = U_blk[i, r]
            V_parts.append(np.zeros(m * p))
            for i, idx in enumerate(v_idx):
                V_parts[-1][idx] = V_blk[i, r]
            W_parts.append(np.zeros(p * n))
            for i, idx in enumerate(w_idx):
                W_parts[-1][idx] = W_blk[i, r]
    
    add_block([0,1,4,5], [0,1,5,6], [0,3,1,4], *strassen_2x2x2())
    add_block([0,1,4,5], [2,3,7,8], [6,9,7,10], *strassen_2x2x2())
    add_block([0,1,4,5], [4,9], [12,13], *schoolbook_2x2x1())
    add_block([2,3,6,7], [10,11,15,16], [0,3,1,4], *strassen_2x2x2())
    add_block([2,3,6,7], [12,13,17,18], [6,9,7,10], *strassen_2x2x2())
    add_block([2,3,6,7], [14,19], [12,13], *schoolbook_2x2x1())
    add_block([8,9], [0,1,5,6], [2,5], *schoolbook_1x2x2())
    add_block([8,9], [2,3,7,8], [8,11], *schoolbook_1x2x2())
    add_block([8,9], [4,9], [14], *schoolbook_1x2x1())
    add_block([10,11], [10,11,15,16], [2,5], *schoolbook_1x2x2())
    add_block([10,11], [12,13,17,18], [8,11], *schoolbook_1x2x2())
    add_block([10,11], [14,19], [14], *schoolbook_1x2x1())
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return U, V, W

def _als_optimize(U, V, W, T_target, n_iters=20):
    """Perform Alternating Least Squares to fit T_target."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for _ in range(n_iters):
        # Solve for U: T_target = U (V^T W^T)
        # T_target is (n*m, m*p, p*n)
        # Reshape V, W to (m*p, p*n, rank) and (p*n, m*p, rank)?
        # T = sum_r u_r v_r^T w_r^T
        # Flatten T to (n*m, (m*p)*(p*n))
        # This is complex. Let's use einsum.
        
        # Fix V, W. Solve for U.
        # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # For fixed i, T[i,:,:] = sum_r U[i,r] (V[:,r] W[:,r]^T)
        # This is a linear system for U[:,r].
        
        # Efficiently:
        # T_flat (n*m, m*p*p*n)
        # Design matrix for U: (m*p*p*n, rank) where entry (idx, r) is V[j,r]W[k,r]
        # This is too big.
        
        # Use gradient descent instead? Or simple least squares on slices.
        # Let's just do a few steps of gradient descent on the error.
        # Error = || T - sum U V W ||^2
        # Gradient wrt U: 2 * (T - T_est) * (V W)^T
        
        T_est = np.einsum('ur,vr,wr->uvw', U, V, W)
        Err = T_target - T_est
        
        # Grad U: Einsum('uvw, vrw -> urw')? No.
        # d/dU[i,r] (Err[i,j,k] V[j,r] W[k,r])
        # Grad_U[i,r] = sum_{j,k} Err[i,j,k] V[j,r] W[k,r]
        Grad_U = np.einsum('ijk,jr,kr->ir', Err, V, W)
        Grad_V = np.einsum('ijk,ir,kr->jr', Err, U, W)
        Grad_W = np.einsum('ijk,ir,jr->kr', Err, U, V)
        
        # Update
        lr = 0.01
        U -= lr * Grad_U
        V -= lr * Grad_V
        W -= lr * Grad_W
        
    return U, V, W

def _round_to_half(U, V, W):
    """Round entries to nearest half-integer."""
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = 3, 4, 5
    
    # Build Target Tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for l in range(m):
            for k in range(p):
                T_target[i*m+l, l*p+k, k*n+i] = 1.0
                
    # Get Seed
    U_seed, V_seed, W_seed = _seed_solution()
    best_U, best_V, best_W = U_seed, V_seed, W_seed
    best_rank = U_seed.shape[1]
    
    # Verify Seed
    if not _verify(best_U, best_V, best_W):
        raise ValueError("Seed solution failed verification")
        
    # Try to find lower rank solutions
    # Target ranks to try
    target_ranks = [52, 50, 48, 46, 44]
    
    for tr in target_ranks:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        if tr >= best_rank:
            continue
            
        # Try multiple random subsets of the seed columns
        for attempt in range(20):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
                
            # Randomly select 'tr' columns from seed
            idx = np.random.choice(best_rank, tr, replace=False)
            U_init = best_U[:, idx]
            V_init = best_V[:, idx]
            W_init = best_W[:, idx]
            
            # Optimize
            U_opt, V_opt, W_opt = _als_optimize(U_init, V_init, W_opt, T_target, n_iters=50)
            
            # Round
            U_r, V_r, W_r = _round_to_half(U_opt, V_opt, W_opt)
            
            # Verify
            if _verify(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = tr
                break # Found a solution for this rank, try lower
        
    # Return best solution
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
