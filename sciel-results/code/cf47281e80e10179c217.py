import numpy as np
import time
import random

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Build the target tensor T
    # U dim: n*m
    # V dim: m*p
    # W dim: p*n
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1.0
                    V[v_idx, col] = 1.0
                    W[w_idx, col] = 1.0
                    col += 1
        return U, V, W, rank

    def check_half_integer(U, V, W):
        # Check if rounding to nearest 0.5 gives exact decomposition
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        
        # Reconstruct and check
        try:
            T_rec = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
            if np.allclose(T_rec, T, atol=1e-9):
                return U_r, V_r, W_r
        except Exception:
            pass
        return None

    def polish(U, V, W, max_iter=1000, tol=1e-12):
        """Polish the solution using higher precision iterations"""
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            # Compute reconstruction
            Rec = np.einsum('ur,vr,wr->uvw', U, V, W)
            Err = T - Rec
            err_norm = np.linalg.norm(Err)
            if err_norm < tol:
                break
            
            # Compute gradients
            G_U = np.einsum('uvw,vr,wr->ur', Err, V, W)
            G_V = np.einsum('uvw,ur,wr->vr', Err, U, W)
            G_W = np.einsum('uvw,ur,vr->wr', Err, U, V)
            
            # Adaptive learning rate based on gradient magnitude
            grad_norm = np.sqrt(np.linalg.norm(G_U)**2 + np.linalg.norm(G_V)**2 + np.linalg.norm(G_W)**2)
            if grad_norm < 1e-15:
                break
            lr = 0.01 / (1 + grad_norm)
            
            U -= lr * G_U
            V -= lr * G_V
            W -= lr * G_W
        
        return check_half_integer(U, V, W)

    def optimize_rank(rank, n_restarts=5):
        """Try to find rank decomposition with multiple restarts"""
        best_result = None
        best_err = float('inf')
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization with different scales
            scale = np.random.uniform(0.5, 2.0)
            U = np.random.randn(dim_u, rank) * scale
            V = np.random.randn(dim_v, rank) * (1.0 / scale)
            W = np.random.randn(dim_w, rank)
            
            # Gradient descent with momentum
            lr = 0.005
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            for step in range(500):
                if time.time() > deadline:
                    break
                
                # Calculate Reconstruction
                Rec = np.einsum('ur,vr,wr->uvw', U, V, W)
                Err = T - Rec
                err_norm = np.linalg.norm(Err)
                
                if err_norm < 1e-10:
                    break
                
                # Compute gradients
                G_U = np.einsum('uvw,vr,wr->ur', Err, V, W)
                G_V = np.einsum('uvw,ur,wr->vr', Err, U, W)
                G_W = np.einsum('uvw,ur,vr->wr', Err, U, V)
                
                # Update with momentum
                vU = momentum * vU - lr * G_U
                vV = momentum * vV - lr * G_V
                vW = momentum * vW - lr * G_W
                
                U += vU
                V += vV
                W += vW
                
                # Periodic normalization to prevent scale drift
                if step % 50 == 0:
                    norm_U = np.linalg.norm(U)
                    norm_V = np.linalg.norm(V)
                    norm_W = np.linalg.norm(W)
                    if norm_U > 1e6 or norm_V > 1e6 or norm_W > 1e6:
                        U /= norm_U
                        V /= norm_V
                        W /= norm_W
            
            # Try to polish
            result = polish(U, V, W)
            if result is not None:
                return result
            
            # Track best error for potential recovery
            Rec = np.einsum('ur,vr,wr->uvw', U, V, W)
            err = np.linalg.norm(T - Rec)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # If we had a good candidate, try harder to polish it
        if best_err < 0.1:
            result = polish(best_U, best_V, best_W, max_iter=2000, tol=1e-13)
            if result is not None:
                return result
        
        return None

    # Try to find lower rank
    # Start from 45, go up to 59
    # If we find one, return it.
    
    # Note: Random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    for rank in range(45, 60):
        if time.time() > deadline:
            break
            
        # Try to find a solution at this rank
        found = optimize_rank(rank)
        if found is not None:
            U, V, W = found
            # Convert to list of lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to Schoolbook
    U, V, W, rank = get_schoolbook()
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
