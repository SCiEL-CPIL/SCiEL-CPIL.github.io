"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x10 matrix.
    
    The schoolbook decomposition has rank n*m*p = 60.
    We attempt to find a lower rank decomposition using adaptive gradient descent
    on the tensor reconstruction loss, followed by a rounding check to
    half-integers. If no valid lower rank is found within the time budget,
    we fall back to the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    # Build the target tensor T
    # T[u, v, w] = 1 if there exists (i, j, k) such that:
    # u = i*m + j, v = j*p + k, w = k*n + i
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
        return U, V, W

    def check_half_integer(U, V, W):
        """
        Checks if rounding U, V, W to nearest 0.5 yields an exact decomposition.
        Returns rounded matrices if valid, else None.
        """
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        
        # Reconstruct tensor: T_rec[u, v, w] = sum_r U[u,r] V[v,r] W[w,r]
        T_rec = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
        
        if np.allclose(T_rec, T, atol=1e-9):
            return U_r, V_r, W_r
        return None

    def optimize_rank(rank, seed=42, max_steps=2000):
        """
        Attempts to find a rank-`rank` decomposition via adaptive gradient descent.
        Returns (U, V, W) if a valid half-integer solution is found, else None.
        """
        if time.time() > deadline:
            return None
            
        np.random.seed(seed)
        
        # Initialize with small random values
        scale = 0.1
        U = np.random.randn(dim_u, rank) * scale
        V = np.random.randn(dim_v, rank) * scale
        W = np.random.randn(dim_w, rank) * scale
        
        # Adam optimizer state
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)
        t = 0
        lr = 0.01
        
        for step in range(max_steps):
            if time.time() > deadline:
                return None
            t += 1
            
            # Calculate Reconstruction
            Rec = np.einsum('ur,vr,wr->uvw', U, V, W)
            
            # Error
            Err = T - Rec
            
            # Gradient
            try:
                G_U = np.einsum('uvw,vr,wr->ur', Err, V, W)
                G_V = np.einsum('uvw,ur,wr->vr', Err, U, W)
                G_W = np.einsum('uvw,ur,vr->wr', Err, U, V)
            except Exception:
                return None

            # Adam update
            mU = beta1 * mU + (1 - beta1) * G_U
            vU = beta2 * vU + (1 - beta2) * (G_U ** 2)
            mU_hat = mU / (1 - beta1 ** t)
            vU_hat = vU / (1 - beta2 ** t)
            U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
            
            mV = beta1 * mV + (1 - beta1) * G_V
            vV = beta2 * vV + (1 - beta2) * (G_V ** 2)
            mV_hat = mV / (1 - beta1 ** t)
            vV_hat = vV / (1 - beta2 ** t)
            V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
            
            mW = beta1 * mW + (1 - beta1) * G_W
            vW = beta2 * vW + (1 - beta2) * (G_W ** 2)
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            
            # Periodic check for convergence to half-integers
            if step % 20 == 0:
                result = check_half_integer(U, V, W)
                if result is not None:
                    return result

        # Final check
        return check_half_integer(U, V, W)

    # Try to find lower rank decompositions
    # Focus most effort on rank 59, then 58, etc.
    # Rank 59 is the most likely to have a solution if any lower rank exists.
    
    best_result = None
    best_rank = n * m * p
    
    # Allocate time: 60% to rank 59, 20% to rank 58, 10% to rank 57, 10% to rank 56
    total_time_budget = deadline - start_time
    time_alloc = {
        59: total_time_budget * 0.6,
        58: total_time_budget * 0.2,
        57: total_time_budget * 0.1,
        56: total_time_budget * 0.1,
    }
    
    for rank in [59, 58, 57, 56]:
        if time.time() > deadline:
            break
            
        rank_start = time.time()
        rank_deadline = rank_start + time_alloc.get(rank, 5.0)
        
        found = None
        restart = 0
        while time.time() < rank_deadline:
            if time.time() > deadline:
                break
            seed = 42 + restart
            res = optimize_rank(rank, seed=seed)
            if res is not None:
                found = res
                break
            restart += 1
            
        if found:
            U, V, W = found
            # Convert to list of lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to Schoolbook
    U, V, W = get_schoolbook()
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
