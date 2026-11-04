import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Construct the target tensor T
    # T is zero everywhere except at (i*m+j, j*p+k, k*n+i) which is 1
    target_indices_u = []
    target_indices_v = []
    target_indices_w = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                target_indices_u.append(idx_u)
                target_indices_v.append(idx_v)
                target_indices_w.append(idx_w)
                
    target_indices_u = np.array(target_indices_u)
    target_indices_v = np.array(target_indices_v)
    target_indices_w = np.array(target_indices_w)
    
    T_target = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    T_target[target_indices_u, target_indices_v, target_indices_w] = 1.0

    def check_exact(U, V, W):
        """Check if U, V, W exactly decompose the target tensor with half-integer entries."""
        # Check half-integer constraint
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        
        if not np.allclose(U2, np.round(U2), atol=1e-6): return False
        if not np.allclose(V2, np.round(V2), atol=1e-6): return False
        if not np.allclose(W2, np.round(W2), atol=1e-6): return False
        
        # Snap to half-integers for exact check
        U_snap = np.round(U2) / 2.0
        V_snap = np.round(V2) / 2.0
        W_snap = np.round(W2) / 2.0
        
        # Check reconstruction
        T_recon = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
        
        if not np.allclose(T_recon, T_target, atol=1e-6):
            return False
            
        return True

    def run_nmtf(rank, iterations=200, seed=42):
        np.random.seed(seed)
        
        # Initialize with non-negative random numbers
        # Scale to roughly match the magnitude of the tensor
        # The sum of squares of T is n*m*p = 80.
        # Rank r. Average energy per component ~ sqrt(80/r)
        scale = np.sqrt(80.0 / rank)
        
        U = np.random.rand(dim_u, rank) * scale
        V = np.random.rand(dim_v, rank) * scale
        W = np.random.rand(dim_w, rank) * scale
        
        epsilon = 1e-12
        
        for it in range(iterations):
            # Update U
            # Num_U[alpha, r] = sum_{beta, gamma} T[alpha, beta, gamma] V[beta, r] W[gamma, r]
            # Den_U[alpha, r] = sum_{beta, gamma} T_recon[alpha, beta, gamma] V[beta, r] W[gamma, r]
            
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            
            # Efficient calculation using matrix multiplications
            # V_r = V[:, r], W_r = W[:, r]
            # Num_U[:, r] = sum_jk T[:, j, k] V[j, r] W[k, r]
            # This is equivalent to: 
            # Temp1 = np.einsum('ijk,kr->ijr', T_target, W)
            # Num_U = np.einsum('ijr,jr->ir', Temp1, V)
            
            Temp1_U = np.einsum('ijk,kr->ijr', T_target, W)
            Num_U = np.einsum('ijr,jr->ir', Temp1_U, V)
            
            Temp1_Recon = np.einsum('ijk,kr->ijr', T_recon, W)
            Den_U = np.einsum('ijr,jr->ir', Temp1_Recon, V)
            
            U = U * (Num_U + epsilon) / (Den_U + epsilon)
            
            # Update V
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            
            Temp1_V = np.einsum('ijk,ir->jkr', T_target, U)
            Num_V = np.einsum('jkr,kr->jr', Temp1_V, W)
            
            Temp1_Recon_V = np.einsum('ijk,ir->jkr', T_recon, U)
            Den_V = np.einsum('jkr,kr->jr', Temp1_Recon_V, W)
            
            V = V * (Num_V + epsilon) / (Den_V + epsilon)
            
            # Update W
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            
            Temp1_W = np.einsum('ijk,jr->ikr', T_target, V)
            Num_W = np.einsum('ikr,ir->kr', Temp1_W, U)
            
            Temp1_Recon_W = np.einsum('ijk,jr->ikr', T_recon, V)
            Den_W = np.einsum('ikr,ir->kr', Temp1_Recon_W, U)
            
            W = W * (Num_W + epsilon) / (Den_W + epsilon)
            
        # Round to half-integers
        U = np.round(U * 2) / 2.0
        V = np.round(V * 2) / 2.0
        W = np.round(W * 2) / 2.0
        
        return U, V, W

    # Search for minimal rank
    # Schoolbook rank is 80.
    # We try ranks from 40 to 80.
    # We use multiple seeds for each rank to escape local minima.
    
    # Fallback to schoolbook
    rank_school = n * m * p
    U_school = np.zeros((dim_u, rank_school))
    V_school = np.zeros((dim_v, rank_school))
    W_school = np.zeros((dim_w, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try ranks from 40 to 79
    # Limit time by reducing iterations or seeds if needed, but 120s is generous.
    # 40 ranks * 10 seeds * 200 iters * small dims = fast.
    
    for rank in range(40, 80):
        for seed in range(10):
            U, V, W = run_nmtf(rank, iterations=100, seed=seed)
            if check_exact(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # If no improvement found, return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-END
