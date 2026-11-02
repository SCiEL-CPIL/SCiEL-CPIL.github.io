"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    deadline = time.time() + 120
    
    # Build the MM tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    U_shape = n * m  # 16
    V_shape = m * p  # 28
    W_shape = p * n  # 28
    
    T = np.zeros((U_shape, V_shape, W_shape), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Schoolbook decomposition
    rank_sb = n * m * p  # 112
    U_sb = np.zeros((U_shape, rank_sb), dtype=np.float64)
    V_sb = np.zeros((V_shape, rank_sb), dtype=np.float64)
    W_sb = np.zeros((W_shape, rank_sb), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Try to find a lower rank decomposition using random search
    # We'll try ranks from 100 down to 80
    for target_rank in range(100, 79, -1):
        if time.time() > deadline - 30:
            break
        
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            
            # Initialize with random sparse vectors
            U = np.zeros((U_shape, target_rank), dtype=np.float64)
            V = np.zeros((V_shape, target_rank), dtype=np.float64)
            W = np.zeros((W_shape, target_rank), dtype=np.float64)
            
            # Each column has a few non-zeros
            for c in range(target_rank):
                # Choose a small number of non-zeros
                num_nonzero_U = np.random.randint(1, 5)
                idx_U = np.random.choice(U_shape, num_nonzero_U, replace=False)
                U[idx_U, c] = np.random.choice([-1, 1], num_nonzero_U)
                
                num_nonzero_V = np.random.randint(1, 5)
                idx_V = np.random.choice(V_shape, num_nonzero_V, replace=False)
                V[idx_V, c] = np.random.choice([-1, 1], num_nonzero_V)
                
                num_nonzero_W = np.random.randint(1, 5)
                idx_W = np.random.choice(W_shape, num_nonzero_W, replace=False)
                W[idx_W, c] = np.random.choice([-1, 1], num_nonzero_W)
            
            # Check if this decomposition is valid
            # Reconstruct T from U, V, W
            T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
            error = np.abs(T_recon - T).max()
            
            if error < 1e-10:
                # Found a valid decomposition!
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
        
        if best_rank < rank_sb:
            break
    
    # Convert to lists with integer/half-integer entries
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
