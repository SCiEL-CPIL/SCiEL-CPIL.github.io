"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 32
    pn = p * n  # 32
    
    # Build target tensor
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    
    def verify_decomposition(U_mat, V_mat, W_mat):
        """Verify exact decomposition."""
        r = U_mat.shape[1]
        T_recon = np.zeros((nm, mp, pn))
        for c in range(r):
            T_recon += np.outer(np.outer(U_mat[:, c], V_mat[:, c]), W_mat[:, c])
        return np.allclose(T_recon, T_target, atol=1e-10)
    
    # Initialize with schoolbook decomposition
    rank_init = n * m * p
    U = np.zeros((nm, rank_init))
    V = np.zeros((mp, rank_init))
    W = np.zeros((pn, rank_init))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = rank_init
    
    # IHT: Iteratively hard threshold the factor matrices
    # At each step, compute the tensor, identify the most important entries
    # in each factor, and zero out the least important ones
    
    for iteration in range(200):
        if current_rank <= 1:
            break
        
        # Compute the current tensor
        T_recon = np.zeros((nm, mp, pn))
        for c in range(current_rank):
            T_recon += np.outer(np.outer(current_U[:, c], current_V[:, c]), current_W[:, c])
        
        # Check if we're still valid
        if not np.allclose(T_recon, T_target, atol=1e-8):
            break
        
        # For each factor, compute row-wise norms to identify important entries
        # and try to zero out small entries to create sparsity, then merge columns
        
        # Compute the "importance" of each column based on its contribution norm
        col_norms = np.zeros(current_rank)
        for c in range(current_rank):
            contrib = np.outer(np.outer(current_U[:, c], current_V[:, c]), current_W[:, c])
            col_norms[c] = np.linalg.norm(contrib)
        
        # Sort columns by norm (ascending)
        sorted_cols = np.argsort(col_norms)
        
        # Try to remove the least important column
        removed = False
        for c_idx in sorted_cols:
            c = c_idx
            if c >= current_rank:
                continue
            
            # Check if removing this column still gives exact decomposition
            mask = [i for i in range(current_rank) if i != c]
            U_test = current_U[:, mask]
            V_test = current_V[:, mask]
            W_test = current_W[:, mask]
            
            if verify_decomposition(U_test, V_test, W_test):
                current_U = U_test
                current_V = V_test
                current_W = W_test
                current_rank -= 1
                removed = True
                break
        
        if not removed:
            break
    
    # Final check
    if not verify_decomposition(current_U, current_V, current_W):
        # Revert to schoolbook
        current_U = U
        current_V = V
        current_W = W
        current_rank = rank_init
    
    U_list = [[float(current_U[i, c]) for c in range(current_rank)] for i in range(nm)]
    V_list = [[float(current_V[i, c]) for c in range(current_rank)] for i in range(mp)]
    W_list = [[float(current_W[i, c]) for c in range(current_rank)] for i in range(pn)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

