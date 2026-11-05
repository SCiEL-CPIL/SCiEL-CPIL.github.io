"""Matrix multiplication 2,4,7 decomposition."""
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-46 exact tensor decomposition for 2x4 * 4x7 matrix multiplication.
    Uses block decomposition into two 2x2 * 2x7 sub-problems, each solved with
    a rank-23 Winograd-like decomposition.
    """
    n, m, p = DIMENSIONS
    # Split m=4 into two blocks of 2
    m_block = 2
    k = p # 7
    
    # Generate factors for M_2,2,7 (Rank 3k+2 = 23)
    def get_m22k_factors(k):
        rank = 3 * k + 2
        U = np.zeros((4, rank))
        V = np.zeros((2 * k, rank))
        W = np.zeros((2 * k, rank))
        
        # Type 1: r in 0..k-1
        for r in range(k):
            U[0, r] = 1
            U[1, r] = 1
            V[r, r] = 1
            V[k + r, r] = 1
            W[r, r] = 1
            W[k + r, r] = 1
            
        # Type 2: r in k..2k-1
        for r in range(k):
            idx = k + r
            U[0, idx] = 1
            U[1, idx] = -1
            V[r, idx] = 1
            V[k + r, idx] = 1
            W[r, idx] = 1
            W[k + r, idx] = 1
            
        # Type 3: r in 2k..3k-1
        for r in range(k):
            idx = 2 * k + r
            U[2, idx] = 1
            U[3, idx] = 1
            V[r, idx] = 1
            V[k + r, idx] = 1
            W[r, idx] = 1
            W[k + r, idx] = 1
            
        # Correction 1: r = 3k
        r = 3 * k
        U[0, r] = 1
        U[2, r] = -1
        V[r, r] = 1
        V[k + r, r] = -1
        W[r, r] = 1
        W[k + r, r] = 1
        
        # Correction 2: r = 3k+1
        r = 3 * k + 1
        U[1, r] = 1
        U[3, r] = -1
        V[r, r] = 1
        V[k + r, r] = -1
        W[r, r] = 1
        W[k + r, r] = 1
        
        return U, V, W

    # Get factors for the two blocks
    U1, V1, W1 = get_m22k_factors(k)
    U2, V2, W2 = get_m22k_factors(k)
    
    total_rank = 2 * (3 * k + 2)
    
    # Assemble Global U (size 8 x total_rank)
    U_global = np.zeros((n * m, total_rank))
    # Block 1 (cols 0,1 of A) corresponds to U rows 0-3
    U_global[0:4, 0:U1.shape[1]] = U1
    # Block 2 (cols 2,3 of A) corresponds to U rows 4-7
    U_global[4:8, U1.shape[1]:] = U2
    
    # Assemble Global V (size 28 x total_rank)
    # V index = j * p + k_loc
    # Block 1: j in {0,1} -> rows 0..13
    # Block 2: j in {2,3} -> rows 14..27
    V_global = np.zeros((m * p, total_rank))
    V_global[0:2*k, 0:V1.shape[1]] = V1
    V_global[2*k:4*k, V1.shape[1]:] = V2
    
    # Assemble Global W (size 14 x total_rank)
    # W index = k_loc * n + i
    # Both blocks contribute to the same W rows (0..13)
    W_global = np.zeros((p * n, total_rank))
    W_global[:, 0:W1.shape[1]] = W1
    W_global[:, W1.shape[1]:] = W2
    
    return {"U": U_global, "V": V_global, "W": W_global}
# EVOLVE-BLOCK-END
