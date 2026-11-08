import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    
    Approach: Represent as direct sum of three 1x4x5 sub-tensors, then apply
    global linear algebraic compression via Khatri-Rao product analysis.
    """
    n, m, p = DIMENSIONS
    
    # Build schoolbook decomposition
    U_parts = []
    V_parts = []
    W_parts = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m)
                u[i * m + j] = 1
                U_parts.append(u)
                
                v = np.zeros(m * p)
                v[j * p + k] = 1
                V_parts.append(v)
                
                w = np.zeros(p * n)
                w[k * n + i] = 1
                W_parts.append(w)
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    # Attempt compression via Khatri-Rao product analysis
    # Form the Khatri-Rao product matrix K where each column is u_t ⊗ v_t
    # for each schoolbook term t. If rank(K) < number of terms, we can compress.
    
    num_terms = n * m * p
    K = np.zeros((n * m * m * p, num_terms))
    
    for t in range(num_terms):
        # Extract the (i,j,k) for this term
        k_idx = t % p
        j_idx = (t // p) % m
        i_idx = (t // (p * m)) % n
        
        u_t = U[:, t]
        v_t = V[:, t]
        
        # Khatri-Rao: column is u_t ⊗ v_t
        K[:, t] = np.kron(u_t, v_t)
    
    # Compute rank of K
    rank_K = np.linalg.matrix_rank(K, tol=1e-10)
    
    if rank_K < num_terms:
        # Find a basis for the column space
        # Use QR decomposition to find independent columns
        Q, R = np.linalg.qr(K)
        # The number of independent columns is rank_K
        # We can use the first rank_K columns of Q as basis
        # But we need to reconstruct the W vectors
        
        # For each basis vector q_r, find coefficients such that
        # q_r = sum_t alpha_{r,t} * (u_t ⊗ v_t)
        # Then the compressed term r has:
        # U_r = sum_t alpha_{r,t} * u_t
        # V_r = sum_t alpha_{r,t} * v_t  
        # W_r = sum_t alpha_{r,t} * w_t
        # This is not quite right - the Khatri-Rao approach needs more care.
        
        # For now, just return the schoolbook if compression doesn't work cleanly
        pass
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
