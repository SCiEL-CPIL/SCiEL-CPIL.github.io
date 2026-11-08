"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Finds a rank-minimal exact tensor decomposition for 2x5x6 matrix multiplication.
    Returns the schoolbook decomposition (rank 60) which is guaranteed to be valid.
    """
    import numpy as np

    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    r = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index for U: row corresponding to A[i,j]
                a_idx = i * m + j
                # Index for V: row corresponding to B[j,k]
                b_idx = j * p + k
                # Index for W: row corresponding to C[k,i] (trace convention W[k*n+i, r])
                # The task specifies output factor W[k*n+i, r].
                # In standard matrix mult C = A @ B, C[i,k] = sum_j A[i,j] B[j,k].
                # Here dimensions are A (n x m), B (m x p), C (n x p).
                # The tensor T is in R^{(n*m) x (m*p) x (p*n)}.
                # Mode 1: A indices (i, j) -> i*m + j
                # Mode 2: B indices (j, k) -> j*p + k
                # Mode 3: C indices (k, i) ? 
                # The prompt says: "The output factor follows the official trace convention W[k*n+i,r]."
                # This implies the third mode index is k*n + i.
                # This corresponds to C[k, i] if we consider C as a p x n matrix?
                # Or is it C[i, k] mapped to k*n + i?
                # Standard C is n x p. Index (i, k) is i*p + k.
                # The prompt explicitly says W[k*n+i, r]. This is a specific layout.
                # Let's trust the prompt's index definition for W.
                c_idx = k * n + i
                
                U[a_idx, col] = 1.0
                V[b_idx, col] = 1.0
                W[c_idx, col] = 1.0
                col += 1
                
    # Convert to list of lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
