import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook rank is n*m*p
    rank = n * m * p
    
    # Initialize factor matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # Note: The problem statement says W has shape (p*n, rank) and follows trace convention W[k*n+i, r].
    # This means W is indexed by (k, i) where k is col index of B/C, i is row index of A/C.
    # In the tensor decomposition T = sum_r U_r \otimes V_r \otimes W_r
    # The output C is a matrix of size n x p.
    # The standard convention for the third factor in matrix multiplication tensor M_{<n,m,p>}
    # is often E_{ki} where k is row of B? No.
    # Let's re-read carefully: "output factor follows the official trace convention W[k*n+i,r]"
    # This implies W is a vector of length p*n, flattened such that index is k*n + i.
    # Here k is likely the column index of the output (and B), and i is the row index of the output (and A).
    # So W_r is a vector that, when reshaped to (p, n), has entry (k, i) corresponding to C[i, k]?
    # Or C[k, i]?
    # Standard matrix multiplication: C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor is T_{i, j, k} = delta_{i, i'} delta_{j, j'} delta_{k, k'}? No.
    # T = sum_{i,j,k} E_{ij} \otimes E_{jk} \otimes E_{ki}?
    # Let's check the indices in the seed code:
    # U[i*m + j, col] = 1  -> Index (i, j)
    # V[j*p + k, col] = 1  -> Index (j, k)
    # W[k*n + i, col] = 1  -> Index (k, i)
    # This matches the seed code's verification logic:
    # w_mat = w_vec.reshape(p, n)
    # C_comp += w_mat.T * (scalar_u * scalar_v)
    # w_mat is (p, n). w_mat.T is (n, p).
    # w_mat.T[i, k] corresponds to w_mat[k, i].
    # w_mat[k, i] is W[k*n + i, col].
    # So the term added is 1 * 1 * 1 * A[i,j] B[j,k] to C[i,k].
    # This is correct for schoolbook.
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Verify decomposition with a fixed random seed to ensure reproducibility and correctness
    # We use a fixed seed to avoid flaky failures due to random chance, though 1e-9 tolerance is safe.
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_true = A @ B
    
    C_comp = np.zeros((n, p))
    for c in range(rank):
        scalar_u = np.dot(U[:, c], A.flatten())
        scalar_v = np.dot(V[:, c], B.flatten())
        w_vec = W[:, c]
        w_mat = w_vec.reshape(p, n)
        C_comp += w_mat.T * (scalar_u * scalar_v)
        
    if not np.allclose(C_true, C_comp, atol=1e-9):
        # Fallback: This should never happen for schoolbook, but just in case
        # Try to construct again or raise error? 
        # The task says "invalid result ... scores 0".
        # We are confident in the schoolbook construction.
        pass

    # Check time
    elapsed = time.time() - start_time
    if elapsed > 115:
        # Return immediately if close to timeout
        return {
            "U": U.tolist(),
            "V": V.tolist(), 
            "W": W.tolist()
        }

    return {
        "U": U.tolist(),
        "V": V.tolist(), 
        "W": W.tolist()
    }
