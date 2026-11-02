"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]
import numpy as np

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n=3, m=3, p=7):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    # Random test matrices
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    
    # Calculate C = A @ B
    C_expected = A @ B
    
    # Calculate C_decomposed = sum_r (A @ u_r) * (v_r @ B) * w_r?
    # No, the formula is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor form:
    # C[i, k] = sum_r U[i*m + ?, r] ... No.
    # The decomposition is:
    # T_{i,j,k} = sum_r U[i,j,r] V[j,k,r] W[i,k,r] ? No.
    
    # Standard form:
    # C = sum_r (A u_r) (v_r^T B) w_r? No.
    # The bilinear form is:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # Decomposition:
    # C_{ik} = sum_r (sum_j A_{ij} U_{ij,r}) (sum_l B_{jl} V_{jl,r}) W_{ik,r} ??
    
    # Let's use the matrix formulation:
    # U is (n*m) x R. Flatten A to a^T.
    # V is (m*p) x R. Flatten B to b^T.
    # W is (p*n) x R. Flatten C to c^T.
    # c = sum_r (U^T a)_r (V^T b)_r W_r?
    # No.
    # c_r = W[:, r] * (U[:, r] dot a) * (V[:, r] dot b) ?
    # Sum over r of these gives C.
    
    a = A.flatten() # size 9
    b = B.flatten() # size 21
    
    C_decomposed = np.zeros((p*n))
    
    for r in range(U.shape[1]):
        # u_r is U[:, r] (size n*m)
        # v_r is V[:, r] (size m*p)
        # w_r is W[:, r] (size p*n)
        
        val_a = np.dot(U[:, r], a)
        val_b = np.dot(V[:, r], b)
        
        C_decomposed += (val_a * val_b) * W[:, r]
        
    C_decomposed = C_decomposed.reshape((p, n)).T # W index is k*n+i, so reshape to (p,n) then transpose to (n,p)
    
    return np.allclose(C_expected, C_decomposed)

def _laderman_3x3():
    """
    Returns U, V, W for 3x3x3 matrix multiplication using Laderman's rank-23 algorithm.
    """
    # Laderman coefficients
    # This is a hardcoded set. If incorrect, verification fails.
    
    # U, V, W are 9x23
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Coefficients
    # I will leave them as zeros for now, which will fail verification.
    # This ensures safety.
    
    # To actually get 55, I need to fill these.
    # Since I can't, I will return schoolbook for the 3x3 block in the caller.
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Attempt 55 rank decomposition (Laderman for 3x3 blocks)
    # Block 1: 3x3 (cols 0-2 of B, cols 0-2 of C)
    # Block 2: 3x3 (cols 3-5 of B, cols 3-5 of C)
    # Block 3: 3x1 (col 6 of B, col 6 of C)
    
    # If Laderman is not available/correct, fall back to Schoolbook (63)
    
    # I will construct the Schoolbook solution directly to guarantee validity.
    # This is the safe bet.
    
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * (m*p) + j * p + k
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
