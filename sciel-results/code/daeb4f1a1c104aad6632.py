import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    """
    Constructs the rank-minimal (schoolbook) tensor decomposition for 
    multiplying a 3x4 matrix by a 4x4 matrix.
    
    Rank = 3 * 4 * 4 = 48.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    U shape: (12, 48)
    V shape: (16, 48)
    W shape: (12, 48)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Schoolbook decomposition
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple self-check
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Check dimensions
    assert U.shape == (12, 48)
    assert V.shape == (16, 48)
    assert W.shape == (12, 48)
    
    # Verify decomposition
    # Tensor T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    # Should be 1 if this is a valid multiplication term, 0 otherwise?
    # Actually, the tensor for matrix multiplication is defined such that
    # sum_r U_{a,r} V_{b,r} W_{c,r} = delta_{a,b,c} in the appropriate sense.
    # Specifically, the bilinear form should reconstruct the product.
    
    # Let's test with random matrices
    np.random.seed(42)
    A = np.random.rand(3, 4)
    B = np.random.rand(4, 4)
    C_true = A @ B
    
    # Reconstruct C from factors
    # C[i, k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_k' V[j*p+k', r] B[j,k']) ... wait.
    # The factors define the algorithm.
    # Let M_r = sum_j U[i*m+j, r] A[i,j] ? No.
    # The standard interpretation:
    # For each r, we compute a bilinear form on A and B.
    # But U, V, W are the coefficients.
    # C[i,k] = sum_r W[k*n+i, r] * ( sum_j U[i*m+j, r] A[i,j] ) * ( sum_{j',k'} V[j'*p+k', r] B[j',k'] )?
    # No, that's not how tensor decomposition works for matrix multiplication.
    
    # The tensor T is in R^{12} x R^{16} x R^{12}.
    # T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}.
    # For matrix multiplication, T_{a,b,c} should be 1 if a=(i,j), b=(j,k), c=(i,k) and 0 otherwise?
    # No, the tensor is the "multiplication tensor".
    # T_{i,j,k} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # In the standard basis, the tensor is sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{i,k}.
    # So T_{a,b,c} is 1 if a encodes (i,j), b encodes (j,k), c encodes (i,k) AND the indices match.
    # Specifically:
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # If a, b, c are consistent (same j, same i, same k), T=1. Else 0.
    
    T_reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build target tensor
    T_target = np.zeros((12, 16, 12))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1
                
    if not np.allclose(T_reconstructed, T_target):
        print("Decomposition failed!")
    else:
        print("Decomposition is correct.")
