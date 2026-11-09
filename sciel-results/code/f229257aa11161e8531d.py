"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p, tol=1e-6):
    # Check that sum_r U[:,r] V[:,r] W[:,r] (outer) equals the multiplication tensor
    # Tensor T[i,j,k] = 1 if we are computing C[i,k] from A[i,j]*B[j,k]
    # Actually the tensor for matmul: T[i,j,k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # T[i,j,k] = 1 for all i,j,k? No.
    # C[i,k] = sum_j A[i,j] B[j,k]. So T[i,j,k] = 1 for all valid i,j,k.
    # The decomposition is: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # where U[i,j,r], V[j,k,r], W[k,i,r] (note W index is k*n+i per contract)
    
    # Reconstruct tensor from factors
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # Index mapping:
    # U_idx = i*m + j
    # V_idx = j*p + k
    # W_idx = k*n + i
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # We can check this efficiently using einsum or loops.
    # T is of shape (n, m, p).
    T_recon = np.zeros((n, m, p))
    # This loop is O(n*m*p*rank). For n=2, m=4, p=8, nmp=64. Rank up to 64.
    # 64*64 = 4096 ops. Very fast.
    for r in range(U.shape[1]):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n) # Note: W is indexed by k*n+i, so reshape to (p, n)
        # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        # This is a sum over r of u[i,j] v[j,k] w[k,i]
        # We can do this with einsum or explicit loops.
        for i in range(n):
            for j in range(m):
                ui_j = u[i, j]
                if ui_j == 0: continue
                for k in range(p):
                    T_recon[i, j, k] += ui_j * v[j, k] * w[k, i]
    
    T_true = np.ones((n, m, p))
    return np.allclose(T_recon, T_true, atol=tol)


def construct():
    start_time = time.time()
    deadline = start_time + 120.0 # 120s limit, keep margin
    
    n, m, p = DIMENSIONS
    best_result = _schoolbook()
    
    # Target: Find rank < n*m*p
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # This is the "all-ones" tensor of shape (n, m, p).
    # The rank of the all-ones tensor is 1?
    # Let's check. T[i,j,k] = 1.
    # Can we write 1 = sum_r u_i v_j w_k?
    # If we pick u_i=1, v_j=1, w_k=1, then sum_r 1*1*1 = 1.
    # So rank 1 is enough?
    # Let's verify the index conventions.
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # If rank=1:
    # U[i*m+j, 0] = 1
    # V[j*p+k, 0] = 1
    # W[k*n+i, 0] = 1
    # Then T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    # T[i,j,k] = 1 * 1 * 1 = 1.
    # This is exactly the tensor for matrix multiplication?
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T is defined such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k].
    # So T[i,j,k] must be 1 for all i,j,k.
    # Yes! The tensor for matrix multiplication is the all-ones tensor.
    # Therefore, the rank is 1.
    
    # Wait, is it that simple?
    # Let's re-read the contract carefully.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 8 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # "Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries that exactly decompose the matrix multiplication tensor"
    
    # If the tensor is the all-ones tensor, then rank 1 works.
    # Let's double check if there are any other constraints.
    # "integer/half-integer entries" - 1 is an integer.
    # "exactly decompose" - 1*1*1 = 1.
    
    # Why did the schoolbook use rank 64? Because it's the standard naive decomposition.
    # Why did Strassen use rank 56? Because it's a clever decomposition for 2x2x2 blocks, but for the *entire* tensor, if it's all-ones, rank 1 is better.
    
    # Is the tensor really all-ones?
    # Matrix multiplication: C = AB.
    # C_{ik} = \sum_j A_{ij} B_{jk}.
    # The "matrix multiplication tensor" T is defined by T_{ijk} = 1 if the term A_{ij}B_{jk} contributes to C_{ik}.
    # Since every A_{ij} and B_{jk} pair contributes to C_{ik} (for fixed i,k), T_{ijk} = 1 for all i,j,k.
    # Yes, the tensor is the all-ones tensor.
    
    # So the minimal rank is 1.
    
    # Let's construct the rank-1 solution.
    rank = 1
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    
    # Verify to be absolutely sure
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails (should not happen)
    return best_result
# EVOLVE-BLOCK-END
