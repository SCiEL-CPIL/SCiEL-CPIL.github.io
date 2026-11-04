"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Leave margin before 150s limit
    
    # Try to find a lower-rank decomposition using linear algebra
    # The tensor has entries T[i,j,k] = 1 if i*m+j and j*p+k are "connected"
    # Actually, the matrix multiplication tensor T[i,j,k,l] = delta_{j,k}
    # For our case: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[i,j,k] where output index is (i,k) = i*p+k... wait
    # W[k*n+i, r] so the output is indexed by (k, i) with row = k*n+i
    
    # Let's build the tensor and try to find its rank
    # Tensor T has shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    
    # Try different ranks and see if we can find a decomposition
    # Use the fact that we need integer/half-integer entries
    
    # First, let's try the known rank-50 decomposition as a baseline
    # Then try to reduce it
    
    # For now, let's try a computational approach:
    # Matricize T along the first mode: shape (n*m, m*p * p*n)
    T_mat = T.reshape(n*m, m*p * p*n)
    
    # The rank of the tensor is at least the rank of any matricization
    # Let's compute the rank of the first matricization
    try:
        rank_lower = np.linalg.matrix_rank(T_mat)
    except:
        rank_lower = 50
    
    # Try to find a decomposition with rank = rank_lower
    # This is a hard problem, so let's try a heuristic approach
    
    # Actually, for small problems, we can try to solve it directly
    # But the search space is huge. Let's try a different approach.
    
    # Use the known fact that for 2x4x7, we can try various tilings
    # Let's try to construct a decomposition with rank < 50
    
    # Try rank 49, 48, ... down to some minimum
    # For each rank, try to find U, V, W such that
    # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # This is a system of linear equations in the entries of U, V, W
    # But it's nonlinear (bilinear in U and V, etc.)
    
    # Let's try a different approach: use the existing rank-50 decomposition
    # and try to find linear dependencies to reduce the rank
    
    # Build the rank-50 decomposition
    rank_50 = 50
    U50 = np.zeros((n*m, rank_50))
    V50 = np.zeros((m*p, rank_50))
    W50 = np.zeros((p*n, rank_50))
    
    col = 0
    def add_strassen(i0, j0, k0):
        nonlocal col
        U50[i0*m + j0, col] += 1
        U50[(i0+1)*m + (j0+1), col] += 1
        V50[j0*p + k0, col] += 1
        V50[(j0+1)*p + (k0+1), col] += 1
        W50[k0*n + i0, col] += 1
        W50[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        U50[(i0+1)*m + j0, col] += 1
        U50[(i0+1)*m + (j0+1), col] += 1
        V50[j0*p + k0, col] += 1
        W50[k0*n + (i0+1), col] += 1
        W50[(k0+1)*n + (i0+1), col] -= 1
        col += 1
        
        U50[i0*m + j0, col] += 1
        V50[j0*p + (k0+1), col] += 1
        V50[(j0+1)*p + (k0+1), col] -= 1
        W50[(k0+1)*n + i0, col] += 1
        W50[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        U50[(i0+1)*m + (j0+1), col] += 1
        V50[(j0+1)*p + k0, col] += 1
        V50[j0*p + k0, col] -= 1
        W50[k0*n + i0, col] += 1
        W50[k0*n + (i0+1), col] += 1
        col += 1
        
        U50[i0*m + j0, col] += 1
        U50[i0*m + (j0+1), col] += 1
        V50[(j0+1)*p + (k0+1), col] += 1
        W50[k0*n + i0, col] -= 1
        W50[(k0+1)*n + i0, col] += 1
        col += 1
        
        U50[(i0+1)*m + j0, col] += 1
        U50[i0*m + j0, col] -= 1
        V50[j0*p + k0, col] += 1
        V50[j0*p + (k0+1), col] += 1
        W50[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        U50[i0*m + (j0+1), col] += 1
        U50[(i0+1)*m + (j0+1), col] -= 1
        V50[(j0+1)*p + k0, col] += 1
        V50[(j0+1)*p + (k0+1), col] += 1
        W50[k0*n + i0, col] += 1
        col += 1

    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    for i in range(2):
        for j in range(4):
            U50[i*m + j, col] = 1
            V50[j*p + 6, col] = 1
            W50[6*n + i, col] = 1
            col += 1
    
    # Verify the rank-50 decomposition
    T_recon = np.einsum('ar,br,cr->abc', U50, V50, W50)
    assert np.allclose(T_recon, T), "Rank-50 decomposition is incorrect"
    
    # Now try to find a lower-rank decomposition
    # Use a heuristic: try to find linear combinations of the rank-50 terms
    # that can be replaced by fewer terms
    
    # For now, return the rank-50 decomposition if we don't find better
    # Convert to list format
    U = U50.tolist()
    V = V50.tolist()
    W = W50.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
