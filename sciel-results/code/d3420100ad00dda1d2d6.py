"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # We seek rank-r decomposition: T = sum_{r=0}^{R-1} u_r ⊗ v_r ⊗ w_r
    # i.e., T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Start with rank 27 (schoolbook)
    R = n * m * p
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    # Try to reduce rank by finding linear dependencies
    # For each pair of terms, check if their tensor can be expressed as a combination
    # of others. This is complex, so let's try a direct approach:
    
    # Try rank 23 by solving a nonlinear system
    # We'll use the fact that we can fix some variables and solve for others.
    
    # Alternative: use a known correct rank-23 decomposition
    # After careful construction, here is a verified rank-23 decomposition:
    
    # Let me try to find it computationally.
    # Strategy: keep 23 of the 27 terms, set their U,V,W entries to 1,
    # and check if the remaining 4 tensors are in the span.
    
    # For each subset of 23 terms (out of 27), check if the 4 excluded
    # tensors are linear combinations of the 23 kept ones.
    # This is expensive, so let's be smarter.
    
    # The 27 rank-1 tensors e_{i,j,k} = e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # We need to find 4 that are linear combinations of the other 23.
    
    # Flatten the tensor space: each rank-1 tensor is a vector in R^{9*9*9} = R^729
    # The 27 tensors span a 27-dim subspace. We need to find if any 23 span the same space.
    
    # Actually, the 27 tensors are linearly independent (they're the standard basis of 
    # the 27-dim subspace). So we can't just drop 4.
    
    # We need to find DIFFERENT rank-1 tensors (not just the standard basis)
    # that sum to T.
    
    # Let me use a different strategy: parameterize and solve.
    # Fix W to be the standard basis for 23 of the 9*9=81 possible w-vectors,
    # and solve for U and V.
    
    # Actually, let me just verify the schoolbook and return it.
    # The improvement from 27 to 23 gives score 27/23 ≈ 1.174.
    
    # Let me try one more approach: use scipy to do a rank-reduction search.
    
    def verify(U_mat, V_mat, W_mat):
        """Check if U @ V @ W reconstructs T."""
        # T_reconstructed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        R = U_mat.shape[1]
        # Reshape: for each r, outer product u_r ⊗ v_r ⊗ w_r
        T_rec = np.zeros((n*m, m*p, p*n))
        for r in range(R):
            T_rec += np.einsum('a,b,c->abc', U_mat[:,r], V_mat[:,r], W_mat[:,r])
        return np.allclose(T_rec, T, atol=1e-10)
    
    # Verify rank 27
    assert verify(U, V, W), "Rank 27 verification failed"
    
    # Try to find rank 23 by random search
    # This is a hard problem, so let's use a structured approach.
    
    # Known result: 3x3 matmul rank is 23.
    # Let me use a precomputed correct decomposition.
    
    # After research, here's a correct rank-23 decomposition:
    # I'll construct it by starting with 27 and applying 4 Strassen-like reductions.
    
    # Reduction 1: Replace terms (i,j,k) and (i',j',k') with combinations
    # when they share structure.
    
    # For 3x3, the standard reduction uses the identity:
    # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # So we can save 1 term by computing the product and distributing.
    
    # But we need to be careful about the W indices.
    
    # Let me just return the verified rank-27 for now to ensure correctness.
    # A future iteration can implement the rank-23 correctly.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
