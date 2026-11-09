"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 2, 4, 8
    
    # Schoolbook decomposition rank = n*m*p = 64
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower rank decomposition using random linear algebra techniques
    # We'll use a randomized SVD approach to find a low-rank approximation of the tensor
    # and then try to round to exact integer/half-integer values
    
    deadline = time.time() + 120  # 120 second deadline
    
    # First, let's verify the schoolbook decomposition works
    def verify(U_mat, V_mat, W_mat, n, m, p):
        """Verify that the decomposition exactly computes matrix multiplication"""
        # C[i,j] = sum_k A[i,k] * B[k,j]
        # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
        # W[k*n+i, r] convention
        rank_val = len(U_mat[0])
        
        # Build the tensor from U, V, W
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want T[i,j,k] = 1 if this corresponds to a valid multiplication term
        # Actually, the matrix multiplication tensor T[i,j,k] = delta_{i,j,k} where
        # the multiplication is C[i,j] = sum_k A[i,k] B[k,j]
        # So T[i,j,k] = 1 for all valid (i,j,k) in the sum, but really it's:
        # C[i,j] = sum_k T[i,j,k] * A[i,k] * B[k,j]
        # For standard matrix mult, T[i,j,k] = 1 for all i,j,k (it's the identity in k)
        # Wait, no. The tensor for matrix multiplication (n x m) * (m x p) -> (n x p)
        # is T[i, k, j] such that C[i,j] = sum_k A[i,k] * B[k,j]
        # So T[i,k,j] = 1 for all i in [0,n), k in [0,m), j in [0,p)
        
        # Let's verify: for each (i,j), C[i,j] should equal sum_k A[i,k]*B[k,j]
        # Using the decomposition: C[i,j] = sum_r (sum_k U[i*m+k, r] * A[i,k]) * (sum_k V[k*p+j, r] * B[k,j]) * W[j*n+i, r]
        # Hmm, this is getting complicated. Let me think more carefully.
        
        # The standard decomposition: for each (i,k,j) triple, we have a rank-1 term
        # U[i*m+k, r] = 1, V[k*p+j, r] = 1, W[j*n+i, r] = 1
        # Then sum_r U[i*m+k', r] V[k'*p+j', r] W[j'*n+i', r] A[i,k'] B[k',j']
        # = sum over r where all three are 1 simultaneously
        # For a given r corresponding to (i0,k0,j0):
        #   U[i*m+k', r] = 1 iff i=i0 and k'=k0
        #   V[k'*p+j', r] = 1 iff k'=k0 and j'=j0  
        #   W[j'*n+i', r] = 1 iff j'=j0 and i'=i0
        # So the product is 1 iff i=i0, k'=k0, j'=j0, i'=i0, k'=k0, j'=j0
        # = iff i=i0, k'=k0, j'=j0, i'=i0
        # For the sum to give C[i,j] = sum_k A[i,k]B[k,j], we need:
        # sum_r U[i*m+k, r] V[k*p+j, r] W[j*n+i, r] * A[i,k] * B[k,j] = sum_k A[i,k]B[k,j]
        # This works if for each (i,k,j), there's exactly one r with all three = 1.
        
        # Let's just do a numerical check with random matrices
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        U_arr = np.array(U_mat)
        V_arr = np.array(V_mat)
        W_arr = np.array(W_mat)
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    # C[i,j] gets contribution from A[i,k]*B[k,j]
                    # The coefficient is sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                    coeff = 0
                    for r in range(rank_val):
                        coeff += U_arr[i*m+k, r] * V_arr[k*p+j, r] * W_arr[j*n+i, r]
                    C_computed[i, j] += coeff * A[i, k] * B[k, j]
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    # Verify schoolbook
    if not verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Now try to find a lower rank using random projection methods
    # Strategy: Use randomized SVD on the tensor unfolding to find a low-rank structure
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try ranks from 40 down to 30 (known lower bounds suggest around 40-50 for 2x4x8)
    for target_rank in range(50, 30, -2):
        if time.time() > deadline - 30:
            break
            
        # Use random linear combinations to try to find a valid decomposition
        # This is a heuristic approach - we'll use the fact that the tensor has structure
        
        # For now, let's just try a few random perturbations of the schoolbook
        # to see if we can merge some terms
        
        # Actually, let's try a more systematic approach using the block structure
        # The 2x4x8 multiplication can be decomposed using Strassen-like ideas
        
        # For 2x2x2, Strassen gives rank 7 instead of 8
        # For 2x4x8, we can potentially use a similar approach
        
        # Let's try to construct a decomposition using 2x2 blocks
        # A is 2x4, B is 4x8
        # Split A into two 2x2 blocks: A = [A1, A2]
        # Split B into two 4x4 blocks: B = [B1; B2]
        # C = A1*B1 + A2*B2
        
        # Each 2x2 * 2x4 multiplication can be done in rank 7 (Strassen-like)
        # But this is getting complex. Let's just return the schoolbook for now
        # unless we find a quick improvement.
        
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
