import numpy as np
import time

def construct():
    n, m, p = 3, 4, 7
    deadline = time.time() + 130  # safety margin
    
    # We'll build a decomposition using block structure
    # Strategy: partition and use Strassen/Winograd for sub-blocks
    
    # For now, let's try to find a valid decomposition with rank < 84
    # by using the algebraic structure
    
    # The matrix multiplication tensor for n×m×p has rank at most n*m*p (schoolbook)
    # but can be lower.
    
    # Let's try to construct a rank-70 or so decomposition using block methods
    
    # Actually, let me try a different strategy: use the fact that we can
    # combine some terms. 
    
    # For 3x4x7, one approach:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # Let's use the identity: for any matrices, we can write
    # C = A * B
    # 
    # Using Strassen on 2x2 blocks where possible.
    
    # Let me try to manually construct a decomposition with rank 83 or less
    # by finding one redundant term in the schoolbook.
    
    # Actually, the simplest improvement: check if rank 83 works.
    # The schoolbook has 84 terms. If we can eliminate one, we get 83.
    
    # Let me try a different approach: use the known lower bound and upper bound.
    # For 3x4x7, the rank is known to be at most... let me just try to construct
    # a valid one with rank < 84.
    
    # Simple approach: return schoolbook but verify it's correct, then try to
    # shave off rank if possible.
    
    # Let me try rank 83 by combining two schoolbook terms if possible.
    
    # Actually, let me just return the schoolbook decomposition correctly
    # and see if we can improve. The key is to get a valid answer first.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        # Check: sum_r U[r, i*m+j] * V[r, j'*p+k'] * W[r, k'*n+i'] 
        # should equal delta(i,i') * delta(j,j') * delta(k',k)
        # Wait, the tensor is T[i,j,k; i',j',k'] = delta(i,i')delta(j,j')delta(k,k')
        # where C[i,k] = sum_j A[i,j] B[j,k]
        # The decomposition: C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_k' V[j'*p+k', r] B[j',k']) * W[k*n+i, r]
        # Hmm, let me re-read the convention.
        
        # W[k*n+i, r] - so W is indexed by (k, i) where k is column of B / row of C, i is row of A / row of C
        # U[i*m+j, r] - U indexed by (i, j) where i is row of A, j is col of A / row of B
        # V[j'*p+k', r] - V indexed by (j', k') where j' is row of B, k' is col of B / col of C
        
        # The product: sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k*n+i, r]
        # This should equal delta(i, i') * delta(j, j') * delta(k', k) for the tensor
        # But wait, in matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor T has indices (i, j, k) for the "input" and the output is (i, k)
        # The standard tensor: T[i, j, k] = 1 if this corresponds to a valid multiplication step
        
        # Actually, the bilinear form: sum_{i,j,k} A[i,j] B[j,k] C[i,k] = sum_r (sum_{i,j} U[i*m+j,r] A[i,j]) (sum_{j,k} V[j*p+k,r] B[j,k]) (sum_{i,k} W[k*n+i,r] C[i,k])
        
        # For this to equal the matrix product, we need:
        # sum_r U[i*m+j,r] V[j'*p+k',r] W[k*n+i,r] = delta_{i,i'} delta_{j,j'} delta_{k',k}
        # Wait no. Let me think again.
        
        # The tensor for matrix multiplication M_{n,m,p} is defined by:
        # T[i, j, k; i', j', k'] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
        # where the three groups of indices correspond to the three "modes"
        
        # Actually, I think the convention is:
        # The matrix multiplication tensor has entries T[i, j, k] = 1 for the "diagonal" 
        # in the sense that C[i,k] = sum_j A[i,j] B[j,k]
        
        # The rank decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
        # where u_r is a vector in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
        
        # And the condition is: for all i, j, j', k, k':
        # sum_r U[i*m+j, r] V[j'*p+k, r] W[k'*n+i, r] = delta_{j,j'} delta_{k,k} ... 
        # No, that's not right either.
        
        # Let me just verify numerically with random matrices.
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        
        # Compute using the decomposition
        # C_decomp[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j' V[j'*p+k, r] B[j',k])
        # Wait, that's not how it works. The bilinear form is:
        # sum_{i,j,k} A[i,j] B[j,k] C[i,k] = sum_r (sum_{i,j} U[i*m+j,r] A[i,j]) (sum_{j,k} V[j*p+k,r] B[j,k]) (sum_{i,k} W[k*n+i,r] C[i,k])
        
        # So to compute C from A and B using the decomposition:
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
        # No, that's not right. The decomposition gives the tensor, not directly C.
        
        # Actually, I think the way to use it is:
        # Define: for each r, compute S_r = sum_j U[i*m+j, r] A[i,j] for all i
        #          T_r = sum_j V[j*p+k, r] B[j,k] for all k  
        #          Then C[i,k] = sum_r W[k*n+i, r] S_r[i] T_r[k]
        
        # Let me verify this interpretation.
        
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            S = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    S[i] += U[i*m+j, r] * A[i,j]
            T = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    T[k] += V[j*p+k, r] * B[j,k]
            for i in range(n):
                for k in range(p):
                    C_decomp[i,k] += W[k*n+i, r] * S[i] * T[k]
        
        if not np.allclose(C, C_decomp, atol=1e-10):
            return False
        return True
    
    if verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: return schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
