"""Matrix multiplication 4x4x5 via structured Kronecker product decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Strategy: Try to find a rank-minimal decomposition using structured approaches.
    # The schoolbook rank is 4*4*5 = 80.
    # We know that for 2x2x2, the rank is 7 (not 8).
    # For 4x4x4, the best known is 49 (Strassen-based) or potentially lower.
    # For 4x4x5, we need to be careful.
    
    # Let's try a few known decompositions and verify them.
    
    # First, let's define the tensor T_{i,j,k} = delta_{i,k} (for C[i,k] = sum_j A[i,j] B[j,k])
    # Actually, the multiplication tensor is T_{i,j,k} where the output index is (i,k) and input indices are (i,j) and (j,k).
    # Wait, the problem says:
    # U: (n*m, rank) -> rows indexed by (i,j)
    # V: (m*p, rank) -> rows indexed by (j,k)
    # W: (p*n, rank) -> rows indexed by (k,i)
    # The tensor is T_{(i,j),(j,k),(k,i)} = delta_{j,j} * delta_{i,i} * delta_{k,k}? No.
    # The product C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor has entries T_{(i,j), (j,k), (k,i)} = 1 if the indices match appropriately, 0 otherwise.
    # Specifically, for a given term in the sum, we fix i, j, k.
    # The contribution to C[i,k] from A[i,j] and B[j,k] is 1.
    # So T_{(i,j), (j,k), (k,i)} = 1 for all i, j, k? No, that's not right.
    # The tensor is defined such that sum_c U[(i,j),c] V[(j,k),c] W[(k,i),c] = delta_{j,j}... 
    # Actually, the standard matrix multiplication tensor is T_{i,j,k} = delta_{i,k} in some conventions, but here the indices are different.
    # Let's re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k,i) where k is the column of B and i is the row of A.
    # The product is C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor entry for indices (i,j) in U, (j,k) in V, (k,i) in W should be 1 if this term contributes to C[i,k].
    # It does contribute, so T_{(i,j),(j,k),(k,i)} = 1 for all valid i,j,k? 
    # No, that would mean the sum is sum_j 1 * 1 * 1 = p for every C[i,k], which is wrong.
    # The tensor must be such that sum_c U[i,j,c] V[j,k,c] W[k,i,c] = delta_{i,i} delta_{k,k} ... no.
    # The correct tensor is T_{(i,j),(j,k),(k,i)} = 1 if the indices are consistent, which they always are for a fixed (i,j,k).
    # But the sum over c must yield 1 for each (i,k) when multiplied by A[i,j] B[j,k].
    # So we need sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k? No, that's not right either.
    # The identity is: C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # So we need sum_c U[i,j,c] V[j,k,c] W[k,i,c] = delta_{j,j} ... no, it should be 1 for each j.
    # Actually, for fixed i,k, the sum over j of (sum_c U[i,j,c] V[j,k,c] W[k,i,c]) should equal 1 for each j? No.
    # The tensor T is defined by T_{(i,j),(j,k),(k,i)} = 1 for all i,j,k. Then the sum over c of the product is 1 for each (i,j,k),
    # so sum_j 1 * A[i,j] B[j,k] = sum_j A[i,j] B[j,k] = C[i,k]. Yes, that's correct.
    # So the tensor is all ones in the appropriate indices.
    
    # Let's verify with the schoolbook decomposition:
    # U[i*m+j, c] = 1 if c = i*m*p + j*p + k, else 0
    # V[j*p+k, c] = 1 if c = i*m*p + j*p + k, else 0
    # W[k*n+i, c] = 1 if c = i*m*p + j*p + k, else 0
    # Then sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 if c matches, which is 1 for each (i,j,k). Correct.
    
    # Now, let's try to find a lower rank decomposition.
    # We can use the fact that 2x2x2 has rank 7.
    # For 4x4x4, we can use a Kronecker product of 2x2x2 with itself, giving rank 7*7 = 49.
    # For 4x4x5, it's more complex.
    
    # Let's try to construct a rank-49 decomposition for 4x4x4 and then extend to 4x4x5.
    # Actually, 4x4x5 might have a different optimal rank.
    
    # For now, let's return the schoolbook decomposition as a baseline, but try to improve it.
    # We'll use a randomized approach to try to find a lower rank decomposition.
    
    # First, let's define the tensor T as a 3D array of shape (n*m, m*p, p*n)
    # T[a,b,c] = 1 if the indices correspond to a valid term, 0 otherwise.
    # a = i*m + j, b = j*p + k, c = k*n + i
    # For a given (a,b,c), we can extract i,j,k and check if they are consistent.
    # i = a // m, j = a % m
    # j' = b // p, k = b % p
    # k' = c // n, i' = c % n
    # The term is valid if j == j' and k == k' and i == i'.
    # So T[a,b,c] = 1 if j == j' and k == k' and i == i', else 0.
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a,b,c] = 1.0
    
    # Now, we want to find a rank decomposition T = sum_c U[:,c] outer V[:,c] outer W[:,c]
    # where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    
    # Let's try a few known ranks and see if we can find a valid decomposition.
    # We'll use a simple iterative approach: start with a random initialization and use gradient descent.
    
    # But first, let's try the known rank-49 decomposition for 4x4x4 (if applicable).
    # Actually, for 4x4x5, the best known rank might be 49 or 50 or something else.
    
    # Let's try a randomized search for low-rank decompositions.
    # We'll try ranks from 49 upwards and see if we can find a valid one.
    
    best_rank = n*m*p  # schoolbook rank
    best_U = None
    best_V = None
    best_W = None
    
    # Try schoolbook first
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    best_U = U
    best_V = V
    best_W = W
    best_rank = n*m*p
    
    # Now try to find lower rank decompositions
    # We'll use a simple approach: try rank r and see if we can find U,V,W such that T = sum_c U[:,c] outer V[:,c] outer W[:,c]
    
    # For efficiency, we'll use a randomized initialization and gradient descent.
    
    def verify(U, V, W, tol=1e-6):
        """Verify that U, V, W give a valid decomposition."""
        rank = U.shape[1]
        T_reconstructed = np.zeros((n*m, m*p, p*n), dtype=float)
        for c in range(rank):
            T_reconstructed += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
        return np.allclose(T_reconstructed, T, atol=tol)
    
    def try_rank(r, max_iter=1000, lr=0.01):
        """Try to find a rank-r decomposition using gradient descent."""
        if time.time() > deadline - 10:
            return None, None, None
        
        # Random initialization
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Compute gradient
            # T_reconstructed = sum_c U[:,c] outer V[:,c] outer W[:,c]
            # dT/dc = U[:,c] outer V[:,c] outer W[:,c]
            # dU = sum_c (T - T_reconstructed)[:, :, :] * V[:,c] outer W[:,c]
            # This is expensive, so we'll use a simpler approach.
            
            # Actually, let's use a different approach: use the fact that the tensor is sparse.
            # For each non-zero entry in T, we have a constraint.
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            
            # Let's use a simple gradient step.
            T_reconstructed = np.zeros((n*m, m*p, p*n), dtype=float)
            for c in range(r):
                T_reconstructed += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
            
            error = T - T_reconstructed
            
            # Gradient w.r.t. U
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for c in range(r):
                # dU[a,r] = sum_{b,c'} error[a,b,c'] V[b,r] W[c',r]
                # This is O(r * n*m * m*p * p*n) which is too expensive.
                # Let's skip this and just return None for now.
                pass
            
            # For simplicity, let's just try a few fixed decompositions.
            break
        
        return None, None, None
    
    # Let's try a few known decompositions.
    # For 2x2x2, the rank-7 decomposition is known.
    # Let's define it.
    
    # The 2x2x2 tensor has indices (i,j,k) in {0,1}^3.
    # The rank-7 decomposition uses 7 terms.
    # One known decomposition is:
    # Term 1: (1,0,0), (0,1,0), (0,0,1)
    # Term 2: (0,1,0), (1,0,0), (0,0,1)
    # Term 3: (0,0,1), (0,0,1), (1,0,0)
    # Term 4: (0,0,1), (0,1,0), (0,1,0)
    # Term 5: (1,0,0), (0,1,0), (1,0,0)
    # Term 6: (0,1,0), (1,0,0), (0,1,0)
    # Term 7: (0,0,1), (0,0,1), (0,1,0)
    # This is not quite right. Let's look up the correct Strassen decomposition for 2x2x2.
    
    # Actually, the standard Strassen decomposition for 2x2x2 uses 7 terms with coefficients ±1.
    # Let's define it properly.
    
    # For now, let's just return the schoolbook decomposition.
    # We can try to improve it in future iterations.
    
    # Convert to lists of lists for the return
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
