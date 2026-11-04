import time
import numpy as np

# Dimensions: n x m by m x p
n, m, p = 2, 3, 9

def construct():
    """
    Returns a tensor decomposition for multiplying a 2x3 matrix by a 3x9 matrix.
    Attempts to find a rank-minimal exact decomposition.
    Falls back to schoolbook rank 54 if no better exact decomposition is found.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    # Schoolbook decomposition (rank 54)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb), dtype=float)
    V_sb = np.zeros((m * p, rank_sb), dtype=float)
    W_sb = np.zeros((p * n, rank_sb), dtype=float)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    def to_lists(mat):
        return mat.tolist()

    # Try to find a better decomposition using random search + local optimization
    # The idea: try random rank-r decompositions and check if they exactly decompose the tensor.
    # For exact integer/half-integer solutions, we can use structured random initialization.

    def verify_decomposition(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        # The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Build the tensor from the decomposition
        r = U.shape[1]
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        
        # Compute T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
        
        # The true tensor for matrix multiplication has T[i,j,k] = 1 if this corresponds to a valid term
        # Actually, the matrix multiplication tensor is: C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T_{i,j,k} = 1 for all valid (i,j,k) indices, representing the bilinear form
        # Wait, let me reconsider. The tensor for matrix multiplication A (n x m) * B (m x p) -> C (n x p)
        # is a trilinear form: T[i, j, k, l] where C[i,l] = sum_{j,k} A[i,j] B[k,l] delta_{j,k}
        # But in the standard tensor decomposition form for matrix multiplication, we have:
        # C[i, k] = sum_r U[i, r] * V[r, k] where the "middle" index is contracted
        # 
        # Let me re-read the problem. The tensor has indices (i, j, k) where:
        # i in range(n), j in range(m), k in range(p)
        # The tensor entry T[i,j,k] = 1 (it's the identity tensor in the middle dimension)
        # Actually no. The matrix multiplication tensor T satisfies:
        # T[i, j, k] is the coefficient such that C[i,k] = sum_{j} A[i,j] * B[j,k]
        # So the tensor is T[i, j, k] = delta_{j, j'} where j' is the column of A and row of B
        # 
        # In the standard formulation: the tensor is T in R^{n x m x p} where T[i,j,k] = 1 if the
        # multiplication uses A[i,j] and B[j,k]. So T[i,j,k] = 1 for all i,j,k.
        # 
        # Wait, that's not right either. Let me think again.
        # 
        # Matrix multiplication: C = A * B where A is n x m, B is m x p, C is n x p
        # C[i, k] = sum_{j=0}^{m-1} A[i, j] * B[j, k]
        # 
        # The tensor T is in R^{(n*m) x (m*p) x (n*p)} but that's not what we have here.
        # 
        # Looking at the shapes: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # So the tensor is in R^{(n*m) x (m*p) x (p*n)}
        # The entry T[a, b, c] where a = i*m + j, b = j'*p + k, c = k'*n + i'
        # T[a, b, c] = 1 if i = i', j = j', k = k' (i.e., all indices match)
        # 
        # So T[(i*m+j), (j'*p+k), (k'*n+i')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
        # 
        # And the decomposition says: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        # Let's verify: T[a,b,c] should be 1 if a=i*m+j, b=j*p+k, c=k*n+i for consistent i,j,k
        # i.e., the index j in a must equal the index j' in b, and k in b must equal k' in c, and i in a must equal i' in c
        
        for a in range(n * m):
            i = a // m
            j = a % m
            for b in range(m * p):
                j2 = b // p
                k = b % p
                for c in range(p * n):
                    k2 = c // n
                    i2 = c % n
                    expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                    actual = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    # Try rank 27 using the Kronecker product structure
    # A is 2x3, B is 3x9. Think of B as three 3x3 blocks: B[:, 0:3], B[:, 3:6], B[:, 6:9]
    # C = A * B = [A*B1, A*B2, A*B3] where Bj is a 3x3 matrix
    # Each A*Bj is a 2x3 matrix product, which has rank 6 schoolbook
    # Total rank would be 3 * 6 = 18? No, that's not how tensor rank works.
    
    # Let me try a different approach: use the structure to get rank 27
    # The tensor is T[i,j,k] = delta_{i,i'}delta_{j,j'}delta_{k,k'} with the constraint that
    # the j index in A's column matches B's row, etc.
    
    # Actually, let's just try random rank-r decompositions for r < 54 and see if any work exactly.
    # For exact solutions with integer/half-integer entries, random search is unlikely to find them
    # unless we use structured approaches.
    
    # Let's try a structured approach for rank 27:
    # U has shape (6, 27), V has shape (27, 27), W has shape (18, 27)
    
    # One idea: exploit the fact that n=2 is small
    # For each pair of rows (i, i'), we can share some computation
    
    # Let me try to use the fact that the tensor is a "slice" structure
    # T[i, j, k] is nonzero only when the indices are consistent
    
    # Given the difficulty of finding a better exact decomposition, let's try a few structured ranks
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb

    # Try rank 27 with a structured pattern
    # Idea: for each k-block (3 values of k), do a 2x3 * 3x3 multiplication
    # But the tensor rank doesn't simply multiply like that
    
    # Let's try random initialization with rounding to half-integers
    for trial_rank in [27, 30, 36, 40, 45]:
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            r = trial_rank
            # Random initialization with small values
            U_try = np.random.randn(n * m, r) * 0.5
            V_try = np.random.randn(m * p, r) * 0.5
            W_try = np.random.randn(p * n, r) * 0.5
            
            # Round to half-integers
            U_try = np.round(U_try * 2) / 2
            V_try = np.round(V_try * 2) / 2
            W_try = np.round(W_try * 2) / 2
            
            if verify_decomposition(U_try, V_try, W_try):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = r
                found = True
                break
        
        if found:
            break

    # If no better rank found, return schoolbook
    return {
        "U": to_lists(best_U),
        "V": to_lists(best_V),
        "W": to_lists(best_W)
    }
