    def verify(U, V, W, n, m, p):
        # T[i, j, k] = sum_l U[i, l] V[j, l] W[k, l]
        # Should match 1 if i,j,k form a valid matmul term.
        # A is n x m, B is m x p, C is n x p.
        # C[i, k] = sum_j A[i, j] B[j, k].
        # Tensor T_{i, k, j} = 1?
        # Indices: U is (n*m, rank). Index i*m+j.
        # V is (m*p, rank). Index j*p+k.
        # W is (p*n, rank). Index k*n+i.
        
        # Check a few random points.
        # Or check all points (27*3*7 = 567 points, very fast).
        import numpy as np
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # T = sum_l U[:, l] V[:, l] W[:, l]
        # T is (n*m, m*p, p*n)? No.
        # T[i, j, k] where i is A index, j is B index, k is C index?
        # U index: a = i*m + j
        # V index: b = j*p + k
        # W index: c = k*n + i
        
        # The tensor is sum_l U[a, l] V[b, l] W[c, l].
        # This should be 1 if b = j*p+k and a = i*m+j and c = k*n+i.
        # i.e., if the indices match the matmul definition.
        
        # We can compute the full tensor and compare to the expected tensor.
        # Expected tensor E:
        # E[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
        
        # Compute T = U @ diag(V, W)? No.
        # T[a, b, c] = sum_l U[a, l] V[b, l] W[c, l].
        # This is a 3-way contraction.
        # Can be done by:
        # Temp = U @ V.T  # (n*m, m*p) ? No.
        # U is (n*m, R). V is (m*p, R).
        # U @ V.T is (n*m, m*p).
        # This gives sum_l U[a, l] V[b, l].
        # Then we need to multiply by W[c, l] and sum?
        # No, W is also contracted.
        
        # T[a, b, c] = sum_l U[a, l] V[b, l] W[c, l].
        # This is the trace of U[a] V[b] W[c]?
        # We can compute this by:
        # for l in range(R):
        #   T += U[:, l] @ V[:, l].T @ W[:, l].T? No.
        #   T[:, :, l] = U[:, l] @ V[:, l].T?
        #   Then T[a, b, c] = sum_l (U[:, l] @ V[:, l].T)[a, b] * W[c, l]?
        #   No, W is also a vector.
        
        # T[a, b, c] = sum_l U[a, l] V[b, l] W[c, l].
        # Let M_l = U[:, l] @ V[:, l].T. (n*m, m*p)
        # Then T[a, b, c] = sum_l M_l[a, b] W[c, l].
        # This is a sum over l of (M_l * W[:, l].T)?
        # We can reshape.
        
        # Simpler:
        # T = 0
        # for l in range(R):
        #   T += np.einsum('a,b,c->abc', U[:, l], V[:, l], W[:, l])
        # This is O(R * n*m * m*p * p*n).
        # 57 * 9 * 21 * 21 = ~230k ops. Very fast.
        
        T = np.zeros((n*m, m*p, p*n))
        for l in range(len(U[0])):
            T += np.einsum('a,b,c->abc', U[:, l], V[:, l], W[:, l])
            
        # Check T against expected.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    if T[a, b, c] != 1:
                        return False
                    # Also check that other entries are 0?
                    # The tensor for matmul is sparse.
                    # T[a, b, c] should be 0 otherwise.
        
        # Check a few zeros.
        # a=0, b=0, c=1. i=0, j=0. b=0 -> j=0, k=0. c=1 -> k=0, i=1.
        # i=0, i=1 conflict. So should be 0.
        if T[0, 0, 1] != 0:
            return False
            
        return True
    