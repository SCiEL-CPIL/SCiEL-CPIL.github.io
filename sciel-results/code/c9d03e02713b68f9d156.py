import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 140.0  # Safety margin under 150s

    n, m, p = 2, 2, 10

    # Build rank-35 Strassen block-diagonal decomposition
    strassen = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,1,0,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,0,1,1]),
        ([0,0,0,1], [-1,0,1,0], [1,1,0,0]),
        ([1,1,0,0], [0,0,0,1], [-1,0,1,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]

    def build_rank35():
        U = np.zeros((n*m, 35))
        V = np.zeros((m*p, 35))
        W = np.zeros((p*n, 35))
        col = 0
        for k in range(5):
            for av, bv, cv in strassen:
                for i in range(4):
                    U[i, col] = av[i]
                V[2*k, col] = bv[0]
                V[2*k+1, col] = bv[1]
                V[10+2*k, col] = bv[2]
                V[10+2*k+1, col] = bv[3]
                W[4*k, col] = cv[0]
                W[4*k+1, col] = cv[1]
                W[4*k+2, col] = cv[2]
                W[4*k+3, col] = cv[3]
                col += 1
        return U, V, W

    def verify(U, V, W):
        """Check that the decomposition is exact."""
        n, m, p = 2, 2, 10
        rng = np.random.RandomState(42)
        for _ in range(5):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_direct = A @ B
            
            A_flat = A.flatten()
            B_flat = B.flatten()
            
            UA = U.T @ A_flat
            VB = V.T @ B_flat
            
            WV = W * (UA * VB)[np.newaxis, :]
            C_decomp_flat = WV.sum(axis=1)
            
            C_decomp = np.zeros((n, p))
            for k in range(p):
                for i in range(n):
                    C_decomp[i, k] = C_decomp_flat[k*n + i]
            
            if not np.allclose(C_direct, C_decomp, atol=1e-10):
                return False
        return True

    def to_list(M):
        return M.tolist()

    # Start with rank-35
    U35, V35, W35 = build_rank35()
    
    best_rank = 35
    best_U, best_V, best_W = U35, V35, W35
    
    if not verify(best_U, best_V, best_W):
        raise RuntimeError("Rank-35 verification failed")
    
    # Try to reduce rank by finding a linear dependency
    # Approach: try to find vectors a (size 35), b (size 35), c (size 35)
    # such that sum_r a[r]*U[:,r] = 0, sum_r b[r]*V[:,r] = 0, sum_r c[r]*W[:,r] = 0
    # This would mean the 35 rank-1 tensors are linearly dependent in a special way.
    # But this is not quite right for tensor rank reduction.
    
    # Better approach: try to find a rank-34 decomposition by solving a system.
    # A rank-r decomposition means: T = sum_{s=1}^r u_s ⊗ v_s ⊗ w_s
    # where u_s in R^4, v_s in R^20, w_s in R^20.
    
    # The tensor T has entries T[i, j, k] = 1 if C_{i,k} = A_{i,j} * B_{j,k} contributes.
    # With W[k*n+i, r] convention: T[i, j, k*n+i] ... let me be careful.
    
    # Actually, let's try a different strategy: use the structure to try rank 34.
    # The 5 blocks each use 7 Strassen products. If we can merge 2 blocks using 13 products
    # instead of 14, we'd get rank 33. If we can merge all 5 using 33, we get 33.
    
    # For now, try a numerical approach: start with rank-35 and try to zero out one column
    # by adjusting the others.
    
    # Try: can we find a vector x in R^35 such that
    # U @ x = 0, V @ x = 0, W @ x = 0?
    # If yes, then the 35 rank-1 terms are linearly dependent and we might reduce rank.
    
    U_stacked = np.vstack([U35, V35, W35])  # (44, 35)
    # Check if there's a non-trivial null vector
    try:
        _, s, vh = np.linalg.svd(U_stacked)
        # The smallest singular value tells us about the null space
        if s[-1] < 1e-10:
            # There's a linear dependency, but this doesn't directly give tensor rank reduction
            pass
    except:
        pass
    
    # Alternative: try to find a rank-34 decomposition by solving nonlinear equations.
    # This is computationally expensive, so let's try a simpler heuristic.
    
    # Heuristic: try to combine two Strassen blocks using a shared product.
    # For blocks k and k+1, the A-side vectors are the same. So the 7 products
    # M_s = (a_s · A) * (b_s · B_k) for block k and M_s' = (a_s · A) * (b_s · B_{k+1})
    # for block k+1. The (a_s · A) part is shared!
    
    # So we can compute a_s · A once and use it for both blocks. But this doesn't reduce
    # the number of bilinear products; it just reduces additions.
    
    # The bilinear products are: for each s and each block k, we have one product
    # (a_s · A) * (b_{s,k} · B_k). That's 7*5 = 35 products.
    
    # To reduce rank, we need to find a way to express some of these products in terms
    # of others, which would require algebraic relations.
    
    # Let's try a completely different approach: use the fact that the tensor is
    # M_{2,2} ⊗ I_5 (in some sense) and look for known results.
    
    # Known: rank(M_{2,2,10}) >= rank(M_{2,2}) * ceil(10/2) = 7*5 = 35? No, that's not right.
    # Actually, there's no simple multiplicative lower bound for non-cubic tensors.
    
    # For now, return rank 35.
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
