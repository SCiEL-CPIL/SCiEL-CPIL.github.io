import numpy as np
import time

DIMENSIONS = [2, 4, 7]
DEADLINE_SECONDS = 140.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    def build_strassen_222(i0, j0, k0, U, V, W, col):
        """Add a 2x2x2 Strassen block. Returns new col index."""
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h)
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1
        
        # M3 = a(f-h)
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1
        return col
    
    def build_schoolbook_221(i0, j0, k0, U, V, W, col):
        """Add a 2x2x1 schoolbook block (4 mults)."""
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        return col
    
    def build_schoolbook_211(i0, j0, k0, U, V, W, col):
        """Add a 2x1x1 schoolbook block (2 mults)."""
        a_idx = i0 * m + j0
        c_idx = (i0 + 1) * m + j0
        e_idx = j0 * p + k0
        c0_idx = k0 * n + i0
        c1_idx = k0 * n + (i0 + 1)
        
        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c0_idx, col] += 1; col += 1
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c1_idx, col] += 1; col += 1
        return col
    
    def build_schoolbook_111(i0, j0, k0, U, V, W, col):
        """Add a 1x1x1 schoolbook block (1 mult)."""
        a_idx = i0 * m + j0
        e_idx = j0 * p + k0
        c_idx = k0 * n + i0
        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c_idx, col] += 1; col += 1
        return col
    
    def verify_decomposition(U, V, W, rank):
        """Check that the decomposition is exact."""
        # Build full matrices and verify
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # C[i,k] should equal sum_j A[i,j]*B[j,k]
                    # In our decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # Actually the tensor is: T[i,j,k] = delta (it's matmul tensor)
                    # The decomposition says: for all i,j,k: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 if consistent
                    # Wait, the matmul tensor M[i,j,k] = 1 for all i,j,k (it's the identity-like tensor)
                    # Actually M is the matrix multiplication tensor: M[i,j,k] = 1
                    # And the decomposition is: M[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                    # Let me verify with random matrices instead
                    pass
        # Better: verify with actual matrix multiplication
        # C = A * B where A is n x m, B is m x p
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # Our decomposition: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
        # Wait no. The standard form is:
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        # Let me just do a numerical check
        np.random.seed(42)
        A = np.random.randint(-5, 6, size=(n, m))
        B = np.random.randint(-5, 6, size=(m, p))
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            U_r = U[:, r]
            V_r = V[:, r]
            W_r = W[:, r]
            if np.all(U_r == 0) or np.all(V_r == 0) or np.all(W_r == 0):
                continue
            # Compute intermediate
            UA = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    if U_r[i * m + j] != 0:
                        UA[i] += U_r[i * m + j] * A[i, j]
            VB = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    if V_r[j * p + k] != 0:
                        VB[k] += V_r[j * p + k] * B[j, k]
            for i in range(n):
                for k in range(p):
                    if W_r[k * n + i] != 0:
                        C_computed[i, k] += W_r[k * n + i] * UA[i] * VB[k]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
        return True
    
    # Strategy: try different block decompositions and pick the best one
    # Current: 6 Strassen (2x2x2) + 2 schoolbook (2x2x1) = 42 + 8 = 50
    
    # Alternative 1: Try splitting 7 as 2+2+2+1 but with different j-splits
    # Alternative 2: Try 3+4 split for the 7 columns
    # Alternative 3: Try 2+3+2 split
    
    best_result = None
    best_rank = 56  # schoolbook upper bound
    
    # Try the current known-good construction first
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            col = build_strassen_222(0, j_start, k_start, U, V, W, col)
        col = build_schoolbook_221(0, j_start, 6, U, V, W, col)
    assert col == 50
    if verify_decomposition(U, V, W, 50):
        best_result = (U.copy(), V.copy(), W.copy(), 50)
        best_rank = 50
    
    # Try alternative: split 7 as 3+4, use 2x2x3 and 2x2x4 blocks
    # For 2x2x3: use Strassen on first 2 cols + schoolbook on last col
    # That's 7 + 2 = 9 per j-block... but we have 2 j-blocks of size 2
    # Total: 2*(9 + 7 + 2) = wait, let me think more carefully
    
    # Actually, let me try: for each j-pair (0-1, 2-3), split k as 3+4
    # 2x2x3: Strassen(2x2x2) + schoolbook(2x2x1) = 7+4 = 11
    # 2x2x4: Strassen(2x2x2) + Strassen(2x2x2) = 7+7 = 14
    # Per j-pair: 11+14 = 25, total = 50. Same as before.
    
    # Try: split k as 2+5 for each j-pair
    # 2x2x2: 7
    # 2x2x5: Strassen(2x2x2) + 2x2x3 = 7 + 11 = 18
    # Per j-pair: 7+18 = 25, total = 50. Same.
    
    # Try a different approach: don't split j at all, use 2x4x7 directly
    # Or split j as 3+1
    # 2x3x7: ?
    # 2x1x7: 14 schoolbook mults
    
    # Let me try: j-split as 3+1, k-split as 2+2+2+1
    # 2x3x2: can we do better than 6? Try Strassen-like
    # For 2x3x2: C[i,k] = sum_{j=0}^{2} A[i,j]*B[j,k]
    # Use: M1=(A00+A11+A22)(B00+B10+B20) - this gets messy
    
    # Actually, let me try a smarter 2x3x2 decomposition
    # 2x3x2 matmul tensor rank: I'll try to find it
    # Schoolbook: 6 mults
    # Can we do 5? Probably not for this small case.
    
    # Let me try yet another split: j as 2+2, but k as 4+3
    # 2x2x4: 2 Strassen = 14
    # 2x2x3: Strassen(2x2x2) + schoolbook(2x2x1) = 7+4 = 11
    # Per j-pair: 14+11 = 25, total = 50. Same again.
    
    # The key question: can we share multiplications across j-blocks?
    # In the current scheme, j=0-1 and j=2-3 are completely independent.
    # What if we use a 2x4x2 Strassen-like block?
    
    # 2x4x2 matmul: C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    # This is like multiplying a 2x4 by 4x2.
    # We can view this as: C = [A_left | A_right] * [B_top; B_bottom]
    # where A_left, A_right are 2x2 and B_top, B_bottom are 2x2.
    # C = A_left*B_top + A_right*B_bottom
    # Each 2x2x2 product takes 7 Strassen mults, total 14.
    # But can we do better by exploiting the sum?
    
    # Actually, there's a known result: the 2x4x2 matmul tensor has rank 14 (same as 2*7).
    # So no savings there.
    
    # What about 2x4x3?
    # C = A_left*B_top3 + A_right*B_bottom3 where B_top3, B_bottom3 are 2x3
    # Each 2x2x3: 11 (as computed above)
    # Total: 22. Schoolbook would be 2*4*3=24. So we save 2.
    # But we need 2x4x7 = 2x4x4 + 2x4x3 = 28 + 22 = 50. Same!
    
    # Hmm, it seems 50 might be optimal for this block-decomposition family.
    # Let me verify with a direct numerical check and return the known-good result.
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS:
        pass
    
    if best_result is None:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        best_result = (U, V, W, rank)
    
    U, V, W, rank = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
