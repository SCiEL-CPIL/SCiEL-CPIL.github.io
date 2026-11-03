import numpy as np
import time

DIMENSIONS = [4, 4, 5]
TIME_LIMIT = 120

def _strassen_2x2x2_factors():
    """
    Returns 7 factors for 2x2x2 matrix multiplication.
    Each factor is (A_dict, B_dict, C_dict) with keys (i,j) for 2x2.
    """
    factors = [
        # M1 = (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (A10+A11)B00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3 = A00(B10-B11)
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = A11(B01-B10)
        ({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (A00+A01)B11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (A10-A00)(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (A01-A11)(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    return factors

def _strassen_4x4x4_factors():
    """
    Returns 49 factors for 4x4x4 matrix multiplication.
    Uses recursive Strassen.
    """
    factors_2x2 = _strassen_2x2x2_factors()
    
    # Strassen combinations for 4x4 blocks
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B10-B11) -> C01 += M3, C11 += M3
    # M4 = A11(B01-B10) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    # For each M_k, we need to expand it into 7 rank-1 terms for the 2x2x2 product.
    # The 2x2x2 product is computed by factors_2x2.
    # We need to map the 2x2 indices to 4x4 indices.
    
    all_factors = []
    
    # Helper to map 2x2 block index (br, bc) and local index (lr, lc) to 4x4 index
    def map_4x4_index(br, bc, lr, lc):
        return (2*br + lr, 2*bc + lc)
    
    # Define the 7 Strassen steps for 4x4
    # Each step defines:
    # 1. Linear forms on A blocks (dict block_idx -> coeff)
    # 2. Linear forms on B blocks (dict block_idx -> coeff)
    # 3. Output C blocks (dict block_idx -> coeff)
    
    strassen_4x4_steps = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4
        ({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):1, (1,0):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    for A_block_coeffs, B_block_coeffs, C_block_coeffs in strassen_4x4_steps:
        # For this step, we compute a 2x2x2 multiplication.
        # We use the 7 rank-1 factors of 2x2x2.
        for f2 in factors_2x2:
            A_2x2, B_2x2, C_2x2 = f2
            
            # Construct A_4x4, B_4x4, C_4x4 dicts
            A_4x4 = {}
            B_4x4 = {}
            C_4x4 = {}
            
            # Expand A
            for (br, bc), coeff_block in A_block_coeffs.items():
                for (lr, lc), coeff_2x2 in A_2x2.items():
                    i, j = map_4x4_index(br, bc, lr, lc)
                    A_4x4[(i,j)] = A_4x4.get((i,j), 0) + coeff_block * coeff_2x2
            
            # Expand B
            for (br, bc), coeff_block in B_block_coeffs.items():
                for (lr, lc), coeff_2x2 in B_2x2.items():
                    i, j = map_4x4_index(br, bc, lr, lc)
                    B_4x4[(i,j)] = B_4x4.get((i,j), 0) + coeff_block * coeff_2x2
            
            # Expand C
            for (br, bc), coeff_block in C_block_coeffs.items():
                for (lr, lc), coeff_2x2 in C_2x2.items():
                    i, j = map_4x4_index(br, bc, lr, lc)
                    C_4x4[(i,j)] = C_4x4.get((i,j), 0) + coeff_block * coeff_2x2
            
            all_factors.append((A_4x4, B_4x4, C_4x4))
            
    return all_factors

def _verify_decomposition(U, V, W, n=4, m=4, p=5, num_tests=10, tol=1e-10):
    rank = U.shape[1]
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for _ in range(num_tests):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    a_val = np.dot(U[i*m:(i+1)*m, r], A[i, :])
                    b_val = np.dot(V[k*m:(k+1)*m, r] if False else V[r, k*m:(k+1)*m], B[:, k]) # V is (mp, rank)
                    # V index: j*p + k. So V[j*p+k, r].
                    # We need sum_j V[j*p+k, r] B[j,k]
                    # This is dot product of V[:, k] (reshaped) and B[:, k]?
                    # V is (m*p, rank). V[j*p+k, r].
                    # For fixed k, V[:, k] is not a column.
                    # V is indexed by (j*p+k).
                    # So for fixed k, the relevant V entries are V[k], V[p+k], V[2p+k]...
                    # Let's just loop or use advanced indexing.
                    b_indices = np.arange(m) * p + k
                    b_val = np.dot(V[b_indices, r], B[:, k])
                    val += W[k*n+i, r] * a_val * b_val
                C_computed[i, k] = val
        
        if not np.allclose(C_computed, C_expected, atol=tol):
            return False
    
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Rank 49 for 4x4x4
    factors_4x4 = _strassen_4x4x4_factors()
    rank_4x4 = len(factors_4x4) # 49
    
    # Rank 16 for 4x4x1
    rank_4x1 = n * m # 16
    
    rank_total = rank_4x4 + rank_4x1 # 65
    
    U = np.zeros((n*m, rank_total))
    V = np.zeros((m*p, rank_total))
    W = np.zeros((p*n, rank_total))
    
    # Fill Strassen part
    for r, (A_lin, B_lin, C_mat) in enumerate(factors_4x4):
        for (i, j), coeff in A_lin.items():
            U[i*m + j, r] += coeff
        for (i, j), coeff in B_lin.items():
            # B is 4x4 (cols 0-3)
            V[i*p + j, r] += coeff
        for (i, j), coeff in C_mat.items():
            # C is 4x4 (cols 0-3)
            # W index: k*n + i. Here k=j, i=i.
            W[j*n + i, r] += coeff
            
    # Fill Schoolbook part for col 4
    for i in range(n):
        for j in range(m):
            col_idx = rank_4x4 + i*m + j
            U[i*m + j, col_idx] = 1
            V[j*p + 4, col_idx] = 1
            W[4*n + i, col_idx] = 1
            
    # Verify
    if time.time() - start_time < TIME_LIMIT - 10:
        try:
            if _verify_decomposition(U, V, W, n, m, p, num_tests=10, tol=1e-10):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            pass
            
    # Fallback
    rank_sb = n*m*p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
