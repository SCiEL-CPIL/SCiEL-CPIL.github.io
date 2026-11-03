import numpy as np
import time

DIMENSIONS = [4, 4, 5]
TIME_LIMIT = 120

def _strassen_2x2x2_factors():
    factors = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    return factors

def _strassen_4x4x4_factors():
    factors_2x2 = _strassen_2x2x2_factors()
    
    def map_4x4_index(br, bc, lr, lc):
        return (2*br + lr, 2*bc + lc)
    
    strassen_4x4_steps = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    all_factors = []
    for A_block_coeffs, B_block_coeffs, C_block_coeffs in strassen_4x4_steps:
        for f2 in factors_2x2:
            A_2x2, B_2x2, C_2x2 = f2
            A_4x4 = {}
            B_4x4 = {}
            C_4x4 = {}
            
            for (br, bc), coeff_block in A_block_coeffs.items():
                for (lr, lc), coeff_2x2 in A_2x2.items():
                    i, j = map_4x4_index(br, bc, lr, lc)
                    A_4x4[(i,j)] = A_4x4.get((i,j), 0) + coeff_block * coeff_2x2
            
            for (br, bc), coeff_block in B_block_coeffs.items():
                for (lr, lc), coeff_2x2 in B_2x2.items():
                    i, j = map_4x4_index(br, bc, lr, lc)
                    B_4x4[(i,j)] = B_4x4.get((i,j), 0) + coeff_block * coeff_2x2
            
            for (br, bc), coeff_block in C_block_coeffs.items():
                for (lr, lc), coeff_2x2 in C_2x2.items():
                    i, j = map_4x4_index(br, bc, lr, lc)
                    C_4x4[(i,j)] = C_4x4.get((i,j), 0) + coeff_block * coeff_2x2
            
            all_factors.append((A_4x4, B_4x4, C_4x4))
            
    return all_factors

def _verify_decomposition(U, V, W, n=4, m=4, p=5, num_tests=20, tol=1e-9):
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
                    b_indices = np.arange(m) * p + k
                    b_val = np.dot(V[b_indices, r], B[:, k])
                    val += W[k*n+i, r] * a_val * b_val
                C_computed[i, k] = val
        
        if not np.allclose(C_computed, C_expected, atol=tol):
            return False
    
    return True

def _try_reduce_rank(U, V, W, n, m, p, max_iter=50, tol=1e-10):
    """Try to reduce rank by finding linear dependencies."""
    rank = U.shape[1]
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Check if any column is linearly dependent on others
    # For each column, check if it can be expressed as a linear combination of others
    # This is expensive, so we'll use a heuristic approach
    
    # Try to find a subset of columns that spans the same tensor
    # Use SVD on the tensor matricizations
    
    # Matricize the tensor: for each output index (i,k), the vector is 
    # sum_r W[i,k,r] * U[:,r] outer V[:,r]
    # This is hard to check directly. Instead, check rank of factor matrices.
    
    # If U has rank < rank, we can reduce. Same for V and W.
    for _ in range(max_iter):
        # Check rank of U
        rU = np.linalg.matrix_rank(U, tol=tol)
        rV = np.linalg.matrix_rank(V, tol=tol)
        rW = np.linalg.matrix_rank(W, tol=tol)
        
        min_rank = min(rU, rV, rW)
        if min_rank < rank:
            # Try to reduce to min_rank
            # Use SVD to find a basis
            U_svd, _, _ = np.linalg.svd(U, full_matrices=False)
            V_svd, _, _ = np.linalg.svd(V, full_matrices=False)
            W_svd, _, _ = np.linalg.svd(W, full_matrices=False)
            
            new_rank = min_rank
            U_new = U_svd[:, :new_rank]
            V_new = V_svd[:, :new_rank]
            W_new = W_svd[:, :new_rank]
            
            # Verify
            if _verify_decomposition(U_new, V_new, W_new, n, m, p, num_tests=5, tol=tol):
                return U_new, V_new, W_new, new_rank
        
        break
    
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build 65-rank decomposition: 49 (Strassen for 4x4x4) + 16 (schoolbook for col 4)
    factors_4x4 = _strassen_4x4x4_factors()
    rank_4x4 = len(factors_4x4)  # 49
    rank_4x1 = n * m  # 16
    rank_total = rank_4x4 + rank_4x1  # 65
    
    U = np.zeros((n*m, rank_total))
    V = np.zeros((m*p, rank_total))
    W = np.zeros((p*n, rank_total))
    
    # Fill Strassen part (columns 0-3 of B)
    for r, (A_lin, B_lin, C_mat) in enumerate(factors_4x4):
        for (i, j), coeff in A_lin.items():
            U[i*m + j, r] += coeff
        for (i, j), coeff in B_lin.items():
            V[i*p + j, r] += coeff
        for (i, j), coeff in C_mat.items():
            W[j*n + i, r] += coeff
    
    # Fill schoolbook part for column 4
    for i in range(n):
        for j in range(m):
            col_idx = rank_4x4 + i*m + j
            U[i*m + j, col_idx] = 1
            V[j*p + 4, col_idx] = 1
            W[4*n + i, col_idx] = 1
    
    # Try to reduce rank
    if time.time() - start_time < TIME_LIMIT - 30:
        try:
            U, V, W, rank = _try_reduce_rank(U, V, W, n, m, p)
        except Exception:
            pass
    
    # Verify
    if time.time() - start_time < TIME_LIMIT - 10:
        try:
            if _verify_decomposition(U, V, W, n, m, p, num_tests=20, tol=1e-9):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            pass
    
    # Fallback: schoolbook rank 80
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
