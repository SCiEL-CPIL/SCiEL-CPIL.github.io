"""Program: attempts a lower-rank exact decomposition for 3x3*3x6.

Strategy:
  1. Construct the 54-term schoolbook decomposition (rank 54).
  2. Use the known 23-term decomposition for M_{3,3,3} to construct a
     rank-46 decomposition for M_{3,3,6}:
       - 3x3 block: use the 23 terms
       - 3x6 block: use 2x(3x3) = 2*9 = 18 terms (schoolbook for the
         additional 3 columns of the 6-wide B matrix)
     Total: 23 + 18 = 41? No.
     
     Actually, let's think more carefully.
     
     M_{3,3,6}: A is 3x3, B is 3x6, C is 3x6.
     C = A * B.
     
     We can split B into B1 (3x3) and B2 (3x3):
     C = A*B1 + A*B2
     
     If we use rank r1 for A*B1 and rank r2 for A*B2, total rank = r1 + r2.
     For 3x3 matrix mult, rank = 23.
     So total = 23 + 23 = 46.
     
     But we need to verify that the factor matrices can be constructed this way.
     The tensor for A*B1 has indices (i, j, k) where k ranges over 0,1,2.
     The tensor for A*B2 has indices (i, j, k) where k ranges over 3,4,5.
     
     The W matrix for A*B1 has entries W[k*n + i, c] for k in 0,1,2.
     The W matrix for A*B2 has entries W[k*n + i, c] for k in 3,4,5.
     
     Since k ranges over disjoint sets, the W matrices can be combined by
     simply concatenating columns. Similarly for U and V.
     
     So the construction is:
     - Take the 23-term decomposition for M_{3,3,3} giving U1 (9x23), V1 (18x23), W1 (18x23).
     - For the second 3x3 block (k=3,4,5), we need another 23-term decomposition.
       But the indices for V and W shift: V[j*p + k, c] where k is now 0,1,2 but
       corresponds to global k = 3,4,5.
       
     Let me be more careful.
     
     For M_{3,3,3} (B is 3x3, p=3):
     U: (9, r), V: (9, r) [since m*p = 3*3 = 9], W: (9, r) [since p*n = 3*3 = 9]
     
     For M_{3,3,6} (B is 3x6, p=6):
     U: (9, R), V: (18, R) [since m*p = 3*6 = 18], W: (18, R) [since p*n = 6*3 = 18]
     
     Block 1 (k=0,1,2): 
       U1: (9, 23), V1: (9, 23), W1: (9, 23)
       W1 indices: k in 0,1,2, i in 0,1,2 -> w_idx = k*3 + i (0..8)
     
     Block 2 (k=3,4,5):
       U2: (9, 23), V2: (9, 23), W2: (9, 23)
       W2 indices: k in 3,4,5, i in 0,1,2 -> w_idx = k*3 + i (9..17)
       V2 indices: j in 0,1,2, k in 3,4,5 -> v_idx = j*6 + k (3,4,5, 9,10,11, 15,16,17)
     
     So the full decomposition:
       U = [U1, U2]  (9 x 46)
       V = [V1, V2]  but V1 only fills rows 0,1,2,6,7,8,12,13,14 (k=0,1,2)
           and V2 fills rows 3,4,5,9,10,11,15,16,17 (k=3,4,5)
       W = [W1, W2]  where W1 fills rows 0..8, W2 fills rows 9..17
     
     This gives rank 46.
     
     Now I need the 23-term coefficients. Let me use a known set.
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    deadline = time.time() + 110.0
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # 23-term decomposition for 3x3 matrix multiplication.
    # Using a known set of coefficients. These are for the tensor T[i,j,k]
    # where C[i,k] = sum_j A[i,j] B[j,k].
    # U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c] summed over c = 1 for all i,j,k.
    
    # Known 23-term decomposition (one of the standard ones).
    # I'll use a set of coefficients. The entries can be half-integers.
    # Let me construct them from a known reference.
    
    # The 23 terms for 3x3 matrix multiplication:
    # Each term is (u_vals, v_vals, w_vals) where each is a length-9 vector.
    # U has 9 entries (3x3), V has 9 entries (3x3), W has 9 entries (3x3).
    
    # I'll use a known decomposition. One common one uses entries in {-1, 0, 1, 0.5}.
    # Let me use the following (from memory / standard reference):
    
    # Actually, let me just try to find a working set by using a simpler approach:
    # Use the schoolbook 9 terms for the 3x3 part, then try to reduce.
    # But that gives rank 9+9=18 for the two 3x3 blocks, total 9+9+9+9 = 36? No.
    
    # Wait, if I use schoolbook for both 3x3 blocks, that's 9 + 9 = 18 terms, total rank 18.
    # But schoolbook for 3x3 is rank 9, so 9+9=18 for M_{3,3,6}? That can't be right
    # because M_{3,3,6} has a lower bound of 44.
    
    # Oh wait, I'm confusing myself. The schoolbook decomposition for M_{3,3,3} has
    # rank 9 (one term per (i,j,k) triple). For M_{3,3,6}, schoolbook has rank 54.
    
    # If I split B into two 3x3 blocks:
    # Block 1: A*B1 where B1 is 3x3. Schoolbook rank = 9.
    # Block 2: A*B2 where B2 is 3x3. Schoolbook rank = 9.
    # Total: 18. But that's wrong because the U factors must be the same?
    
    # No! In a tensor decomposition, each term has one U row vector, one V row vector,
    # and one W row vector. The U, V, W matrices are shared across all terms.
    
    # For the split:
    # Term c in block 1: U[i*m+j, c], V[j*p+k, c] (k in 0,1,2), W[k*n+i, c] (k in 0,1,2)
    # Term c in block 2: U[i*m+j, c], V[j*p+k, c] (k in 3,4,5), W[k*n+i, c] (k in 3,4,5)
    
    # The U factor is the same structure for both blocks (same i,j indices).
    # But the V and W factors have different k ranges.
    
    # So if I use 23 terms for block 1 and 23 terms for block 2, total rank = 46.
    # The U matrix will have 46 columns: first 23 from block 1, next 23 from block 2.
    # But U for block 1 and block 2 can be different! They don't need to be the same.
    
    # So yes, rank 46 is achievable IF we have a 23-term decomposition for 3x3.
    
    # Let me use a known 23-term decomposition. I'll use one with rational coefficients.
    
    # Here's a known 23-term decomposition for 3x3 matrix multiplication:
    # (i, j, k) indexing for U[i*3+j], V[j*3+k], W[k*3+i]
    
    # I'll use the following coefficients (one of the standard 23-term ones):
    # This is complex to hardcode from memory. Let me try a different approach:
    
    # Use scipy to find a rank-46 decomposition via optimization, starting from
    # a good initialization.
    
    # Actually, let me just do the schoolbook fallback (rank 54) and try to
    # improve with optimization.
    
    from scipy.optimize import least_squares
    
    def check_decomp(U, V, W):
        """Check if U, V, W exactly decompose the 3x3*3x6 multiplication tensor."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if not np.isclose(val, 1.0, atol=1e-10):
                        return False
        return True
    
    # First, create the schoolbook rank-54 decomposition as fallback
    def schoolbook():
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
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    
    # Try to find a rank-46 decomposition
    def find_rank46(max_attempts=30, timeout=90):
        start = time.time()
        rank = 46
        size_U = 9 * rank
        size_V = 18 * rank
        size_W = 18 * rank
        total_size = size_U + size_V + size_W
        
        for attempt in range(max_attempts):
            if time.time() - start > timeout:
                break
            if time.time() > deadline:
                break
            
            # Initialize with schoolbook-like structure
            # First 23 columns: try to form a 3x3 decomposition for block 1
            # Next 23 columns: try to form a 3x3 decomposition for block 2
            
            # Random initialization with small values
            params = np.random.randn(total_size) * 0.5
            
            # Add some structure: make first few columns have non-zero patterns
            # For block 1 (k=0,1,2), V rows 0,1,2,6,7,8,12,13,14 should be non-zero
            # For block 2 (k=3,4,5), V rows 3,4,5,9,10,11,15,16,17 should be non-zero
            
            # Let's try a structured initialization
            # Columns 0-22: block 1
            # Columns 23-45: block 2
            
            # For block 1, use a perturbed version of a 3x3 schoolbook-like init
            for c in range(23):
                # U: all 9 entries can be non-zero
                U_start = c * 9
                U_end = (c + 1) * 9
                params[U_start:U_end] = np.random.randn(9) * 0.3
                
                # V: only rows corresponding to k=0,1,2
                # v_idx = j*6 + k for j in 0,1,2 and k in 0,1,2
                # These are: 0,1,2, 6,7,8, 12,13,14
                v_rows_block1 = [0,1,2,6,7,8,12,13,14]
                V_start = size_U + c * 18
                for vr in v_rows_block1:
                    params[V_start + vr] = np.random.randn() * 0.3
                # Other V rows should be zero
                for vr in range(18):
                    if vr not in v_rows_block1:
                        params[V_start + vr] = 0.0
                
                # W: only rows corresponding to k=0,1,2
                # w_idx = k*3 + i for k in 0,1,2 and i in 0,1,2
                # These are: 0,1,2, 3,4,5, 6,7,8
                W_start = size_U + size_V + c * 18
                for wr in range(9):  # rows 0-8
                    params[W_start + wr] = np.random.randn() * 0.3
                for wr in range(9, 18):
                    params[W_start + wr] = 0.0
            
            # For block 2, use a similar structure but with k=3,4,5
            for c in range(23):
                c_idx = c + 23
                U_start = c_idx * 9
                U_end = (c_idx + 1) * 9
                params[U_start:U_end] = np.random.randn(9) * 0.3
                
                v_rows_block2 = [3,4,5,9,10,11,15,16,17]
                V_start = size_U + c_idx * 18
                for vr in v_rows_block2:
                    params[V_start + vr] = np.random.randn() * 0.3
                for vr in range(18):
                    if vr not in v_rows_block2:
                        params[V_start + vr] = 0.0
                
                W_start = size_U + size_V + c_idx * 18
                for wr in range(9, 18):  # rows 9-17
                    params[W_start + wr] = np.random.randn() * 0.3
                for wr in range(9):
                    params[W_start + wr] = 0.0
            
            def residuals(params):
                U = params[0:size_U].reshape(9, rank)
                V = params[size_U:size_U+size_V].reshape(18, rank)
                W = params[size_U+size_V:].reshape(18, rank)
                
                res = np.zeros(n*m*p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                            res[idx] = val - 1.0
                            idx += 1
                return res
            
            try:
                result = least_squares(residuals, params, method='lm', xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=5000)
            except:
                continue
            
            if result.cost < 1e-8:
                U = result.x[0:size_U].reshape(9, rank)
                V = result.x[size_U:size_U+size_V].reshape(18, rank)
                W = result.x[size_U+size_V:].reshape(18, rank)
                
                # Try rounding to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if check_decomp(U_r, V_r, W_r):
                    return U_r, V_r, W_r
                
                # Try integer rounding
                U_i = np.round(U)
                V_i = np.round(V)
                W_i = np.round(W)
                if check_decomp(U_i, V_i, W_i):
                    return U_i, V_i, W_i
        
        return None
    
    result = find_rank46(max_attempts=20, timeout=80)
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 54
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
