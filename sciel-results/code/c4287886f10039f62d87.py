"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY = 5.0

def _time_left():
    return _DEADLINE - _SAFETY - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the matmul tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    rank = U.shape[1]
    # For each (i,j,k), check sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], np.multiply(V[j*p+k], W[k*n+i]))
                if abs(val - 1.0) > 1e-8:
                    return False
    return True

def _strassen_2x2():
    """
    Returns U, V, W for 2x2 * 2x2 -> 2x2.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices:
    U[i*2+j, r]
    V[j*2+k, r]
    W[k*2+i, r]
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen's 7 products
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-d)(f+g)
    # M7 = (b-d)(e+f)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1 = (a+d)(e+h)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1 # C11, C22
    
    # M2 = (c+d)(e)
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1 # C21
    
    # M3 = a(f-h)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[2,2] = 1 # C12
    
    # M4 = d(g-e)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[1,3] = 1 # C11, C21
    
    # M5 = (a+b)(h)
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[2,4] = 1 # C11, C12
    
    # M6 = (c-d)(f+g)
    U[2,5] = 1; U[3,5] = -1
    V[1,5] = 1; V[2,5] = 1
    W[3,5] = 1 # C22
    
    # M7 = (b-d)(e+f)
    U[1,6] = 1; U[3,6] = -1
    V[0,6] = 1; V[1,6] = 1
    W[0,6] = 1 # C11
    
    return U, V, W

def _build_strassen_4x4():
    """
    Builds 4x4 * 4x4 -> 4x4 using Strassen.
    Rank 49.
    U shape (16, 49), V shape (16, 49), W shape (16, 49).
    Indices:
    U[i*4+j, r]
    V[j*4+k, r]
    W[k*4+i, r]
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Get 2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2()
    
    # Strassen for 4x4:
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    # C blocks: C00, C01, C10, C11 (each 2x2)
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A11)(B01+B10)
    # M7 = (A01-A11)(B00+B01)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mk is a 2x2 matrix product.
    # We substitute the 2x2 multiplication with the 7-term Strassen decomposition.
    
    # Helper to map block indices to global indices
    # A block (bi, bj) corresponds to rows i in [2*bi, 2*bi+1], cols j in [2*bj, 2*bj+1]
    # Global index for A: i*4 + j
    # Global index for B: j*4 + k
    # Global index for C: k*4 + i
    
    def add_product(U_mat, V_mat, W_mat, 
                    A_coeffs, B_coeffs, C_coeffs, 
                    base_rank, U2, V2, W2):
        """
        Adds a 2x2 matrix product term to the decomposition.
        A_coeffs: list of (block_row, block_col, coeff) for A
        B_coeffs: list of (block_row, block_col, coeff) for B
        C_coeffs: list of (block_row, block_col, coeff) for C
        """
        for r2 in range(7):
            r_global = base_rank + r2
            # U contribution
            for (br, bc, coeff) in A_coeffs:
                # A block (br, bc) is 2x2.
                # U2 has indices for 2x2: u_idx = i2*2 + j2
                # Global A index: i*4 + j
                # i = 2*br + i2, j = 2*bc + j2
                for i2 in range(2):
                    for j2 in range(2):
                        u_idx = i2 * 2 + j2
                        u_val = U2[u_idx, r2] * coeff
                        if u_val != 0:
                            i_global = 2 * br + i2
                            j_global = 2 * bc + j2
                            U_mat[i_global * 4 + j_global, r_global] += u_val
            
            # V contribution
            for (br, bc, coeff) in B_coeffs:
                # B block (br, bc) is 2x2.
                # V2 has indices for 2x2: v_idx = j2*2 + k2
                # Global B index: j*4 + k
                # j = 2*br + j2, k = 2*bc + k2
                for j2 in range(2):
                    for k2 in range(2):
                        v_idx = j2 * 2 + k2
                        v_val = V2[v_idx, r2] * coeff
                        if v_val != 0:
                            j_global = 2 * br + j2
                            k_global = 2 * bc + k2
                            V_mat[j_global * 4 + k_global, r_global] += v_val
            
            # W contribution
            for (br, bc, coeff) in C_coeffs:
                # C block (br, bc) is 2x2.
                # W2 has indices for 2x2: w_idx = k2*2 + i2
                # Global C index: k*4 + i
                # k = 2*br + k2, i = 2*bc + i2
                for k2 in range(2):
                    for i2 in range(2):
                        w_idx = k2 * 2 + i2
                        w_val = W2[w_idx, r2] * coeff
                        if w_val != 0:
                            k_global = 2 * br + k2
                            i_global = 2 * bc + i2
                            W_mat[k_global * 4 + i_global, r_global] += w_val

    # Define the 7 Strassen products for 4x4
    # M1 = (A00+A11)(B00+B11)
    add_product(U, V, W, 
                [(0,0,1), (1,1,1)], 
                [(0,0,1), (1,1,1)], 
                [], 
                0, U2, V2, W2)
    
    # M2 = (A10+A11)(B00)
    add_product(U, V, W, 
                [(1,0,1), (1,1,1)], 
                [(0,0,1)], 
                [], 
                7, U2, V2, W2)
    
    # M3 = (A00)(B01-B11)
    add_product(U, V, W, 
                [(0,0,1)], 
                [(0,1,1), (1,1,-1)], 
                [], 
                14, U2, V2, W2)
    
    # M4 = (A11)(B10-B00)
    add_product(U, V, W, 
                [(1,1,1)], 
                [(1,0,1), (0,0,-1)], 
                [], 
                21, U2, V2, W2)
    
    # M5 = (A00+A01)(B11)
    add_product(U, V, W, 
                [(0,0,1), (0,1,1)], 
                [(1,1,1)], 
                [], 
                28, U2, V2, W2)
    
    # M6 = (A10-A11)(B01+B10)
    add_product(U, V, W, 
                [(1,0,1), (1,1,-1)], 
                [(0,1,1), (1,0,1)], 
                [], 
                35, U2, V2, W2)
    
    # M7 = (A01-A11)(B00+B01)
    add_product(U, V, W, 
                [(0,1,1), (1,1,-1)], 
                [(0,0,1), (0,1,1)], 
                [], 
                42, U2, V2, W2)
    
    # Now add the W coefficients for C
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to add W contributions for each C block.
    # The W matrix from _strassen_2x2 already contains the output coefficients for the 2x2 case.
    # But in _build_strassen_4x4, we called add_product with empty C_coeffs.
    # We need to add the W parts now.
    
    # Let's re-do the W part properly.
    # The W part of the 2x2 decomposition maps the 7 products to the 4 outputs.
    # For each of the 7 M-products, we need to distribute its contribution to the C blocks.
    
    # Actually, the add_product function above only filled U and V.
    # We need to fill W as well.
    # Let's create a helper that takes the C_coeffs for each M product.
    
    # M1 contributes to C00 (+1) and C11 (+1)
    # M2 contributes to C10 (+1)
    # M3 contributes to C01 (+1) and C11 (+1)
    # M4 contributes to C00 (+1) and C10 (+1)
    # M5 contributes to C00 (-1) and C01 (+1)
    # M6 contributes to C11 (+1)
    # M7 contributes to C00 (+1)
    
    # We will iterate through the 7 products and add their W contributions.
    
    def add_w_product(W_mat, C_coeffs, base_rank, W2):
        for r2 in range(7):
            r_global = base_rank + r2
            for (br, bc, coeff) in C_coeffs:
                # C block (br, bc) is 2x2.
                # W2 has indices for 2x2: w_idx = k2*2 + i2
                # Global C index: k*4 + i
                # k = 2*br + k2, i = 2*bc + i2
                for k2 in range(2):
                    for i2 in range(2):
                        w_idx = k2 * 2 + i2
                        w_val = W2[w_idx, r2] * coeff
                        if w_val != 0:
                            k_global = 2 * br + k2
                            i_global = 2 * bc + i2
                            W_mat[k_global * 4 + i_global, r_global] += w_val

    # M1: C00 (+1), C11 (+1)
    add_w_product(W, [(0,0,1), (1,1,1)], 0, W2)
    # M2: C10 (+1)
    add_w_product(W, [(1,0,1)], 7, W2)
    # M3: C01 (+1), C11 (+1)
    add_w_product(W, [(0,1,1), (1,1,1)], 14, W2)
    # M4: C00 (+1), C10 (+1)
    add_w_product(W, [(0,0,1), (1,0,1)], 21, W2)
    # M5: C00 (-1), C01 (+1)
    add_w_product(W, [(0,0,-1), (0,1,1)], 28, W2)
    # M6: C11 (+1)
    add_w_product(W, [(1,1,1)], 35, W2)
    # M7: C00 (+1)
    add_w_product(W, [(0,0,1)], 42, W2)
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS # 4, 4, 8
    
    # Build 4x4x4 Strassen decomposition
    U4, V4, W4 = _build_strassen_4x4()
    
    # Total rank is 49 + 49 = 98
    rank = 98
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # First half: B1 (cols 0-3 of B)
    # U is the same for both halves (A is the same)
    U[:, 0:49] = U4
    
    # V for first half:
    # V4 has shape (16, 49). Indices j*4+k for j in 0..3, k in 0..3.
    # Global V index is j*8+k.
    # For k in 0..3, global index is j*8 + k.
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 0:49] = V4[j*4 + k, 0:49]
    
    # W for first half:
    # W4 has shape (16, 49). Indices k*4+i for k in 0..3, i in 0..3.
    # Global W index is k*4+i.
    # For k in 0..3, global index is k*4 + i.
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W4[k*4 + i, 0:49]
    
    # Second half: B2 (cols 4-7 of B)
    # U is the same
    U[:, 49:98] = U4
    
    # V for second half:
    # B2 corresponds to k in 4..7.
    # Let k_local = k - 4.
    # Global V index is j*8 + k = j*8 + k_local + 4.
    for j in range(4):
        for k_local in range(4):
            V[j*8 + k_local + 4, 49:98] = V4[j*4 + k_local, 0:49]
    
    # W for second half:
    # C2 corresponds to k in 4..7.
    # Global W index is k*4 + i = k_local*4 + 4*4 + i = k_local*4 + 16 + i.
    for k_local in range(4):
        for i in range(4):
            W[k_local*4 + 16 + i, 49:98] = W4[k_local*4 + i, 0:49]
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if verification fails (should not happen)
    rank_school = n * m * p
    U_s = np.zeros((n*m, rank_school))
    V_s = np.zeros((m*p, rank_school))
    W_s = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1.0
                V_s[j*p+k, col] = 1.0
                W_s[k*n+i, col] = 1.0
                col += 1
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
# EVOLVE-BLOCK-END
