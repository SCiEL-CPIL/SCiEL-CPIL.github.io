import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 4]

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    if len(U) == 0:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return _verify_full(U, V, W)

def _verify_full(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    # result[u,v,w] = sum_c U[u,c]*V[v,c]*W[w,c]
    result = np.einsum('uc,vc,wc->uvw', U_arr, V_arr, W_arr)
    
    if np.max(np.abs(result - T)) > 1e-9:
        return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank

def _block_partition_search():
    """
    Partition the 3x4 matrix A and 4x4 matrix B into overlapping 2x2 blocks.
    A is 3x4: split rows into [0:2] and [1:3], cols into [0:2] and [2:4]
    B is 4x4: split rows into [0:2] and [2:4], cols into [0:2] and [2:4]
    
    The product C = A*B is 3x4.
    We use 2x2 block structure where each block multiplication of 2x2 by 2x2
    can be done with 7 multiplications (Strassen) instead of 8.
    
    Block layout:
    A = [[A11, A12],     B = [[B11, B12],
         [A21, A22]]          [B21, B22]]
    
    where A11 is 2x2 (rows 0-1, cols 0-1)
          A12 is 2x2 (rows 0-1, cols 2-3)
          A21 is 1x2 (row 2, cols 0-1)  -- note: A is 3x4, so bottom row has 2 cols
          A22 is 1x2 (row 2, cols 2-3)
    
    B11 is 2x2 (rows 0-1, cols 0-1)
    B12 is 2x2 (rows 0-1, cols 2-3)
    B21 is 2x2 (rows 2-3, cols 0-1)
    B22 is 2x2 (rows 2-3, cols 2-3)
    
    C = [[C11, C12],
         [C21, C22]]
    
    C11 = A11*B11 + A12*B21  (2x2 result)
    C12 = A11*B12 + A12*B22  (2x2 result)
    C21 = A21*B11 + A22*B21  (1x2 result)
    C22 = A21*B12 + A22*B22  (1x2 result)
    
    For the 2x2 block multiplications, we can use Strassen (7 mults each).
    For the 1x2 by 2x2 multiplications, we need 2 mults each (each element of 1x2
    times each element of 2x2, but actually it's a dot product per column).
    
    Actually, let me reconsider. The A21 and A22 are 1x2 vectors, and B21, B22 are 2x2.
    A21 * B11 means (1x2) * (2x2) = 1x2. This requires 2 multiplications per column
    of the result, so 2*2 = 4 mults total for A21*B11.
    
    Let me count more carefully:
    - A11*B11: 2x2 * 2x2 = 2x2, needs 8 schoolbook or 7 Strassen
    - A12*B21: 2x2 * 2x2 = 2x2, needs 8 or 7
    - A11*B12: 2x2 * 2x2 = 2x2, needs 8 or 7
    - A12*B22: 2x2 * 2x2 = 2x2, needs 8 or 7
    - A21*B11: 1x2 * 2x2 = 1x2, needs 4 mults (2 per output element)
    - A22*B21: 1x2 * 2x2 = 1x2, needs 4 mults
    - A21*B12: 1x2 * 2x2 = 1x2, needs 4 mults
    - A22*B22: 1x2 * 2x2 = 1x2, needs 4 mults
    
    Total schoolbook: 4*8 + 4*4 = 32 + 16 = 48 (matches n*m*p = 3*4*4 = 48)
    
    With Strassen for the four 2x2*2x2 products: 4*7 + 4*4 = 28 + 16 = 44
    
    So we can potentially get 44 with this block structure!
    
    Let me implement this properly.
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # We'll build the decomposition using Strassen for 2x2 blocks
    # and direct multiplication for 1x2 by 2x2 blocks.
    
    # Strassen for 2x2:
    # Given P (2x2) and Q (2x2), compute R = P*Q
    # P = [[p00, p01],     Q = [[q00, q01],
    #      [p10, p11]]          [q10, q11]]
    #
    # M1 = (p00 + p11) * (q00 + q11)
    # M2 = (p10 + p11) * q00
    # M3 = p00 * (q01 - q11)
    # M4 = p11 * (q10 - q00)
    # M5 = (p00 + p01) * q11
    # M6 = (p10 - p00) * (q00 + q01)
    # M7 = (p01 - p11) * (q10 + q11)
    #
    # R00 = M1 + M4 - M5 + M7
    # R01 = M3 + M5
    # R10 = M2 + M4
    # R11 = M1 - M2 + M3 + M6
    
    # Each Strassen multiplication uses 7 scalar multiplications.
    # In our tensor decomposition, each "multiplication" corresponds to one rank-1 term.
    
    # For the 1x2 * 2x2 case:
    # v (1x2) = [v0, v1], M (2x2) = [[m00, m01], [m10, m11]]
    # result = [v0*m00 + v1*m10, v0*m01 + v1*m11]
    # This is 4 scalar multiplications, each contributing to one output.
    # So we need 4 rank-1 terms.
    
    # Let's enumerate all the rank-1 terms needed.
    
    terms = []  # Each term is (u_indices, v_indices, w_indices) where each is a dict mapping position->coefficient
    
    # Helper: add a Strassen-based 2x2*2x2 product
    # a_block: 2x2 submatrix of A, specified by (row_start, col_start)
    # b_block: 2x2 submatrix of B, specified by (row_start, col_start)
    # The result contributes to C at (row_start, col_start) with a 2x2 block
    # But we need to handle the fact that A is 3x4, so some "2x2" blocks are actually 1x2
    
    def add_2x2_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """Add rank-1 terms for Strassen multiplication of A[a_rows, a_cols] * B[b_rows, b_cols] -> C[c_rows, c_cols]"""
        # a_rows, a_cols, b_rows, b_cols, c_rows, c_cols are lists of indices
        # For 2x2: len=2 for each
        # For 1x2: a_rows has len 1, a_cols has len 2, etc.
        
        # We handle the general case by decomposing into scalar products
        # Actually, let me just do direct multiplication for simplicity first,
        # then optimize with Strassen where applicable.
        pass
    
    # Let me take a different approach: directly enumerate all the scalar multiplications
    # needed, then try to combine them using Strassen identities.
    
    # Actually, the cleanest approach: build the full tensor decomposition by
    # enumerating all (i,j,k) triples and then apply Strassen to reduce.
    
    # For now, let me implement the block structure with Strassen for the 2x2*2x2 parts.
    
    # The four 2x2*2x2 products:
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # 2. A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # 3. A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # 4. A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    # The four 1x2*2x2 products:
    # 5. A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    # 6. A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    # 7. A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4]
    # 8. A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4]
    
    # For products 1-4, use Strassen (7 mults each) = 28 mults
    # For products 5-8, use direct (4 mults each) = 16 mults
    # Total: 44
    
    U = [[0] * 44 for _ in range(n * m)]
    V = [[0] * 44 for _ in range(m * p)]
    W = [[0] * 44 for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        """Add 7 rank-1 terms for Strassen multiplication of 2x2 blocks.
        A block: rows [a_r0, a_r0+1], cols [a_c0, a_c0+1]
        B block: rows [b_r0, b_r0+1], cols [b_c0, b_c0+1]
        C block: rows [c_r0, c_r0+1], cols [c_c0, c_c0+1]
        """
        nonlocal col
        # P indices in A: (a_r0, a_c0)=p00, (a_r0, a_c0+1)=p01, (a_r0+1, a_c0)=p10, (a_r0+1, a_c0+1)=p11
        # Q indices in B: (b_r0, b_c0)=q00, (b_r0, b_c0+1)=q01, (b_r0+1, b_c0)=q10, (b_r0+1, b_c0+1)=q11
        # R indices in C: (c_r0, c_c0)=r00, (c_r0, c_c0+1)=r01, (c_r0+1, c_c0)=r10, (c_r0+1, c_c0+1)=r11
        
        def A_idx(r, c):
            return r * m + c
        
        def B_idx(r, c):
            return r * p + c
        
        def C_idx(r, c):
            return r * n + c
        
        # M1 = (p00 + p11) * (q00 + q11)
        # U contribution: A[a_r0, a_c0] and A[a_r0+1, a_c0+1] get coefficient 1
        # V contribution: B[b_r0, b_c0] and B[b_r0+1, b_c0+1] get coefficient 1
        # W contribution: C[c_r0, c_c0] += 1, C[c_r0, c_c0+1] += 0, C[c_r0+1, c_c0] += 0, C[c_r0+1, c_c0+1] += 1
        # Wait, R00 = M1 + M4 - M5 + M7, so M1 contributes +1 to R00 and +1 to R11
        
        # Let me be more careful. In Strassen:
        # M1 = (P00+P11)(Q00+Q11)
        # M2 = (P10+P11)Q00
        # M3 = P00(Q01-Q11)
        # M4 = P11(Q10-Q00)
        # M5 = (P00+P01)Q11
        # M6 = (P10-P00)(Q00+Q01)
        # M7 = (P01-P11)(Q10+Q11)
        #
        # R00 = M1 + M4 - M5 + M7
        # R01 = M3 + M5
        # R10 = M2 + M4
        # R11 = M1 - M2 + M3 + M6
        
        # Each M is a product of a linear combination of A entries and a linear combination of B entries.
        # In the tensor decomposition, each M corresponds to one rank-1 term where:
        # U has coefficients for the A entries involved
        # V has coefficients for the B entries involved
        # W has coefficients for the C entries that this M contributes to
        
        # M1: A coeffs: A[a_r0,a_c0]=1, A[a_r0+1,a_c0+1]=1; B coeffs: B[b_r0,b_c0]=1, B[b_r0+1,b_c0+1]=1
        #     C coeffs: C[c_r0,c_c0]=1, C[c_r0+1,c_c0+1]=1
        U[A_idx(a_r0, a_c0)][col] += 1
        U[A_idx(a_r0+1, a_c0+1)][col] += 1
        V[B_idx(b_r0, b_c0)][col] += 1
        V[B_idx(b_r0+1, b_c0+1)][col] += 1
        W[C_idx(c_r0, c_c0)][col] += 1
        W[C_idx(c_r0+1, c_c0+1)][col] += 1
        col += 1
        
        # M2: A coeffs: A[a_r0+1,a_c0]=1, A[a_r0+1,a_c0+1]=1; B coeffs: B[b_r0,b_c0]=1
        #     C coeffs: C[c_r0+1,c_c0]=1, C[c_r0+1,c_c0+1]=-1
        U[A_idx(a_r0+1, a_c0)][col] += 1
        U[A_idx(a_r0+1, a_c0+1)][col] += 1
        V[B_idx(b_r0, b_c0)][col] += 1
        W[C_idx(c_r0+1, c_c0)][col] += 1
        W[C_idx(c_r0+1, c_c0+1)][col] -= 1
        col += 1
        
        # M3: A coeffs: A[a_r0,a_c0]=1; B coeffs: B[b_r0,b_c0+1]=1, B[b_r0+1,b_c0+1]=-1
        #     C coeffs: C[c_r0,c_c0+1]=1, C[c_r0+1,c_c0+1]=1
        U[A_idx(a_r0, a_c0)][col] += 1
        V[B_idx(b_r0, b_c0+1)][col] += 1
        V[B_idx(b_r0+1, b_c0+1)][col] -= 1
        W[C_idx(c_r0, c_c0+1)][col] += 1
        W[C_idx(c_r0+1, c_c0+1)][col] += 1
        col += 1
        
        # M4: A coeffs: A[a_r0+1,a_c0+1]=1; B coeffs: B[b_r0+1,b_c0]=1, B[b_r0,b_c0]=-1
        #     C coeffs: C[c_r0,c_c0]=1, C[c_r0+1,c_c0]=1
        U[A_idx(a_r0+1, a_c0+1)][col] += 1
        V[B_idx(b_r0+1, b_c0)][col] += 1
        V[B_idx(b_r0, b_c0)][col] -= 1
        W[C_idx(c_r0, c_c0)][col] += 1
        W[C_idx(c_r0+1, c_c0)][col] += 1
        col += 1
        
        # M5: A coeffs: A[a_r0,a_c0]=1, A[a_r0,a_c0+1]=1; B coeffs: B[b_r0+1,b_c0+1]=1
        #     C coeffs: C[c_r0,c_c0]=-1, C[c_r0,c_c0+1]=1
        U[A_idx(a_r0, a_c0)][col] += 1
        U[A_idx(a_r0, a_c0+1)][col] += 1
        V[B_idx(b_r0+1, b_c0+1)][col] += 1
        W[C_idx(c_r0, c_c0)][col] -= 1
        W[C_idx(c_r0, c_c0+1)][col] += 1
        col += 1
        
        # M6: A coeffs: A[a_r0+1,a_c0]=1, A[a_r0,a_c0]=-1; B coeffs: B[b_r0,b_c0]=1, B[b_r0,b_c0+1]=1
        #     C coeffs: C[c_r0+1,c_c0+1]=1
        U[A_idx(a_r0+1, a_c0)][col] += 1
        U[A_idx(a_r0, a_c0)][col] -= 1
        V[B_idx(b_r0, b_c0)][col] += 1
        V[B_idx(b_r0, b_c0+1)][col] += 1
        W[C_idx(c_r0+1, c_c0+1)][col] += 1
        col += 1
        
        # M7: A coeffs: A[a_r0,a_c0+1]=1, A[a_r0+1,a_c0+1]=-1; B coeffs: B[b_r0+1,b_c0]=1, B[b_r0+1,b_c0+1]=1
        #     C coeffs: C[c_r0,c_c0]=1
        U[A_idx(a_r0, a_c0+1)][col] += 1
        U[A_idx(a_r0+1, a_c0+1)][col] -= 1
        V[B_idx(b_r0+1, b_c0)][col] += 1
        V[B_idx(b_r0+1, b_c0+1)][col] += 1
        W[C_idx(c_r0, c_c0)][col] += 1
        col += 1
    
    def add_direct_1x2(a_r, a_c0, b_r0, b_c0, c_r, c_c0):
        """Add 4 rank-1 terms for direct multiplication of 1x2 * 2x2 -> 1x2.
        A entry: A[a_r, a_c0] and A[a_r, a_c0+1]
        B block: rows [b_r0, b_r0+1], cols [b_c0, b_c0+1]
        C entries: C[c_r, c_c0] and C[c_r, c_c0+1]
        """
        nonlocal col
        # C[c_r, c_c0] = A[a_r, a_c0]*B[b_r0, b_c0] + A[a_r, a_c0+1]*B[b_r0+1, b_c0]
        # C[c_r, c_c0+1] = A[a_r, a_c0]*B[b_r0, b_c0+1] + A[a_r, a_c0+1]*B[b_r0+1, b_c0+1]
        
        # Term 1: A[a_r,a_c0] * B[b_r0,b_c0] -> C[c_r,c_c0]
        U[a_r * m + a_c0][col] += 1
        V[b_r0 * p + b_c0][col] += 1
        W[c_r * n + c_c0][col] += 1
        col += 1
        
        # Term 2: A[a_r,a_c0+1] * B[b_r0+1,b_c0] -> C[c_r,c_c0]
        U[a_r * m + (a_c0+1)][col] += 1
        V[(b_r0+1) * p + b_c0][col] += 1
        W[c_r * n + c_c0][col] += 1
        col += 1
        
        # Term 3: A[a_r,a_c0] * B[b_r0,b_c0+1] -> C[c_r,c_c0+1]
        U[a_r * m + a_c0][col] += 1
        V[b_r0 * p + (b_c0+1)][col] += 1
        W[c_r * n + (c_c0+1)][col] += 1
        col += 1
        
        # Term 4: A[a_r,a_c0+1] * B[b_r0+1,b_c0+1] -> C[c_r,c_c0+1]
        U[a_r * m + (a_c0+1)][col] += 1
        V[(b_r0+1) * p + (b_c0+1)][col] += 1
        W[c_r * n + (c_c0+1)][col] += 1
        col += 1
    
    # Product 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_2x2(0, 0, 0, 0, 0, 0)
    
    # Product 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_2x2(0, 2, 2, 0, 0, 0)
    
    # Product 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_2x2(0, 0, 0, 2, 0, 2)
    
    # Product 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_2x2(0, 2, 2, 2, 0, 2)
    
    # Product 5: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    add_direct_1x2(2, 0, 0, 0, 2, 0)
    
    # Product 6: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    add_direct_1x2(2, 2, 2, 0, 2, 0)
    
    # Product 7: A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4]
    add_direct_1x2(2, 0, 0, 2, 2, 2)
    
    # Product 8: A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4]
    add_direct_1x2(2, 2, 2, 2, 2, 2)
    
    return U, V, W, col

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school, rank_school = _schoolbook()
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try block partition with Strassen
    if time.time() < deadline:
        U_block, V_block, W_block, rank_block = _block_partition_search()
        if _verify_full(U_block, V_block, W_block):
            best_U, best_V, best_W = U_block, V_block, W_block
            best_rank = rank_block
    
    # Try to reduce rank by finding linear dependencies among terms
    if time.time() < deadline:
        U_red, V_red, W_red, rank_red = _try_reduce_rank(best_U, best_V, best_W, deadline)
        if rank_red < best_rank:
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = rank_red
    
    # Try an alternative: use 4x4 Strassen-like decomposition on B
    # 4x4 matrix multiplication can be done with 49 mults (instead of 64)
    # using recursive Strassen: 4x4 = 2*(2x2) blocks, each 2x2*2x2 with 7 mults
    # Total for 4x4*4x4: 7*7 = 49 mults for the 4x4 product
    # But we have 3x4 * 4x4, not 4x4 * 4x4
    #
    # Let's try: treat A as having 3 rows, each 1x4
    # For each row a (1x4), compute a * B (4x4) = 1x4
    # We can split a into [a1, a2] where a1, a2 are 1x2
    # And B into [[B11, B12], [B21, B22]] where each is 2x2
    # a * B = [a1*B11 + a2*B21, a1*B12 + a2*B22]
    # Each 1x2 * 2x2 = 1x2 needs 4 scalar mults
    # So per row: 4 + 4 + 4 + 4 = 16 mults
    # Total: 3 * 16 = 48. Same as schoolbook.
    #
    # Alternative: use the 49-mult 4x4*4x4 algorithm
    # If we pad A to 4x4 by adding a zero row, then compute 4x4*4x4 with 49 mults,
    # we get rank 49 which is worse than 44.
    #
    # Let's try a different partition: split A into [1x4; 1x4; 1x4] 
    # and use a smarter B decomposition.
    #
    # Actually, let's try: split A into [2x4; 1x4] but use a different B split
    # B = [B_left 2x4, B_right 2x4] where B_left is 4x2, B_right is 4x2
    # A_top (2x4) * B_left (4x2) = 2x2, needs 8 or 7 mults
    # A_top (2x4) * B_right (4x2) = 2x2, needs 8 or 7 mults
    # A_bottom (1x4) * B_left (4x2) = 1x2, needs 4 mults
    # A_bottom (1x4) * B_right (4x2) = 1x2, needs 4 mults
    # Total: 7 + 7 + 4 + 4 = 22? No, that's not right.
    # A is 3x4, B is 4x4.
    # A = [A_top 2x4; A_bottom 1x4]
    # B = [B11 2x2, B12 2x2; B21 2x2, B22 2x2]
    # C = [C_top 2x4; C_bottom 1x4]
    # C_top = A_top * B = [A_top*B11 + A_top*B21 (left 2x2), A_top*B12 + A_top*B22 (right 2x2)]
    #   Wait, A_top is 2x4, B11 is 2x2. A_top*B11 doesn't make sense dimensionally.
    # A_top (2x4) * B (4x4) = 2x4
    # = [A_top[:,0:2]*B[0:2,0:2] + A_top[:,2:4]*B[2:4,0:2] (left 2x2),
    #    A_top[:,0:2]*B[0:2,2:4] + A_top[:,2:4]*B[2:4,2:4] (right 2x2)]
    # Each 2x2*2x2 = 2x2 needs 7 mults with Strassen
    # So C_top needs 7 + 7 = 14 mults
    # C_bottom (1x4) = A_bottom (1x4) * B (4x4)
    # = [A_bottom[0:2]*B[0:2,0:2] + A_bottom[2:4]*B[2:4,0:2] (left 1x2),
    #    A_bottom[0:2]*B[0:2,2:4] + A_bottom[2:4]*B[2:4,2:4] (right 1x2)]
    # Each 1x2*2x2 = 1x2 needs 4 mults
    # So C_bottom needs 4 + 4 = 8 mults
    # Total: 14 + 8 = 22? That can't be right since schoolbook is 48.
    #
    # I'm confusing myself. Let me just stick with the 44-mult solution.
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _try_reduce_rank(U, V, W, deadline):
    """Try to reduce rank by finding linear dependencies among rank-1 terms.
    
    Key insight: If two terms (c1, c2) have U[:,c1] = α*U[:,c2], 
    V[:,c1] = β*V[:,c2], W[:,c1] = γ*W[:,c2] with α*β*γ = 1,
    then the two terms are identical and one can be removed.
    
    More generally, if we can find a linear dependency where the 
    product of coefficients equals 1 for all matching entries, we can merge.
    
    Practical approach: check all pairs for proportional columns.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    def find_proportional_pair(U1, V1, W1):
        """Find a pair of columns that are proportional in all three matrices."""
        r = U1.shape[1]
        for i in range(r):
            if time.time() > deadline - 10:
                return None
            for j in range(i + 1, r):
                # Check if U[:,i] and U[:,j] are proportional
                u_i = U1[:, i]
                u_j = U1[:, j]
                if np.max(np.abs(u_i)) < 1e-12 or np.max(np.abs(u_j)) < 1e-12:
                    continue
                
                # Find ratio
                nonzero = np.abs(u_i) > 1e-12
                if np.sum(nonzero) == 0:
                    continue
                alpha = u_i[nonzero][0] / u_j[nonzero][0]
                if np.max(np.abs(u_i - alpha * u_j)) > 1e-8:
                    continue
                
                # Check V
                v_i = V1[:, i]
                v_j = V1[:, j]
                nonzero_v = np.abs(v_i) > 1e-12
                if np.sum(nonzero_v) == 0:
                    continue
                beta = v_i[nonzero_v][0] / v_j[nonzero_v][0]
                if np.max(np.abs(v_i - beta * v_j)) > 1e-8:
                    continue
                
                # Check W
                w_i = W1[:, i]
                w_j = W1[:, j]
                nonzero_w = np.abs(w_i) > 1e-12
                if np.sum(nonzero_w) == 0:
                    continue
                gamma = w_i[nonzero_w][0] / w_j[nonzero_w][0]
                if np.max(np.abs(w_i - gamma * w_j)) > 1e-8:
                    continue
                
                # Check if alpha * beta * gamma ≈ 1
                if abs(alpha * beta * gamma - 1) > 1e-6:
                    continue
                
                return (i, j, alpha, beta, gamma)
        return None
    
    # Try to find and eliminate proportional pairs
    U_new = U_arr.copy()
    V_new = V_arr.copy()
    W_new = W_arr.copy()
    kept = list(range(rank))
    
    changed = True
    while changed:
        changed = False
        pair = find_proportional_pair(
            U_new[:, kept], V_new[:, kept], W_new[:, kept])
        if pair is None:
            break
        
        i, j, alpha, beta, gamma = pair
        # Remove column j (keep i)
        # The contribution of term j is U_j⊗V_j⊗W_j = (1/alpha)U_i ⊗ (1/beta)V_i ⊗ (1/gamma)W_i
        # Since alpha*beta*gamma = 1, (1/alpha)*(1/beta)*(1/gamma) = 1
        # So term j has the same tensor contribution as term i
        # We can just remove term j
        
        kept = [k for k in kept if k != j]
        changed = True
    
    if len(kept) < rank:
        U_final = U_new[:, kept]
        V_final = V_new[:, kept]
        W_final = W_new[:, kept]
        
        # Convert back to list format
        U_list = U_final.tolist()
        V_list = V_final.tolist()
        W_list = W_final.tolist()
        
        # Verify
        if _verify_full(U_list, V_list, W_list):
            return U_list, V_list, W_list, len(kept)
    
    return U, V, W, rank
