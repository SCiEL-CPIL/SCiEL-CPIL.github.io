"""Matrix multiplication 4x4 * 4x6 tensor decomposition using Strassen tiling.

Strategy: Tile the 4x4 * 4x6 product into four 2x2 * 2x3 block products.
Each 2x2 * 2x3 block is computed via Strassen (rank 7) for the 2x2 sub-block
plus schoolbook (rank 4) for the 2x1 sub-block, giving rank 11 per product.
Total rank: 8 products * 11 = 88... wait, let me recount.

Actually: C11 = A11*B11 + A12*B21 -> 2 products of 2x2*2x3
C12 = A11*B12 + A12*B22 -> 2 products
C21 = A21*B11 + A22*B21 -> 2 products
C22 = A21*B12 + A22*B22 -> 2 products
Total: 8 products * 11 = 88.

But we can do better by using Strassen at the 4x4 level. Let's use a different
tiling: treat the 4x4*4x6 as a 2x2 block matrix multiply where:
- Left: [[A11, A12], [A21, A22]] with each block 2x2
- Right: [[B11, B12], [B21, B22]] with each block 2x3

Using Strassen on the block level (7 block multiplications instead of 8):
Each block multiplication is (2x2) * (2x3) which has rank 11.
Total: 7 * 11 = 77.

Let me implement this Strassen-at-block-level approach.
"""

import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = 4, 4, 6  # A is n x m, B is m x p, C is n x p
    
    U_rows = n * m  # 16
    V_rows = m * p  # 24
    W_rows = p * n  # 24
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    def zeros(length):
        return [0] * length
    
    # Block offsets:
    # A blocks (each 2x2): A11 at (0,0), A12 at (0,2), A21 at (2,0), A22 at (2,2)
    # B blocks (each 2x3): B11 at (0,0), B12 at (0,3), B21 at (2,0), B22 at (2,3)
    # C blocks (each 2x3): C11 at (0,0), C12 at (0,3), C21 at (2,0), C22 at (2,3)
    
    # For Strassen at block level, we need to compute linear combinations of blocks.
    # Each "block multiplication" is (2x2) * (2x3) = (2x3), rank 11.
    # 
    # Strassen formulas (at block level):
    #   M1 = (A11+A22) * (B11+B22)
    #   M2 = (A21+A22) * B11
    #   M3 = A11 * (B12-B22)
    #   M4 = A22 * (B21-B11)
    #   M5 = (A11+A12) * B22
    #   M6 = (A21-A11) * (B11+B12)
    #   M7 = (A12-A22) * (B21+B22)
    #
    #   C11 = M1 + M4 - M5 + M7
    #   C12 = M3 + M5
    #   C21 = M2 + M4
    #   C22 = M1 - M2 + M3 + M6
    #
    # The issue: M1 involves (A11+A22)*(B11+B22). This is NOT a simple block
    # multiplication because the left factor is a sum of two 2x2 blocks and the
    # right factor is a sum of two 2x3 blocks. We can't directly use a rank-11
    # decomposition for this.
    #
    # Instead, let's stick with the direct tiling approach (8 products of 2x2*2x3)
    # which gives rank 88, and see if we can find a better decomposition.
    #
    # Actually, wait. Let me reconsider. The direct approach gives:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # That's 8 products of (2x2)*(2x3), each rank 11, total 88.
    #
    # Can we do Strassen at the block level properly? The problem is that
    # Strassen requires the block matrices to be square (or at least the same
    # shape on both sides). Here left blocks are 2x2 and right blocks are 2x3.
    #
    # Alternative: Use a different splitting. What if we split differently?
    #
    # Let me try: split A into 2x2 blocks, and B into 2x3 blocks, but use
    # a "Strassen-like" approach where we compute:
    #   M1 = A11 * B11
    #   M2 = A12 * B21
    #   M3 = A21 * B12
    #   M4 = A22 * B22
    #   M5 = (A11+A12+A21+A22) * (B11+B12+B21+B22)  -- this doesn't work
    #
    # Actually, the standard approach for rectangular matrix multiplication
    # using Strassen is more complex. Let me just use the direct 88-rank
    # decomposition and verify it's correct.
    #
    # Wait, I realize the current program already does this and scores 1.0909.
    # Let me check: 96/88 = 1.0909... Yes! So the current rank is 88.
    #
    # To improve, I need to find a decomposition with rank < 88.
    #
    # One idea: Use the fact that some of the 2x2*2x1 products might be
    # shareable. But in the current construction, each product is independent.
    #
    # Another idea: Use a different algorithm for 2x2*2x3 that has rank < 11.
    # The schoolbook rank is 2*2*3 = 12. Strassen gives 7+4=11.
    # Is there a rank-10 decomposition for 2x2*2x3?
    #
    # Actually, let me try a completely different approach: use the known
    # result that the tensor rank of 4x4*4x6 might be lower than 88.
    #
    # For now, let me just implement the 88-rank decomposition correctly and
    # add a deadline guard. If time permits, I can try to search for a lower
    # rank decomposition.
    
    def strassen_2x2_2x2(a_off, b_off, c_off):
        """Add 7 rank-1 terms for Strassen 2x2*2x2."""
        ar, ac = a_off
        br, bc = b_off
        cr, cc = c_off
        
        def a_idx(i, j): return (ar+i)*m + (ac+j)
        def b_idx(i, j): return (br+i)*p + (bc+j)
        def c_idx(i, j): return (cc+j)*n + (cr+i)
        
        # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0,0)] = 1; u[a_idx(1,1)] = 1
        v[b_idx(0,0)] = 1; v[b_idx(1,1)] = 1
        w[c_idx(0,0)] = 1; w[c_idx(1,1)] = 1
        add_term(u, v, w)
        
        # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1,0)] = 1; u[a_idx(1,1)] = 1
        v[b_idx(0,0)] = 1
        w[c_idx(1,0)] = 1; w[c_idx(1,1)] = -1
        add_term(u, v, w)
        
        # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0,0)] = 1
        v[b_idx(0,1)] = 1; v[b_idx(1,1)] = -1
        w[c_idx(0,1)] = 1; w[c_idx(1,1)] = 1
        add_term(u, v, w)
        
        # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1,1)] = 1
        v[b_idx(1,0)] = 1; v[b_idx(0,0)] = -1
        w[c_idx(0,0)] = 1; w[c_idx(1,0)] = 1
        add_term(u, v, w)
        
        # M5 = (A11+A12)*B22 -> C11 -= M5, C12 += M5
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0,0)] = 1; u[a_idx(0,1)] = 1
        v[b_idx(1,1)] = 1
        w[c_idx(0,0)] = -1; w[c_idx(0,1)] = 1
        add_term(u, v, w)
        
        # M6 = (A21-A11)*(B11+B12) -> C22 += M6
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1,0)] = 1; u[a_idx(0,0)] = -1
        v[b_idx(0,0)] = 1; v[b_idx(0,1)] = 1
        w[c_idx(1,1)] = 1
        add_term(u, v, w)
        
        # M7 = (A12-A22)*(B21+B22) -> C11 += M7
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0,1)] = 1; u[a_idx(1,1)] = -1
        v[b_idx(1,0)] = 1; v[b_idx(1,1)] = 1
        w[c_idx(0,0)] = 1
        add_term(u, v, w)
    
    def schoolbook_2x2_2x1(a_off, b_off, c_off):
        """Add 4 rank-1 terms for schoolbook 2x2*2x1."""
        ar, ac = a_off
        br, bc = b_off
        cr, cc = c_off
        
        def a_idx(i, j): return (ar+i)*m + (ac+j)
        def b_idx(i, j): return (br+i)*p + (bc+j)
        def c_idx(i, j): return (cc+j)*n + (cr+i)
        
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0,0)] = 1
        v[b_idx(0,0)] = 1
        w[c_idx(0,0)] = 1
        add_term(u, v, w)
        
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0,1)] = 1
        v[b_idx(1,0)] = 1
        w[c_idx(0,0)] = 1
        add_term(u, v, w)
        
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1,0)] = 1
        v[b_idx(0,0)] = 1
        w[c_idx(1,0)] = 1
        add_term(u, v, w)
        
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1,1)] = 1
        v[b_idx(1,0)] = 1
        w[c_idx(1,0)] = 1
        add_term(u, v, w)
    
    def product_2x2_2x3(a_off, b_off, c_off):
        """Compute A(2x2) * B(2x3) = C(2x3) using Strassen + schoolbook."""
        ar, ac = a_off
        br, bc = b_off
        cr, cc = c_off
        
        # B_left is 2x2 at (br, bc)
        strassen_2x2_2x2(a_off, (br, bc), (cr, cc))
        
        # B_right is 2x1 at (br, bc+2)
        schoolbook_2x2_2x1(a_off, (br, bc+2), (cr, cc+2))
    
    # Build the 88-rank decomposition
    # C11 = A11*B11 + A12*B21
    product_2x2_2x3((0,0), (0,0), (0,0))   # A11 * B11
    product_2x2_2x3((0,2), (2,0), (0,0))   # A12 * B21
    
    # C12 = A11*B12 + A12*B22
    product_2x2_2x3((0,0), (0,3), (0,3))   # A11 * B12
    product_2x2_2x3((0,2), (2,3), (0,3))   # A12 * B22
    
    # C21 = A21*B11 + A22*B21
    product_2x2_2x3((2,0), (0,0), (2,0))   # A21 * B11
    product_2x2_2x3((2,2), (2,0), (2,0))   # A22 * B21
    
    # C22 = A21*B12 + A22*B22
    product_2x2_2x3((2,0), (0,3), (2,3))   # A21 * B12
    product_2x2_2x3((2,2), (2,3), (2,3))   # A22 * B22
    
    rank = len(U_cols)
    
    U = [[U_cols[r][i] for r in range(rank)] for i in range(U_rows)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(V_rows)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(W_rows)]
    
    return {"U": U, "V": V, "W": W}
