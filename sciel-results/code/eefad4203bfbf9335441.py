"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # n=2, m=4, p=6
    # U: 8 x r
    # V: 24 x r
    # W: 12 x r
    
    # Schoolbook solution
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}, rank

    best_result, best_rank = schoolbook()
    
    if time.time() > deadline - 10:
        return best_result

    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Strategy: Construct a Strassen-like decomposition.
    # We divide the 4x6 matrix into three 2x6 blocks? No, let's divide into 2x2 blocks.
    # A is 2x4. B is 4x6.
    # Let A = [[A11, A12], [A21, A22]] where each is 1x2? No, 2x4 can be split into two 2x2 blocks horizontally.
    # A = [A_left, A_right] where A_left, A_right are 2x2.
    # B is 4x6. Split vertically into three 2x2 blocks? 4 rows / 2 = 2 blocks. 6 cols / 2 = 3 blocks.
    # So B is a 2x3 grid of 2x2 blocks.
    # B = [[B11, B12, B13], [B21, B22, B23]] where each Bxy is 2x2.
    
    # The product C = A * B is 2x6.
    # C = [A_left * B_top_left_part + A_right * B_bottom_left_part, ...]
    # Actually, standard block multiplication:
    # C = [ A_left * B_col1 + A_right * B_col2, A_left * B_col3 + A_right * B_col4, ... ]
    # Wait, B is 4x6.
    # Let's partition B into 2x2 blocks.
    # B = [ B11 B12 B13 ]
    #     [ B21 B22 B23 ]
    # A = [ A11 A12 ]  (2x4, so A11, A12 are 2x2)
    
    # C = A * B = [ A11*B11 + A12*B21, A11*B12 + A12*B22, A11*B13 + A12*B23 ]
    # Each term is a 2x2 matrix product.
    # We have 3 such terms. Each term is a sum of two 2x2 matrix products.
    # Total 6 2x2 matrix products.
    # Using Strassen, each 2x2 product takes 7 mults.
    # But we can share computations?
    # Let's define the 6 products:
    # P1 = A11 * B11
    # P2 = A12 * B21
    # P3 = A11 * B12
    # P4 = A12 * B22
    # P5 = A11 * B13
    # P6 = A12 * B23
    
    # C1 = P1 + P2
    # C2 = P3 + P4
    # C3 = P5 + P6
    
    # If we use Strassen for each of the 6 products, that's 6*7 = 42 multiplications.
    # Can we do better?
    # Notice A11 appears in P1, P3, P5. A12 appears in P2, P4, P6.
    # B11, B21, B12, B22, B13, B23 are distinct.
    # Maybe we can group them differently?
    
    # Let's try to implement the 42-rank Strassen decomposition.
    # This is a known result: rank(2x4x6) <= 42?
    # Actually, the rank of 2x2x2 is 7.
    # If we view 2x4x6 as three 2x2x2 blocks sharing the A matrix?
    # No, A is shared.
    
    # Let's construct the 42-rank decomposition explicitly.
    # We will use 7 columns for each of the 6 "virtual" 2x2 products.
    # But we can optimize the additions.
    
    # Let's just build the 42-rank solution first to ensure we beat 48.
    # Then we can try to merge.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for A (2x4)
    # A11: rows 0,1, cols 0,1 -> indices 0,1,2,3 in U (flattened row-major? No, U is (n*m, rank))
    # U row index = i*m + j.
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 2
    # i=1, j=1 -> 3
    # i=0, j=2 -> 4
    # i=0, j=3 -> 5
    # i=1, j=2 -> 6
    # i=1, j=3 -> 7
    
    # A11 (2x2): U rows 0,1,2,3 corresponding to A[0,0], A[0,1], A[1,0], A[1,1]?
    # Wait, A is 2x4.
    # A[0,0] is U[0]
    # A[0,1] is U[1]
    # A[1,0] is U[2]
    # A[1,1] is U[3]
    # A[0,2] is U[4]
    # A[0,3] is U[5]
    # A[1,2] is U[6]
    # A[1,3] is U[7]
    
    # A11 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # A12 = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
    
    # B is 4x6.
    # V row index = j*p + k.
    # j=0, k=0..5 -> V[0..5]
    # j=1, k=0..5 -> V[6..11]
    # j=2, k=0..5 -> V[12..17]
    # j=3, k=0..5 -> V[18..23]
    
    # B11 (2x2): rows 0,1, cols 0,1.
    # B[0,0]=V[0], B[0,1]=V[1]
    # B[1,0]=V[6], B[1,1]=V[7]
    
    # B21 (2x2): rows 2,3, cols 0,1.
    # B[2,0]=V[12], B[2,1]=V[13]
    # B[3,0]=V[18], B[3,1]=V[19]
    
    # And so on.
    
    # Strassen 2x2x2 decomposition:
    # Given X (2x2) and Y (2x2), Z = X*Y.
    # M1 = (X00 + X11) * (Y00 + Y11)
    # M2 = (X10 + X11) * Y00
    # M3 = X00 * (Y01 - Y11)
    # M4 = X11 * (Y10 - Y00)
    # M5 = (X00 + X01) * Y11
    # M6 = (X10 - X00) * (Y00 + Y01)
    # M7 = (X01 - X11) * (Y10 + Y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # We need to map these to our U, V, W.
    # For each "virtual" product, we allocate 7 columns.
    # Let's define a helper to add a Strassen product.
    
    def add_strassen_product(U, V, W, rank_offset, A_rows, B_rows, C_rows):
        # A_rows: indices in U for the 2x2 matrix A_block (flattened row-major: [a00, a01, a10, a11])
        # B_rows: indices in V for the 2x2 matrix B_block (flattened row-major: [b00, b01, b10, b11])
        # C_rows: indices in W for the 2x2 matrix C_block (flattened row-major: [c00, c01, c10, c11])
        # Note: W index is k*n + i.
        # C_block is 2x2.
        # C[0,0] is W[C_rows[0]]
        # C[0,1] is W[C_rows[1]]
        # C[1,0] is W[C_rows[2]]
        # C[1,1] is W[C_rows[3]]
        
        a00, a01, a10, a11 = A_rows
        b00, b01, b10, b11 = B_rows
        c00, c01, c10, c11 = C_rows
        
        # M1 = (a00 + a11) * (b00 + b11)
        U[a00][rank_offset] = 1
        U[a11][rank_offset] = 1
        V[b00][rank_offset] = 1
        V[b11][rank_offset] = 1
        W[c00][rank_offset] = 1
        W[c11][rank_offset] = 1
        
        # M2 = (a10 + a11) * b00
        U[a10][rank_offset+1] = 1
        U[a11][rank_offset+1] = 1
        V[b00][rank_offset+1] = 1
        W[c10][rank_offset+1] = 1
        
        # M3 = a00 * (b01 - b11)
        U[a00][rank_offset+2] = 1
        V[b01][rank_offset+2] = 1
        V[b11][rank_offset+2] = -1
        W[c01][rank_offset+2] = 1
        W[c11][rank_offset+2] = 1 # Wait, M3 contributes to Z01 and Z11?
        # Z01 = M3 + M5. So M3 contributes to Z01.
        # Z11 = M1 - M2 + M3 + M6. So M3 contributes to Z11.
        # So W[c01] gets 1, W[c11] gets 1.
        
        # M4 = a11 * (b10 - b00)
        U[a11][rank_offset+3] = 1
        V[b10][rank_offset+3] = 1
        V[b00][rank_offset+3] = -1
        W[c00][rank_offset+3] = 1
        W[c10][rank_offset+3] = 1
        
        # M5 = (a00 + a01) * b11
        U[a00][rank_offset+4] = 1
        U[a01][rank_offset+4] = 1
        V[b11][rank_offset+4] = 1
        W[c00][rank_offset+4] = -1
        W[c01][rank_offset+4] = 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        U[a10][rank_offset+5] = 1
        U[a00][rank_offset+5] = -1
        V[b00][rank_offset+5] = 1
        V[b01][rank_offset+5] = 1
        W[c11][rank_offset+5] = 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        U[a01][rank_offset+6] = 1
        U[a11][rank_offset+6] = -1
        V[b10][rank_offset+6] = 1
        V[b11][rank_offset+6] = 1
        W[c00][rank_offset+6] = 1
        W[c11][rank_offset+6] = -1
        
        return rank_offset + 7

    # Define the 6 products
    # 1. A11 * B11 -> contributes to C11 (top-left 2x2 of C)
    # A11: U rows [0, 1, 2, 3]
    # B11: V rows [0, 1, 6, 7]
    # C11: W rows?
    # C is 2x6.
    # C[0,0] is W[0*2+0] = W[0]
    # C[0,1] is W[1*2+0] = W[2]
    # C[1,0] is W[0*2+1] = W[1]
    # C[1,1] is W[1*2+1] = W[3]
    # So C_rows for C11 are [0, 2, 1, 3]
    
    offset = 0
    offset = add_strassen_product(U, V, W, offset, [0, 1, 2, 3], [0, 1, 6, 7], [0, 2, 1, 3])
    
    # 2. A12 * B21 -> contributes to C11
    # A12: U rows [4, 5, 6, 7]
    # B21: V rows [12, 13, 18, 19]
    # C11: W rows [0, 2, 1, 3]
    offset = add_strassen_product(U, V, W, offset, [4, 5, 6, 7], [12, 13, 18, 19], [0, 2, 1, 3])
    
    # 3. A11 * B12 -> contributes to C12 (top-middle 2x2 of C)
    # B12: V rows [2, 3, 8, 9]
    # C12: 
    # C[0,2] is W[2*2+0] = W[4]
    # C[0,3] is W[3*2+0] = W[6]
    # C[1,2] is W[2*2+1] = W[5]
    # C[1,3] is W[3*2+1] = W[7]
    # C_rows: [4, 6, 5, 7]
    offset = add_strassen_product(U, V, W, offset, [0, 1, 2, 3], [2, 3, 8, 9], [4, 6, 5, 7])
    
    # 4. A12 * B22 -> contributes to C12
    # B22: V rows [14, 15, 20, 21]
    offset = add_strassen_product(U, V, W, offset, [4, 5, 6, 7], [14, 15, 20, 21], [4, 6, 5, 7])
    
    # 5. A11 * B13 -> contributes to C13 (top-right 2x2 of C)
    # B13: V rows [4, 5, 10, 11]
    # C13:
    # C[0,4] is W[4*2+0] = W[8]
    # C[0,5] is W[5*2+0] = W[10]
    # C[1,4] is W[4*2+1] = W[9]
    # C[1,5] is W[5*2+1] = W[11]
    # C_rows: [8, 10, 9, 11]
    offset = add_strassen_product(U, V, W, offset, [0, 1, 2, 3], [4, 5, 10, 11], [8, 10, 9, 11])
    
    # 6. A12 * B23 -> contributes to C13
    # B23: V rows [16, 17, 22, 23]
    offset = add_strassen_product(U, V, W, offset, [4, 5, 6, 7], [16, 17, 22, 23], [8, 10, 9, 11])
    
    # Verify
    if verify(U, V, W):
        best_result = {"U": U, "V": V, "W": W}
        best_rank = 42
    else:
        # Fallback to schoolbook if construction failed (should not happen)
        best_result, best_rank = schoolbook()

    return best_result

# EVOLVE-BLOCK-END
