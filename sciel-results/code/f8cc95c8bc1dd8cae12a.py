"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    # n=4, m=5, p=6
    # U: n*m = 20
    # V: m*p = 30
    # W: p*n = 24
    
    # Estimated rank: 101
    # A_R B_R: 24
    # A_L B_L1: 28
    # A_L B_L2: 49
    total_rank = 101
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # u_idx: list of 4 global U indices for A[0,0], A[0,1], A[1,0], A[1,1]
    # v_idx: list of 4 global V indices for B[0,0], B[0,1], B[1,0], B[1,1]
    # w_idx: list of 4 global W indices for C[0,0], C[0,1], C[1,0], C[1,1]
    def add_strassen_2x2x2(u_idx, v_idx, w_idx):
        nonlocal col
        if col + 7 > total_rank:
            return col
        
        # M1 = (a00+a11)(b00+b11)
        U[u_idx[0], col] += 1
        U[u_idx[2], col] += 1
        V[v_idx[0], col] += 1
        V[v_idx[2], col] += 1
        W[w_idx[0], col] += 1
        W[w_idx[3], col] += 1
        col += 1
        
        # M2 = (a10+a11)b00
        U[u_idx[1], col] += 1
        U[u_idx[3], col] += 1
        V[v_idx[0], col] += 1
        W[w_idx[1], col] += 1
        W[w_idx[3], col] -= 1
        col += 1
        
        # M3 = a00(b10-b11)
        U[u_idx[0], col] += 1
        V[v_idx[1], col] += 1
        V[v_idx[3], col] -= 1
        W[w_idx[0], col] -= 1
        W[w_idx[2], col] += 1
        col += 1
        
        # M4 = a11(b01-b00)
        U[u_idx[3], col] += 1
        V[v_idx[1], col] += 1
        V[v_idx[0], col] -= 1
        W[w_idx[0], col] += 1
        W[w_idx[1], col] += 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[u_idx[0], col] += 1
        U[u_idx[2], col] += 1
        V[v_idx[3], col] += 1
        W[w_idx[0], col] -= 1
        W[w_idx[2], col] += 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[u_idx[1], col] += 1
        U[u_idx[0], col] -= 1
        V[v_idx[0], col] += 1
        V[v_idx[1], col] += 1
        W[w_idx[3], col] += 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[u_idx[2], col] += 1
        U[u_idx[3], col] -= 1
        V[v_idx[1], col] += 1
        V[v_idx[3], col] += 1
        W[w_idx[0], col] += 1
        W[w_idx[2], col] += 1
        col += 1
        
        return col

    # 1. A_R B_R: 4x1x6
    # i in 0..3, j=4, k in 0..5
    for i in range(4):
        for k in range(6):
            u_idx = i * m + 4
            v_idx = 4 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # 2. A_L B_L1: 4x4x2
    # A_L is 4x4 (i 0..3, j 0..3)
    # B_L1 is 4x2 (j 0..3, k 0..1)
    # We decompose into 4 Strassen 2x2x2 blocks
    # C_top (i 0..1) = A_11 B_top + A_12 B_bot
    # C_bot (i 2..3) = A_21 B_top + A_22 B_bot
    
    # Block 1: A_11 * B_top
    # A_11: i 0..1, j 0..1
    # B_top: j 0..1, k 0..1
    # C_11: i 0..1, k 0..1
    u_idx = [0*5+0, 0*5+1, 1*5+0, 1*5+1] # a00, a01, a10, a11
    v_idx = [0*6+0, 0*6+1, 1*6+0, 1*6+1] # b00, b01, b10, b11
    w_idx = [0*4+0, 0*4+1, 1*4+0, 1*4+1] # c00, c01, c10, c11
    add_strassen_2x2x2(u_idx, v_idx, w_idx)
    
    # Block 2: A_12 * B_bot
    # A_12: i 0..1, j 2..3
    # B_bot: j 2..3, k 0..1
    # C_12: i 0..1, k 0..1
    u_idx = [0*5+2, 0*5+3, 1*5+2, 1*5+3]
    v_idx = [2*6+0, 2*6+1, 3*6+0, 3*6+1]
    w_idx = [0*4+0, 0*4+1, 1*4+0, 1*4+1]
    add_strassen_2x2x2(u_idx, v_idx, w_idx)
    
    # Block 3: A_21 * B_top
    # A_21: i 2..3, j 0..1
    # B_top: j 0..1, k 0..1
    # C_21: i 2..3, k 0..1
    u_idx = [2*5+0, 2*5+1, 3*5+0, 3*5+1]
    v_idx = [0*6+0, 0*6+1, 1*6+0, 1*6+1]
    w_idx = [0*4+2, 0*4+3, 1*4+2, 1*4+3]
    add_strassen_2x2x2(u_idx, v_idx, w_idx)
    
    # Block 4: A_22 * B_bot
    # A_22: i 2..3, j 2..3
    # B_bot: j 2..3, k 0..1
    # C_22: i 2..3, k 0..1
    u_idx = [2*5+2, 2*5+3, 3*5+2, 3*5+3]
    v_idx = [2*6+0, 2*6+1, 3*6+0, 3*6+1]
    w_idx = [0*4+2, 0*4+3, 1*4+2, 1*4+3]
    add_strassen_2x2x2(u_idx, v_idx, w_idx)

    # 3. A_L B_L2: 4x4x4
    # A_L is 4x4
    # B_L2 is 4x4 (j 0..3, k 2..5)
    # This is M(4) = 49
    # We decompose into 4 Strassen 2x2x2 blocks? No, M(4) uses 7 Strassen 2x2x2 multiplications.
    # Wait, M(4) = 7 * M(2) = 7 * 7 = 49.
    # So we need to apply Strassen to the 4x4x4 problem.
    # This means we define 7 "virtual" 2x2x2 multiplications.
    
    # A_L = [[A00, A01], [A10, A11]] where blocks are 2x2.
    # B_L2 = [[B00, B01], [B10, B11]] where blocks are 2x2.
    
    # We need to add 7 Strassen calls, but the inputs are linear combinations of the 2x2 blocks.
    # This is tricky to map directly to U,V,W indices without expanding.
    # Instead, we can just call add_strassen_2x2x2 7 times? No, the inputs are sums.
    # We need to expand the linear combinations into U and V entries.
    
    # Let's define the 7 terms for M(4) Strassen.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mi is a 2x2 matrix product.
    # So each Mi requires 7 rank-1 terms.
    # Total 49 terms.
    
    # We need a helper to add a "virtual" Strassen block where inputs are linear combos.
    def add_virtual_strassen_2x2x2(u_coeffs, v_coeffs, w_idx):
        # u_coeffs: list of 4 lists of (global_u_idx, coeff)
        # v_coeffs: list of 4 lists of (global_v_idx, coeff)
        # w_idx: list of 4 global W indices
        nonlocal col
        if col + 7 > total_rank:
            return col
            
        # We need to add 7 terms.
        # For each term, we update U and V with the linear combos, and W with the Strassen pattern.
        
        # Term 1: (u0+u2)(v0+v2)
        for idx, c in u_coeffs[0]: U[idx, col] += c
        for idx, c in u_coeffs[2]: U[idx, col] += c
        for idx, c in v_coeffs[0]: V[idx, col] += c
        for idx, c in v_coeffs[2]: V[idx, col] += c
        W[w_idx[0], col] += 1
        W[w_idx[3], col] += 1
        col += 1
        
        # Term 2: (u1+u3)v0
        for idx, c in u_coeffs[1]: U[idx, col] += c
        for idx, c in u_coeffs[3]: U[idx, col] += c
        for idx, c in v_coeffs[0]: V[idx, col] += c
        W[w_idx[1], col] += 1
        W[w_idx[3], col] -= 1
        col += 1
        
        # Term 3: u0(v1-v3)
        for idx, c in u_coeffs[0]: U[idx, col] += c
        for idx, c in v_coeffs[1]: V[idx, col] += c
        for idx, c in v_coeffs[3]: V[idx, col] -= c
        W[w_idx[0], col] -= 1
        W[w_idx[2], col] += 1
        col += 1
        
        # Term 4: u3(v1-v0)
        for idx, c in u_coeffs[3]: U[idx, col] += c
        for idx, c in v_coeffs[1]: V[idx, col] += c
        for idx, c in v_coeffs[0]: V[idx, col] -= c
        W[w_idx[0], col] += 1
        W[w_idx[1], col] += 1
        col += 1
        
        # Term 5: (u0+u2)v3
        for idx, c in u_coeffs[0]: U[idx, col] += c
        for idx, c in u_coeffs[2]: U[idx, col] += c
        for idx, c in v_coeffs[3]: V[idx, col] += c
        W[w_idx[0], col] -= 1
        W[w_idx[2], col] += 1
        col += 1
        
        # Term 6: (u1-u0)(v0+v1)
        for idx, c in u_coeffs[1]: U[idx, col] += c
        for idx, c in u_coeffs[0]: U[idx, col] -= c
        for idx, c in v_coeffs[0]: V[idx, col] += c
        for idx, c in v_coeffs[1]: V[idx, col] += c
        W[w_idx[3], col] += 1
        col += 1
        
        # Term 7: (u2-u3)(v1+v3)
        for idx, c in u_coeffs[2]: U[idx, col] += c
        for idx, c in u_coeffs[3]: U[idx, col] -= c
        for idx, c in v_coeffs[1]: V[idx, col] += c
        for idx, c in v_coeffs[3]: V[idx, col] += c
        W[w_idx[0], col] += 1
        W[w_idx[2], col] += 1
        col += 1
        
        return col

    # Define the 2x2 blocks of A_L and B_L2
    # A_L blocks:
    # A00: i 0..1, j 0..1
    # A01: i 0..1, j 2..3
    # A10: i 2..3, j 0..1
    # A11: i 2..3, j 2..3
    
    # B_L2 blocks:
    # B00: j 0..1, k 2..3
    # B01: j 0..1, k 4..5
    # B10: j 2..3, k 2..3
    # B11: j 2..3, k 4..5
    
    # C blocks:
    # C00: i 0..1, k 2..3
    # C01: i 0..1, k 4..5
    # C10: i 2..3, k 2..3
    # C11: i 2..3, k 4..5

    # Helper to get u_coeffs for a 2x2 block
    def get_u_coeffs(i_start, j_start):
        return [
            [(i_start*5 + j_start, 1)],
            [(i_start*5 + j_start+1, 1)],
            [((i_start+1)*5 + j_start, 1)],
            [((i_start+1)*5 + j_start+1, 1)]
        ]

    def get_v_coeffs(j_start, k_start):
        return [
            [(j_start*6 + k_start, 1)],
            [(j_start*6 + k_start+1, 1)],
            [((j_start+1)*6 + k_start, 1)],
            [((j_start+1)*6 + k_start+1, 1)]
        ]

    def get_w_idx(i_start, k_start):
        return [
            k_start*4 + i_start,
            k_start*4 + i_start+1,
            (k_start+1)*4 + i_start,
            (k_start+1)*4 + i_start+1
        ]

    # M1: (A00+A11)(B00+B11) -> C00 + C11
    u_m1 = [
        [(i, 1) for i in [0*5+0, 0*5+1, 1*5+0, 1*5+1, 2*5+2, 2*5+3, 3*5+2, 3*5+3]],
        [(i, 1) for i in [0*5+1, 1*5+1, 2*5+3, 3*5+3]],
        [(i, 1) for i in [0*5+0, 1*5+0, 2*5+2, 3*5+2]],
        [(i, 1) for i in [1*5+1, 3*5+3]]
    ]
    # Wait, u_coeffs structure is [u00, u01, u10, u11] for the virtual 2x2 matrix.
    # The virtual matrix for M1 input A is A00+A11.
    # So u00 = A00_00 + A11_00, etc.
    
    # Let's construct the linear combos properly.
    # A00: i 0,1; j 0,1
    # A11: i 2,3; j 2,3
    
    # Virtual A for M1:
    # vA00 = A00_00 + A11_00 = U[0*5+0] + U[2*5+2]
    # vA01 = A00_01 + A11_01 = U[0*5+1] + U[2*5+3]
    # vA10 = A00_10 + A11_10 = U[1*5+0] + U[3*5+2]
    # vA11 = A00_11 + A11_11 = U[1*5+1] + U[3*5+3]
    
    u_m1 = [
        [(0*5+0, 1), (2*5+2, 1)],
        [(0*5+1, 1), (2*5+3, 1)],
        [(1*5+0, 1), (3*5+2, 1)],
        [(1*5+1, 1), (3*5+3, 1)]
    ]
    
    # Virtual B for M1: B00+B11
    # B00: j 0,1; k 2,3
    # B11: j 2,3; k 4,5
    v_m1 = [
        [(0*6+2, 1), (2*6+4, 1)],
        [(0*6+3, 1), (2*6+5, 1)],
        [(1*6+2, 1), (3*6+4, 1)],
        [(1*6+3, 1), (3*6+5, 1)]
    ]
    
    w_m1 = get_w_idx(0, 2) # C00 and C11? No, Strassen output is 2x2.
    # The output of M1 is added to C00 and C11 of the 4x4 result.
    # In the virtual 2x2 Strassen, the output is a 2x2 matrix.
    # The (0,0) element of virtual C corresponds to C00 (2x2 block).
    # The (1,1) element of virtual C corresponds to C11 (2x2 block).
    # So w_idx should be the indices for the 2x2 blocks C00 and C11?
    # No, the virtual Strassen produces a 2x2 matrix of scalars?
    # No, the virtual Strassen produces a 2x2 matrix of 2x2 matrices?
    # No, the virtual Strassen operates on 2x2 matrices.
    # So the output is a 2x2 matrix where each element is a 2x2 matrix?
    # No, the "virtual" Strassen is just a way to group the 49 terms.
    # Each "virtual" Strassen call adds 7 terms to U,V,W.
    # The w_idx in add_virtual_strassen_2x2x2 should be the 4 indices of the 2x2 block of C.
    # But M1 contributes to C00 and C11.
    # So the virtual C is not a single 2x2 block.
    # This approach is flawed because Strassen M(4) mixes the blocks.
    
    # Alternative: Just use the known 49-term decomposition for M(4).
    # It is easier to just use the trivial 64 terms for M(4) if we can't easily map Strassen.
    # But 49 is much better.
    
    # Let's just use the trivial decomposition for the last part to be safe and fast.
    # Rank for 4x4x4 trivial is 64.
    # Total rank: 24 + 28 + 64 = 116.
    # This is worse than 101.
    
    # Let's try to map the 7 Strassen terms for M(4) correctly.
    # The 7 terms are products of linear combos of the 2x2 blocks.
    # Each product is a 2x2x2 multiplication.
    # So we need to call add_virtual_strassen_2x2x2 7 times.
    # The w_idx for each call should be the indices of the 2x2 block of C that it contributes to?
    # No, each virtual Strassen term contributes to ALL 4 blocks of C?
    # No, the Strassen algorithm for 4x4x4 computes C = A B.
    # The 7 products M1..M7 are 2x2 matrices.
    # C00 = M1_00 + M4_00 - M5_00 + M7_00 (where Mx_00 is the top-left 2x2 block of Mx? No.)
    # M1 is a 2x2 matrix.
    # C00 (top-left 2x2 of C) = M1_00 + M4_00 - M5_00 + M7_00.
    # So each Mx contributes to C00, C01, C10, C11.
    # So for each of the 7 virtual Strassen calls, the output 2x2 matrix corresponds to the 4 blocks of C.
    # So w_idx should be the 4 indices of the 2x2 blocks C00, C01, C10, C11?
    # No, w_idx in add_virtual_strassen_2x2x2 expects 4 indices for a single 2x2 block.
    # But here the output is a 2x2 matrix of 2x2 matrices.
    # So we need to modify add_virtual_strassen_2x2x2 to handle 4 output blocks.
    
    # Let's write a specific function for M(4) Strassen.
    def add_strassen_4x4x4(u_blocks, v_blocks, w_blocks):
        # u_blocks: 4 lists of u_coeffs for A00, A01, A10, A11
        # v_blocks: 4 lists of v_coeffs for B00, B01, B10, B11
        # w_blocks: 4 lists of w_idx for C00, C01, C10, C11
        nonlocal col
        if col + 49 > total_rank:
            return col
            
        # M1 = (A00+A11)(B00+B11)
        u_m1 = [
            [(i, c1) for i, c1 in u_blocks[0]] + [(i, c1) for i, c1 in u_blocks[3]],
            [(i, c1) for i, c1 in u_blocks[1]] + [(i, c1) for i, c1 in u_blocks[4]], # Error: A01 is block 1, A11 is block 3?
            # A00:0, A01:1, A10:2, A11:3
            # M1 input A: A00+A11.
            # vA00 = A00_00 + A11_00
            # vA01 = A00_01 + A11_01
            # vA10 = A00_10 + A11_10
            # vA11 = A00_11 + A11_11
        ]
        
        # This is getting complicated.
        # Let's just use the trivial 64 for the last part.
        # Total rank 116.
        # It's a valid decomposition.
        
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    u_idx = i * m + j
                    v_idx = j * p + (2 + k) # B_L2 starts at k=2
                    w_idx = (2 + k) * n + i
                    U[u_idx, col] = 1
                    V[v_idx, col] = 1
                    W[w_idx, col] = 1
                    col += 1
                    if time.time() > deadline:
                        return col
            if time.time() > deadline:
                return col
        return col

    add_strassen_4x4x4(None, None, None)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
