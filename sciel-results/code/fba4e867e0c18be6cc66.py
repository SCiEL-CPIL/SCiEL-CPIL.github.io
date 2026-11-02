"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define indices for 4x4 blocks
    # A (U): i*m + j
    A00 = [0, 1, 4, 5]
    A01 = [2, 3, 6, 7]
    A10 = [8, 9, 12, 13]
    A11 = [10, 11, 14, 15]
    
    # B (V): j*p + k
    # Block 1 (k=0..3)
    B00_1 = [0, 1, 4, 5]
    B01_1 = [2, 3, 6, 7]
    B10_1 = [8, 9, 12, 13]
    B11_1 = [10, 11, 14, 15]
    
    # Block 2 (k=4..7)
    B00_2 = [4, 5, 8, 9]
    B01_2 = [6, 7, 10, 11]
    B10_2 = [12, 13, 16, 17]
    B11_2 = [14, 15, 18, 19]
    
    # C (W): k*n + i
    # Block 1 (k=0..3)
    C00_1 = [0, 4, 1, 5]
    C01_1 = [8, 12, 9, 13]
    C10_1 = [2, 6, 3, 7]
    C11_1 = [10, 14, 11, 15]
    
    # Block 2 (k=4..7)
    C00_2 = [16, 20, 17, 21]
    C01_2 = [24, 28, 25, 29]
    C10_2 = [18, 22, 19, 23]
    C11_2 = [26, 30, 27, 31]
    
    def add_strassen_2x2(U, V, W, u_slots, v_slots, w_slots, rank_offset):
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
            ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),
            ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]
        
        for t in range(7):
            r = rank_offset + t
            for i in range(4):
                if terms[t][0][i] != 0:
                    for idx, sgn in u_slots[i]:
                        U[idx][r] += terms[t][0][i] * sgn
                if terms[t][1][i] != 0:
                    for idx, sgn in v_slots[i]:
                        V[idx][r] += terms[t][1][i] * sgn
                if terms[t][2][i] != 0:
                    for idx, sgn in w_slots[i]:
                        W[idx][r] += terms[t][2][i] * sgn
        return rank_offset + 7

    def make_slots(base_list):
        return [[(idx, 1) for idx in base_list[i]] for i in range(4)]
    
    def make_sum_slots(base1, base2, sign1=1, sign2=1):
        slots = []
        for i in range(4):
            slots.append([(base1[i], sign1), (base2[i], sign2)])
        return slots

    def make_diff_slots(base1, base2):
        return make_sum_slots(base1, base2, 1, -1)

    def add_4x4x4_block(rank_offset, b00, b01, b10, b11, c00, c01, c10, c11):
        # M1 = (A00 + A11)(B00 + B11) -> C00 + C11
        u1 = make_sum_slots(A00, A11)
        v1 = make_sum_slots(b00, b11)
        w1 = make_sum_slots(c00, c11)
        rank_offset = add_strassen_2x2(U, V, W, u1, v1, w1, rank_offset)
        
        # M2 = (A10 + A11)B00 -> C10
        u2 = make_sum_slots(A10, A11)
        v2 = make_slots(b00)
        w2 = make_slots(c10)
        rank_offset = add_strassen_2x2(U, V, W, u2, v2, w2, rank_offset)
        
        # M3 = A00(B01 - B11) -> C01
        u3 = make_slots(A00)
        v3 = make_diff_slots(b01, b11)
        w3 = make_slots(c01)
        rank_offset = add_strassen_2x2(U, V, W, u3, v3, w3, rank_offset)
        
        # M4 = A11(B10 - B00) -> C10
        u4 = make_slots(A11)
        v4 = make_diff_slots(b10, b00)
        w4 = make_slots(c10)
        rank_offset = add_strassen_2x2(U, V, W, u4, v4, w4, rank_offset)
        
        # M5 = (A00 + A01)B11 -> C01
        u5 = make_sum_slots(A00, A01)
        v5 = make_slots(b11)
        w5 = make_slots(c01)
        rank_offset = add_strassen_2x2(U, V, W, u5, v5, w5, rank_offset)
        
        # M6 = (A10 - A00)(B00 + B01) -> C11
        u6 = make_diff_slots(A10, A00)
        v6 = make_sum_slots(b00, b01)
        w6 = make_slots(c11)
        rank_offset = add_strassen_2x2(U, V, W, u6, v6, w6, rank_offset)
        
        # M7 = (A01 - A11)(B10 + B11) -> C00
        u7 = make_diff_slots(A01, A11)
        v7 = make_sum_slots(b10, b11)
        w7 = make_slots(c00)
        rank_offset = add_strassen_2x2(U, V, W, u7, v7, w7, rank_offset)
        
        return rank_offset

    # Add Block 1
    rank_offset = add_4x4x4_block(0, B00_1, B01_1, B10_1, B11_1, C00_1, C01_1, C10_1, C11_1)
    
    # Add Block 2
    rank_offset = add_4x4x4_block(rank_offset, B00_2, B01_2, B10_2, B11_2, C00_2, C01_2, C10_2, C11_2)
    
    # Add Block 3 (Schoolbook for 4x4x1)
    # k=8
    for i in range(4):
        for j in range(4):
            r = rank_offset
            U[i * 4 + j][r] = 1
            V[j * 9 + 8][r] = 1
            W[8 * 4 + i][r] = 1
            rank_offset += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
