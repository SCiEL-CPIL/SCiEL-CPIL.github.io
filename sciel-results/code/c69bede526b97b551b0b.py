import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 30
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Helper to add a Strassen block
    def add_strassen_block(i_off, j_off_a, j_off_b, k_off, r_start):
        # i_off: row offset in A and C
        # j_off_a: col offset in A
        # j_off_b: row offset in B
        # k_off: col offset in B and C
        # r_start: starting index in U, V, W
        
        # Strassen coefficients for A (2x2)
        # M1: (0,0)+ (1,1)
        # M2: (1,0)+ (1,1)
        # M3: (0,0)
        # M4: (1,1)
        # M5: (0,0)+ (0,1)
        # M6: (1,0)- (0,0)
        # M7: (0,1)- (1,1)
        
        a_coeffs = [
            [(1, 0), (1, 1)],  # M1
            [(1, 0), (1, 1)],  # M2
            [(1, 0)],          # M3
            [(1, 1)],          # M4
            [(1, 0), (1, 1)],  # M5
            [(-1, 0), (1, 1)], # M6
            [(-1, 1), (1, 0)]  # M7
        ]
        
        # Strassen coefficients for B (2x2)
        # M1: (0,0)+ (1,1)
        # M2: (0,0)
        # M3: (0,1)- (1,1)
        # M4: (1,0)- (0,0)
        # M5: (1,1)
        # M6: (0,0)+ (0,1)
        # M7: (1,0)+ (1,1)
        
        b_coeffs = [
            [(1, 0), (1, 1)],  # M1
            [(1, 0)],          # M2
            [(-1, 1), (1, 0)], # M3
            [(-1, 0), (1, 1)], # M4
            [(1, 1)],          # M5
            [(1, 0), (1, 1)],  # M6
            [(1, 0), (1, 1)]   # M7
        ]
        
        # Output coefficients for C (2x2)
        # C00: M1 - M4 + M6 - M7
        # C01: M3 + M5
        # C10: M2 + M4
        # C11: M1 - M3 + M2 - M5
        
        c_coeffs = [
            # M1, M2, M3, M4, M5, M6, M7
            [(1, 0), (-1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)], # C00 part 1? No, we sum later.
            # Let's just store the contribution of each M to each C entry.
        ]
        
        # Actually, it's easier to just define the 7 rank-1 tensors directly.
        # Term t has u_t, v_t, w_t.
        # w_t is the vector of coefficients for C.
        
        # C00 gets M1, -M4, M6, -M7
        # C01 gets M3, M5
        # C10 gets M2, M4
        # C11 gets M1, -M3, M2, -M5
        
        w_coeffs = [
            # M1
            [(1, 0), (0, 1)],  # C00 += 1, C11 += 1
            # M2
            [(0, 0), (1, 1)],  # C10 += 1, C11 += 1
            # M3
            [(0, 1), (-1, 1)], # C01 += 1, C11 -= 1
            # M4
            [(-1, 0), (1, 1)], # C00 -= 1, C10 += 1, C11 += 1
            # M5
            [(0, 1), (-1, 1)], # C01 += 1, C11 -= 1
            # M6
            [(1, 0), (0, 0)],  # C00 += 1
            # M7
            [(-1, 0), (0, 0)]  # C00 -= 1
        ]
        
        for t in range(7):
            r = r_start + t
            
            # Fill U
            for (val, di, dj) in a_coeffs[t]:
                i = i_off + di
                j = j_off_a + dj
                U[i * m + j][r] += val
                
            # Fill V
            for (val, di, dj) in b_coeffs[t]:
                j = j_off_b + di
                k = k_off + dj
                V[j * p + k][r] += val
                
            # Fill W
            for (val, di, dj) in w_coeffs[t]:
                k = k_off + di
                i = i_off + dj
                W[k * n + i][r] += val
                
        return r_start + 7

    # Block 1: A(0,0) x B(0,0) -> C(0,0)
    current_rank = add_strassen_block(0, 0, 0, 0, current_rank)
    
    # Block 2: A(0,0) x B(0,2) -> C(0,2)
    current_rank = add_strassen_block(0, 0, 0, 2, current_rank)
    
    # Block 3: A(0,2) x B(2,0) -> C(0,0)
    current_rank = add_strassen_block(0, 2, 2, 0, current_rank)
    
    # Block 4: A(0,2) x B(2,2) -> C(0,2)
    current_rank = add_strassen_block(0, 2, 2, 2, current_rank)
    
    # Part 5: Last column of B (k=4)
    # C(:, 4) = A * B(:, 4)
    # Rank 2
    # Term 5.1: i=0
    r = current_rank
    for j in range(m):
        U[0 * m + j][r] = 1
        V[j * p + 4][r] = 1
    W[4 * n + 0][r] = 1
    current_rank += 1
    
    # Term 5.2: i=1
    r = current_rank
    for j in range(m):
        U[1 * m + j][r] = 1
        V[j * p + 4][r] = 1
    W[4 * n + 1][r] = 1
    current_rank += 1
    
    return {"U": U, "V": V, "W": W}
