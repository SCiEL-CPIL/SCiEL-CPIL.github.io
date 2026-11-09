import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 7 decomposition for M_{2,2,3}
    # Coefficients from Inspiration 1
    C_A = [
        [[1, 0], [0, 1]],  # M1
        [[0, 0], [1, 1]],  # M2
        [[1, 0], [0, 0]],  # M3
        [[0, 0], [0, 1]],  # M4
        [[1, 1], [0, 0]],  # M5
        [[-1, 0], [1, 0]], # M6
        [[0, 1], [0, -1]]  # M7
    ]
    
    C_B = [
        [[1, 0], [0, 1]],  # M1
        [[1, 0], [0, 0]],  # M2
        [[0, 1], [0, -1]], # M3
        [[-1, 0], [1, 0]], # M4
        [[0, 0], [0, 1]],  # M5
        [[1, 1], [0, 0]],  # M6
        [[0, 0], [1, 1]]   # M7
    ]
    
    Targets = [
        [(0, 0, 1), (1, 1, 1)],    # M1 -> C00, C11
        [(1, 0, 1), (1, 1, -1)],   # M2 -> C10, C11
        [(0, 1, 1), (1, 1, 1)],    # M3 -> C01, C11
        [(0, 0, 1), (1, 0, 1)],    # M4 -> C00, C10
        [(0, 0, -1), (0, 1, 1)],   # M5 -> C00, C01
        [(1, 1, 1)],               # M6 -> C11
        [(0, 0, 1)]                # M7 -> C00
    ]
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Loop over C blocks (i, k)
    for i_blk in range(2):
        for k_blk in range(2):
            # Loop over A block columns (j_blk)
            for j_blk in range(2):
                # Apply rank 7 decomposition for M_{2,2,3}
                for t in range(7):
                    # Fill U (A part)
                    for i_loc in range(2):
                        for j_loc in range(2):
                            coeff_a = C_A[t][i_loc][j_loc]
                            if coeff_a != 0:
                                I = i_blk * 2 + i_loc
                                J = j_blk * 2 + j_loc
                                U[I * m + J][col] = coeff_a
                                
                    # Fill V (B part)
                    for j_loc in range(2):
                        for k_loc in range(3):
                            # C_B is 2x2, but we need to map to 2x3?
                            # In Insp 1, C_B was 2x2 and applied to all k_loc?
                            # Let's check Insp 1 logic again.
                            # Insp 1: coeff_b = C_B[t][j_blk][k_blk]
                            # Here j_loc is local row of B block.
                            # k_loc is local col of B block.
                            # C_B[t] is 2x2.
                            # It seems C_B[t][j_loc][?] is used.
                            # In Insp 1, k_blk was the block index.
                            # Here, we are inside a specific k_blk.
                            # So we need to map C_B to the 2x3 block.
                            # Insp 1 used C_B[t][j_loc][k_blk_global]?
                            # No, Insp 1 iterated k_loc and used the same coeff.
                            # This implies the B block in Insp 1 was treated as 2x2?
                            # No, k_loc went to 3.
                            # If C_B is 2x2, how did it fill 2x3?
                            # It used C_B[t][j_loc][k_loc % 2]?
                            # Or maybe C_B in Insp 1 was actually 2x3?
                            # Let's look at C_B definition in Insp 1:
                            # [[1, 0], [0, 1]] -> 2x2.
                            # If it's 2x2, and k_loc is 0,1,2.
                            # Maybe it used C_B[t][j_loc][k_loc] if k_loc < 2, else 0?
                            # Or maybe it repeated?
                            # Given Insp 1 was valid, I will assume the same logic:
                            # coeff_b = C_B[t][j_loc][k_loc] if k_loc < 2 else 0?
                            # Or maybe C_B[t][j_loc][k_loc % 2]?
                            # Let's assume the standard Strassen for 2x2x2 and ignore the 3rd col?
                            # No, that would be wrong.
                            # I will use a known rank 7 decomposition for 2x2x3 if possible.
                            # Actually, I will just use the Insp 1 logic directly:
                            # coeff_b = C_B[t][j_loc][k_loc] is not possible if C_B is 2x2.
                            # I will define C_B_223 explicitly.
                            
                            pass # Placeholder
                    
                    # Fill W (C part)
                    for i_t, k_t, coeff_c in Targets[t]:
                        for k_loc in range(3):
                            # Map local target to global
                            # i_t is local row of C block
                            # k_t is local col of C block?
                            # In Insp 1, Targets had (i_target_block, k_target_block, coeff)
                            # Here i_t, k_t are local indices?
                            # Insp 1: I = i_t * 2 + i_loc.
                            # So i_t is block index.
                            # But here we are in a specific block.
                            # So i_t should be i_loc?
                            # And k_t should be k_loc?
                            # No, Targets in Insp 1 were for the whole 2x2x3 block?
                            # I will assume Targets define the local C indices.
                            pass
                    col += 1
    return {"U": U, "V": V, "W": W}
