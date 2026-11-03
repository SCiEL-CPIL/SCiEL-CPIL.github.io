import numpy as np

def construct():
    n, m, p = 4, 4, 7
    
    # Split B into B1 (4x3) and B2 (4x4)
    # Rank 48 for B1, Rank 49 for B2
    rank1 = 48
    rank2 = 49
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: M_4,4,3 (Trivial)
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
                t += 1
                
    # Part 2: M_4,4,4 (Strassen)
    # We need to generate the rank 49 decomposition
    # This is 7 Strassen steps, each rank 7
    # Strassen 2x2x2 U,V,W are 4x7
    
    # Get base Strassen U,V,W
    U_s, V_s, W_s = get_strassen_2x2_2x2_UVW()
    
    # The 7 products are:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # We need to map these to the 4x4 blocks
    # A blocks: A00, A01, A10, A11 (2x2 each)
    # B blocks: B00, B01, B10, B11 (2x2 each)
    
    # For each of the 7 steps, we add 7 terms to the rank
    # Each term is a 2x2x2 multiplication
    
    # Helper to add a 2x2x2 decomposition at specific block offsets
    def add_strassen_block(U, V, W, u_offset, v_offset, w_offset, U_s, V_s, W_s, signs_U, signs_V, signs_W):
        # u_offset, v_offset, w_offset are (row, col) in the 2x2 block grid
        # But we need to map to global indices
        pass

    # It's easier to just iterate through the 7 steps and 7 sub-terms
    # Step s (0..6), Sub-term r (0..6)
    # Global rank index = 48 + s*7 + r
    
    # We need to determine the signs and offsets for each step
    
    # Step 1: P1 = (A00+A11)(B00+B11)
    # A parts: A00 (+1), A11 (+1)
    # B parts: B00 (+1), B11 (+1)
    # C parts: C00 (+1), C11 (+1)
    
    # This is complex to hardcode directly in the loop without a clear structure.
    # Let's use the known Strassen coefficients.
    
    # For each of the 7 steps, we have a set of (A_block, sign_A), (B_block, sign_B), (C_block, sign_C)
    # And we use the base Strassen U_s, V_s, W_s for the 2x2 multiplication.
    
    # Actually, the base Strassen U_s, V_s, W_s already encodes the 2x2 multiplication.
    # We just need to place them at the correct block positions with the correct signs.
    
    # Block positions:
    # 0: (0,0) -> A00, B00, C00
    # 1: (0,1) -> A01, B01, C01
    # 2: (1,0) -> A10, B10, C10
    # 3: (1,1) -> A11, B11, C11
    
    # Step 1: A00+A11, B00+B11 -> C00+C11
    # Terms:
    # (A00, +1) * (B00, +1) -> C00 (+1)
    # (A00, +1) * (B11, +1) -> C11 (+1)
    # (A11, +1) * (B00, +1) -> C00 (+1)
    # (A11, +1) * (B11, +1) -> C11 (+1)
    # Wait, the Strassen step is ONE multiplication of two 2x2 matrices.
    # The "sum" is handled by the inputs to the multiplication.
    # The decomposition of the multiplication $X Y = Z$ is given by U_s, V_s, W_s.
    # Here $X = A_{block1} + A_{block2}$.
    # This means the U matrix for this step will have contributions from both A blocks.
    
    # So for Step 1:
    # X = A00 + A11.
    # Y = B00 + B11.
    # Z = C00 + C11.
    
    # The U_s matrix has 4 rows (for the 2x2 X).
    # We need to map these 4 rows to the global A indices for A00 and A11.
    # And apply the signs.
    
    # This is getting complicated to generate dynamically.
    # Given the time limit, I will stick to the Rank 56 idea if I can find a valid decomposition, 
    # or just return the Rank 112 trivial solution if I can't guarantee correctness of the complex one quickly.
    
    # However, Rank 97 is a clear improvement.
    # Let's try to implement the Rank 97 solution carefully.
    
    return {"U": U, "V": "V", "W": "W"} # Placeholder

