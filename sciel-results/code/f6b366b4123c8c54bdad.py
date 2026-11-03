import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # Rank 56 decomposition
    rank = 56
    U = [[0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 16 x 56
    
    # Define Strassen factors for 2x2x2
    # U_strassen: 4x7, V_strassen: 4x7, W_strassen: 4x7
    # Indices for 2x2 matrix: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1: (X00+X11)(Y00+Y11) -> Z00, Z11
    # M2: (X01+X11)Y00 -> Z10
    # M3: X00(Y01-Y11) -> Z01, -Z11
    # M4: X11(Y10-Y00) -> Z00, -Z11
    # M5: (X00+X01)Y11 -> Z01
    # M6: (X10-X00)(Y00+Y01) -> Z00
    # M7: (X10-X11)(Y10+Y11) -> Z10
    
    # We will generate 4 sets of 7 factors for the 4 Strassen blocks
    # Block 1: A1, C11 -> P1 (cols 0-1 of Y)
    # Block 2: A2, C21 -> P2 (cols 0-1 of Y)
    # Block 3: A1, C12 -> P3 (cols 2-3 of Y)
    # Block 4: A2, C22 -> P4 (cols 2-3 of Y)
    
    # A indices:
    # A1: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # A2: (0,2)->4, (0,3)->5, (1,2)->6, (1,3)->7
    
    # C indices (for 4x4):
    # C11: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # C12: (0,2)->4, (0,3)->5, (1,2)->6, (1,3)->7
    # C21: (2,0)->8, (2,1)->9, (3,0)->10, (3,1)->11
    # C22: (2,2)->12, (2,3)->13, (3,2)->14, (3,3)->15
    
    # Y indices (for 2x4, k*n+i):
    # Col 0 (k=0): i=0->0, i=1->1
    # Col 1 (k=1): i=0->2, i=1->3
    # Col 2 (k=2): i=0->4, i=1->5
    # Col 3 (k=3): i=0->6, i=1->7
    
    # Strassen factor definitions (local indices 0..3)
    # U factors
    sU = [
        [1, 0, 0, 1], # M1
        [0, 1, 0, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, -1], # M6
        [0, 0, 1, -1]  # M7 (Wait, M7 is X10-X11. X10 is idx 2, X11 is idx 3. So [0,0,1,-1])
    ]
    # Correction for M7: X10 is (1,0) -> idx 2. X11 is (1,1) -> idx 3.
    # M6: X10-X00. X10 idx 2, X00 idx 0. [0,0,1,-1] is wrong for M6?
    # M6: (X10-X00). X10=1, X00=-1. [0,0,1,-1] at indices 2,0?
    # Vector: [ -1, 0, 1, 0 ].
    
    # Let's re-verify Strassen indices.
    # X = [[x00, x01], [x10, x11]]
    # M1: x00+x11. V: y00+y11. W: z00, z11.
    # M2: x01+x11. V: y00. W: z10.
    # M3: x00. V: y01-y11. W: z01, -z11.
    # M4: x11. V: y10-y00. W: z00, -z11.
    # M5: x00+x01. V: y11. W: z01.
    # M6: x10-x00. V: y00+y01. W: z00.
    # M7: x10-x11. V: y10+y11. W: z10.
    
    sU = [
        [1, 0, 0, 1],   # M1
        [0, 1, 0, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 0, 1, -1]   # M7
    ]
    
    sV = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [0, 0, 1, -1],  # M4 (y10-y00: y10 idx 2, y00 idx 0. [ -1, 0, 1, 0 ])
        # Correction M4: y10 is (1,0) idx 2. y00 is (0,0) idx 0.
        # Vector: [-1, 0, 1, 0].
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7 (y10+y11: idx 2, 3)
    ]
    # Fix M4 V
    sV[3] = [-1, 0, 1, 0]
    
    sW = [
        [1, 0, 0, 1],   # M1 (z00, z11)
        [0, 0, 1, 0],   # M2 (z10)
        [0, 1, 0, -1],  # M3 (z01, -z11)
        [1, 0, 0, -1],  # M4 (z00, -z11)
        [0, 1, 0, 0],   # M5 (z01)
        [1, 0, 0, 0],   # M6 (z00)
        [0, 0, 1, 0]    # M7 (z10)
    ]
    
    # Mapping of local 2x2 indices to global indices
    # A1 local: 0,1,2,3 -> Global: 0,1,2,3
    # A2 local: 0,1,2,3 -> Global: 4,5,6,7
    
    # C11 local: 0,1,2,3 -> Global: 0,1,2,3
    # C21 local: 0,1,2,3 -> Global: 8,9,10,11
    # C12 local: 0,1,2,3 -> Global: 4,5,6,7
    # C22 local: 0,1,2,3 -> Global: 12,13,14,15
    
    # Y local: 0,1,2,3 -> Global:
    # P1, P2 map to Y cols 0,1.
    # Y col 0 (k=0): i=0->0, i=1->1.
    # Y col 1 (k=1): i=0->2, i=1->3.
    # So local 0(z00)->0, 1(z01)->2, 2(z10)->1, 3(z11)->3.
    
    # P3, P4 map to Y cols 2,3.
    # Y col 2 (k=2): i=0->4, i=1->5.
    # Y col 3 (k=3): i=0->6, i=1->7.
    # So local 0(z00)->4, 1(z01)->6, 2(z10)->5, 3(z11)->7.
    
    y_map_1 = [0, 2, 1, 3] # for P1, P2
    y_map_2 = [4, 6, 5, 7] # for P3, P4
    
    # Define the 4 blocks
    # (A_offset, C_offset, Y_map, Rank_Offset)
    blocks = [
        (0, 0, y_map_1, 0),   # P1: A1, C11
        (4, 8, y_map_1, 7),   # P2: A2, C21
        (0, 4, y_map_2, 14),  # P3: A1, C12
        (4, 12, y_map_2, 21)  # P4: A2, C22
    ]
    
    # We have 2 copies of B (B1, B2)
    # B1 uses C indices 0..15? No.
    # B is 4x8.
    # B1 is 4x4 (cols 0-3). C indices 0..15 correspond to B1?
    # C index j*p+k.
    # B1: j=0..3, k=0..3. Indices 0..15.
    # B2: j=0..3, k=4..7. Indices 16..31.
    
    # So for B1, C_offset is as defined above.
    # For B2, C_offset is shifted by 16.
    
    all_blocks = []
    for b_idx in range(2):
        c_shift = b_idx * 16
        rank_shift = b_idx * 28
        for a_off, c_off, y_map, r_off in blocks:
            all_blocks.append((a_off, c_off + c_shift, y_map, r_off + rank_shift))
            
    # Fill U, V, W
    for a_off, c_off, y_map, r_off in all_blocks:
        for s in range(7):
            col = r_off + s
            # U
            for i in range(4):
                val = sU[s][i]
                if val != 0:
                    U[a_off + i][col] += val
            # V
            for i in range(4):
                val = sV[s][i]
                if val != 0:
                    V[c_off + i][col] += val
            # W
            for i in range(4):
                val = sW[s][i]
                if val != 0:
                    W[y_map[i]][col] += val
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
