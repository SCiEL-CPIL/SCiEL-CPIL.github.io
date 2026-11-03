import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen on 2x2x2 blocks
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 16 x 56
    
    col = 0
    # Strassen coefficients for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # A indices for sub-block 2x2: (0,0), (0,1), (1,0), (1,1)
    # B indices for sub-block 2x2: (0,0), (0,1), (1,0), (1,1)
    # C indices for sub-block 2x2: (0,0), (0,1), (1,0), (1,1)
    
    # Strassen U coefficients (for A)
    # A11, A12, A21, A22
    # M1: A11=1, A22=1
    # M2: A21=1, A22=1
    # M3: A11=1
    # M4: A22=1
    # M5: A11=1, A12=1
    # M6: A21=1, A11=-1
    # M7: A12=1, A22=-1
    
    # Strassen V coefficients (for B)
    # B11, B12, B21, B22
    # M1: B11=1, B22=1
    # M2: B11=1
    # M3: B12=1, B22=-1
    # M4: B21=1, B11=-1
    # M5: B22=1
    # M6: B11=1, B12=1
    # M7: B21=1, B22=1
    
    # Strassen W coefficients (for C)
    # C11, C12, C21, C22
    # M1: C11=1, C22=1
    # M2: C21=1, C22=-1
    # M3: C12=1, C22=1
    # M4: C11=1, C21=1
    # M5: C11=-1, C12=1
    # M6: C22=1
    # M7: C11=1, C22=-1? No.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's verify C22:
    # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # -M2 = -(A21+A22)B11 = -A21B11 - A22B11
    # M3 = A11(B12-B22) = A11B12 - A11B22
    # M6 = (A21-A11)(B11+B12) = A21B11 + A21B12 - A11B11 - A11B12
    # Sum:
    # A11B11: 1 - 1 = 0
    # A11B22: 1 - 1 = 0
    # A22B11: 1 - 1 = 0
    # A22B22: 1
    # A21B11: -1 + 1 = 0
    # A21B12: 1
    # A11B12: -1 - 1 = -2? No.
    # Let's re-check Strassen formulas.
    
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Check C22:
    # M1 = A11B11 + A11B22 + A22B11 + A22B22
    # -M2 = -A21B11 - A22B11
    # M3 = A11B12 - A11B22
    # M6 = A21B11 + A21B12 - A11B11 - A11B12
    # Sum:
    # A11B11: 1 - 1 = 0
    # A11B22: 1 - 1 = 0
    # A22B11: 1 - 1 = 0
    # A22B22: 1
    # A21B11: -1 + 1 = 0
    # A21B12: 1
    # A11B12: -1 - 1 = -2?
    # Wait, A22B22 is not in C22?
    # C22 should be A11B12 + A12B22? No, C is 2x2.
    # C = [[A11B11+A12B21, A11B12+A12B22], [A21B11+A22B21, A21B12+A22B22]]
    # So C22 = A21B12 + A22B22.
    # My sum gave A21B12 + A22B22.
    # A11B12 terms: -1 (from M3) - 1 (from M6) = -2.
    # A11B11 terms: 1 (M1) - 1 (M6) = 0.
    # A21B11 terms: -1 (M2) + 1 (M6) = 0.
    # A22B11 terms: 1 (M1) - 1 (M2) = 0.
    # A11B22 terms: 1 (M1) - 1 (M3) = 0.
    # So it works.
    
    # W coefficients:
    # M1: C11=1, C22=1
    # M2: C21=1, C22=-1
    # M3: C12=1, C22=1
    # M4: C11=1, C21=1
    # M5: C11=-1, C12=1
    # M6: C22=1
    # M7: C11=1, C22=-1?
    # C11 = M1 + M4 - M5 + M7
    # C22 = M1 - M2 + M3 + M6
    # So M7 contributes to C11=1. Does it contribute to C22?
    # C22 formula doesn't have M7.
    # So M7: C11=1.
    
    # Let's list them properly.
    # s = 0..6
    # U_s, V_s, W_s
    
    # s=0 (M1):
    # U: A11=1, A22=1
    # V: B11=1, B22=1
    # W: C11=1, C22=1
    # s=1 (M2):
    # U: A21=1, A22=1
    # V: B11=1
    # W: C21=1, C22=-1
    # s=2 (M3):
    # U: A11=1
    # V: B12=1, B22=-1
    # W: C12=1, C22=1
    # s=3 (M4):
    # U: A22=1
    # V: B21=1, B11=-1
    # W: C11=1, C21=1
    # s=4 (M5):
    # U: A11=1, A12=1
    # V: B22=1
    # W: C11=-1, C12=1
    # s=5 (M6):
    # U: A21=1, A11=-1
    # V: B11=1, B12=1
    # W: C22=1
    # s=6 (M7):
    # U: A12=1, A22=-1
    # V: B21=1, B22=1
    # W: C11=1
    
    # Indices in 2x2 block:
    # 0: 11, 1: 12, 2: 21, 3: 22
    
    strassen_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    strassen_v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    strassen_w = [
        [1, 0, 0, 1], # M1: C11, C22
        [0, 0, 1, -1], # M2: C21, C22
        [0, 1, 0, 1], # M3: C12, C22
        [1, 0, 1, 0], # M4: C11, C21
        [-1, 1, 0, 0], # M5: C11, C12
        [0, 0, 0, 1], # M6: C22
        [1, 0, 0, 0]  # M7: C11
    ]
    
    # Block mapping
    # a in {1, 2}
    # b in {1, 2, 3, 4}
    
    for a in range(2):
        for b in range(4):
            for s in range(7):
                # U
                # A rows for block a
                # a=0: cols 0,1 -> U rows 0,1,4,5
                # a=1: cols 2,3 -> U rows 2,3,6,7
                # Sub-block indices: 0:11, 1:12, 2:21, 3:22
                # 11: i=0, j=0
                # 12: i=0, j=1
                # 21: i=1, j=0
                # 22: i=1, j=1
                
                if a == 0:
                    u_rows = [0, 1, 4, 5]
                else:
                    u_rows = [2, 3, 6, 7]
                
                for idx in range(4):
                    val = strassen_u[s][idx]
                    if val != 0:
                        U[u_rows[idx]][col] = val
                
                # V
                # B rows for block b
                # b=0: cols 0,1 -> V rows k=0,1
                # b=1: cols 2,3 -> V rows k=2,3
                # ...
                # Sub-block B is 2x2.
                # B11: j=0, k=2b
                # B12: j=0, k=2b+1
                # B21: j=1, k=2b
                # B22: j=1, k=2b+1
                
                # But B is split into B1 (rows 0,1) and B2 (rows 2,3).
                # We are multiplying A(a) by B(part).
                # If a=0, we use B1 (rows 0,1).
                # If a=1, we use B2 (rows 2,3).
                
                if a == 0:
                    v_base_rows = [0, 8] # j=0, j=1
                else:
                    v_base_rows = [16, 24] # j=2, j=3
                
                v_k_offset = 2 * b
                
                for idx in range(4):
                    val = strassen_v[s][idx]
                    if val != 0:
                        # idx 0: B11 (j=0, k=0)
                        # idx 1: B12 (j=0, k=1)
                        # idx 2: B21 (j=1, k=0)
                        # idx 3: B22 (j=1, k=1)
                        
                        j_offset = 0 if idx < 2 else 1
                        k_offset = 0 if idx % 2 == 0 else 1
                        
                        v_row = v_base_rows[j_offset] + v_k_offset + k_offset
                        V[v_row][col] = val
                        
                # W
                # C rows for block b
                # C is 2x8.
                # Block b covers cols 2b, 2b+1.
                # C11: i=0, k=2b
                # C12: i=0, k=2b+1
                # C21: i=1, k=2b
                # C22: i=1, k=2b+1
                
                # W index: k * n + i = k * 2 + i
                # C11: k=2b, i=0 -> 4b
                # C12: k=2b+1, i=0 -> 4b+2
                # C21: k=2b, i=1 -> 4b+1
                # C22: k=2b+1, i=1 -> 4b+3
                
                w_rows = [4*b, 4*b+2, 4*b+1, 4*b+3]
                
                for idx in range(4):
                    val = strassen_w[s][idx]
                    if val != 0:
                        W[w_rows[idx]][col] = val
                        
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
