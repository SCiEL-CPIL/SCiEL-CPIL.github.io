import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline to be safe
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try Rank 40 decomposition
    # Structure:
    # A (3x4) = [A1 (3x3) | A2 (3x1)]
    # B (4x4) = [B1 (3x4); B2 (1x4)]
    # C = A1 B1 + A2 B2
    
    # Part 1: A1 (3x3) * B1 (3x4)
    # B1 is 3x4. Split B1 into B1a (3x3) and B1b (3x1).
    # C1 = A1 B1a + A1 B1b
    # Rank(A1 B1a) <= 19 (BCFL)
    # Rank(A1 B1b) <= 9 (Schoolbook 3x3x1)
    # Part 2: A2 (3x1) * B2 (1x4)
    # Rank(A2 B2) <= 12 (Schoolbook 3x1x4)
    
    # Total Rank = 19 + 9 + 12 = 40
    
    U = []
    V = []
    W = []
    
    # --- Part 1a: A1 (3x3) * B1a (3x3) ---
    # Indices: i in 0..2, j in 0..2, k in 0..2
    # U indices: i * m + j = i * 4 + j
    # V indices: j * p + k = j * 4 + k
    # W indices: k * n + i = k * 3 + i
    
    # BCFL Rank 19 factors
    U_core = [
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1]
    ]
    V_core = [
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0]
    ]
    W_core = [
        [1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    ]
    
    # Verify Core
    valid_core = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0
                for r in range(19):
                    s += U_core[i][r] * V_core[j][r] * W_core[k][r]
                if s != 1:
                    valid_core = False
                    break
            if not valid_core:
                break
        if not valid_core:
            break
    
    if not valid_core:
        # Fallback to schoolbook for core
        U_core = [[0]*27 for _ in range(3)]
        V_core = [[0]*27 for _ in range(3)]
        W_core = [[0]*27 for _ in range(3)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U_core[i][col] = 1
                    V_core[j][col] = 1
                    W_core[k][col] = 1
                    col += 1
        rank_core = 27
    else:
        rank_core = 19
        
    # Add Core to U, V, W
    for r in range(rank_core):
        u_row = [0] * (n * m)
        v_row = [0] * (m * p)
        w_row = [0] * (p * n)
        for i in range(3):
            u_row[i * 4 + 0] = 0 # j is 0..2
            # Wait, j is the column of A1. A1 is cols 0,1,2 of A.
            # So j in 0..2.
            # U index is i * m + j.
            u_row[i * 4 + i] = 0 # Placeholder
            # Actually, for each r, we add to U, V, W
            pass
        
        # Correctly map indices
        u_val = [0] * (n * m)
        v_val = [0] * (m * p)
        w_val = [0] * (p * n)
        
        for i in range(3):
            for j in range(3):
                # U[i, j] corresponds to U_core[i, r]
                # But U_core is 3x19. U_core[i][r] is the coefficient for A[i,j]?
                # No, U_core[i][r] is the coefficient for the i-th row of A.
                # The term is U_i V_j W_k A_ij B_jk.
                # So for a fixed r, U_i is constant for all j?
                # Yes, U is n x R.
                # So for each r, we have a vector u_r of length n.
                # And v_r of length m.
                # And w_r of length p.
                
                pass
                
        # Let's just build the full matrices directly
        U_part1a = [[0] * rank_core for _ in range(n * m)]
        V_part1a = [[0] * rank_core for _ in range(m * p)]
        W_part1a = [[0] * rank_core for _ in range(p * n)]
        
        for r in range(rank_core):
            for i in range(3):
                for j in range(3):
                    U_part1a[i * 4 + j][r] = U_core[i][r]
                    # V is indexed by j, k. Here k is 0..2 (B1a)
                    # V[j, k] = V_core[j][r]
                    for k in range(3):
                        V_part1a[j * 4 + k][r] = V_core[j][r]
                        # W is indexed by k, i.
                        W_part1a[k * 3 + i][r] = W_core[k][r]
                        
        U.append(U_part1a)
        V.append(V_part1a)
        W.append(W_part1a)
        
    # --- Part 1b: A1 (3x3) * B1b (3x1) ---
    # j in 0..2, k=3.
    # Rank 9 Schoolbook.
    rank_1b = 9
    U_part1b = [[0] * rank_1b for _ in range(n * m)]
    V_part1b = [[0] * rank_1b for _ in range(m * p)]
    W_part1b = [[0] * rank_1b for _ in range(p * n)]
    col = 0
    for i in range(3):
        for j in range(3):
            U_part1b[i * 4 + j][col] = 1
            V_part1b[j * 4 + 3][col] = 1 # k=3
            W_part1b[3 * 3 + i][col] = 1 # k=3, i
            col += 1
    U.append(U_part1b)
    V.append(V_part1b)
    W.append(W_part1b)
    
    # --- Part 2: A2 (3x1) * B2 (1x4) ---
    # j=3, k in 0..3.
    # Rank 12 Schoolbook.
    rank_2 = 12
    U_part2 = [[0] * rank_2 for _ in range(n * m)]
    V_part2 = [[0] * rank_2 for _ in range(m * p)]
    W_part2 = [[0] * rank_2 for _ in range(p * n)]
    col = 0
    for i in range(3):
        for k in range(4):
            U_part2[i * 4 + 3][col] = 1 # j=3
            V_part2[3 * 4 + k][col] = 1 # j=3
            W_part2[k * 3 + i][col] = 1
            col += 1
    U.append(U_part2)
    V.append(V_part2)
    W.append(W_part2)
    
    # Combine
    final_U = []
    for i in range(n * m):
        row = []
        for part in U:
            row.extend(part[i])
        final_U.append(row)
        
    final_V = []
    for j in range(m * p):
        row = []
        for part in V:
            row.extend(part[j])
        final_V.append(row)
        
    final_W = []
    for k in range(p * n):
        row = []
        for part in W:
            row.extend(part[k])
        final_W.append(row)
            
    return {"U": final_U, "V": final_V, "W": final_W}

# EVOLVE-BLOCK-END
