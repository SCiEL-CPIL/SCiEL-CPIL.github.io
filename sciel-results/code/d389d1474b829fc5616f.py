"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose B into two 3x2 blocks: B1 (cols 0,1) and B2 (cols 2,3)
    # A is 2x3. We split A into A1 (cols 0,1) 2x2 and A2 (col 2) 2x1.
    # C = A1 B1 + A2 B2_part? No.
    # C = A B = A1 B1 + A2 B2?
    # A is 2x3. B is 3x4.
    # A = [A_L (2x2), A_R (2x1)]
    # B = [B_T (2x4); B_B (1x4)]
    # This split is not what I used above.
    # My previous split:
    # B = [B1 (3x2), B2 (3x2)]
    # A = [A1 (2x2), A2 (2x1)]
    # C = A1 B1 + A2 B2?
    # A1 B1 is 2x2. A2 B2 is 2x2.
    # But A1 B1 uses A1 (cols 0,1) and B1 (rows 0,1).
    # What about A1 B2? A1 is 2x2, B2 is 3x2. Dimensions don't match for direct mult.
    # A1 (2x2) * B2 (3x2) is invalid.
    # A (2x3) * B (3x4).
    # C_ij = sum_k A_ik B_kj.
    # Split B into B1 (rows 0,1) and B2 (row 2).
    # B1 is 2x4. B2 is 1x4.
    # C = A1 B1 + A2 B2.
    # A1 is 2x2 (cols 0,1 of A). A2 is 2x1 (col 2 of A).
    # A1 B1: 2x2 * 2x4 -> 2x4.
    # A2 B2: 2x1 * 1x4 -> 2x4.
    # This works.
    
    # Rank of A1 B1 (2x2 * 2x4):
    # Split B1 into B1a (2x2) and B1b (2x2).
    # A1 B1a: 2x2 * 2x2 -> Rank 7 (Strassen).
    # A1 B1b: 2x2 * 2x2 -> Rank 7 (Strassen).
    # Total 14?
    # Wait, A1 is shared. Can we combine?
    # Strassen uses linear combos of A1.
    # If we do two separate Strassens, we use 7+7=14 terms.
    # Can we do better?
    # Rank of 2x2 * 2x4 is 12?
    # If rank is 12, then A2 B2 (2x1 * 1x4) is rank 2?
    # A2 B2: C_ij = A_i2 B_2j.
    # This is rank 1? No, it's 2x4.
    # C_ij = A_i2 B_2j.
    # This is an outer product of A2 (2x1) and B2 (1x4).
    # Rank 1.
    # So if Rank(2x2*2x4) is 12, total is 13.
    # Is Rank(2x2*2x4) 12?
    # Schoolbook is 8. Strassen for two 2x2 blocks is 14.
    # Actually, Rank(2x2*2x4) is 12?
    # Let's check.
    # If I use Strassen for each 2x2 block, I get 14.
    # If I can find a rank 12 decomposition for 2x2*2x4, I save 2.
    # But 14 is a safe upper bound.
    # Total 14 + 1 = 15.
    
    # Let's implement the 15-rank decomposition.
    # It is better than 19 and 24.
    
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block 1: A1 (2x2) * B1a (2x2) -> C1a (2x2)
    # Strassen for 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Indices for A1: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4 (since m=3, stride 3)
    # Indices for B1a: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5 (since p=4, stride 4)
    # Output C1a: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3 (since n=2, stride 2)
    
    def add_strassen(a_idx, b_idx, c_idx, offset):
        nonlocal col
        # a_idx: list of 4 indices [11, 12, 21, 22] in flattened A
        # b_idx: list of 4 indices [11, 12, 21, 22] in flattened B
        # c_idx: list of 4 indices [11, 12, 21, 22] in flattened C
        
        # M1
        U[a_idx[0]][col] += 1; U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1; V[b_idx[3]][col] += 1
        W[c_idx[0]][col] += 1; W[c_idx[1]][col] += 1; W[c_idx[2]][col] += 1; W[c_idx[3]][col] += 1
        col += 1
        
        # M2
        U[a_idx[2]][col] += 1; U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        W[c_idx[2]][col] += 1; W[c_idx[3]][col] += 1
        col += 1
        
        # M3
        U[a_idx[0]][col] += 1
        V[b_idx[2]][col] += 1; V[b_idx[3]][col] -= 1
        W[c_idx[0]][col] += 1; W[c_idx[2]][col] += 1
        col += 1
        
        # M4
        U[a_idx[3]][col] += 1
        V[b_idx[1]][col] += 1; V[b_idx[0]][col] -= 1
        W[c_idx[1]][col] += 1; W[c_idx[3]][col] += 1
        col += 1
        
        # M5
        U[a_idx[0]][col] += 1; U[a_idx[1]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[1]][col] += 1; W[c_idx[3]][col] += 1
        col += 1
        
        # M6
        U[a_idx[2]][col] += 1; U[a_idx[0]][col] -= 1
        V[b_idx[0]][col] += 1; V[b_idx[1]][col] += 1
        W[c_idx[0]][col] += 1; W[c_idx[2]][col] += 1
        col += 1
        
        # M7
        U[a_idx[1]][col] += 1; U[a_idx[3]][col] -= 1
        V[b_idx[2]][col] += 1; V[b_idx[3]][col] += 1
        W[c_idx[1]][col] += 1; W[c_idx[3]][col] += 1
        col += 1

    # A1 indices: (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4
    a1_idx = [0, 1, 3, 4]
    
    # B1a indices (cols 0,1 of B, rows 0,1):
    # (0,0)=0, (0,1)=1, (1,0)=4, (1,1)=5
    b1a_idx = [0, 1, 4, 5]
    
    # C1a indices (cols 0,1 of C, rows 0,1):
    # (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    c1a_idx = [0, 1, 2, 3]
    
    add_strassen(a1_idx, b1a_idx, c1a_idx, 0)
    
    # B1b indices (cols 2,3 of B, rows 0,1):
    # (0,2)=2, (0,3)=3, (1,2)=6, (1,3)=7
    b1b_idx = [2, 3, 6, 7]
    
    # C1b indices (cols 2,3 of C, rows 0,1):
    # (0,2)=2, (0,3)=3, (1,2)=4, (1,3)=5
    c1b_idx = [2, 3, 4, 5]
    
    add_strassen(a1_idx, b1b_idx, c1b_idx, 0)
    
    # Block 2: A2 (2x1) * B2 (1x4)
    # A2 indices: (0,2)=2, (1,2)=5
    # B2 indices: (2,0)=8, (2,1)=9, (2,2)=10, (2,3)=11
    # C2 indices: all 8 indices of C
    
    # C_ij = A_i2 B_2j
    # This is rank 1.
    # U has 1 at 2 and 5.
    # V has 1 at 8,9,10,11.
    # W has 1 at all c indices.
    
    U[2][col] = 1
    U[5][col] = 1
    V[8][col] = 1
    V[9][col] = 1
    V[10][col] = 1
    V[11][col] = 1
    for i in range(n):
        for j in range(p):
            W[j * n + i][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
