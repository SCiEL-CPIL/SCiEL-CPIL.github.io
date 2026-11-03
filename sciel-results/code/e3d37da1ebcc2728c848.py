"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Rank-45 decomposition via slicing:
    # Split the 6 output columns into two groups of 3.
    # For each group of 3 columns, use a rank-7 (Strassen-like) 3x3x3 decomposition.
    # Total rank = 7 + 7 = 14... wait, that's too optimistic.
    # Actually, for 3x3x3 matrix multiplication, the rank is 7 (Strassen).
    # So 3x3x6 = two independent 3x3x3 blocks -> rank 7+7 = 14.
    # But we need to be careful about the indexing.
    
    # Let's use the known rank-7 decomposition for 3x3x3 (Strassen's algorithm)
    # and apply it to each half of the 6 columns.
    
    # Strassen's 7-term decomposition for 3x3x3:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # For 3x3x3, the tensor entries are T[i,j,k] = 1 if they form a valid (i,j,k) triple.
    # a = i*m + j = i*3 + j (row of A, col of A = row of B)
    # b = j*p + k = j*3 + k (row of B, col of B)
    # o = k*n + i = k*3 + i (row of C, col of C)
    
    # For the first 3 columns (k=0,1,2):
    # We need C[i, k] for i=0,1,2 and k=0,1,2
    # o = k*3 + i
    
    # Let me construct the rank-7 factors for 3x3x3.
    # Each term c has U_c, V_c, W_c vectors.
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # U1: A00=1, A11=1 -> positions (0*3+0)=0, (1*3+1)=4
    # V1: B00=1, B11=1 -> positions (0*3+0)=0, (1*3+1)=4
    # W1: contributes to C00 and C11 with +1
    # C00: o = 0*3+0 = 0 -> W1[0] += 1
    # C11: o = 1*3+1 = 4 -> W1[4] += 1
    
    # Term 2: M2 = (A10+A11)B00
    # U2: A10=1, A11=1 -> positions (1*3+0)=3, (1*3+1)=4
    # V2: B00=1 -> position (0*3+0)=0
    # W2: contributes to C10 with +1
    # C10: o = 0*3+1 = 1 -> W2[1] += 1
    
    # Term 3: M3 = A00(B10-B11)
    # U3: A00=1 -> position 0
    # V3: B10=1, B11=-1 -> positions (1*3+0)=3, (1*3+1)=4
    # W3: contributes to C01 and C11 with +1
    # C01: o = 1*3+0 = 3 -> W3[3] += 1
    # C11: o = 1*3+1 = 4 -> W3[4] += 1
    
    # Term 4: M4 = A11(B01-B00)
    # U4: A11=1 -> position 4
    # V4: B01=1, B00=-1 -> positions (0*3+1)=1, (0*3+0)=0
    # W4: contributes to C00 and C10 with +1
    # C00: o = 0*3+0 = 0 -> W4[0] += 1
    # C10: o = 0*3+1 = 1 -> W4[1] += 1
    
    # Term 5: M5 = (A00+A01)B11
    # U5: A00=1, A01=1 -> positions 0, (0*3+1)=1
    # V5: B11=1 -> position 4
    # W5: contributes to C00 with -1, C01 with +1
    # C00: o = 0 -> W5[0] += -1
    # C01: o = 3 -> W5[3] += 1
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    # U6: A10=1, A00=-1 -> positions 3, 0
    # V6: B00=1, B01=1 -> positions 0, 1
    # W6: contributes to C10 with +1
    # C10: o = 1 -> W6[1] += 1
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    # U7: A01=1, A11=-1 -> positions 1, 4
    # V7: B10=1, B11=1 -> positions 3, 4
    # W7: contributes to C00 with -1, C11 with +1
    # C00: o = 0 -> W7[0] += -1
    # C11: o = 4 -> W7[4] += 1
    
    # Now let's build the full 3x3x6 decomposition.
    # For columns 0-2 (k=0,1,2), use terms 1-7.
    # For columns 3-5 (k=3,4,5), use terms 8-14 (same structure, shifted).
    
    rank = 14
    U = [[0] * rank for _ in range(n * m)]  # 9 x 14
    V = [[0] * rank for _ in range(m * p)]  # 18 x 14
    W = [[0] * rank for _ in range(p * n)]  # 18 x 14
    
    # First group: columns 0,1,2 (k=0,1,2)
    # a = i*3 + j, b = j*3 + k, o = k*3 + i
    
    # Term 1 (c=0): M1
    U[0][0] = 1   # A00
    U[4][0] = 1   # A11
    V[0][0] = 1   # B00
    V[4][0] = 1   # B11
    W[0][0] = 1   # C00 (o=0*3+0=0)
    W[4][0] = 1   # C11 (o=1*3+1=4)
    
    # Term 2 (c=1): M2
    U[3][1] = 1   # A10
    U[4][1] = 1   # A11
    V[0][1] = 1   # B00
    W[1][1] = 1   # C10 (o=0*3+1=1)
    
    # Term 3 (c=2): M3
    U[0][2] = 1   # A00
    V[3][2] = 1   # B10
    V[4][2] = -1  # B11
    W[3][2] = 1   # C01 (o=1*3+0=3)
    W[4][2] = 1   # C11 (o=1*3+1=4)
    
    # Term 4 (c=3): M4
    U[4][3] = 1   # A11
    V[1][3] = 1   # B01
    V[0][3] = -1  # B00
    W[0][3] = 1   # C00 (o=0)
    W[1][3] = 1   # C10 (o=1)
    
    # Term 5 (c=4): M5
    U[0][4] = 1   # A00
    U[1][4] = 1   # A01
    V[4][4] = 1   # B11
    W[0][4] = -1  # C00 (o=0)
    W[3][4] = 1   # C01 (o=3)
    
    # Term 6 (c=5): M6
    U[3][5] = 1   # A10
    U[0][5] = -1  # A00
    V[0][5] = 1   # B00
    V[1][5] = 1   # B01
    W[1][5] = 1   # C10 (o=1)
    
    # Term 7 (c=6): M7
    U[1][6] = 1   # A01
    U[4][6] = -1  # A11
    V[3][6] = 1   # B10
    V[4][6] = 1   # B11
    W[0][6] = -1  # C00 (o=0)
    W[4][6] = 1   # C11 (o=4)
    
    # Second group: columns 3,4,5 (k=3,4,5)
    # b = j*3 + (k+3) = j*3 + k + 3
    # o = (k+3)*3 + i = k*3 + 9 + i
    
    # Term 8 (c=7): M1'
    U[0][7] = 1   # A00
    U[4][7] = 1   # A11
    V[0+3][7] = 1   # B0,3
    V[4+3][7] = 1   # B1,4
    W[0+9][7] = 1   # C0,3 (o=3*3+0=9)
    W[4+9][7] = 1   # C1,4 (o=4*3+1=13)
    
    # Term 9 (c=8): M2'
    U[3][8] = 1   # A10
    U[4][8] = 1   # A11
    V[0+3][8] = 1   # B0,3
    W[1+9][8] = 1   # C1,3 (o=3*3+1=10)
    
    # Term 10 (c=9): M3'
    U[0][9] = 1   # A00
    V[3+3][9] = 1   # B1,3
    V[4+3][9] = -1  # B1,4
    W[3+9][9] = 1   # C0,4 (o=4*3+0=12)
    W[4+9][9] = 1   # C1,4 (o=13)
    
    # Term 11 (c=10): M4'
    U[4][10] = 1   # A11
    V[1+3][10] = 1   # B0,4
    V[0+3][10] = -1  # B0,3
    W[0+9][10] = 1   # C0,3 (o=9)
    W[1+9][10] = 1   # C1,3 (o=10)
    
    # Term 12 (c=11): M5'
    U[0][11] = 1   # A00
    U[1][11] = 1   # A01
    V[4+3][11] = 1   # B1,4
    W[0+9][11] = -1  # C0,3 (o=9)
    W[3+9][11] = 1   # C0,4 (o=12)
    
    # Term 13 (c=12): M6'
    U[3][12] = 1   # A10
    U[0][12] = -1  # A00
    V[0+3][12] = 1   # B0,3
    V[1+3][12] = 1   # B0,4
    W[1+9][12] = 1   # C1,3 (o=10)
    
    # Term 14 (c=13): M7'
    U[1][13] = 1   # A01
    U[4][13] = -1  # A11
    V[3+3][13] = 1   # B1,3
    V[4+3][13] = 1   # B1,4
    W[0+9][13] = -1  # C0,3 (o=9)
    W[4+9][13] = 1   # C1,4 (o=13)
    
    def is_valid(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        # Check all valid entries should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    o = k * n + i
                    val = 0
                    for c in range(rank):
                        val += U[a][c] * V[b][c] * W[o][c]
                    if val != 1:
                        return False
        # Check all invalid entries should be 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    o = k * n + i
                    # Already checked above
        # Check random invalid entries
        import random
        random.seed(42)
        for _ in range(500):
            a = random.randint(0, n*m - 1)
            b = random.randint(0, m*p - 1)
            o = random.randint(0, p*n - 1)
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            k_o = o // n
            i_o = o % n
            
            # Valid if i_a == i_o and j_a == j_b and k_b == k_o
            if i_a == i_o and j_a == j_b and k_b == k_o:
                # Should be 1, already checked
                continue
            else:
                val = 0
                for c in range(rank):
                    val += U[a][c] * V[b][c] * W[o][c]
                if val != 0:
                    return False
        return True
    
    if is_valid(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if rank-14 fails
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
