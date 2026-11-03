"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # We decompose the 2x3x5 multiplication into:
    # - Left 2x2x5 block: A[0:2, 0:2] * B[0:2, :] -> C[0:2, :]
    # - Right 2x1x5 block: A[0:2, 2:3] * B[2:3, :] -> C[0:2, :]
    #
    # For the 2x2x5 block, split B's 5 columns into 2+3:
    #   2x2x2 (Strassen, rank 7) + 2x2x3 (rank 11) = rank 18
    # For the 2x1x5 block: rank 10
    # Total rank = 28
    
    U_rows = n * m  # 6
    V_rows = m * p  # 15
    W_rows = p * n  # 10
    
    rank = 28
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    col = 0
    
    # === Part 1: 2x2x2 Strassen block (A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]) ===
    # Strassen formulas for 2x2:
    # M1 = (A11)(B11+B22)
    # M2 = (A11+A22)(B22)
    # M3 = (A21)(B11+B12)
    # M4 = (A22)(B21+B22)
    # M5 = (A11+A22)(B11+B22)
    # M6 = (A11-A22)(B21+B22)
    # M7 = (A21-A11)(B11+B12)
    # C11 = M5 + M4 - M2 - M4 -> wait let me use standard Strassen
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
    
    # Indices: A row i, col j; B row j, col k; C row i, col k
    # For 2x2x2 sub-block: i in {0,1}, j in {0,1}, k in {0,1}
    # U index: i*m + j = i*3 + j
    # V index: j*p + k = j*5 + k
    # W index: k*n + i = k*2 + i
    
    # Strassen 2x2x2:
    # M1 = (A11+A22)(B11+B22): U[0,0]+U[1,1] * V[0,0]+V[1,1] -> contributes to C
    # Let me define the 7 rank-1 terms for 2x2x2:
    # Term 1: (A00+A11)(B00+B11) -> C00 += 1, C01 += 1, C10 += 1, C11 -= 1
    # Wait, let me be more careful.
    
    # Standard Strassen for C = A*B where A,B,C are 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each rank-1 term: U vector, V vector, W vector
    # For term t, U[i,j], V[j,k], W[k,i]
    
    # Term 1: (A00+A11)(B00+B11)
    # U: [0,0]=1, [1,1]=1
    # V: [0,0]=1, [1,1]=1
    # W: [0,0]=1, [0,1]=1, [1,0]=1, [1,1]=-1  (C00=1,C01=1,C10=1,C11=-1)
    U[0][col] = 1; U[3][col] = 1  # A00 (i=0,j=0 -> idx 0*3+0=0), A11 (i=1,j=1 -> idx 1*3+1=4)
    # Wait: U index is i*m+j. For i=0,j=0: 0. For i=1,j=1: 1*3+1=4.
    U[4][col] = 1
    V[0][col] = 1; V[6][col] = 1  # B00 (j=0,k=0 -> 0*5+0=0), B11 (j=1,k=1 -> 1*5+1=6)
    W[0][col] = 1; W[1][col] = 1; W[2][col] = 1; W[3][col] = -1
    # W index: k*n+i. C00: k=0,i=0 -> 0. C01: k=1,i=0 -> 2. C10: k=0,i=1 -> 1. C11: k=1,i=1 -> 3.
    # So W[0]=C00, W[1]=C10, W[2]=C01, W[3]=C11
    # C00 += 1, C01 += 1, C10 += 1, C11 -= 1
    # W[0]=1, W[2]=1, W[1]=1, W[3]=-1
    col += 1
    
    # Term 2: (A10+A11)(B00)
    # U: [1,0]=1, [1,1]=1 -> idx 1*3+0=3, 1*3+1=4
    # V: [0,0]=1 -> idx 0
    # W: C10 += 1, C11 -= 1 -> W[1]=1, W[3]=-1
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1
    col += 1
    
    # Term 3: (A00)(B01-B11)
    # U: [0,0]=1 -> idx 0
    # V: [0,1]=1, [1,1]=-1 -> idx 1, 6
    # W: C01 += 1 -> W[2]=1
    U[0][col] = 1
    V[1][col] = 1; V[6][col] = -1
    W[2][col] = 1
    col += 1
    
    # Term 4: (A11)(B10-B00)
    # U: [1,1]=1 -> idx 4
    # V: [1,0]=1, [0,0]=-1 -> idx 5, 0
    # W: C00 += 1, C10 += 1 -> W[0]=1, W[1]=1
    U[4][col] = 1
    V[5][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # Term 5: (A00+A01)(B11)
    # U: [0,0]=1, [0,1]=1 -> idx 0, 1
    # V: [1,1]=1 -> idx 6
    # W: C00 -= 1, C01 += 1 -> W[0]=-1, W[2]=1
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    W[0][col] = -1; W[2][col] = 1
    col += 1
    
    # Term 6: (A10-A00)(B00+B01)
    # U: [1,0]=1, [0,0]=-1 -> idx 3, 0
    # V: [0,0]=1, [0,1]=1 -> idx 0, 1
    # W: C11 += 1 -> W[3]=1
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1
    
    # Term 7: (A01-A11)(B10+B11)
    # U: [0,1]=1, [1,1]=-1 -> idx 1, 4
    # V: [1,0]=1, [1,1]=1 -> idx 5, 6
    # W: C00 += 1 -> W[0]=1
    U[1][col] = 1; U[4][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[0][col] = 1
    col += 1
    
    # col is now 7
    
    # === Part 2: 2x2x3 block (A[0:2,0:2] * B[0:2,2:5] -> C[0:2,2:5]) ===
    # This is multiplying 2x2 by 2x3. Known optimal rank is 11.
    # I need a rank-11 decomposition for 2x2x3.
    # 
    # Let me use a known construction. For 2x2x3:
    # A is 2x2, B is 2x3, C is 2x3.
    # One approach: think of it as two 2x2x2 blocks sharing some structure.
    # 
    # Actually, let me use a direct construction. The tensor is T[i,j,k] = delta[i,j,k] 
    # for i in {0,1}, j in {0,1}, k in {0,1,2}.
    # 
    # A known rank-11 decomposition can be constructed. Let me use the following approach:
    # Split into k=0,1 (2x2x2, rank 7) and k=2 (2x2x1, rank 4), total 11.
    # But that's not minimal in general. Actually 7+4=11 which is the known bound.
    
    # For 2x2x1: A[2,2] * B[2,1] = C[2,1], rank 4 (each C[i] = sum_j A[i,j]*B[j,0])
    # Actually rank of 2x2x1 is 4 (matrix multiplication of 2x2 by 2x1).
    
    # So 2x2x3 = 2x2x2 (rank 7) + 2x2x1 (rank 4) = rank 11.
    
    # 2x2x2 part for k=0,1 (B columns 2,3 in original, which are k=2,3 in the 5-col B):
    # Wait, I need to be careful with indexing.
    # In the full problem: A is 2x3, B is 3x5, C is 2x5.
    # The 2x2x5 block uses A[0:2, 0:2] and B[0:2, 0:5].
    # I split B[0:2, 0:5] into B[0:2, 0:2] and B[0:2, 2:5].
    # So for the 2x2x2 part: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (done above)
    # For the 2x2x3 part: A[0:2,0:2] * B[0:2,2:5] -> C[0:2,2:5]
    
    # In the 2x2x3 sub-problem:
    # A' is 2x2 (rows 0,1 of A, cols 0,1)
    # B' is 2x3 (rows 0,1 of B, cols 2,3,4)
    # C' is 2x3 (rows 0,1 of C, cols 2,3,4)
    
    # Split B' into B'[0:2,0:1] and B'[0:2,1:3]:
    # 2x2x1: A' * B'[0:2,0] -> C'[0:2,0] (i.e., C[0:2,2])
    # 2x2x2: A' * B'[0:2,1:3] -> C'[0:2,1:3] (i.e., C[0:2,3:5])
    
    # 2x2x1: rank 4
    # For A[2,2] * B[2,1] = C[2,1]:
    # C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # 4 rank-1 terms:
    # (A00)(B00) -> C0 += 1
    # (A01)(B10) -> C0 += 1
    # (A10)(B00) -> C1 += 1
    # (A11)(B10) -> C1 += 1
    
    # In full indices:
    # A[i,j] -> U[i*3+j]
    # B[j,k] -> V[j*5+k]
    # C[i,k] -> W[k*2+i]
    
    # For 2x2x1 part (B col 2 in full, i.e., k=2):
    # (A00)(B02) -> C02: U[0]=1, V[2]=1, W[4]=1  (W[2*2+0]=W[4])
    U[0][col] = 1; V[2][col] = 1; W[4][col] = 1
    col += 1
    # (A01)(B12) -> C02: U[1]=1, V[7]=1, W[4]=1
    U[1][col] = 1; V[7][col] = 1; W[4][col] = 1
    col += 1
    # (A10)(B02) -> C12: U[3]=1, V[2]=1, W[5]=1  (W[2*2+1]=W[5])
    U[3][col] = 1; V[2][col] = 1; W[5][col] = 1
    col += 1
    # (A11)(B12) -> C12: U[4]=1, V[7]=1, W[5]=1
    U[4][col] = 1; V[7][col] = 1; W[5][col] = 1
    col += 1
    
    # col is now 11
    
    # 2x2x2 part for B cols 3,4 (k=3,4 in full):
    # A[0:2,0:2] * B[0:2,3:5] -> C[0:2,3:5]
    # This is another 2x2x2, use Strassen again.
    # B'[0,0]=B[0,3]=V[3], B'[0,1]=B[0,4]=V[4]
    # B'[1,0]=B[1,3]=V[8], B'[1,1]=B[1,4]=V[9]
    # C'[0,0]=C[0,3]=W[6], C'[0,1]=C[0,4]=W[7]
    # C'[1,0]=C[1,3]=W[8], C'[1,1]=C[1,4]=V[9]... wait W[4*2+1]=W[9]
    
    # Strassen for this 2x2x2:
    # Term 1: (A00+A11)(B00+B11) where B00=V[3], B11=V[9]
    U[0][col] = 1; U[4][col] = 1
    V[3][col] = 1; V[9][col] = 1
    W[6][col] = 1; W[7][col] = 1; W[8][col] = 1; W[9][col] = -1
    col += 1
    
    # Term 2: (A10+A11)(B00)
    U[3][col] = 1; U[4][col] = 1
    V[3][col] = 1
    W[8][col] = 1; W[9][col] = -1
    col += 1
    
    # Term 3: (A00)(B01-B11) where B01=V[4], B11=V[9]
    U[0][col] = 1
    V[4][col] = 1; V[9][col] = -1
    W[7][col] = 1
    col += 1
    
    # Term 4: (A11)(B10-B00) where B10=V[8], B00=V[3]
    U[4][col] = 1
    V[8][col] = 1; V[3][col] = -1
    W[6][col] = 1; W[8][col] = 1
    col += 1
    
    # Term 5: (A00+A01)(B11)
    U[0][col] = 1; U[1][col] = 1
    V[9][col] = 1
    W[6][col] = -1; W[7][col] = 1
    col += 1
    
    # Term 6: (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[9][col] = 1
    col += 1
    
    # Term 7: (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[8][col] = 1; V[9][col] = 1
    W[6][col] = 1
    col += 1
    
    # col is now 18
    
    # === Part 3: 2x1x5 block (A[0:2,2] * B[2,0:5] -> C[0:2,0:5]) ===
    # A[0,2]*B[2,k] -> C[0,k]
    # A[1,2]*B[2,k] -> C[1,k]
    # 10 rank-1 terms
    
    for k in range(5):
        # A[0,2] * B[2,k] -> C[0,k]: U[2]=1, V[10+k]=1, W[k*2+0]=W[k*2]=1
        U[2][col] = 1; V[10 + k][col] = 1; W[k * 2][col] = 1
        col += 1
        # A[1,2] * B[2,k] -> C[1,k]: U[5]=1, V[10+k]=1, W[k*2+1]=1
        U[5][col] = 1; V[10 + k][col] = 1; W[k * 2 + 1][col] = 1
        col += 1
    
    # col should be 28
    assert col == rank, f"col={col}, rank={rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
