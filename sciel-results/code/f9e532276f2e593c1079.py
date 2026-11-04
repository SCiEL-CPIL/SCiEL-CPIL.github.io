import time
import sys

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # We'll build a rank-45 decomposition:
    # Split the middle dimension (3) into 2+1.
    # A is 2x3: A = [A1 (2x2), A2 (2x1)]
    # B is 3x10: B = [B1 (2x10); B2 (1x10)]
    # C = A1*B1 + A2*B2
    # 
    # A1*B1: 2x2 * 2x10 = 5 blocks of 2x2*2x2
    # Use Strassen for each 2x2x2 block: rank 7 each, total 35
    # A2*B2: 2x1 * 1x10 = 2x10, rank 10 (schoolbook)
    # Total rank: 35 + 10 = 45
    
    rank = 45
    U = [[0] * rank for _ in range(n * m)]  # 6 x 45
    V = [[0] * rank for _ in range(m * p)]  # 30 x 45
    W = [[0] * rank for _ in range(p * n)]  # 20 x 45
    
    # Strassen decomposition for 2x2x2:
    # C00 = M1 - M4 + M6 + M7
    # C01 = M5 + M4
    # C10 = M1 + M2
    # C11 = M3 - M2 + M5 + M7
    # where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # We'll handle 5 blocks of 2x2x2 for A1*B1
    # Block b corresponds to columns 2*b and 2*b+1 of B1
    # A1 is rows 0,1 of A (columns 0,1 of the 3-column matrix)
    # B1 is rows 0,1 of B (all 10 columns)
    
    # U indexing: U[i*m + j, r] where i in [0,1], j in [0,1] for A1 part
    # V indexing: V[j*p + k, r] where j in [0,1], k in [0,9] for B1 part
    # W indexing: W[k*n + i, r] where k in [0,9], i in [0,1] for C part
    
    # For the 2x1x10 part (A2*B2):
    # A2 is row 2 of A (column 2), B2 is row 2 of B
    # C = A2 * B2, so C[i,k] = A2[i] * B2[k]
    # U[2*m + i, r] = A2[i] = U[2*3 + i, r] = U[6+i, r] but wait, n*m = 6, so indices 0-5
    # Actually U has 6 rows: i in [0,1], j in [0,2], so i*m+j: (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5
    # A1 uses j=0,1: rows 0,1,3,4
    # A2 uses j=2: rows 2,5
    # V has 30 rows: j in [0,2], k in [0,9], j*p+k
    # B1 uses j=0,1: rows 0-19
    # B2 uses j=2: rows 20-29
    # W has 20 rows: k in [0,9], i in [0,1], k*n+i
    
    # Strassen for block b (columns 2b, 2b+1 of B1):
    # r_start = 7*b
    for b in range(5):
        if time.time() > deadline:
            break
        r0 = 7 * b
        # M1 = (A00+A11)(B00+B11)
        # A00 = U[0], A11 = U[4]
        # B00 = V[2b], B11 = V[2b+1]
        U[0][r0] = 1; U[4][r0] = 1
        V[2*b][r0] = 1; V[2*b+1][r0] = 1
        # W: C00 += M1, C11 += M1
        # C00 is W[2b*2+0] = W[4b], C11 is W[2b+1, 1] = W[4b+3]
        W[4*b][r0] = 1; W[4*b+3][r0] = 1
        
        # M2 = (A10+A11)B00
        # A10 = U[3], A11 = U[4]
        # B00 = V[2b]
        U[3][r0+1] = 1; U[4][r0+1] = 1
        V[2*b][r0+1] = 1
        # W: C10 += M2
        # C10 is W[2b, 1] = W[4b+1]
        W[4*b+1][r0+1] = 1
        
        # M3 = A00(B10-B11)
        # A00 = U[0]
        # B10 = V[2b+1], B11 = V[2b+2]... wait, B10 is row 1, col 2b, B11 is row 1, col 2b+1
        # V[j*p + k]: B10 = V[1*10 + 2b] = V[10+2b], B11 = V[1*10 + 2b+1] = V[10+2b+1]
        U[0][r0+2] = 1
        V[10+2*b][r0+2] = 1; V[10+2*b+1][r0+2] = -1
        # W: C10 += M3
        W[4*b+1][r0+2] = 1
        
        # M4 = A11(B01-B10)
        # A11 = U[4]
        # B01 = V[0*10 + 2b+1] = V[2b+1], B10 = V[10+2b]
        U[4][r0+3] = 1
        V[2*b+1][r0+3] = 1; V[10+2*b][r0+3] = -1
        # W: C01 += M4, C10 += M4... wait
        # C01 = M5 + M4, C10 = M1 + M2
        # So M4 only contributes to C01
        # C01 is W[2b+1, 0] = W[4b+2]
        W[4*b+2][r0+3] = 1
        
        # M5 = (A00+A01)B11
        # A00 = U[0], A01 = U[1]
        # B11 = V[10+2b+1]
        U[0][r0+4] = 1; U[1][r0+4] = 1
        V[10+2*b+1][r0+4] = 1
        # W: C01 += M5, C11 += M5
        # C01 = W[4b+2], C11 = W[4b+3]
        W[4*b+2][r0+4] = 1; W[4*b+3][r0+4] = 1
        
        # M6 = (A10-A00)(B00+B10)
        # A10 = U[3], A00 = U[0]
        # B00 = V[2b], B10 = V[10+2b]
        U[3][r0+5] = 1; U[0][r0+5] = -1
        V[2*b][r0+5] = 1; V[10+2*b][r0+5] = 1
        # W: C00 += M6
        # C00 = W[4b]
        W[4*b][r0+5] = 1
        
        # M7 = (A01-A11)(B01+B11)
        # A01 = U[1], A11 = U[4]
        # B01 = V[2b+1], B11 = V[10+2b+1]
        U[1][r0+6] = 1; U[4][r0+6] = -1
        V[2*b+1][r0+6] = 1; V[10+2*b+1][r0+6] = 1
        # W: C00 += M7, C11 += M7
        # C00 = W[4b], C11 = W[4b+3]
        W[4*b][r0+6] = 1; W[4*b+3][r0+6] = 1
    
    # Now the 2x1x10 part: A2*B2
    # A2 = rows 2,5 of U (i=0,j=2 and i=1,j=2)
    # B2 = rows 20-29 of V (j=2, k=0-9)
    # C[i,k] = A2[i] * B2[k]
    # For each k in 0-9:
    #   r = 35 + k
    #   U[2][r] = 1 (A2[0])
    #   U[5][r] = 1 (A2[1])
    #   V[20+k][r] = 1 (B2[k])
    #   W[k*2+0][r] = 1 (C[0,k])
    #   W[k*2+1][r] = 1 (C[1,k])
    for k in range(10):
        if time.time() > deadline:
            break
        r = 35 + k
        U[2][r] = 1
        U[5][r] = 1
        V[20+k][r] = 1
        W[k*2][r] = 1
        W[k*2+1][r] = 1
    
    return {"U": U, "V": V, "W": W}
