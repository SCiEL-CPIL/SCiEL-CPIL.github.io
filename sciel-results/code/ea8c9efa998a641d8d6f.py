"""Matrix multiplication 2x3 by 3x5 - rank 28 decomposition using Strassen for 2x2x2 sub-blocks."""
DIMENSIONS = [2, 3, 5]

def construct():
    """
    Rank-28 decomposition for 2x3 * 3x5 matrix multiplication.
    
    Strategy: Split the 3x3 middle dimension into 2+1.
    - For the 2x2x2 sub-blocks: use Strassen (rank 7 instead of 8), saving 1 per block
    - For the 2x2x1, 2x1x2, 1x1x2, 1x2x1, 1x1x1 blocks: schoolbook
    
    The 3x3 middle is split as (2,1) for both A's columns and B's rows.
    This creates sub-problems:
    - (2 cols of A) x (2 rows of B): 2x2x2 -> Strassen rank 7
    - (2 cols of A) x (1 row of B): 2x1x2 -> schoolbook rank 4
    - (1 col of A) x (2 rows of B): 1x2x2 -> schoolbook rank 4
    - (1 col of A) x (1 row of B): 1x1x2 -> schoolbook rank 2
    
    Wait, that gives 7+4+4+2 = 17, not 28.
    
    Let me reconsider. The full problem is 2x3 * 3x5.
    n=2, m=3, p=5.
    
    Split m=3 into m1=2, m2=1.
    Split n=2 into n1=2, n2=0 (no split needed for n since it's already 2).
    Split p=5 into p1=2, p2=3? Or p1=3, p2=2?
    
    Actually, the key insight from the knowledge: "rank-11 decomposition for the 2x2x3 sub-block 
    (by combining Strassen for 2x2x2 and schoolbook for 2x2x1)".
    
    So the 2x2x3 sub-block has rank 11 = 7 (Strassen 2x2x2) + 4 (schoolbook 2x2x1).
    
    For 2x3x5: split the 3 (m dimension) into 2+1, giving:
    - 2x2x5 sub-block: split 5 into 2+3? No...
    
    Let me think again. The tensor is T[i,j,k] = delta_{i,k} for the multiplication.
    Actually T[i,j,k] = 1 if the output position (i,k) depends on A[i,j]*B[j,k].
    
    For 2x3 * 3x5: output is 2x5, so i in {0,1}, k in {0,1,2,3,4}, j in {0,1,2}.
    
    Split j into j1 in {0,1} and j2 = {2}.
    
    Sub-block 1: i in {0,1}, j1 in {0,1}, k in {0,1,2,3,4} -> 2x2x5
    Sub-block 2: i in {0,1}, j2={2}, k in {0,1,2,3,4} -> 2x1x5
    
    For sub-block 1 (2x2x5): split k into k1 in {0,1} and k2 in {2,3,4}
    - 2x2x2: Strassen rank 7
    - 2x2x3: schoolbook rank 12
    Total for sub-block 1: 7 + 12 = 19
    
    For sub-block 2 (2x1x5): schoolbook rank 10
    
    Total: 19 + 10 = 29. Not 28.
    
    Alternative: split k into k1 in {0,1,2} and k2 in {3,4}
    - 2x2x3: ? 
    - 2x2x2: Strassen rank 7
    
    Hmm. Let me try: for 2x2x3, can we do better than 12?
    2x2x3 = 2x2x2 + 2x2x1 = 7 + 4 = 11 (Strassen + schoolbook)
    
    So for sub-block 1 (2x2x5): split k into k1 in {0,1,2} and k2 in {3,4}
    - 2x2x3: rank 11
    - 2x2x2: rank 7
    Total: 18
    
    Sub-block 2 (2x1x5): rank 10
    Total: 28. Yes!
    
    So the decomposition:
    - i in {0,1}, j in {0,1}, k in {0,1,2}: 2x2x3, rank 11 (Strassen 2x2x2 + schoolbook 2x2x1)
    - i in {0,1}, j in {0,1}, k in {3,4}: 2x2x2, rank 7 (Strassen)
    - i in {0,1}, j={2}, k in {0,1,2,3,4}: 2x1x5, rank 10 (schoolbook)
    
    Total rank: 11 + 7 + 10 = 28
    """
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # U index: i*m + j (i in 0..1, j in 0..2)
    # V index: j*p + k (j in 0..2, k in 0..4)
    # W index: k*n + i (k in 0..4, i in 0..1)
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    col = 0
    
    # Part 1: 2x2x3 sub-block (i in {0,1}, j in {0,1}, k in {0,1,2})
    # Split k into k1 in {0,1} (2x2x2, Strassen) and k2={2} (2x2x1, schoolbook)
    
    # 2x2x2 Strassen part: i in {0,1}, j in {0,1}, k in {0,1}
    # Strassen products for A[i,j] * B[j,k] where i,j,k in {0,1}
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a01(b11-b01)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # U entries for Strassen (U[i*m+j, col])
    # V entries (V[j*p+k, col])
    # W entries (W[k*n+i, col])
    
    # M1 = (a00+a11)(b00+b11)
    # U: a00 -> U[0*3+0]=U[0], a11 -> U[1*3+1]=U[4]
    # V: b00 -> V[0*5+0]=V[0], b11 -> V[1*5+1]=V[6]
    # W: contributes to C[0,0], C[0,1], C[1,0], C[1,1]
    # C[0,0] += M1, C[0,1] += M1, C[1,0] += M1, C[1,1] += M1
    # W[k*n+i]: C[0,0]->W[0*2+0]=W[0], C[0,1]->W[1*2+0]=W[2], C[1,0]->W[0*2+1]=W[1], C[1,1]->W[1*2+1]=W[3]
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[2][col] = 1; W[1][col] = 1; W[3][col] = 1
    col += 1
    
    # M2 = (a10+a11)b00
    # U: a10 -> U[1*3+0]=U[3], a11 -> U[4]
    # V: b00 -> V[0]
    # C[1,0] += M2, C[1,1] += M2
    # W: C[1,0]->W[1], C[1,1]->W[3]
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = 1
    col += 1
    
    # M3 = a01(b11-b01)
    # U: a01 -> U[0*3+1]=U[1]
    # V: b11 -> V[6], b01 -> V[1]
    # C[0,1] += M3, C[1,1] += M3
    # W: C[0,1]->W[2], C[1,1]->W[3]
    U[1][col] = 1
    V[6][col] = 1; V[1][col] = -1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # M4 = a11(b10-b00)
    # U: a11 -> U[4]
    # V: b10 -> V[5], b00 -> V[0]
    # C[0,0] += M4, C[1,0] += M4
    # W: C[0,0]->W[0], C[1,0]->W[1]
    U[4][col] = 1
    V[5][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5 = (a00+a01)b11
    # U: a00 -> U[0], a01 -> U[1]
    # V: b11 -> V[6]
    # C[0,0] -= M5, C[1,0] -= M5
    # W: C[0,0]->W[0], C[1,0]->W[1]
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    W[0][col] = -1; W[1][col] = -1
    col += 1
    
    # M6 = (a10-a00)(b00+b01)
    # U: a10 -> U[3], a00 -> U[0]
    # V: b00 -> V[0], b01 -> V[1]
    # C[1,1] += M6
    # W: C[1,1]->W[3]
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1
    
    # M7 = (a01-a11)(b10+b11)
    # U: a01 -> U[1], a11 -> U[4]
    # V: b10 -> V[5], b11 -> V[6]
    # C[0,0] += M7
    # W: C[0,0]->W[0]
    U[1][col] = 1; U[4][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[0][col] = 1
    col += 1
    
    # 2x2x1 schoolbook part: i in {0,1}, j in {0,1}, k={2}
    # 4 terms: (i,j) pairs
    for i in range(2):
        for j in range(2):
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
    
    # Part 2: 2x2x2 sub-block (i in {0,1}, j in {0,1}, k in {3,4})
    # Strassen again, with k offset by 3
    # A = [[a00, a01], [a10, a11]], B = [[b03, b04], [b13, b14]]
    # Same Strassen structure, but B indices shifted
    
    # M1' = (a00+a11)(b03+b14)
    U[0][col] = 1; U[4][col] = 1
    V[0*5+3][col] = 1; V[1*5+4][col] = 1
    # C[0,3], C[0,4], C[1,3], C[1,4]
    # W: C[0,3]->W[3*2+0]=W[6], C[0,4]->W[4*2+0]=W[8], C[1,3]->W[3*2+1]=W[7], C[1,4]->W[4*2+1]=W[9]
    W[6][col] = 1; W[8][col] = 1; W[7][col] = 1; W[9][col] = 1
    col += 1
    
    # M2' = (a10+a11)b03
    U[3][col] = 1; U[4][col] = 1
    V[0*5+3][col] = 1
    W[7][col] = 1; W[9][col] = 1
    col += 1
    
    # M3' = a01(b14-b04)
    U[1][col] = 1
    V[1*5+4][col] = 1; V[0*5+4][col] = -1
    W[8][col] = 1; W[9][col] = 1
    col += 1
    
    # M4' = a11(b13-b03)
    U[4][col] = 1
    V[1*5+3][col] = 1; V[0*5+3][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # M5' = (a00+a01)b14
    U[0][col] = 1; U[1][col] = 1
    V[1*5+4][col] = 1
    W[6][col] = -1; W[7][col] = -1
    col += 1
    
    # M6' = (a10-a00)(b03+b04)
    U[3][col] = 1; U[0][col] = -1
    V[0*5+3][col] = 1; V[0*5+4][col] = 1
    W[9][col] = 1
    col += 1
    
    # M7' = (a01-a11)(b13+b14)
    U[1][col] = 1; U[4][col] = -1
    V[1*5+3][col] = 1; V[1*5+4][col] = 1
    W[6][col] = 1
    col += 1
    
    # Part 3: 2x1x5 schoolbook (i in {0,1}, j={2}, k in {0,1,2,3,4})
    for i in range(2):
        for k in range(5):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}

