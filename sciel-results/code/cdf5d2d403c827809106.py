import time

def construct():
    """
    Rank-65 exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    Uses recursive Strassen (rank 7 for 2x2x2, rank 49 for 4x4x4) for the first 4 columns,
    plus 16 scalar terms for the 5th column.
    
    Returns:
        dict with 'U' (16 x 65), 'V' (20 x 65), 'W' (20 x 65)
    """
    start_time = time.time()
    
    n, m, p = 4, 4, 5
    rank = 65
    
    # Initialize matrices
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0.0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0.0] * rank for _ in range(p * n)]  # 20 x 65
    
    # Strassen 7 terms for 2x2x2 multiplication
    # Each term: U_factor (2x2), V_factor (2x2), W_factor (2x2)
    # Indices: U[i,j], V[j,k], W[k,i]
    
    # M1 = (P00+P11)*(Q00+Q11) -> R00 += 1, R11 += 1
    sU0 = {(0,0): 1, (1,1): 1}
    sV0 = {(0,0): 1, (1,1): 1}
    sW0 = {(0,0): 1, (1,1): 1}
    
    # M2 = (P10+P11)*Q00 -> R10 += 1, R11 -= 1
    sU1 = {(1,0): 1, (1,1): 1}
    sV1 = {(0,0): 1}
    sW1 = {(0,1): 1, (1,1): -1}
    
    # M3 = P00*(Q01-Q11) -> R01 += 1, R11 += 1
    sU2 = {(0,0): 1}
    sV2 = {(0,1): 1, (1,1): -1}
    sW2 = {(1,0): 1, (1,1): 1}
    
    # M4 = P11*(Q10-Q00) -> R00 += 1, R10 += 1
    sU3 = {(1,1): 1}
    sV3 = {(1,0): 1, (0,0): -1}
    sW3 = {(0,0): 1, (0,1): 1}
    
    # M5 = (P00+P01)*Q11 -> R00 -= 1, R01 += 1
    sU4 = {(0,0): 1, (0,1): 1}
    sV4 = {(1,1): 1}
    sW4 = {(0,0): -1, (1,0): 1}
    
    # M6 = (P10-P00)*(Q00+Q01) -> R11 += 1
    sU5 = {(1,0): 1, (0,0): -1}
    sV5 = {(0,0): 1, (0,1): 1}
    sW5 = {(1,1): 1}
    
    # M7 = (P01-P11)*(Q10+Q11) -> R00 += 1
    sU6 = {(0,1): 1, (1,1): -1}
    sV6 = {(1,0): 1, (1,1): 1}
    sW6 = {(0,0): 1}
    
    strassen_U = [sU0, sU1, sU2, sU3, sU4, sU5, sU6]
    strassen_V = [sV0, sV1, sV2, sV3, sV4, sV5, sV6]
    strassen_W = [sW0, sW1, sW2, sW3, sW4, sW5, sW6]
    
    # Top-level Strassen for 4x4x4 (using 2x2 blocks)
    # P_blocks[t]: which A blocks (2x2) are combined
    # Q_blocks[t]: which B blocks (2x2) are combined
    # W_top[t]: which C blocks (2x2) the result contributes to
    
    # M1_top = (A11+A22)(B11+B22)
    P0 = {(0,0): 1, (1,1): 1}
    Q0 = {(0,0): 1, (1,1): 1}
    Wtop0 = {(0,0): 1, (1,1): 1}  # C11 += 1, C22 += 1
    
    # M2_top = (A21+A22)B11
    P1 = {(1,0): 1, (1,1): 1}
    Q1 = {(0,0): 1}
    Wtop1 = {(1,0): 1, (1,1): -1}  # C21 += 1, C22 -= 1
    
    # M3_top = A11(B12-B22)
    P2 = {(0,0): 1}
    Q2 = {(0,1): 1, (1,1): -1}
    Wtop2 = {(0,1): 1, (1,1): 1}  # C12 += 1, C22 += 1
    
    # M4_top = A22(B21-B11)
    P3 = {(1,1): 1}
    Q3 = {(1,0): 1, (0,0): -1}
    Wtop3 = {(0,0): 1, (1,0): 1}  # C11 += 1, C21 += 1
    
    # M5_top = (A11+A12)B22
    P4 = {(0,0): 1, (0,1): 1}
    Q4 = {(1,1): 1}
    Wtop4 = {(0,0): -1, (0,1): 1}  # C11 -= 1, C12 += 1
    
    # M6_top = (A21-A11)(B11+B12)
    P5 = {(1,0): 1, (0,0): -1}
    Q5 = {(0,0): 1, (0,1): 1}
    Wtop5 = {(1,1): 1}  # C22 += 1
    
    # M7_top = (A12-A22)(B21+B22)
    P6 = {(0,1): 1, (1,1): -1}
    Q6 = {(1,0): 1, (1,1): 1}
    Wtop6 = {(0,0): 1}  # C11 += 1
    
    P_blocks = [P0, P1, P2, P3, P4, P5, P6]
    Q_blocks = [Q0, Q1, Q2, Q3, Q4, Q5, Q6]
    W_top = [Wtop0, Wtop1, Wtop2, Wtop3, Wtop4, Wtop5, Wtop6]
    
    # Build the 49 terms for 4x4x4 multiplication (first 4 columns of B)
    term_col = 0
    for t in range(7):
        for s in range(7):
            col = term_col
            term_col += 1
            
            # Build U factor: combine sub-level U with top-level P blocks
            for (si, sj), cu in strassen_U[s].items():
                for (bi, bj), cp in P_blocks[t].items():
                    global_i = 2 * bi + si
                    global_j = 2 * bj + sj
                    U[global_i * m + global_j][col] += cu * cp
            
            # Build V factor: combine sub-level V with top-level Q blocks
            for (sj, sk), cv in strassen_V[s].items():
                for (bj, bk), cq in Q_blocks[t].items():
                    global_j = 2 * bj + sj
                    global_k = 2 * bk + sk
                    if global_k < 4:  # Only first 4 columns of B
                        V[global_j * p + global_k][col] += cv * cq
            
            # Build W factor: combine sub-level W with top-level W distribution
            for (sk, si), cw in strassen_W[s].items():
                for (bk, bi), ct in W_top[t].items():
                    global_k = 2 * bk + sk
                    global_i = 2 * bi + si
                    W[global_k * n + global_i][col] += cw * ct
    
    # Add 16 scalar terms for the 5th column (k=4)
    # C[i,4] = sum_j A[i,j] * B[j,4]
    for i in range(n):
        for j in range(m):
            col = 49 + i * m + j
            U[i * m + j][col] = 1.0
            V[j * p + 4][col] = 1.0
            W[4 * n + i][col] = 1.0
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > 140:  # Safety margin
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
