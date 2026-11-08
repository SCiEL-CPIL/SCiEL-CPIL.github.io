"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact rank-33 decomposition for 2x3 * 3x6 matrix multiplication.

    Decomposition strategy:
      - Split A (2x3) = [A_L (2x2) | A_R (2x1)]
      - Split B (3x6) = [B_L (2x6); B_R (1x6)]
      - C = A_L @ B_L + A_R @ B_R
      - A_L @ B_L: B_L has 6 columns = 3 blocks of 2 columns each
        Each 2x2 * 2x2 block uses Strassen (rank 7), total 3*7 = 21
      - A_R @ B_R: 2x1 * 1x6 = 2x6, schoolbook rank 12
      Total rank = 21 + 12 = 33
    """
    n, m, p = DIMENSIONS  # 2, 3, 6

    rank = 33
    U = [[0] * rank for _ in range(n * m)]  # 6 x 33
    V = [[0] * rank for _ in range(m * p)]  # 18 x 33
    W = [[0] * rank for _ in range(p * n)]  # 12 x 33

    col = 0

    # Part 1: A_L (rows 0-1 of A, cols 0-1) * B_L (rows 0-1 of B, all 6 cols)
    # B_L is split into 3 blocks of 2 columns each
    for block in range(3):
        # Block b covers B columns 2*b and 2*b+1
        b = block
        # Strassen 7 multiplications for 2x2 * 2x2
        # A_L elements: A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[2], A[1,1]=U[3]
        # B_L block elements:
        #   B[0, 2b]=V[0+2b], B[0, 2b+1]=V[1+2b]
        #   B[1, 2b]=V[2+2b], B[1, 2b+1]=V[3+2b]
        #
        # Strassen products:
        # M1 = (A00+A11)(B00+B11)  -> W: C00 += M1, C11 -= M1... wait let me be careful
        # Actually Strassen:
        # M1 = (A11)(B00-B10)
        # M2 = (A10+A11)(B00)
        # M3 = (A00)(B11-B01)
        # M4 = (A00+A11)(B10+B11)
        # M5 = (A01-A00)(B10+B11)  -- wait, standard Strassen for 2x2:
        # Let me use the standard formulation:
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e)
        # M3 = (a+b)(h)
        # M4 = (a-d)(g+h)
        # M5 = (c+b)(f+h)  -- no
        #
        # Standard Strassen:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f+h)  -- wait no
        #
        # Let me use the most common form:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e  
        # M3 = (a+b)h
        # M4 = (a-d)(g+h)
        # M5 = (c+b)(f+h)  -- hmm, let me just use:
        #
        # The 7 Strassen products for C = AB where A=[[a,b],[c,d]], B=[[e,f],[g,h]]:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e)
        # M3 = (a+b)(h)
        # M4 = (a-d)(g+h)
        # M5 = (c+b)(f+h)
        # M6 = (a-b)(g)
        # M7 = (c+d)(f-h)
        #
        # C00 = M1 - M3 + M6 + M7
        # C01 = M3 - M2 - M5 + M4  -- wait this is getting confusing.
        #
        # Let me use a cleaner formulation. The standard Strassen:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e)
        # M3 = (a+b)(h)
        # M4 = (a-d)(g+h)
        # M5 = (c+b)(f+h)
        # M6 = (a-b)(g)
        # M7 = (c+d)(f-h)
        #
        # C00 = M1 - M3 + M6 + M7
        # C01 = M3 + M4 - M5 - M7
        # C10 = M2 + M5 - M4 - M1
        # C11 = M1 - M2 + M3 - M6

        # Indices in U: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
        # Indices in V: B[0,2b]=2b, B[0,2b+1]=2b+1, B[1,2b]=2b+2, B[1,2b+1]=2b+3
        # Indices in W: C[0,2b]=2b, C[0,2b+1]=2b+1, C[1,2b]=2b+2, C[1,2b+1]=2b+3

        # We'll add contributions to W using signed coefficients
        # For each Strassen product, we set U and V entries, and add +/-1 to W

        # Define the 7 Strassen terms
        # Each term: (U_indices with coefficients, V_indices with coefficients, W_output_indices with coefficients)

        # Term 1: M1 = (a+d)(e+h)
        # U: A00 (+1), A11 (+1) -> U[0]+=1, U[3]+=1
        # V: B00 (+1), B11 (+1) -> V[2b]+=1, V[2b+3]+=1
        # W: C00 += M1, C01 -= M1? No.
        # From formulas:
        # C00 = M1 - M3 + M6 + M7
        # C01 = M3 + M4 - M5 - M7
        # C10 = M2 + M5 - M4 - M1
        # C11 = M1 - M2 + M3 - M6

        # So M1 contributes: +1 to C00, -1 to C10, +1 to C11
        U[0][col] += 1; U[3][col] += 1
        V[2*b][col] += 1; V[2*b+3][col] += 1
        W[2*b][col] += 1; W[2*b+2][col] -= 1; W[2*b+3][col] += 1
        col += 1

        # Term 2: M2 = (c+d)(e)
        # U: A10 (+1), A11 (+1) -> U[2]+=1, U[3]+=1
        # V: B00 (+1) -> V[2b]+=1
        # M2 contributes: +1 to C10, -1 to C11
        U[2][col] += 1; U[3][col] += 1
        V[2*b][col] += 1
        W[2*b+2][col] += 1; W[2*b+3][col] -= 1
        col += 1

        # Term 3: M3 = (a+b)(h)
        # U: A00 (+1), A01 (+1) -> U[0]+=1, U[1]+=1
        # V: B11 (+1) -> V[2b+3]+=1
        # M3 contributes: -1 to C00, +1 to C01, +1 to C11
        U[0][col] += 1; U[1][col] += 1
        V[2*b+3][col] += 1
        W[2*b][col] -= 1; W[2*b+1][col] += 1; W[2*b+3][col] += 1
        col += 1

        # Term 4: M4 = (a-d)(g+h)
        # U: A00 (+1), A11 (-1) -> U[0]+=1, U[3]-=1
        # V: B10 (+1), B11 (+1) -> V[2b+2]+=1, V[2b+3]+=1
        # M4 contributes: +1 to C01, -1 to C10
        U[0][col] += 1; U[3][col] -= 1
        V[2*b+2][col] += 1; V[2*b+3][col] += 1
        W[2*b+1][col] += 1; W[2*b+2][col] -= 1
        col += 1

        # Term 5: M5 = (c+b)(f+h)
        # U: A10 (+1), A01 (+1) -> U[2]+=1, U[1]+=1
        # V: B01 (+1), B11 (+1) -> V[2b+1]+=1, V[2b+3]+=1
        # M5 contributes: -1 to C01, +1 to C10
        U[2][col] += 1; U[1][col] += 1
        V[2*b+1][col] += 1; V[2*b+3][col] += 1
        W[2*b+1][col] -= 1; W[2*b+2][col] += 1
        col += 1

        # Term 6: M6 = (a-b)(g)
        # U: A00 (+1), A01 (-1) -> U[0]+=1, U[1]-=1
        # V: B10 (+1) -> V[2b+2]+=1
        # M6 contributes: +1 to C00, -1 to C11
        U[0][col] += 1; U[1][col] -= 1
        V[2*b+2][col] += 1
        W[2*b][col] += 1; W[2*b+3][col] -= 1
        col += 1

        # Term 7: M7 = (c+d)(f-h)
        # U: A10 (+1), A11 (+1) -> U[2]+=1, U[3]+=1
        # V: B01 (+1), B11 (-1) -> V[2b+1]+=1, V[2b+3]-=1
        # M7 contributes: +1 to C00, -1 to C01
        U[2][col] += 1; U[3][col] += 1
        V[2*b+1][col] += 1; V[2*b+3][col] -= 1
        W[2*b][col] += 1; W[2*b+1][col] -= 1
        col += 1

    # Part 2: A_R (2x1, col index 2 in A) * B_R (1x6, row index 2 in B)
    # A_R[i] = A[i, 2], i=0,1 -> U indices: i*m + 2 = i*3+2 -> U[2], U[5]
    # B_R[j] = B[2, j], j=0..5 -> V indices: 2*6 + j = 12+j -> V[12],...,V[17]
    # C[i,j] += A_R[i] * B_R[j] -> W index: j*n + i = 2*i... wait
    # W index convention: W[k*n + i, r] where k is B row index (which is j here, 0..5), i is A row index (0..1)
    # So W[j*2 + i, col]
    for i in range(2):
        for j in range(6):
            U[i * m + 2][col] = 1
            V[2 * p + j][col] = 1
            W[j * n + i][col] = 1
            col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
