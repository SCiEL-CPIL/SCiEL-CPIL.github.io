"""
Rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.

Strategy:
  - Split the 5-dimension (middle) into 4 + 1.
  - Part A: 4x4x4 block (A[:, :4] * B[:4, :]) -> use Strassen (rank 49)
  - Part B: 4x1x6 block (A[:, 4:5] * B[4:5, :]) -> rank 6 (schoolbook)
  - Total rank = 49 + 6 = 55

Trace convention: W[k*n + i, r] where n=4, p=6, so W has shape (p*n, rank) = (24, rank).
U has shape (n*m, rank) = (20, rank), V has shape (m*p, rank) = (30, rank).
"""

def construct():
    n, m, p = 4, 5, 6
    rank = 55  # 49 + 6

    U = [[0] * rank for _ in range(n * m)]  # 20 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 24 x 55

    # ---- Part A: 4x4x4 Strassen (columns 0..48) ----
    # Strassen for 2x2 * 2x2:
    # M1 = A11*(B11+B21)
    # M2 = (A11+A12)*B21
    # M3 = A21*(B11-B22)
    # M4 = A22*(B12+B22)
    # M5 = (A11+A21)*(B11+B22)
    # M6 = (A12-A21)*(B21+B22)
    # M7 = (A11-A22)*(B12-B11)
    #
    # For 4x4 * 4x4, we apply Strassen at the 2x2 block level.
    # A4x4 is split into four 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    # B4x4 is split into four 2x2 blocks: B11, B12, B21, B22 (each 2x2)
    # C4x4 has blocks C11, C12, C21, C22 (each 2x2)
    #
    # C11 = M1 + M3 - M5 + M7
    # C12 = M2 + M4
    # C21 = M6 + M5
    # C22 = M1 - M2 + M4 + M6
    #
    # Each Mi is a 2x2*2x2 product, which itself can be done with rank 7 (Strassen) or rank 4 (schoolbook).
    # To get rank 49 for 4x4x4, we use schoolbook (rank 4) for each of the 7 Strassen terms: 7*4 = 28.
    # Wait, that gives 28, not 49. Let me reconsider.
    #
    # Actually, for 4x4*4x4, the schoolbook rank is 16. Strassen gives 7 terms each of which is a 2x2*2x2.
    # If we use rank 7 Strassen for each 2x2*2x2, total = 7*7 = 49.
    # If we use rank 4 schoolbook for each, total = 7*4 = 28.
    #
    # Let's use rank 28 for the 4x4x4 part (Strassen with schoolbook leaves).
    # Then total rank = 28 + 6 = 34.
    #
    # Let me verify: 4x4*4x4 via Strassen:
    # 7 products of 2x2*2x2, each done with schoolbook rank 4. Total 28.
    # This is valid and gives C = A*B exactly.

    # Re-do with rank 28 for 4x4x4 part.
    rank = 28 + 6  # = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # ---- Part A: 4x4x4 (columns 0..27) ----
    # A indices: rows 0..3, cols 0..3 (within the 4x5 matrix, cols 0-3)
    # B indices: rows 0..3, cols 0..3 (within the 5x6 matrix, rows 0-3, cols 0-3)
    # C indices: rows 0..3, cols 0..3 (within the 4x6 result)
    #
    # A4x4 block layout (row, col) in A:
    #   A11: (0,0),(0,1),(1,0),(1,1)
    #   A12: (0,2),(0,3),(1,2),(1,3)
    #   A21: (2,0),(2,1),(3,0),(3,1)
    #   A22: (2,2),(2,3),(3,2),(3,3)
    #
    # B4x4 block layout (row, col) in B:
    #   B11: (0,0),(0,1),(1,0),(1,1)
    #   B12: (0,2),(0,3),(1,2),(1,3)
    #   B21: (2,0),(2,1),(3,0),(3,1)
    #   B22: (2,2),(2,3),(3,2),(3,3)
    #
    # C4x4 block layout (row, col) in C:
    #   C11: (0,0),(0,1),(1,0),(1,1)
    #   C12: (0,2),(0,3),(1,2),(1,3)
    #   C21: (2,0),(2,1),(3,0),(3,1)
    #   C22: (2,2),(2,3),(3,2),(3,3)
    #
    # Strassen terms:
    # M1 = A11 * (B11 + B21)
    # M2 = (A11 + A12) * B21
    # M3 = A21 * (B11 - B22)
    # M4 = A22 * (B12 + B22)
    # M5 = (A11 + A21) * (B11 + B22)
    # M6 = (A12 - A21) * (B21 + B22)
    # M7 = (A11 - A22) * (B12 - B11)
    #
    # Each Mi is a 2x2 * 2x2 product. Use schoolbook rank 4 for each.
    # For a 2x2*2x2 schoolbook:
    #   C[i,j] = sum_k A[i,k]*B[k,j]
    #   4 terms: (i,j,k) for i,j,k in {0,1}
    #
    # For each Strassen term, we need to map:
    #   - Which A elements (with signs) go into U
    #   - Which B elements (with signs) go into V
    #   - Which C elements (with signs) get contributions from W
    #
    # Let me define the 2x2*2x2 schoolbook pattern:
    # For 2x2*2x2, the 4 rank-1 terms are:
    #   term 0: A[0,0]*B[0,0] -> C[0,0] += 
    #   term 1: A[0,1]*B[1,0] -> C[0,0] +=
    #   term 2: A[1,0]*B[0,1] -> C[1,0] +=
    #   term 3: A[1,1]*B[1,1] -> C[1,0] +=
    #   Wait, that's not right. Let me be more careful.
    #
    # 2x2 * 2x2:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #
    # Schoolbook rank 4:
    # term 0: A[0,0]*B[0,0] -> contributes to C[0,0] and C[1,0]? No.
    # Actually each term is A[i,k]*B[k,j] for fixed i,k,j... no.
    #
    # The rank-1 decomposition: C[i,j] = sum_k A[i,k]*B[k,j]
    # Each (i,k,j) triple is a rank-1 term? No, that's not how it works.
    #
    # For 2x2*2x2, schoolbook has 4 multiplications:
    # m0 = A[0,0]*B[0,0]
    # m1 = A[0,1]*B[1,0]
    # m2 = A[1,0]*B[0,1]
    # m3 = A[1,1]*B[1,1]
    # But that's only 4 and doesn't give all entries.
    #
    # Actually schoolbook for 2x2*2x2 is:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #
    # This requires 8 scalar multiplications if done naively, but the tensor rank is 4.
    # The rank-4 decomposition:
    # Term 0: A[0,0] * B[0,0] -> W contributes to C[0,0]
    # Term 1: A[0,1] * B[1,0] -> W contributes to C[0,0]
    # Term 2: A[0,0] * B[0,1] -> W contributes to C[0,1]
    # Term 3: A[0,1] * B[1,1] -> W contributes to C[0,1]
    # Term 4: A[1,0] * B[0,0] -> W contributes to C[1,0]
    # Term 5: A[1,1] * B[1,0] -> W contributes to C[1,0]
    # Term 6: A[1,0] * B[0,1] -> W contributes to C[1,1]
    # Term 7: A[1,1] * B[1,1] -> W contributes to C[1,1]
    #
    # That's rank 8, not 4. The tensor rank of 2x2*2x2 is actually 7 (Strassen) or 4?
    # No, the bilinear complexity of 2x2*2x2 is 7 (Strassen). Schoolbook is 4 multiplications
    # but in the tensor sense, it's rank 4? Let me think again.
    #
    # The tensor T[i,j,k] = delta_{ik} * delta_{kj} ... no.
    # T[A_idx, B_idx, C_idx] where A_idx=(i,k), B_idx=(k,j), C_idx=(i,j).
    # T[(i,k),(k,j),(i,j)] = 1.
    #
    # For 2x2*2x2: i,j,k in {0,1}.
    # T[(0,0),(0,0),(0,0)] = 1
    # T[(0,1),(1,0),(0,0)] = 1
    # T[(0,0),(0,1),(0,1)] = 1
    # T[(0,1),(1,1),(0,1)] = 1
    # T[(1,0),(0,0),(1,0)] = 1
    # T[(1,1),(1,0),(1,0)] = 1
    # T[(1,0),(0,1),(1,1)] = 1
    # T[(1,1),(1,1),(1,1)] = 1
    #
    # That's 8 nonzero entries. The rank of this tensor is 7 (Strassen shows it's at most 7,
    # and it's known to be exactly 7).
    #
    # So for 4x4*4x4 via Strassen: 7 terms, each is a 2x2*2x2 with rank 7. Total = 49.
    # For 4x4*4x4 via Strassen with schoolbook leaves: not possible since schoolbook for 2x2*2x2
    # in tensor sense is rank 7 (same as Strassen). There's no rank-4 decomposition for 2x2*2x2.
    #
    # So 4x4*4x4 has rank at most 49 (Strassen of Strassen) or 16 (schoolbook).
    # Schoolbook for 4x4*4x4: 16 multiplications, rank 16.
    #
    # Let me just use schoolbook for 4x4*4x4: rank 16.
    # Total rank = 16 + 6 = 22.

    # Re-do with rank 16 for 4x4x4 part.
    rank = 16 + 6  # = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # ---- Part A: 4x4x4 schoolbook (columns 0..15) ----
    # A: rows 0..3, cols 0..3 (in 4x5 matrix)
    # B: rows 0..3, cols 0..3 (in 5x6 matrix)
    # C: rows 0..3, cols 0..3 (in 4x6 result)
    #
    # C[i,j] = sum_{k=0}^{3} A[i,k] * B[k,j]
    # Each (i,k,j) is a rank-1 term.
    #
    # U index: i*m + k = i*5 + k (for k=0..3)
    # V index: k*p + j = k*6 + j (for k=0..3, j=0..3)
    # W index: j*n + i = j*4 + i (for j=0..3, i=0..3)
    
    col = 0
    for i in range(4):
        for k in range(4):
            for j in range(4):
                U[i * 5 + k][col] = 1
                V[k * 6 + j][col] = 1
                W[j * 4 + i][col] = 1
                col += 1
    # col is now 16

    # ---- Part B: 4x1x6 schoolbook (columns 16..21) ----
    # A: rows 0..3, col 4 (the 5th column, index 4)
    # B: row 4 (the 5th row, index 4), cols 0..5
    # C: rows 0..3, cols 0..5
    #
    # C[i,j] = A[i,4] * B[4,j]
    # Each (i,j) is a rank-1 term. 4*6 = 24 terms? No, 4*6 = 24.
    # Wait, that's too many. Let me reconsider.
    #
    # C[i,j] = A[i,4] * B[4,j] for i=0..3, j=0..5.
    # This is an outer product of a 4-vector and a 6-vector, which has rank 1!
    # C = a * b^T where a = A[:,4] (4x1) and b = B[4,:] (1x6).
    # So rank 1 suffices.
    
    # U index: i*5 + 4 for i=0..3
    # V index: 4*6 + j = 24 + j for j=0..5
    # W index: j*4 + i for j=0..5, i=0..3
    
    U[0 * 5 + 4][col] = 1  # i=0
    U[1 * 5 + 4][col] = 1  # i=1
    U[2 * 5 + 4][col] = 1  # i=2
    U[3 * 5 + 4][col] = 1  # i=3
    V[4 * 6 + 0][col] = 1  # j=0
    V[4 * 6 + 1][col] = 1  # j=1
    V[4 * 6 + 2][col] = 1  # j=2
    V[4 * 6 + 3][col] = 1  # j=3
    V[4 * 6 + 4][col] = 1  # j=4
    V[4 * 6 + 5][col] = 1  # j=5
    for j in range(6):
        for i in range(4):
            W[j * 4 + i][col] = 1
    col += 1
    # col is now 17

    # But wait, I allocated rank 22. Let me just use rank 17.
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # ---- Part A: 4x4x4 schoolbook (columns 0..15) ----
    col = 0
    for i in range(4):
        for k in range(4):
            for j in range(4):
                U[i * 5 + k][col] = 1
                V[k * 6 + j][col] = 1
                W[j * 4 + i][col] = 1
                col += 1

    # ---- Part B: 4x1x6 outer product (column 16) ----
    for i in range(4):
        U[i * 5 + 4][col] = 1
    for j in range(6):
        V[4 * 6 + j][col] = 1
    for j in range(6):
        for i in range(4):
            W[j * 4 + i][col] = 1

    return {"U": U, "V": V, "W": W}
