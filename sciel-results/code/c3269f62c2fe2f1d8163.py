"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    # Schoolbook rank = n*m*p = 50
    # We'll attempt a rank-31 decomposition using Strassen for the 4x4 core
    # plus schoolbook for boundary terms.
    #
    # Strategy:
    # - The full tensor T[i,j,k] = delta (contributes to C[i,k] via A[i,j]*B[j,k])
    # - For j in {0,1,2,3} and k in {0,1,2,3}: use a rank-7 decomposition
    #   treating the 2x4x4 sub-tensor. Since 2x4x4 isn't a standard Strassen block,
    #   we instead pad: create a 4x4x4 problem by duplicating A's two rows into
    #   a 4x4 matrix with zeros, compute Strassen, and extract.
    # - For j=4 or k=4: schoolbook terms.
    #
    # Boundary terms: j=4 (all k) -> 2*5=10 terms, k=4 (j in 0..3) -> 2*4=8 terms
    # Total boundary: 18 schoolbook terms
    # Core: 2x4x4 = 32 schoolbook terms, replaced by rank-7 Strassen
    # Total rank: 18 + 7 = 25? Let's verify.
    #
    # Actually, the 2x4x4 sub-tensor: T[i,j,k] for i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # We can embed this in a 4x4x4 tensor by setting rows 2,3 of the "left" to zero.
    # Strassen rank 7 for 4x4x4 gives us 7 terms that compute all 16 outputs of a 4x4 product.
    # We only need outputs for i in {0,1}, so we use all 7 Strassen terms.
    #
    # U has shape (n*m, rank) = (10, rank)
    # V has shape (m*p, rank) = (25, rank)
    # W has shape (p*n, rank) = (25, rank), indexed as W[k*n+i, c]
    
    import time
    start = time.time()
    deadline = start + 120.0
    
    # Strassen 4x4x4 decomposition (rank 7)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 4x4x4, the Strassen sub-blocks are 2x2, each of which uses its own 3-term Strassen.
    # Full rank-7 4x4x4 decomposition has 7 * 3 = 21 elementary terms? No.
    # Standard 4x4x4 Strassen has rank 7 where each of the 7 "products" is a 2x2 matrix product,
    # each requiring rank 3. So total rank = 7*3 = 21 for the 4x4x4.
    #
    # That's more complex. Let me just use schoolbook rank 50 for correctness.
    
    rank = n * m * p  # 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    if time.time() > deadline - 5.0:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
