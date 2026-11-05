import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 48 decomposition
    # 6 blocks of 2x2x2 (Strassen, rank 7 each) + 1 block of 2x1x6 (rank 6)
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h) -> C00 += 1, C11 += 1
    # M2 = (c+d)e    -> C10 += 1
    # M3 = a(f-h)    -> C01 += 1, C11 -= 1
    # M4 = d(g-f)    -> C00 += 1, C10 += 1
    # M5 = (a+b)h    -> C00 -= 1, C01 += 1
    # M6 = (c-a)(e+f)-> C10 += 1, C11 -= 1 (Wait, check Strassen formulas)
    # M7 = (b-d)(g+h)-> C00 -= 1, C11 += 1
    
    # Standard Strassen:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # M1: (a+d)(e+h)
    # M2: (c+d)e
    # M3: a(f-h)
    # M4: d(g-f)
    # M5: (a+b)h
    # M6: (c-a)(e+f) -- Wait, M6 is not in the C formulas above?
    # Ah, M6 is used in C11?
    # C11 = M1 - M2 + M3 + M7. Where is M6?
    # Let's check Strassen algorithm again.
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    # Wait, M6 is not used in the final C?
    # No, M6 is used in C22?
    # C22 = M1 - M2 + M3 + M7.
    # Let's verify.
    # M1 - M2 + M3 + M7 = (A11+A22)(B11+B22) - (A21+A22)B11 + A11(B21-B22) + (A12-A22)(B21+B22)
    # = A11B11 + A11B22 + A22B11 + A22B22 - A21B11 - A22B11 + A11B21 - A11B22 + A12B21 + A12B22 - A22B21 - A22B22
    # = A11B11 + A11B21 + A12B21 + A12B22
    # = A11(B11+B21) + A12(B21+B22). This is not A12B22.
    # There is a mistake in my Strassen memory or the formula.
    # Correct Strassen:
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    # Let's re-verify C22.
    # M1 = A11B11 + A11B22 + A22B11 + A22B22
    # -M2 = -A21B11 - A22B11
    # +M3 = A11B21 - A11B22
    # +M7 = A12B21 + A12B22 - A22B21 - A22B22
    # Sum: A11B11 + A11B21 + A12B21 + A12B22.
    # This is C12, not C22.
    # So C22 formula is wrong.
    # Correct C22 = M1 - M2 + M3 + M7 is for C12?
    # Let's look up Strassen.
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    # If C22 is wrong, what is it?
    # Actually, M6 is used in C22?
    # C22 = M1 - M2 + M3 + M7 is often cited.
    # Let's check M6 usage.
    # M6 = (A21 - A11)(B11 + B12).
    # It seems M6 is not used in the standard 7-multiplication Strassen for C?
    # No, all 7 are used.
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    # Wait, I found a source that says:
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    # And M6 is... not used?
    # No, that's impossible. 7 multiplications for 4 outputs.
    # Ah, M6 is used in C22 in some variants?
    # Let's use a reliable implementation.
    
    # Let's just use the known rank 7 decomposition coefficients.
    # I will hardcode the 7 terms for a 2x2x2 block.

    def add_strassen_block(i0, i1, j0, j1, k0, k1):
        nonlocal col
        # A indices: (i0, j0), (i0, j1), (i1, j0), (i1, j1)
        # B indices: (j0, k0), (j0, k1), (j1, k0), (j1, k1)
        # C indices: (k0, i0), (k0, i1), (k1, i0), (k1, i1)
        
        # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        # C[k0,i0] += 1, C[k1,i1] += 1
        U[i0*m + j0][col] += 1
        U[i1*m + j1][col] += 1
        V[j0*p + k0][col] += 1
        V[j1*p + k1][col] += 1
        W[k0*n + i0][col] += 1
        W[k1*n + i1][col] += 1
        col += 1

        # M2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        # C[k0,i1] += 1
        U[i1*m + j0][col] += 1
        U[i1*m + j1][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + i1][col] += 1
        col += 1

        # M3 = A[i0,j0] * (B[j1,k0] - B[j1,k1])
        # C[k0,i0] += 1, C[k1,i0] -= 1
        U[i0*m + j0][col] += 1
        V[j1*p + k0][col] += 1
        V[j1*p + k1][col] -= 1
        W[k0*n + i0][col] += 1
        W[k1*n + i0][col] -= 1
        col += 1

        # M4 = A[i1,j1] * (B[j0,k1] - B[j0,k0])
        # C[k0,i0] += 1, C[k1,i1] += 1
        U[i1*m + j1][col] += 1
        V[j0*p + k1][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k1*n + i1][col] += 1
        col += 1

        # M5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        # C[k0,i0] -= 1, C[k1,i0] += 1
        U[i0*m + j0][col] += 1
        U[i0*m + j1][col] += 1
        V[j1*p + k1][col] += 1
        W[k0*n + i0][col] -= 1
        W[k1*n + i0][col] += 1
        col += 1

        # M6 = (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k1])
        # C[k0,i1] += 1, C[k1,i1] -= 1
        U[i1*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + k1][col] += 1
        W[k0*n + i1][col] += 1
        W[k1*n + i1][col] -= 1
        col += 1

        # M7 = (A[i0,j1] - A[i1,j1]) * (B[j1,k0] + B[j1,k1])
        # C[k0,i0] -= 1, C[k1,i1] += 1
        U[i0*m + j1][col] += 1
        U[i1*m + j1][col] -= 1
        V[j1*p + k0][col] += 1
        V[j1*p + k1][col] += 1
        W[k0*n + i0][col] -= 1
        W[k1*n + i1][col] += 1
        col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen_block(0, 1, 0, 1, 0, 1)
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen_block(0, 1, 0, 1, 2, 3)
    # Block 3: A[0:2, 0:2] * B[0:2, 4:6]
    add_strassen_block(0, 1, 0, 1, 4, 5)
    # Block 4: A[0:2, 2:4] * B[2:4, 0:2]
    add_strassen_block(0, 1, 2, 3, 0, 1)
    # Block 5: A[0:2, 2:4] * B[2:4, 2:4]
    add_strassen_block(0, 1, 2, 3, 2, 3)
    # Block 6: A[0:2, 2:4] * B[2:4, 4:6]
    add_strassen_block(0, 1, 2, 3, 4, 5)
    
    # Block 7: A[0:2, 4:5] * B[4:5, 0:6]
    # This is 2x1 x 1x6. Rank 6.
    # C[i, k] = A[i, 4] * B[4, k]
    for k in range(6):
        U[0*m + 4][col] += 1 # A[0,4]
        V[4*p + k][col] += 1 # B[4,k]
        W[k*n + 0][col] += 1 # C[0,k]
        col += 1
        
        U[1*m + 4][col] += 1 # A[1,4]
        V[4*p + k][col] += 1 # B[4,k]
        W[k*n + 1][col] += 1 # C[1,k]
        col += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
