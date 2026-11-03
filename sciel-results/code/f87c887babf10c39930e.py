import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen(j_off, k_off, col_start):
        # 7 steps of Strassen for 2x2x2
        c = col_start
        # P1 = (A00 + A11)(B00 + B11)
        U[0 * m + j_off][c] += 1
        U[1 * m + (j_off + 1)][c] += 1
        V[j_off * p + k_off][c] += 1
        V[(j_off + 1) * p + (k_off + 1)][c] += 1
        W[k_off * n + 0][c] += 1
        W[(k_off + 1) * n + 1][c] += 1
        c += 1

        # P2 = (A10 + A11) B00
        U[1 * m + j_off][c] += 1
        U[1 * m + (j_off + 1)][c] += 1
        V[j_off * p + k_off][c] += 1
        W[k_off * n + 1][c] += 1
        W[(k_off + 1) * n + 1][c] -= 1
        c += 1

        # P3 = A00 (B01 - B11)
        U[0 * m + j_off][c] += 1
        V[j_off * p + (k_off + 1)][c] += 1
        V[(j_off + 1) * p + (k_off + 1)][c] -= 1
        W[(k_off + 1) * n + 0][c] += 1
        W[(k_off + 1) * n + 1][c] += 1
        c += 1

        # P4 = A11 (B10 - B00)
        U[1 * m + (j_off + 1)][c] += 1
        V[(j_off + 1) * p + k_off][c] += 1
        V[j_off * p + k_off][c] -= 1
        W[k_off * n + 0][c] += 1
        W[k_off * n + 1][c] += 1
        c += 1

        # P5 = (A00 + A01) B11
        U[0 * m + j_off][c] += 1
        U[0 * m + (j_off + 1)][c] += 1
        V[(j_off + 1) * p + (k_off + 1)][c] += 1
        W[k_off * n + 0][c] -= 1
        W[(k_off + 1) * n + 0][c] += 1
        c += 1

        # P6 = (A10 - A00)(B00 + B01)
        U[1 * m + j_off][c] += 1
        U[0 * m + j_off][c] -= 1
        V[j_off * p + k_off][c] += 1
        V[j_off * p + (k_off + 1)][c] += 1
        W[(k_off + 1) * n + 1][c] += 1
        c += 1

        # P7 = (A01 - A11)(B10 + B11)
        U[0 * m + (j_off + 1)][c] += 1
        U[1 * m + (j_off + 1)][c] -= 1
        V[(j_off + 1) * p + k_off][c] += 1
        V[(j_off + 1) * p + (k_off + 1)][c] += 1
        W[k_off * n + 0][c] += 1

    # Block 1: 2x4x4 using 4x Strassen (28 rank)
    add_strassen(0, 0, 0)
    add_strassen(0, 2, 7)
    add_strassen(2, 0, 14)
    add_strassen(2, 2, 21)

    # Block 2: A_1 B_14 (j=4, k in 0..3) -> 4 terms (was 8)
    col = 28
    # Combine A[i][4]*B[4][k] for k=0,1 using Strassen-like trick:
    # (A[0][4]+A[1][4])*(B[4][0]+B[4][1]) and (A[0][4]-A[1][4])*(B[4][0]-B[4][1])
    # This gives us 4 terms instead of 8 for the 2x4 block
    # But actually we need to be more careful. Let's use a different approach.
    
    # For row i=0: A[0][4] * B[4][k] for k=0,1,2,3
    # For row i=1: A[1][4] * B[4][k] for k=0,1,2,3
    # We can pair them up:
    # s0 = A[0][4] + A[1][4]
    # s1 = A[0][4] - A[1][4]
    # Then:
    # C[0][k] = A[0][4]*B[4][k]
    # C[1][k] = A[1][4]*B[4][k]
    # A[0][4] = (s0 + s1)/2, A[1][4] = (s0 - s1)/2
    # C[0][k] = (s0+s1)/2 * B[4][k] = s0*B[4][k]/2 + s1*B[4][k]/2
    # C[1][k] = (s0-s1)/2 * B[4][k] = s0*B[4][k]/2 - s1*B[4][k]/2
    
    # This doesn't save rank directly. Let's just use the 8 terms but check if we can merge with Strassen.
    
    # Actually, let's reconsider. The 2x4x4 block uses 4*Strassen = 28.
    # The remaining part is 2x1x4 + 2x4x1 + 2x1x1 = 8 + 8 + 2 = 18, total 46.
    # Can we do better?
    
    # Let me try a different partition. Use 3x3 Strassen? No, 5 is not 3*2.
    
    # Let's try to combine the j=4 row with the Strassen blocks more cleverly.
    
    # Alternative: Use a 2x5x5 approach where we treat it as two 1x5x5 problems that share some structure.
    
    # For now, let's keep the 46-rank solution and try to find a rank-reduction via algebraic manipulation.
    
    # Let me try a completely different approach: enumerate all possible rank-r decompositions
    # using the fact that the tensor is structured.
    
    # Actually, let's just return the known good 46-rank solution first, then try to optimize.
    
    col = 28
    # Block 2: A_1 B_14 (j=4, k in 0..3)
    for i in range(2):
        for k in range(4):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Block 3: A_4 B_41 (j in 0..3, k=4)
    for i in range(2):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1

    # Block 4: A_1 B_11 (j=4, k=4)
    for i in range(2):
        U[i * m + 4][col] = 1
        V[4 * p + 4][col] = 1
        W[4 * n + i][col] = 1
        col += 1

    # Now try to reduce rank by finding linear dependencies
    # Convert to numpy for easier manipulation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # The tensor condition is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    # Let's verify
    def check(Ua, Va, Wa):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(Ua[i*m+j,c] * Va[j*p+k,c] * Wa[k*n+i,c] for c in range(rank))
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True

    if not check(U_np, V_np, W_np):
        # Fallback to schoolbook
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

