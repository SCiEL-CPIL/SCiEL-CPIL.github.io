<<<<<<< SEARCH
"""Program for 3x4x5 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 5]

# Strassen's 2x2x2 rank-7 decomposition factors
# A = A11 A12; B = B11 B12; C = C11 C12 (each 2x2 block)
# M1 = (A11+A22)*(B11+B22)
# M2 = (A11+A22)*B21
# M3 = A12*(B21-B22)
# M4 = A11*B11
# M5 = A22*(B12-B11)
# M6 = (A12+A21)*B11
# M7 = (A21-A11)*(B12+B21)
# C11 = M1+M4-M5+M7
# C12 = M3+M4+M6
# C21 = M2+M4+M5
# C22 = M1-M2+M3+M6

STRASSEN_U = [
    [1, 0, 0, 1, 0, 0, 0],  # A11
    [0, 1, 1, 0, 1, 0, 1],  # A12
    [1, 0, 0, 0, 0, 1, -1], # A21
    [0, 1, 0, 0, 1, 1, 0],  # A22
]

STRASSEN_V = [
    [1, 1, 0, 1, 0, 1, 1],  # B11
    [0, 0, 1, 0, -1, 0, 0], # B12
    [1, 1, -1, 0, 0, 0, 0], # B21
    [1, 0, -1, 0, 0, 1, 1], # B22
]

STRASSEN_W = [
    [1, 0, 0, 1, -1, 0, 1],  # C11
    [0, 0, 1, 1, 0, 1, 0],   # C12
    [0, 1, 0, 1, 1, 0, 0],   # C21
    [1, -1, 1, 0, 0, 1, 0],  # C22
]


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Build a decomposition using Strassen for the 4x4 middle part
    # Split rows of A (3x4) into 2+1, columns of A into 2+2
    # Split rows of B (4x5) into 2+2, columns of B (5) as is
    
    # A is 3x4: split into blocks
    # A11: 2x2 (rows 0-1, cols 0-1)
    # A12: 2x2 (rows 0-1, cols 2-3)
    # A21: 1x2 (row 2, cols 0-1)
    # A22: 1x2 (row 2, cols 2-3)
    
    # B is 4x5: split into blocks
    # B11: 2x5 (rows 0-1, cols 0-4)
    # B12: 2x5 (rows 2-3, cols 0-4)
    # B21: 2x5 (rows 0-1, cols 0-4) -- wait, let me reconsider
    
    # Actually, let's use a different approach.
    # Use schoolbook for the non-square parts and Strassen for the square part.
    
    # Split the 4x4 middle dimension:
    # A: 3x4 -> rows: 2+1, cols: 2+2
    # B: 4x5 -> rows: 2+2, cols: 5
    
    # Let's think of this as:
    # C = A * B where A is 3x4, B is 4x5
    # Split A into [A_top (2x4); A_bot (1x4)]
    # Split B into [B_left (4x5)]
    # Split A's columns into A11 (2x2), A12 (2x2), A21 (1x2), A22 (1x2)
    # Split B's rows into B_top (2x5), B_bot (2x5)
    
    # This gives:
    # C_top (2x5) = A11*B_top + A12*B_bot
    # C_bot (1x5) = A21*B_top + A22*B_bot
    
    # We can use Strassen for the 2x2 matrix multiplications:
    # A11*B_top: this is (2x2) * (2x5) -> not square
    # Hmm, Strassen requires square matrices.
    
    # Alternative: Pad to make square, or use a different decomposition.
    
    # Let's just use the schoolbook decomposition which is guaranteed to work.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
=======
"""Program for 3x4x5 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 5]

# Correct Strassen 2x2x2 rank-7 decomposition
# A = [[a,b],[c,d]], B = [[e,f],[g,h]]
# p1 = (a+d)(e+h)
# p2 = (a+d)(g-h)
# p3 = (a-c)(e+f)
# p4 = a(g-h)
# p5 = c(h-f)
# p6 = (b+c)(h-f)
# p7 = (a+b)(f-h)
# C11 = p1 + p4 - p5 + p7
# C12 = p3 + p4
# C21 = p2 + p5
# C22 = p1 - p3 + p6 + p7

STRASSEN_U = [
    [1, 1, 1, 1, 0, 0, 1],  # a: p1,p2,p3,p4,p7
    [0, 0, 0, 0, 0, 1, 1],  # b: p6,p7
    [0, 0, -1, 0, 1, 1, 0], # c: p3(-1),p5,p6
    [1, 1, 0, 0, 0, 0, 0],  # d: p1,p2
]

STRASSEN_V = [
    [1, 0, 1, 0, 0, 0, 0],  # e: p1,p3
    [0, 0, 1, 0, -1, -1, -1], # f: p3,p5(-1),p6(-1),p7(-1)
    [0, 1, 0, 1, 0, 0, 0],  # g: p2,p4
    [1, -1, 0, 0, 1, 1, 0], # h: p1,p2(-1),p5,p6
]

STRASSEN_W = [
    [1, 0, 0, 1, -1, 0, 1],  # C11: p1,p4,p5(-1),p7
    [0, 0, 1, 1, 0, 0, 0],   # C12: p3,p4
    [0, 1, 0, 0, 1, 0, 0],   # C21: p2,p5
    [1, 0, -1, 0, 0, 1, 1],  # C22: p1,p3(-1),p6,p7
]


def _verify_strassen():
    """Verify Strassen factors produce correct matrix multiplication."""
    n = m = p = 2
    U = np.array(STRASSEN_U, dtype=float)
    V = np.array(STRASSEN_V, dtype=float)
    W = np.array(STRASSEN_W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(7))
                if abs(val - 1.0) > 1e-10:
                    raise ValueError(f"Strassen verification failed at ({i},{j},{k}): {val}")


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Verify Strassen factors are correct
    _verify_strassen()
    
    # Split: A rows 2+1, A cols 2+2, B rows 2+2, B cols 3+2
    # Sub-problems and their ranks:
    # 1. A11(2x2)*B11(2x3): [2,2,3] schoolbook rank 12
    # 2. A12(2x2)*B21(2x3): [2,2,3] schoolbook rank 12
    # 3. A11(2x2)*B12(2x2): [2,2,2] Strassen rank 7
    # 4. A12(2x2)*B22(2x2): [2,2,2] Strassen rank 7
    # 5. A21(1x2)*B11(2x3): [1,2,3] schoolbook rank 6
    # 6. A22(1x2)*B21(2x3): [1,2,3] schoolbook rank 6
    # 7. A21(1x2)*B12(2x2): [1,2,2] schoolbook rank 4
    # 8. A22(1x2)*B22(2x2): [1,2,2] schoolbook rank 4
    # Total: 12+12+7+7+6+6+4+4 = 58
    
    total_rank = 58
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook(nn, mm, pp, i_off, j_off, b_j_off, k_off, col_start):
        """Add schoolbook for A[i_off:i_off+nn, j_off:j_off+mm] * B[b_j_off:b_j_off+mm, k_off:k_off+pp]"""
        c = col_start
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[(i_off+i)*m + (j_off+j)][c] = 1
                    V[(b_j_off+j)*p + (k_off+k)][c] = 1
                    W[(k_off+k)*n + (i_off+i)][c] = 1
                    c += 1
        return c
    
    def add_strassen(i_off, j_off, b_j_off, k_off, col_start):
        """Add Strassen for A[i_off:i_off+2, j_off:j_off+2] * B[b_j_off:b_j_off+2, k_off:k_off+2]"""
        c = col_start
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    val = STRASSEN_U[i*2+j][r]
                    if val != 0:
                        U[(i_off+i)*m + (j_off+j)][c] += val
            for j in range(2):
                for k in range(2):
                    val = STRASSEN_V[j*2+k][r]
                    if val != 0:
                        V[(b_j_off+j)*p + (k_off+k)][c] += val
            for k in range(2):
                for i in range(2):
                    val = STRASSEN_W[k*2+i][r]
                    if val != 0:
                        W[(k_off+k)*n + (i_off+i)][c] += val
            c += 1
        return c
    
    # 1. A11*B11: [2,2,3] at i_off=0, j_off=0, b_j_off=0, k_off=0
    col = add_schoolbook(2, 2, 3, 0, 0, 0, 0, col)
    
    # 2. A12*B21: [2,2,3] at i_off=0, j_off=2, b_j_off=2, k_off=0
    col = add_schoolbook(2, 2, 3, 0, 2, 2, 0, col)
    
    # 3. A11*B12: [2,2,2] at i_off=0, j_off=0, b_j_off=0, k_off=3
    col = add_strassen(0, 0, 0, 3, col)
    
    # 4. A12*B22: [2,2,2] at i_off=0, j_off=2, b_j_off=2, k_off=3
    col = add_strassen(0, 2, 2, 3, col)
    
    # 5. A21*B11: [1,2,3] at i_off=2, j_off=0, b_j_off=0, k_off=0
    col = add_schoolbook(1, 2, 3, 2, 0, 0, 0, col)
    
    # 6. A22*B21: [1,2,3] at i_off=2, j_off=2, b_j_off=2, k_off=0
    col = add_schoolbook(1, 2, 3, 2, 2, 2, 0, col)
    
    # 7. A21*B12: [1,2,2] at i_off=2, j_off=0, b_j_off=0, k_off=3
    col = add_schoolbook(1, 2, 2, 2, 0, 0, 3, col)
    
    # 8. A22*B22: [1,2,2] at i_off=2, j_off=2, b_j_off=2, k_off=3
    col = add_schoolbook(1, 2, 2, 2, 2, 2, 3, col)
    
    assert col == total_rank, f"Expected {total_rank}, got {col}"
    
    # Final verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r] for r in range(total_rank))
                if abs(val - 1.0) > 1e-10:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): {val}")
    
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE

Wait, I need to double-check my Strassen factors. Let me verify them carefully.

Using:
p1 = (a+d)(e+h)
p2 = (a+d)(g-h)
p3 = (a-c)(e+f)
p4 = a(g-h)
p5 = c(h-f)
p6 = (b+c)(h-f)
p7 = (a+b)(f-h)

C11 = p1 + p4 - p5 + p7
C12 = p3 + p4
C21 = p2 + p5
C22 = p1 - p3 + p6 + p7

Let me verify C11 = ae + bg:
p1 = ae+ah+de+dh
p4 = ag-ah
p5 = ch-cf
p7 = af-ah+bf-bh

C11 = (ae+ah+de+dh) + (ag-ah) - (ch-cf) + (af-ah+bf-bh)
= ae+ah+de+dh+ag-ah-ch+cf+af-ah+bf-bh
= ae + de + dh + ag - ch + cf + af - ah + bf - bh

This should equal ae+bg. Let me check with specific values. Let a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8.
ae+bg = 5+14 = 19

p1 = (1+4)(5+8) = 5*13 = 65
p4 = 1*(7-8) = -1
p5 = 3*(8-6) = 6
p7 = (1+2)(6-8) = 3*(-2) = -6

C11 = 65 + (-1) - 6 + (-6) = 65-1-6-6 = 52. That's not 19!

My Strassen formulas are definitely wrong. Let me look this up properly.

After more careful thought, the correct Strassen formulas are:
p1 = (a + d)(e + h)
p2 = (a + d)(g − i)  -- no, for 2x2 there's no i.

I think the issue is that I'm confusing different conventions. Let me use a definitive source.

From the standard reference:
Given A = [[a, b], [c, d]] and B = [[e, f], [g, h]]:

M1 = (a + d)(e + h)
M2 = (a + d)(g − h)  
M3 = (a − c)(e + f)
M4 = a(g − h)
M5 = c(h − f)
M6 = (b + c)(h − f)
M7 = (a + b)(f − h)

C11 = M1 + M4 − M5 + M7
C12 = M3 + M4
C21 = M2 + M5
C22 = M1 − M3 + M6 + M7

Let me verify with a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8:
C should be [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]

M1 = (1+4)(5+8) = 65
M2 = (1+4)(7-8) = -5
M3 = (1-3)(5+6) = -22
M4 = 1*(7-8) = -1
M5 = 3*(8-6) = 6
M6 = (2+3)(8-6) = 10
M7 = (1+2)(6-8) = -6

C11 = 65 + (-1) - 6 + (-6) = 52 ≠ 19

This is still wrong! I must be using incorrect formulas. Let me try yet another set.

Actually, I think the problem is that different sources define the blocks differently. Let me just implement it and test numerically. I'll try several candidate factor sets and verify which one works.

Let me take a step back. Instead of trying to get Strassen right from memory, let me just use schoolbook for all sub-problems. The total rank would be 12+12+12+12+6+6+4+4 = 68, which is worse than 60. That's not good.

Alternatively, I can just use the full schoolbook decomposition (rank 60) which is what the current program does, and that scores 1.0.

Let me try a different approach: use a known-correct Strassen implementation by testing it in code.

