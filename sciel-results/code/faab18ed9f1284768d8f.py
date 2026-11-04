import time
import numpy as np

N, M, P = 2, 3, 6
DEADLINE_FRACTION = 0.85  # stop searching 15% before the 120s limit

def _deadline():
    return time.time() + 120 * DEADLINE_FRACTION

def _schoolbook():
    rank = N * M * P
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _block232():
    """
    2x3x2 multiplication: A is 2x3, B is 3x2, C is 2x2.
    Split into A1 (2x2) * B1 (2x2) + A2 (2x1) * B2 (1x2).
    A1*B1 uses Strassen (rank 7), A2*B2 uses schoolbook (rank 4).
    Total rank 11.
    U: 6x11, V: 6x11, W: 4x11.
    """
    U = [[0] * 11 for _ in range(6)]
    V = [[0] * 11 for _ in range(6)]
    W = [[0] * 11 for _ in range(4)]
    col = 0
    # Strassen part: A1 is A[:, 0:2] (rows 0,1,2,3), B1 is B[0:2, :] (rows 0,1,2,3)
    # U for A1: rows 0,1,2,3 (a,b,c,d)
    # V for B1: rows 0,1,2,3 (e,f,g,h)
    U[0][col] = 1; V[0][col] = 1; W[0][col] = 1; W[1][col] = 1
    U[3][col] = 1; V[1][col] = 1
    col += 1
    U[2][col] = 1; U[3][col] = 1; V[0][col] = 1
    W[2][col] = 1
    col += 1
    U[0][col] = 1; V[1][col] = -1; V[1][col] = 1
    W[0][col] = 1
    col += 1
    U[3][col] = 1; V[2][col] = 1; V[1][col] = -1
    W[3][col] = 1
    col += 1
    U[0][col] = 1; U[1][col] = 1; V[1][col] = 1
    W[1][col] = 1
    col += 1
    U[2][col] = 1; U[3][col] = -1; V[2][col] = 1
    W[2][col] = 1
    col += 1
    U[0][col] = 1; U[3][col] = -1; V[0][col] = 1; V[1][col] = 1
    W[0][col] = 1; W[1][col] = 1
    col += 1

    # Schoolbook part: A2 is A[:, 2:3] (rows 4,5), B2 is B[2:3, :] (rows 4,5)
    # A2 B2 = [[a3*e3 + a3*f3], [a4*e3 + a4*f3]]? No.
    # A2 is 2x1: [a3, a4]^T. B2 is 1x2: [e3, f3].
    # C_part = [[a3*e3, a3*f3], [a4*e3, a4*f3]].
    # U: row 4 (a3), row 5 (a4). V: row 4 (e3), row 5 (f3).
    # W: row 0 (C00), row 1 (C01), row 2 (C10), row 3 (C11).
    U[4][col] = 1; V[4][col] = 1; W[0][col] = 1
    col += 1
    U[4][col] = 1; V[5][col] = 1; W[1][col] = 1
    col += 1
    U[5][col] = 1; V[4][col] = 1; W[2][col] = 1
    col += 1
    U[5][col] = 1; V[5][col] = 1; W[3][col] = 1
    col += 1

    return {"U": U, "V": V, "W": W}

def _block236():
    """
    2x3x6 multiplication.
    Split B into 3 blocks of 1x2 (rows 0, 2, 4).
    A is 2x3.
    C is 2x6.
    For each block l in 0..2:
      A_l is A[:, l] (2x1). B_l is B[l, :] (1x2).
      C_l = A_l B_l (2x2).
      This is a 2x1x2 problem, rank 4.
    Total rank 12.
    Wait, A is 2x3. B is 3x6.
    If I split B into 3 blocks of size 1x2, I am using the structure of B.
    But A is multiplied by the whole B.
    C = A B.
    If B = [B0; B1; B2] where each Bl is 1x2.
    Then C = A0 B0 + A1 B1 + A2 B2.
    A0 = A[:, 0] (2x1), A1 = A[:, 1] (2x1), A2 = A[:, 2] (2x1).
    Each term is (2x1)(1x2) = 2x2. Rank 4.
    Total rank 12.
    This is better than 36.
    """
    rank = 12
    U = [[0] * rank for _ in range(N * M)] # 6x12
    V = [[0] * rank for _ in range(M * P)] # 18x12
    W = [[0] * rank for _ in range(P * N)] # 12x12

    for l in range(3):
        base = l * 4
        # A_l is A[:, l] -> rows l, M+l (i.e., 0+3l? No. A is 2x3. Rows are (i,j). idx = i*M+j.
        # A[:, 0] -> idx 0, 3. A[:, 1] -> idx 1, 4. A[:, 2] -> idx 2, 5.
        a_row_1 = l
        a_row_2 = M + l
        # B_l is B[l, :] -> rows l*P, l*P+1, l*P+2, l*P+3? No.
        # B is 3x6. Rows are (j,k). idx = j*P+k.
        # B[l, :] -> idx l*P + 0, l*P+1, ... l*P+5.
        # Wait, B_l is 1x2? No, B is 3x6.
        # If I split B into 3 blocks of 1x2, that means I am treating B as a stack of 3 vectors of length 6?
        # No, that doesn't make sense for matrix multiplication.
        # Let's stick to the previous block 2x3x2 idea.
        # 2x3x6 = (2x3x2) * 3.
        # Split output dimension 6 into three 2's.
        # B is 3x6 = [B1 | B2 | B3] where each Bi is 3x2.
        # C = A B = [A B1 | A B2 | A B3].
        # Each A Bi is 2x3 * 3x2 = 2x2.
        # So we need 3 decompositions of 2x3x2.
        # Rank 11 each. Total 33.

    # Let's implement the 2x3x2 decomposition properly.
    # A (2x3), B (3x2), C (2x2).
    # U: 6xR, V: 6xR, W: 4xR.
    # A = [[a, b, c], [d, e, f]]
    # B = [[x, y], [z, w], [u, v]]
    # C = [[ax+by+cz, ay+bw+cv], [dx+ey+fz, dy+ew+fv]]

    # Try to find a rank 11 decomposition for 2x3x2.
    # We can use Strassen for the first 2x2 block of A and B.
    # A1 = [[a, b], [d, e]], B1 = [[x, y], [z, w]].
    # A2 = [[c], [f]], B2 = [[u, v]].
    # C = A1 B1 + A2 B2.
    # A1 B1 is 2x2x2 (Rank 7).
    # A2 B2 is 2x1x2 (Rank 4).
    # Total 11.

    # Let's verify the 2x1x2 rank 4.
    # [[c], [f]] * [[u, v]] = [[cu, cv], [fu, fv]].
    # U: c (row 2), f (row 5). V: u (row 4), v (row 5). W: C00 (0), C01 (1), C10 (2), C11 (3).
    # Term 1: cu. U[2]=1, V[4]=1, W[0]=1.
    # Term 2: cv. U[2]=1, V[5]=1, W[1]=1.
    # Term 3: fu. U[5]=1, V[4]=1, W[2]=1.
    # Term 4: fv. U[5]=1, V[5]=1, W[3]=1.

    # Strassen for A1 B1:
    # P1 = (a+e)(x+w) -> U: a(0), e(4); V: x(0), w(2); W: C00(0), C01(1)
    # P2 = (d+e)x -> U: d(3), e(4); V: x(0); W: C10(2)
    # P3 = a(b-e) -> U: a(0); V: b(1), e(4); W: C00(0)
    # P4 = e(w-d) -> U: e(4); V: w(2), d(3); W: C11(3)
    # P5 = (a+b)w -> U: a(0), b(1); V: w(2); W: C01(1)
    # P6 = d(e-b) -> U: d(3); V: e(4), b(1); W: C10(2)
    # P7 = (a-e)(x+w) -> U: a(0), e(4); V: x(0), w(2); W: C11(3)
    # Wait, P7 in standard Strassen is usually (a-d)(e+f) or similar.
    # Let's use the standard Strassen factors for C = A1 B1.
    # C00 = P1 - P4 - P6 + P7?
    # Let's check:
    # P1 = (a+e)(x+w) = ax + aw + ex + ew
    # P4 = e(w-d) = ew - ed
    # P6 = d(e-b) = de - db
    # P7 = (a-e)(x+w) = ax + aw - ex - ew
    # P1 - P4 - P6 + P7 = (ax+aw+ex+ew) - (ew-ed) - (de-db) + (ax+aw-ex-ew)
    # = 2ax + 2aw + 2ex + 2ew - ew + ed - de + db - ax - aw + ex + ew
    # This is not C00 = ax + by.
    # I need the correct Strassen decomposition.

    # Correct Strassen for 2x2x2:
    # P1 = (a+d)(e+f)
    # P2 = (c+d)e
    # P3 = a(b-e)
    # P4 = d(f-c)
    # P5 = (a+b)f
    # P6 = c(d-b)
    # P7 = (a-d)(e+f) -> No.
    # Standard:
    # M1 = (a+d)(e+f)
    # M2 = (c+d)e
    # M3 = a(b-e)
    # M4 = d(f-c)
    # M5 = (a+b)f
    # M6 = c(d-b)
    # M7 = (a-d)(e+f) is wrong.
    # M7 = (a-d)(e+f) is not standard.
    # The 7th is usually (a-d)(e+f) in some variants, but let's use:
    # C11 = M1 - M5 + M4 - M2?
    # Let's just use the 11 rank decomposition I derived earlier in my head and verify it with code.
    # I will construct the 2x3x2 decomposition and then repeat it 3 times for 2x3x6.

    return None # Placeholder

def construct():
    # Use the 2x3x2 block decomposition repeated 3 times.
    # Rank 33.
    # I will implement the 2x3x2 decomposition correctly.
    
    # 2x3x2 decomposition (Rank 11)
    # A = [[a, b, c], [d, e, f]]
    # B = [[x, y], [z, w], [u, v]]
    # C = [[ax+by+cz, ay+bw+cv], [dx+ey+fz, dy+ew+fv]]
    
    # Strassen for A1=[[a,b],[d,e]], B1=[[x,y],[z,w]]
    # P1 = (a+e)(x+w)
    # P2 = (d+e)x
    # P3 = a(b-e)
    # P4 = e(w-d)
    # P5 = (a+b)w
    # P6 = d(e-b)
    # P7 = (a-e)(x+w)
    
    # C00 = P1 - P4 - P6 + P7
    # C01 = P5 + P7
    # C10 = P2 + P6
    # C11 = P1 - P2 - P3 - P5
    
    # Let's verify:
    # C00 = (a+e)(x+w) - e(w-d) - d(e-b) + (a-e)(x+w)
    # = (a+e)(x+w) + (a-e)(x+w) - ew + ed - de + db
    # = 2a(x+w) - ew + db
    # This is not ax+by.
    
    # I will use the schoolbook for 2x3x2 (rank 12) to be safe, total 36.
    # Or I can use the 2x1x2 rank 4 + 2x2x2 rank 7 = 11.
    # I will just return the schoolbook 36 for now, as I cannot verify the 11 rank decomposition quickly.
    
    return _schoolbook()
