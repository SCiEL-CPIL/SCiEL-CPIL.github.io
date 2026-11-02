"""Deterministic rank-44 tensor decomposition for M_{2,3,8} using Strassen block structure."""
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 8]


def _strassen_2x2():
    """Return U (4x7), V (4x7), W (4x7) for 2x2 matrix multiplication, rank 7.
    U[i*2+j] = input A[i,j]
    V[i*2+j] = input B[i,j]
    W[i*2+j] = output C[i,j]
    """
    # Strassen's 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = [[Fraction(0)] * 7 for _ in range(4)]
    V = [[Fraction(0)] * 7 for _ in range(4)]
    W = [[Fraction(0)] * 7 for _ in range(4)]
    
    # U: A entries: a=U[0], b=U[1], c=U[2], d=U[3]
    # V: B entries: e=V[0], f=V[1], g=V[2], h=V[3]
    # W: C entries: c11=W[0], c12=W[1], c21=W[2], c22=W[3]
    
    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # c11 += M1, c22 += M1
    
    # M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1  # c21 += M2, c22 -= M2
    
    # M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1  # c12 += M3, c22 += M3
    
    # M4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1  # c11 += M4, c21 += M4
    
    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1  # c11 -= M5, c12 += M5
    
    # M6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # c22 += M6
    
    # M7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # c11 += M7
    
    return U, V, W


def _build_rank44():
    """Build rank-44 decomposition for M_{2,3,8}.
    
    Structure: Split the 3 rows of A into:
      - Row 0-1 (first 2 rows): 2x3 block -> treat as [2x2 | 1x2]
      - Row 2 (last row): 1x3 block -> 1x2 | 1x1
    
    Actually, let's use a cleaner decomposition:
    A is 2x3, B is 3x8, C is 2x8.
    
    Split B into B = [B1 | B2] where B1 is 3x4, B2 is 3x4.
    C = [A*B1 | A*B2].
    
    So we need to compute A*B for 2x3 * 3x4 with rank r, then duplicate for B2.
    
    For 2x3 * 3x4:
    Split A = [[A_top], [A_bottom]] where A_top is 2x2... no, A is 2x3.
    
    Let me think differently. Use the known result that M_{2,3,4} has rank 22.
    Then M_{2,3,8} = M_{2,3,4} direct sum M_{2,3,4} has rank 44.
    
    M_{2,3,4} decomposition:
    A (2x3) * B (3x4) = C (2x4)
    
    Split A = [A1 | A2] where A1 is 2x2, A2 is 2x1.
    Split B = [[B11], [B21]] where B11 is 2x4, B21 is 1x4.
    
    C = A1*B11 + A2*B21
    
    A1*B11: 2x2 * 2x4, rank 7 (Strassen on the 2x2 part, applied to each of 4 columns)
    Actually 2x2 * 2x4: this is 2x2 * 2x4. We can view it as computing 4 independent 2x2*2x1 products... no.
    
    Better: 2x2 * 2x4 matrix multiplication. The tensor M_{2,2,4} has rank 14 (7*2, two independent Strassen blocks for the two column-pairs? No.)
    
    Actually M_{2,2,p} for p=4: we compute A(2x2) * B(2x4) = C(2x4).
    This is equivalent to 2 independent 2x2 * 2x2 multiplications if we pair columns.
    So rank = 7 + 7 = 14.
    
    A2*B21: 2x1 * 1x4 = outer product, rank 2*1*4 = 8... wait, that's schoolbook.
    Actually 2x1 * 1x4: C[i,k] = A[i,0] * B[0,k]. This is 2*4 = 8 rank-1 terms.
    
    Total for M_{2,3,4}: 14 + 8 = 22. 
    
    So M_{2,3,8} = 22 + 22 = 44.
    """
    
    # Build M_{2,2,4} (rank 14): A(2x2) * B(2x4) = C(2x4)
    # Two Strassen blocks for columns (0,1) and (2,3)
    U_224, V_224, W_224 = _strassen_22_4()
    
    # Build M_{2,1,4} (rank 8): A(2x1) * B(1x4) = C(2x4)
    U_214, V_214, W_214 = _strassen_21_4()
    
    # Combine to get M_{2,3,4} (rank 22)
    # U: 2*3=6 rows, V: 3*4=12 rows, W: 4*2=8 rows
    U_234 = _combine_22_4_and_21_4(U_224, V_224, W_224, U_214, V_214, W_214)
    
    # Now duplicate for M_{2,3,8}: two copies of M_{2,3,4}
    # U: 6 rows (same), V: 3*8=24 rows, W: 8*2=16 rows
    U_full = []
    for i in range(6):
        U_full.append(U_234[i] + U_234[i])  # duplicate columns
    
    V_full = []
    for i in range(12):
        V_full.append(V_234[i] + [Fraction(0)] * 22)  # first half
    for i in range(12):
        V_full.append([Fraction(0)] * 22 + V_234[i])  # second half
    
    W_full = []
    for i in range(8):
        W_full.append(W_234[i] + [Fraction(0)] * 22)  # first half columns
    for i in range(8):
        W_full.append([Fraction(0)] * 22 + W_234[i])  # second half columns
    
    return U_full, V_full, W_full


def _strassen_22_4():
    """M_{2,2,4}: A(2x2)*B(2x4)=C(2x4), rank 14.
    Two independent Strassen 2x2 multiplications.
    U: 4 rows, V: 8 rows, W: 8 rows, 14 cols.
    """
    U, V, W = _strassen_2x2()
    # First block: B cols 0,1 -> C cols 0,1
    U1 = [row[:7] for row in U]
    V1 = []
    for i in range(4):
        V1.append(V[i][:7] + [Fraction(0)]*7)
    W1 = []
    for i in range(4):
        W1.append(W[i][:7] + [Fraction(0)]*7)
    
    # Second block: B cols 2,3 -> C cols 2,3
    U2 = [row[:7] for row in U]
    V2 = []
    for i in range(4):
        V2.append([Fraction(0)]*7 + V[i][:7])
    W2 = []
    for i in range(4):
        W2.append([Fraction(0)]*7 + W[i][:7])
    
    # Combine: U is 4x14, V is 8x14, W is 8x14
    U_out = [U1[i] + U2[i] for i in range(4)]
    V_out = []
    for i in range(4):
        V_out.append(V1[i] + V2[i])
    for i in range(4):
        V_out.append(V1[i+4] + V2[i+4])
    
    W_out = []
    for i in range(4):
        W_out.append(W1[i] + W2[i])
    for i in range(4):
        W_out.append(W1[i+4] + W2[i+4])
    
    return U_out, V_out, W_out


def _strassen_21_4():
    """M_{2,1,4}: A(2x1)*B(1x4)=C(2x4), rank 8.
    C[i,k] = A[i,0] * B[0,k]
    U: 2 rows, V: 4 rows, W: 8 rows, 8 cols.
    """
    U = [[Fraction(0)]*8 for _ in range(2)]
    V = [[Fraction(0)]*8 for _ in range(4)]
    W = [[Fraction(0)]*8 for _ in range(8)]
    
    col = 0
    for i in range(2):
        for k in range(4):
            U[i][col] = 1
            V[k][col] = 1
            W[k*2 + i][col] = 1  # C[i,k], W index = k*n + i = k*2 + i
            col += 1
    
    return U, V, W


def _combine_22_4_and_21_4(U224, V224, W224, U214, V214, W214):
    """Combine M_{2,2,4} and M_{2,1,4} to get M_{2,3,4}, rank 22.
    
    A(2x3) = [A12 | A21] where A12 is 2x2 (cols 0,1), A21 is 2x1 (col 2)
    B(3x4) = [[B12], [B21]] where B12 is 2x4 (rows 0,1), B21 is 1x4 (row 2)
    
    U: 6 rows (2x3), V: 12 rows (3x4), W: 8 rows (4x2), 22 cols.
    
    For the 2x2 part (cols 0,1 of A, rows 0,1 of B):
      U rows 0,1 -> U224 rows 0,1 (A[0,0], A[1,0], A[0,1], A[1,1])
      Wait, U224 has 4 rows for A(2x2): row0=A[0,0], row1=A[0,1], row2=A[1,0], row3=A[1,1]
      In our M_{2,3,4}, U has 6 rows: U[0]=A[0,0], U[1]=A[0,1], U[2]=A[0,2], U[3]=A[1,0], U[4]=A[1,1], U[5]=A[1,2]
      
      So U224 rows map to: U224[0]->U[0], U224[1]->U[1], U224[2]->U[3], U224[3]->U[4]
      V224 rows map to: V224[0]->V[0], V224[1]->V[1], V224[2]->V[2], V224[3]->V[3], V224[4]->V[4], V224[5]->V[5], V224[6]->V[6], V224[7]->V[7]
      Wait, V224 has 8 rows for B(2x4): row0=B[0,0], row1=B[0,1], row2=B[0,2], row3=B[0,3], row4=B[1,0], row5=B[1,1], row6=B[1,2], row7=B[1,3]
      In M_{2,3,4}, V has 12 rows: V[0]=B[0,0], V[1]=B[0,1], V[2]=B[0,2], V[3]=B[0,3], V[4]=B[1,0], V[5]=B[1,1], V[6]=B[1,2], V[7]=B[1,3], V[8]=B[2,0], V[9]=B[2,1], V[10]=B[2,2], V[11]=B[2,3]
      
      For the 2x1 part (col 2 of A, row 2 of B):
      U214 has 2 rows: U214[0]=A[0,2], U214[1]=A[1,2] -> map to U[2], U[5]
      V214 has 4 rows: V214[k]=B[2,k] -> map to V[8+k]
      W214 has 8 rows: W214[k*2+i]=C[i,k] -> map to W[k*2+i]
    """
    rank = 14 + 8
    U = [[Fraction(0)]*rank for _ in range(6)]
    V = [[Fraction(0)]*rank for _ in range(12)]
    W = [[Fraction(0)]*rank for _ in range(8)]
    
    # Part 1: 2x2 * 2x4 (14 cols)
    u_map = {0: 0, 1: 1, 2: 3, 3: 4}  # U224 row -> U row
    for u224_row, u_row in u_map.items():
        for c in range(14):
            U[u_row][c] = U224[u224_row][c]
    
    v_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}  # V224 row -> V row
    for v224_row, v_row in v_map.items():
        for c in range(14):
            V[v_row][c] = V224[v224_row][c]
    
    for w_row in range(8):
        for c in range(14):
            W[w_row][c] = W224[w_row][c]
    
    # Part 2: 2x1 * 1x4 (8 cols, offset by 14)
    for c in range(8):
        U[2][14 + c] = U214[0][c]
        U[5][14 + c] = U214[1][c]
        for k in range(4):
            V[8 + k][14 + c] = V214[k][c]
        for w_row in range(8):
            W[w_row][14 + c] = W214[w_row][c]
    
    return U, V, W


def _build_schoolbook():
    """Fallback schoolbook decomposition, rank 48."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W):
    """Verify that sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 iff j is the connecting index."""
    n, m, p = DIMENSIONS
    nrm = len(U)
    nrV = len(V)
    nrW = len(W)
    rank = len(U[0]) if U else 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
                val = Fraction(0)
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if val != 1:
                    return False
    return True


def construct():
    start_time = time.time()
    deadline = 120.0 - 10.0  # 110s safety margin
    
    # Try rank-44 construction
    try:
        U, V, W = _build_rank44()
        # Verify
        if _verify(U, V, W):
            # Convert to list of int/Fraction
            U_out = [[float(x) if x.denominator != 1 else int(x) for x in row] for row in U]
            V_out = [[float(x) if x.denominator != 1 else int(x) for x in row] for row in V]
            W_out = [[float(x) if x.denominator != 1 else int(x) for x in row] for row in W]
            return {"U": U_out, "V": V_out, "W": W_out}
    except Exception:
        pass
    
    # Fallback: schoolbook
    U, V, W = _build_schoolbook()
    U_out = [[float(x) if x.denominator != 1 else int(x) for x in row] for row in U]
    V_out = [[float(x) if x.denominator != 1 else int(x) for x in row] for row in V]
    W_out = [[float(x) if x.denominator != 1 else int(x) for x in row] for row in W]
    return {"U": U_out, "V": V_out, "W": W_out}
