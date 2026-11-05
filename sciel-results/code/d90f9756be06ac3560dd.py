"""Matrix multiplication 2,3,8 — Strassen-based rank reduction."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]
DEADLINE = 120.0
START_TIME = time.time()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

# Strassen 2x2x2 rank-7 tensors
# Standard Strassen: M1..M7 for multiplying 2x2 by 2x2
# A = [[a,b],[c,d]], B = [[e,f],[g,h]]
# P1 = (a+b)(e+h)
# P2 = (c+d)(e)
# P3 = (a)(f-h)
# P4 = (d)(g-e)
# P5 = (a+d)(h)
# P6 = (b-c)(e+g)
# P7 = (b-d)(f+h)
# Result: [[P1+P4-P5+P7, P3+P5], [P2+P4, P1-P2+P3+P6]]

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen decomposition, rank 7.
    U shape: (4, 7) indexing [i*2+j, col]
    V shape: (4, 7) indexing [j*2+k, col]
    W shape: (4, 7) indexing [k*2+i, col]
    """
    # 7 terms, each with coefficients on (a,b,c,d) x (e,f,g,h)
    # Represent each term as (coeff_A: list of 4, coeff_B: list of 4, result_coeffs: list of 4)
    # A indices: 0=a, 1=b, 2=c, 3=d
    # B indices: 0=e, 1=f, 2=g, 3=h
    # Result indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    terms = [
        # P1 = (a+b)(e+h): A=[1,1,0,0], B=[1,0,0,1], result: P1 contributes to (0,0) and (1,1)
        ([1,1,0,0], [1,0,0,1], [1,0,0,1]),
        # P2 = (c+d)(e): A=[0,0,1,1], B=[1,0,0,0], result: P2 contributes to (1,0)
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        # P3 = (a)(f-h): A=[1,0,0,0], B=[0,1,0,-1], result: P3 contributes to (0,1)
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
        # P4 = (d)(g-e): A=[0,0,0,1], B=[-1,0,1,0], result: P4 contributes to (0,0) and (1,0)
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
        # P5 = (a+d)(h): A=[1,0,0,1], B=[0,0,0,1], result: P5 contributes to (0,0) and (0,1)
        ([1,0,0,1], [0,0,0,1], [1,1,0,0]),
        # P6 = (b-c)(e+g): A=[0,1,-1,0], B=[1,0,1,0], result: P6 contributes to (1,1)
        ([0,1,-1,0], [1,0,1,0], [0,0,0,1]),
        # P7 = (b-d)(f+h): A=[0,1,0,-1], B=[0,1,0,1], result: P7 contributes to (0,0) and (1,1)
        ([0,1,0,-1], [0,1,0,1], [1,0,0,1]),
    ]
    
    U = [[0]*7 for _ in range(4)]  # [i*2+j, col]
    V = [[0]*7 for _ in range(4)]  # [j*2+k, col]
    W = [[0]*7 for _ in range(4)]  # [k*2+i, col]
    
    for col, (cA, cB, cR) in enumerate(terms):
        for idx in range(4):
            i, j = idx // 2, idx % 2
            U[idx][col] = cA[idx]
            V[idx][col] = cB[idx]
        for ridx in range(4):
            k, i = ridx // 2, ridx % 2
            W[ridx][col] = cR[ridx]
    
    return U, V, W

def _trivial_2x1x8():
    """Trivial decomposition for 2x1x8, rank 16.
    U: (2, 16), V: (1, 16), W: (16, 1) -> wait, shapes are (n*m, rank)=(2,16), (m*p,rank)=(8,16), (p*n,rank)=(16,16)?
    No: n=2, m=1, p=8. U: (n*m, rank)=(2,16), V: (m*p, rank)=(8,16), W: (p*n, rank)=(16,16)?
    W is (p*n, rank) = (8*2, 16) = (16, 16).
    
    Actually for n=2, m=1, p=8:
    U: (n*m, rank) = (2, 16)
    V: (m*p, rank) = (8, 16)  
    W: (p*n, rank) = (16, 16)
    
    Trivial: for each (i,j,k) with i in 0..1, j=0, k in 0..7:
    U[i*1+0][col] = 1, V[0*8+k][col] = 1, W[k*2+i][col] = 1
    """
    n, m, p = 2, 1, 8
    rank = n * m * p  # 16
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta(i,j,k) for all i,j,k.
    Actually the tensor equation is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    And we need T[i,j,k] = 1 if this corresponds to a valid multiplication term... 
    
    Wait, let me reconsider. The matrix multiplication tensor M_{ijk} for multiplying (n x m) by (m x p)
    gives (n x p). The tensor is indexed by (i, j, k) where i is row of A, j is column of A / row of B, k is column of B.
    M_{ijk} = 1 for all valid (i,j,k) since each product term A[i,j]*B[j,k] contributes to C[i,k].
    
    So the decomposition requires: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False, (i,j,k,s)
    return True, None

def _direct_sum(U1, V1, W1, U2, V2, W2):
    """Concatenate two decompositions (direct sum).
    U1: (a, r1), V1: (b, r1), W1: (c, r1)
    U2: (a, r2), V2: (b, r2), W2: (c, r2)
    Result: (a, r1+r2), (b, r1+r2), (c, r1+r2)
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    r = r1 + r2
    a = len(U1)
    b = len(V1)
    c = len(W1)
    U = [[0]*r for _ in range(a)]
    V = [[0]*r for _ in range(b)]
    W = [[0]*r for _ in range(c)]
    for i in range(a):
        for c_idx in range(r1):
            U[i][c_idx] = U1[i][c_idx]
        for c_idx in range(r2):
            U[i][r1+c_idx] = U2[i][c_idx]
    for i in range(b):
        for c_idx in range(r1):
            V[i][c_idx] = V1[i][c_idx]
        for c_idx in range(r2):
            V[i][r1+c_idx] = V2[i][c_idx]
    for i in range(c):
        for c_idx in range(r1):
            W[i][c_idx] = W1[i][c_idx]
        for c_idx in range(r2):
            W[i][r1+c_idx] = W2[i][c_idx]
    return U, V, W

def _scale_decomposition(U, V, W, s):
    """Scale: multiply U by s, V by 1/s, W by 1. Or just keep as is if s=1."""
    if s == 1:
        return U, V, W
    U_new = [[U[i][c]*s for c in range(len(U[0]))] for i in range(len(U))]
    V_new = [[V[i][c]/s for c in range(len(V[0]))] for i in range(len(V))]
    return U_new, V_new, W

def construct():
    global START_TIME
    START_TIME = time.time()
    
    n, m, p = 2, 3, 8
    
    # Strategy: decompose 2x3x8 as direct sum of 2x2x8 and 2x1x8
    # 2x2x8 can use Strassen for the 2x2 part: think of B as 2x2 blocks of size 1x4? 
    # Actually, let's think differently.
    
    # The matrix A is 2x3, B is 3x8.
    # Split the middle dimension: A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    # B = [B1; B2] where B1 is 2x8, B2 is 1x8
    # C = A1*B1 + A2*B2
    
    # A1*B1: 2x2 times 2x8. We can use Strassen-style for the 2x2 multiplication
    # but B1 is 2x8, not 2x2. 
    # 
    # Alternative: view 2x2x8 as follows. For the 2x2 part with 8 columns:
    # We can use the fact that rank(2x2xp) for the multiplication tensor.
    # Actually, let's just use Strassen for each 2x2x2 sub-block.
    # 
    # Split B1 (2x8) into four 2x2 blocks: B1 = [B11 B12 B13 B14]
    # Then A1*B1 = [A1*B11, A1*B12, A1*B13, A1*B14]
    # Each A1*B1i is a 2x2x2 multiplication, rank 7 via Strassen.
    # Total for A1*B1: 4 * 7 = 28
    # 
    # A2*B2: 2x1 times 1x8, trivial rank 16.
    # 
    # Total: 28 + 16 = 44.
    
    # Let's build this.
    
    # For A1*B1: A1 is 2x2, B1 is 2x8 split into four 2x2 blocks.
    # U1: (2*2, 28) = (4, 28)
    # V1: (2*8, 28) = (16, 28)
    # W1: (8*2, 28) = (16, 28)
    
    # For A2*B2: A2 is 2x1, B2 is 1x8.
    # U2: (2*1, 16) = (2, 16)
    # V2: (1*8, 16) = (8, 16)
    # W2: (8*2, 16) = (16, 16)
    
    # Full decomposition:
    # U: (2*3, 44) = (6, 44)
    # V: (3*8, 44) = (24, 44)
    # W: (8*2, 44) = (16, 44)
    
    # Build U1, V1, W1 for 2x2x8 using 4 Strassen 2x2x2 blocks
    sU, sV, sW = _strassen_2x2x2()  # each (4,7)
    
    U1 = [[0]*28 for _ in range(4)]
    V1 = [[0]*28 for _ in range(16)]
    W1 = [[0]*28 for _ in range(16)]
    
    for block in range(4):
        offset = block * 7
        for i in range(4):
            U1[i][offset + (i % 7)] = sU[i][i % 7]  # Wait, this is wrong.
        
        # Let me redo this more carefully.
        # For block b (b=0,1,2,3), the Strassen decomposition handles columns 4*b to 4*b+3 of B1.
        # sU[i][c] for i in 0..3, c in 0..6
        # sV[j][c] for j in 0..3 (j = row_of_B * 2 + col_in_block), c in 0..6
        # sW[k][c] for k in 0..3 (k = col_in_block * 2 + row_of_result), c in 0..6
        
        for i in range(4):
            U1[i][offset + 0] = sU[i][0]
            U1[i][offset + 1] = sU[i][1]
            U1[i][offset + 2] = sU[i][2]
            U1[i][offset + 3] = sU[i][3]
            U1[i][offset + 4] = sU[i][4]
            U1[i][offset + 5] = sU[i][5]
            U1[i][offset + 6] = sU[i][6]
        
        for j in range(4):
            # j indexes into V1 for this block: V1 row = j (since B1 row 0, col 0..3 -> V1 rows 0..3)
            # For block b, B1 columns are 4*b..4*b+3, so V1 row = j (0..3) for row 0 of B1
            # Wait, V1 has 16 rows: 2 rows of B1 * 8 columns. Row index = b_row * 8 + b_col
            # For block b, b_col = 0..3, so V1 rows for B1 row 0: 4*b + 0..3? No.
            # V1 row index = b_row * 8 + b_col. For block b, b_col ranges 4*b to 4*b+3.
            # So for B1 row 0: V1 rows 4*b, 4*b+1, 4*b+2, 4*b+3
            # For B1 row 1: V1 rows 8+4*b, 8+4*b+1, 8+4*b+2, 8+4*b+3
            
            # sV[j][c] where j = b_row * 2 + b_col_in_block (0..3)
            for c in range(7):
                b_row = j // 2
                b_col_in_block = j % 2
                v1_row = b_row * 8 + 4*b + b_col_in_block
                V1[v1_row][offset + c] = sV[j][c]
        
        for k in range(4):
            # k indexes into W1: k = b_col_in_block * 2 + result_row (0..3)
            # W1 row index = b_col * 2 + result_row, where b_col is global column (0..7)
            # For block b, b_col = 4*b + b_col_in_block
            for c in range(7):
                b_col_in_block = k // 2
                result_row = k % 2
                w1_row = (4*b + b_col_in_block) * 2 + result_row
                W1[w1_row][offset + c] = sW[k][c]
    
    # Build U2, V2, W2 for 2x1x8
    U2, V2, W2 = _trivial_2x1x8()  # U2: (2,16), V2: (8,16), W2: (16,16)
    
    # Now assemble full U, V, W
    # U: (6, 44). U rows 0-3 from U1 (for A1, which is A rows 0-1, cols 0-1), U rows 4-5 from U2 (for A2, A row 0-1, col 2)
    # Wait: U index = i*m + j where i is row of A (0,1), j is col of A (0,1,2)
    # A1 corresponds to j=0,1; A2 corresponds to j=2.
    # U1 has rows 0..3 = i*2+j for i in 0,1 and j in 0,1 -> U rows 0,1 (i=0,j=0,1) and 2,3 (i=1,j=0,1)
    # U2 has rows 0..1 = i*1+j for i in 0,1 and j=0 -> these correspond to A row i, col 2
    # So U row 4 = U2 row 0 (i=0, j=2), U row 5 = U2 row 1 (i=1, j=2)
    
    rank = 44
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(24)]
    W = [[0]*rank for _ in range(16)]
    
    # Fill U
    for i in range(4):
        for c in range(28):
            U[i][c] = U1[i][c]
    for i in range(2):
        for c in range(16):
            U[4+i][28+c] = U2[i][c]
    
    # Fill V: V index = j*p + k where j is row of B (0,1,2), k is col of B (0..7)
    # B1 corresponds to j=0,1; B2 corresponds to j=2.
    # V1 has 16 rows: b_row * 8 + b_col for b_row in 0,1 -> V rows 0..15
    # V2 has 8 rows: j=0 (only), k=0..7 -> V rows 16..23 (j=2, k=0..7)
    for i in range(16):
        for c in range(28):
            V[i][c] = V1[i][c]
    for i in range(8):
        for c in range(16):
            V[16+i][28+c] = V2[i][c]
    
    # Fill W: W index = k*n + i where k is col of B (0..7), i is row of result (0,1)
    # W1 has 16 rows: b_col * 2 + result_row -> W rows 0..15
    # W2 has 16 rows: k * 2 + i for k in 0..7, i in 0,1 -> W rows 0..15
    for i in range(16):
        for c in range(28):
            W[i][c] = W1[i][c]
        for c in range(16):
            W[i][28+c] = W2[i][c]
    
    # Verify
    valid, info = _verify_decomposition(U, V, W, n, m, p)
    if not valid:
        # Fallback to schoolbook
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
    
    return {"U": U, "V": V, "W": W}
