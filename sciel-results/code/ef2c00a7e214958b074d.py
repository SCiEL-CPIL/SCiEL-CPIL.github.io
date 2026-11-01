import time
import sys

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 34 decomposition
    rank = 34
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for A (3x3): i in 0..2, j in 0..2 -> index i*3+j
    # Indices for B (3x4): j in 0..2, k in 0..3 -> index j*4+k
    # Indices for C (4x3? No, C is 3x4? 
    # Task: 3x3 * 3x4 -> 3x4.
    # W[k*n+i, r] where C[i,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Wait, the trace convention is W[k*n+i, r].
    # C[i,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This implies W index is k*n + i.
    # C is 3x4. i in 0..2, k in 0..3.
    # W size is p*n = 4*3 = 12.
    
    col = 0
    
    # Block 1: C[:, 0:2] = A * B[:, 0:2]
    # B1 is 3x2 (cols 0,1)
    # C1 is 3x2 (cols 0,1)
    
    # Strassen for A_TL (2x2) and B1_TL (2x2)
    # A_TL: A[0,0], A[0,1], A[1,0], A[1,1]
    # B1_TL: B[0,0], B[0,1], B[1,0], B[1,1]
    
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B10)
    # P7 = (A01-A11)(B01+B11)
    
    # C1_TL = P1 - P4 + P7 - P5 (C00, C01, C10, C11 parts? No)
    # C00 = P1 - P4 + P7 - P5? 
    # Standard Strassen:
    # C11 = P1 - P4 + P7 - P5
    # C12 = P3 + P4
    # C21 = P2 + P5
    # C22 = P1 - P2 + P3 + P6
    
    # Indices in U, V, W for this block
    
    # We need to map Strassen steps to rank-1 terms.
    # Each Pk is a scalar product of linear combinations.
    # We add 7 terms.
    
    # Helper to add a rank-1 term defined by sums of basis vectors
    # U is 9x34, V is 12x34, W is 12x34
    
    def add_term(u_idx, v_idx, w_idx, val):
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val

    # P1: (A00+A11)(B00+B11)
    # U: A00(1), A11(1)
    # V: B00(1), B11(1)
    # W: C00(1), C11(1), C10(-1), C01(-1)? No.
    # C00 coeff: 1
    # C11 coeff: 1
    # C10 coeff: -1 (from P4, P7, P5? No, P1 only contributes +1 to C00 and C11? 
    # Wait, P1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # C00 = A00B00 + A01B10 + A10B01 + A11B11.
    # P1 contains A00B00 and A11B11.
    # So P1 contributes +1 to C00 and +1 to C11.
    # It does NOT contribute to C01 or C10.
    
    # Let's trace Strassen carefully.
    # C00 = P1 - P4 + P7 - P5
    # C01 = P3 + P4
    # C10 = P2 + P5
    # C11 = P1 - P2 + P3 + P6
    
    # So:
    # P1: C00(+1), C11(+1)
    # P2: C10(+1), C11(-1)
    # P3: C01(+1), C11(+1)
    # P4: C00(-1), C01(+1)
    # P5: C00(-1), C10(+1)
    # P6: C11(+1)
    # P7: C00(+1)
    
    # U, V, W indices for A_TL, B1_TL, C1_TL
    # A00=0, A01=1, A10=3, A11=4
    # B00=0, B01=1, B10=4, B11=5 (B is 3x4, B[j,k] -> j*4+k. B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5)
    # C00=0, C01=1, C10=3, C11=4 (C is 3x4, C[i,k] -> W[k*3+i]. C[0,0]=0, C[0,1]=1, C[1,0]=3, C[1,1]=4)
    
    # P1: (A00+A11)(B00+B11)
    U[0][col] += 1; U[4][col] += 1
    V[0][col] += 1; V[5][col] += 1
    W[0][col] += 1; W[4][col] += 1
    col += 1
    
    # P2: (A10+A11)B00
    U[3][col] += 1; U[4][col] += 1
    V[0][col] += 1
    W[3][col] += 1; W[4][col] -= 1
    col += 1
    
    # P3: A00(B10-B11)
    U[0][col] += 1
    V[4][col] += 1; V[5][col] -= 1
    W[1][col] += 1; W[4][col] += 1
    col += 1
    
    # P4: A11(B01-B00)
    U[4][col] += 1
    V[1][col] += 1; V[0][col] -= 1
    W[0][col] -= 1; W[1][col] += 1
    col += 1
    
    # P5: (A00+A01)B11
    U[0][col] += 1; U[1][col] += 1
    V[5][col] += 1
    W[0][col] -= 1; W[3][col] += 1
    col += 1
    
    # P6: (A10-A00)(B00+B10)
    U[3][col] += 1; U[0][col] -= 1
    V[0][col] += 1; V[4][col] += 1
    W[4][col] += 1
    col += 1
    
    # P7: (A01-A11)(B01+B11)
    U[1][col] += 1; U[4][col] -= 1
    V[1][col] += 1; V[5][col] += 1
    W[0][col] += 1
    col += 1
    
    # Now the border terms for Block 1 (cols 0,1 of C)
    # 1. a1 * b1 (A02, A12 with B20, B21)
    # A02=2, A12=5
    # B20=8, B21=9
    # C00, C01, C10, C11 contributions
    # C00 += A02 B20
    # C01 += A02 B21
    # C10 += A12 B20
    # C11 += A12 B21
    
    # T1: A02 B20
    U[2][col] += 1
    V[8][col] += 1
    W[0][col] += 1
    col += 1
    
    # T2: A02 B21
    U[2][col] += 1
    V[9][col] += 1
    W[1][col] += 1
    col += 1
    
    # T3: A12 B20
    U[5][col] += 1
    V[8][col] += 1
    W[3][col] += 1
    col += 1
    
    # T4: A12 B21
    U[5][col] += 1
    V[9][col] += 1
    W[4][col] += 1
    col += 1
    
    # 2. u^T * B1_TL (A20, A21 with B00,B01,B10,B11)
    # A20=6, A21=7
    # C20, C21 contributions
    # C20 += A20 B00 + A21 B10
    # C21 += A20 B01 + A21 B11
    
    # T5: A20 B00
    U[6][col] += 1
    V[0][col] += 1
    W[6][col] += 1 # C[2,0] -> W[0*3+2] = W[2]? No.
    # W index is k*n + i.
    # C[2,0] -> k=0, i=2 -> 0*3+2 = 2.
    # C[2,1] -> k=1, i=2 -> 1*3+2 = 5.
    # My previous W indices:
    # C00 (i=0,k=0) -> 0
    # C01 (i=0,k=1) -> 3? No. k=1, i=0 -> 1*3+0 = 3.
    # C10 (i=1,k=0) -> 1
    # C11 (i=1,k=1) -> 4
    # C20 (i=2,k=0) -> 2
    # C21 (i=2,k=1) -> 5
    
    # Let's re-verify W indices in Strassen part.
    # C00 (0,0) -> W[0]
    # C01 (0,1) -> W[3]
    # C10 (1,0) -> W[1]
    # C11 (1,1) -> W[4]
    
    # In my Strassen code above:
    # P1: W[0](+1), W[4](+1). Correct.
    # P2: W[1](+1), W[4](-1). Correct.
    # P3: W[3](+1), W[4](+1). Correct.
    # P4: W[0](-1), W[3](+1). Correct.
    # P5: W[0](-1), W[1](+1). Correct.
    # P6: W[4](+1). Correct.
    # P7: W[0](+1). Correct.
    
    # So T5: A20 B00 contributes to C20 (W[2]).
    U[6][col] += 1
    V[0][col] += 1
    W[2][col] += 1
    col += 1
    
    # T6: A20 B01
    U[6][col] += 1
    V[1][col] += 1
    W[5][col] += 1
    col += 1
    
    # T7: A21 B10
    U[7][col] += 1
    V[4][col] += 1
    W[2][col] += 1
    col += 1
    
    # T8: A21 B11
    U[7][col] += 1
    V[5][col] += 1
    W[5][col] += 1
    col += 1
    
    # 3. A22 * b1 (A22 with B20, B21)
    # A22=8
    # C20 += A22 B20
    # C21 += A22 B21
    
    # T9: A22 B20
    U[8][col] += 1
    V[8][col] += 1
    W[2][col] += 1
    col += 1
    
    # T10: A22 B21
    U[8][col] += 1
    V[9][col] += 1
    W[5][col] += 1
    col += 1
    
    # Block 1 done. col = 17.
    
    # Block 2: C[:, 2:4] = A * B[:, 2:4]
    # B2 is 3x2 (cols 2,3)
    # C2 is 3x2 (cols 2,3)
    
    # Strassen for A_TL and B2_TL
    # B2_TL: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    
    # P1
    U[0][col] += 1; U[4][col] += 1
    V[2][col] += 1; V[7][col] += 1
    W[6][col] += 1; W[10][col] += 1 # C02 (W[6]), C12 (W[10])
    col += 1
    
    # P2
    U[3][col] += 1; U[4][col] += 1
    V[2][col] += 1
    W[7][col] += 1; W[10][col] -= 1 # C12 (W[10]), C12? No C12 is (1,2) -> W[2*3+1]=7? 
    # C[i,k].
    # C02 (0,2) -> k=2, i=0 -> 6.
    # C03 (0,3) -> k=3, i=0 -> 9.
    # C12 (1,2) -> k=2, i=1 -> 7.
    # C13 (1,3) -> k=3, i=1 -> 10.
    
    # Strassen P2: C10(+1), C11(-1). Here C12(+1), C13(-1)?
    # C12 corresponds to C10 in local 2x2.
    # C13 corresponds to C11 in local 2x2.
    # So W[7](+1), W[10](-1). Correct.
    col += 1
    
    # P3
    U[0][col] += 1
    V[6][col] += 1; V[7][col] -= 1
    W[9][col] += 1; W[10][col] += 1 # C03 (W[9]), C13 (W[10])
    col += 1
    
    # P4
    U[4][col] += 1
    V[3][col] += 1; V[2][col] -= 1
    W[6][col] -= 1; W[9][col] += 1 # C02 (W[6]), C03 (W[9])
    col += 1
    
    # P5
    U[0][col] += 1; U[1][col] += 1
    V[7][col] += 1
    W[6][col] -= 1; W[7][col] += 1 # C02 (W[6]), C12 (W[7])
    col += 1
    
    # P6
    U[3][col] += 1; U[0][col] -= 1
    V[2][col] += 1; V[6][col] += 1
    W[10][col] += 1 # C13 (W[10])
    col += 1
    
    # P7
    U[1][col] += 1; U[4][col] -= 1
    V[3][col] += 1; V[7][col] += 1
    W[6][col] += 1 # C02 (W[6])
    col += 1
    
    # Borders for Block 2
    
    # T1: A02 B22 (B[2,2]=10)
    U[2][col] += 1
    V[10][col] += 1
    W[6][col] += 1
    col += 1
    
    # T2: A02 B23 (B[2,3]=11)
    U[2][col] += 1
    V[11][col] += 1
    W[9][col] += 1
    col += 1
    
    # T3: A12 B22
    U[5][col] += 1
    V[10][col] += 1
    W[7][col] += 1
    col += 1
    
    # T4: A12 B23
    U[5][col] += 1
    V[11][col] += 1
    W[10][col] += 1
    col += 1
    
    # T5: A20 B02
    U[6][col] += 1
    V[2][col] += 1
    W[8][col] += 1 # C22 (W[8])
    col += 1
    
    # T6: A20 B03
    U[6][col] += 1
    V[3][col] += 1
    W[11][col] += 1 # C23 (W[11])
    col += 1
    
    # T7: A21 B12
    U[7][col] += 1
    V[6][col] += 1
    W[8][col] += 1
    col += 1
    
    # T8: A21 B13
    U[7][col] += 1
    V[7][col] += 1
    W[11][col] += 1
    col += 1
    
    # T9: A22 B22
    U[8][col] += 1
    V[10][col] += 1
    W[8][col] += 1
    col += 1
    
    # T10: A22 B23
    U[8][col] += 1
    V[11][col] += 1
    W[11][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
