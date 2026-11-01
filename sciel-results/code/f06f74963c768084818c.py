"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Rank 46 decomposition using Strassen on 2x2 blocks and direct handling of 2x3/2x1/3x2/1x2/1x1 blocks.
    #
    # Matrix A is 2x5, matrix B is 5x5, result C is 2x5.
    # We split A = [A11, A12] where A11 is 2x2, A12 is 2x3.
    # We split B into:
    #   B11 (2x2), B12 (2x3), B13 (2x1)
    #   B21 (3x2), B22 (3x3), B23 (3x1)
    #   B31 (1x2), B32 (1x3), B33 (1x1)
    #
    # C11 = A11*B11 + A12*B21 + A11*B31  (2x2)
    # C12 = A11*B12 + A12*B22 + A11*B32  (2x3)
    # C13 = A11*B13 + A12*B23 + A11*B33  (2x1)
    # C21 = A12*B21 + A12*B21... wait no.
    #
    # Let me re-derive the blocks more carefully.
    # A is 2x5. Split columns: A = [A_left (2x2), A_right (2x3)].
    # B is 5x5. Split rows and columns:
    #   Rows: first 2, next 3. So B = [[B11, B12, B13], [B21, B22, B23]] where
    #     B11 is 2x2, B12 is 2x3, B13 is 2x1
    #     B21 is 3x2, B22 is 3x3, B23 is 3x1
    #   Wait, B is 5x5. Split rows into 2 and 3, columns into 2, 3, 1? That gives:
    #     B11: 2x2, B12: 2x3, B13: 2x1
    #     B21: 3x2, B22: 3x3, B23: 3x1
    #   Total rows: 2+3=5. Total cols: 2+3+1=6. That's wrong, B is 5x5.
    #
    # Let me split columns of B into 2, 2, 1:
    #   B11: 2x2, B12: 2x2, B13: 2x1
    #   B21: 3x2, B22: 3x2, B23: 3x1
    #   Total cols: 2+2+1=5. Good.
    #
    # C = A*B = [A_left, A_right] * B
    # C is 2x5. Split columns of C into 2, 2, 1:
    #   C11 = A_left*B11 + A_right*B21  (2x2)
    #   C12 = A_left*B12 + A_right*B22  (2x2)
    #   C13 = A_left*B13 + A_right*B23  (2x1)
    #
    # So we need:
    #   2x2 x 2x2: C11 part 1 (A_left*B11) - Strassen rank 7
    #   2x2 x 2x2: C12 part 1 (A_left*B12) - Strassen rank 7
    #   2x2 x 2x1: C13 part 1 (A_left*B13) - direct rank 4
    #   2x3 x 3x2: C11 part 2 (A_right*B21) - need to compute
    #   2x3 x 3x2: C12 part 2 (A_right*B22) - need to compute
    #   2x3 x 3x1: C13 part 2 (A_right*B23) - need to compute
    #
    # For 2x3 x 3x2:
    #   A_right is 2x3, B21 is 3x2. Result is 2x2.
    #   Split A_right = [A_r1 (2x2), A_r2 (2x1)]
    #   Split B21 = [B21a (2x2), B21b (1x2)] (rows: first 2, last 1)
    #   A_right*B21 = A_r1*B21a + A_r2*B21b
    #   A_r1*B21a: 2x2 x 2x2, Strassen rank 7
    #   A_r2*B21b: 2x1 x 1x2, direct rank 4 (outer product)
    #   Total: 11
    #
    # Similarly for 2x3 x 3x2 (C12 part 2): 11
    # For 2x3 x 3x1 (C13 part 2):
    #   A_right is 2x3, B23 is 3x1. Result is 2x1.
    #   Split A_right = [A_r1 (2x2), A_r2 (2x1)]
    #   Split B23 = [B23a (2x1), B23b (1x1)]
    #   A_right*B23 = A_r1*B23a + A_r2*B23b
    #   A_r1*B23a: 2x2 x 2x1, direct rank 4
    #   A_r2*B23b: 2x1 x 1x1, direct rank 2
    #   Total: 6
    #
    # Grand total: 7 + 7 + 4 + 11 + 11 + 6 = 46. 
    #
    # Now let's build the U, V, W matrices.
    # U is (2*5) x 46 = 10 x 46
    # V is (5*5) x 46 = 25 x 46
    # W is (5*2) x 46 = 10 x 46
    #
    # Indexing:
    # U[i*5+j, c] corresponds to A[i,j]
    # V[j*5+k, c] corresponds to B[j,k]
    # W[k*2+i, c] corresponds to C[i,k] (trace convention W[k*n+i])
    #
    # Let me define the blocks:
    # A indices: A[0,0], A[0,1] are in A_left (col 0,1). A[0,2],A[0,3],A[0,4] in A_right.
    #            A[1,0], A[1,1] in A_left. A[1,2],A[1,3],A[1,4] in A_right.
    # B indices: B[j,k]. Rows j=0,1 are in top block; j=2,3,4 in bottom block.
    #            Cols k=0,1 in first; k=2,3 in second; k=4 in third.
    # C indices: C[i,k]. i=0,1. k=0,1 in first; k=2,3 in second; k=4 in third.
    #
    # Let me create helper functions to add terms.
    
    n, m, p = 2, 5, 5
    rank = 46
    U = [[0.0] * rank for _ in range(n * m)]  # 10 x 46
    V = [[0.0] * rank for _ in range(m * p)]  # 25 x 46
    W = [[0.0] * rank for _ in range(p * n)]  # 10 x 46
    
    col = 0
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, c):
        U[u_idx][c] += u_val
        V[v_idx][c] += v_val
        W[w_idx][c] += w_val
    
    # Block 1: A_left * B11 (2x2 x 2x2) - Strassen, rank 7
    # A_left: A[0,0]=A00, A[0,1]=A01, A[1,0]=A10, A[1,1]=A11
    # B11: B[0,0]=B00, B[0,1]=B01, B[1,0]=B10, B[1,1]=B11
    # C result: C[0,0], C[0,1], C[1,0], C[1,1]
    # U indices: A[i,j] -> U[i*5+j]
    #   A00: U[0], A01: U[1], A10: U[5], A11: U[6]
    # V indices: B[j,k] -> V[j*5+k]
    #   B00: V[0], B01: V[1], B10: V[5], B11: V[6]
    # W indices: C[i,k] -> W[k*2+i]
    #   C[0,0]: W[0], C[0,1]: W[1], C[1,0]: W[2], C[1,1]: W[3]
    
    # M1 = (A00+A11)(B00+B11)
    add_term(0, 1, 0, 1, 0, 1, col); add_term(6, 1, 6, 1, 0, 1, col)
    add_term(0, 1, 0, 1, 2, 1, col); add_term(6, 1, 6, 1, 2, 1, col)
    add_term(0, 1, 0, 1, 1, -1, col); add_term(6, 1, 6, 1, 1, -1, col)
    add_term(0, 1, 0, 1, 3, 1, col); add_term(6, 1, 6, 1, 3, 1, col)
    col += 1
    
    # M2 = (A10+A11)B00
    add_term(5, 1, 0, 1, 2, 1, col); add_term(6, 1, 0, 1, 2, 1, col)
    add_term(5, 1, 0, 1, 3, -1, col); add_term(6, 1, 0, 1, 3, -1, col)
    col += 1
    
    # M3 = A00(B01-B11)
    add_term(0, 1, 1, 1, 1, 1, col)
    add_term(0, 1, 6, -1, 1, 1, col)
    add_term(0, 1, 1, 1, 3, 1, col)
    add_term(0, 1, 6, -1, 3, 1, col)
    col += 1
    
    # M4 = A11(B10-B00)
    add_term(6, 1, 5, 1, 0, 1, col)
    add_term(6, 1, 0, -1, 0, 1, col)
    add_term(6, 1, 5, 1, 2, 1, col)
    add_term(6, 1, 0, -1, 2, 1, col)
    col += 1
    
    # M5 = (A00+A01)B11
    add_term(0, 1, 6, 1, 1, 1, col); add_term(1, 1, 6, 1, 1, 1, col)
    add_term(0, 1, 6, 1, 3, -1, col); add_term(1, 1, 6, 1, 3, -1, col)
    add_term(0, 1, 6, 1, 0, -1, col); add_term(1, 1, 6, 1, 0, -1, col)
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    add_term(5, 1, 0, 1, 3, 1, col); add_term(0, -1, 0, 1, 3, 1, col)
    add_term(5, 1, 1, 1, 3, 1, col); add_term(0, -1, 1, 1, 3, 1, col)
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    add_term(1, 1, 5, 1, 0, 1, col); add_term(6, -1, 5, 1, 0, 1, col)
    add_term(1, 1, 6, 1, 0, 1, col); add_term(6, -1, 6, 1, 0, 1, col)
    col += 1
    
    # Block 2: A_left * B12 (2x2 x 2x2) - Strassen, rank 7
    # B12: B[0,2]=B02, B[0,3]=B03, B[1,2]=B12, B[1,3]=B13
    # V indices: B02: V[2], B03: V[3], B12: V[7], B13: V[8]
    # C result: C[0,2], C[0,3], C[1,2], C[1,3]
    # W indices: C[0,2]: W[4], C[0,3]: W[5], C[1,2]: W[6], C[1,3]: W[7]
    
    # M1 = (A00+A11)(B02+B13)
    add_term(0, 1, 2, 1, 4, 1, col); add_term(6, 1, 8, 1, 4, 1, col)
    add_term(0, 1, 2, 1, 6, 1, col); add_term(6, 1, 8, 1, 6, 1, col)
    add_term(0, 1, 2, 1, 5, -1, col); add_term(6, 1, 8, 1, 5, -1, col)
    add_term(0, 1, 2, 1, 7, 1, col); add_term(6, 1, 8, 1, 7, 1, col)
    col += 1
    
    # M2 = (A10+A11)B02
    add_term(5, 1, 2, 1, 6, 1, col); add_term(6, 1, 2, 1, 6, 1, col)
    add_term(5, 1, 2, 1, 7, -1, col); add_term(6, 1, 2, 1, 7, -1, col)
    col += 1
    
    # M3 = A00(B03-B13)
    add_term(0, 1, 3, 1, 5, 1, col)
    add_term(0, 1, 8, -1, 5, 1, col)
    add_term(0, 1, 3, 1, 7, 1, col)
    add_term(0, 1, 8, -1, 7, 1, col)
    col += 1
    
    # M4 = A11(B12-B02)
    add_term(6, 1, 7, 1, 4, 1, col)
    add_term(6, 1, 2, -1, 4, 1, col)
    add_term(6, 1, 7, 1, 6, 1, col)
    add_term(6, 1, 2, -1, 6, 1, col)
    col += 1
    
    # M5 = (A00+A01)B13
    add_term(0, 1, 8, 1, 5, 1, col); add_term(1, 1, 8, 1, 5, 1, col)
    add_term(0, 1, 8, 1, 7, -1, col); add_term(1, 1, 8, 1, 7, -1, col)
    add_term(0, 1, 8, 1, 4, -1, col); add_term(1, 1, 8, 1, 4, -1, col)
    col += 1
    
    # M6 = (A10-A00)(B02+B03)
    add_term(5, 1, 2, 1, 7, 1, col); add_term(0, -1, 2, 1, 7, 1, col)
    add_term(5, 1, 3, 1, 7, 1, col); add_term(0, -1, 3, 1, 7, 1, col)
    col += 1
    
    # M7 = (A01-A11)(B12+B13)
    add_term(1, 1, 7, 1, 4, 1, col); add_term(6, -1, 7, 1, 4, 1, col)
    add_term(1, 1, 8, 1, 4, 1, col); add_term(6, -1, 8, 1, 4, 1, col)
    col += 1
    
    # Block 3: A_left * B13 (2x2 x 2x1) - direct, rank 4
    # B13: B[0,4]=B04, B[1,4]=B14
    # V indices: B04: V[4], B14: V[9]
    # C result: C[0,4], C[1,4]
    # W indices: C[0,4]: W[8], C[1,4]: W[9]
    # C[0,4] = A00*B04 + A01*B14
    # C[1,4] = A10*B04 + A11*B14
    
    add_term(0, 1, 4, 1, 8, 1, col)
    col += 1
    add_term(1, 1, 9, 1, 8, 1, col)
    col += 1
    add_term(5, 1, 4, 1, 9, 1, col)
    col += 1
    add_term(6, 1, 9, 1, 9, 1, col)
    col += 1
    
    # Block 4: A_right * B21 (2x3 x 3x2) - rank 11
    # A_right: A[0,2]=A02, A[0,3]=A03, A[0,4]=A04, A[1,2]=A12, A[1,3]=A13, A[1,4]=A14
    # U indices: A02: U[2], A03: U[3], A04: U[4], A12: U[7], A13: U[8], A14: U[9]
    # B21: B[2,0]=B20, B[2,1]=B21, B[3,0]=B30, B[3,1]=B31, B[4,0]=B40, B[4,1]=B41
    # V indices: B20: V[10], B21: V[11], B30: V[15], B31: V[16], B40: V[20], B41: V[21]
    # C result: C[0,0], C[0,1], C[1,0], C[1,1]
    # W indices: C[0,0]: W[0], C[0,1]: W[1], C[1,0]: W[2], C[1,1]: W[3]
    #
    # Split: A_r1 = [A02, A03; A12, A13] (2x2), A_r2 = [A04; A14] (2x1)
    #        B21a = [B20, B21; B30, B31] (2x2), B21b = [B40, B41] (1x2)
    #
    # A_r1 * B21a (2x2 x 2x2) - Strassen, rank 7
    # A_r1: A02=A00, A03=A01, A12=A10, A13=A11 (renaming for Strassen)
    # B21a: B20=B00, B21=B01, B30=B10, B31=B11
    
    # M1 = (A02+A13)(B20+B31)
    add_term(2, 1, 10, 1, 0, 1, col); add_term(8, 1, 16, 1, 0, 1, col)
    add_term(2, 1, 10, 1, 2, 1, col); add_term(8, 1, 16, 1, 2, 1, col)
    add_term(2, 1, 10, 1, 1, -1, col); add_term(8, 1, 16, 1, 1, -1, col)
    add_term(2, 1, 10, 1, 3, 1, col); add_term(8, 1, 16, 1, 3, 1, col)
    col += 1
    
    # M2 = (A12+A13)B20
    add_term(7, 1, 10, 1, 2, 1, col); add_term(8, 1, 10, 1, 2, 1, col)
    add_term(7, 1, 10, 1, 3, -1, col); add_term(8, 1, 10, 1, 3, -1, col)
    col += 1
    
    # M3 = A02(B21-B31)
    add_term(2, 1, 11, 1, 1, 1, col)
    add_term(2, 1, 16, -1, 1, 1, col)
    add_term(2, 1, 11, 1, 3, 1, col)
    add_term(2, 1, 16, -1, 3, 1, col)
    col += 1
    
    # M4 = A13(B30-B20)
    add_term(8, 1, 15, 1, 0, 1, col)
    add_term(8, 1, 10, -1, 0, 1, col)
    add_term(8, 1, 15, 1, 2, 1, col)
    add_term(8, 1, 10, -1, 2, 1, col)
    col += 1
    
    # M5 = (A02+A03)B31
    add_term(2, 1, 16, 1, 1, 1, col); add_term(3, 1, 16, 1, 1, 1, col)
    add_term(2, 1, 16, 1, 3, -1, col); add_term(3, 1, 16, 1, 3, -1, col)
    add_term(2, 1, 16, 1, 0, -1, col); add_term(3, 1, 16, 1, 0, -1, col)
    col += 1
    
    # M6 = (A12-A02)(B20+B21)
    add_term(7, 1, 10, 1, 3, 1, col); add_term(2, -1, 10, 1, 3, 1, col)
    add_term(7, 1, 11, 1, 3, 1, col); add_term(2, -1, 11, 1, 3, 1, col)
    col += 1
    
    # M7 = (A03-A13)(B30+B31)
    add_term(3, 1, 15, 1, 0, 1, col); add_term(8, -1, 15, 1, 0, 1, col)
    add_term(3, 1, 16, 1, 0, 1, col); add_term(8, -1, 16, 1, 0, 1, col)
    col += 1
    
    # A_r2 * B21b (2x1 x 1x2) - outer product, rank 4
    # A_r2: A04, A14. B21b: B40, B41
    # C[0,0] += A04*B40, C[0,1] += A04*B41
    # C[1,0] += A14*B40, C[1,1] += A14*B41
    
    add_term(4, 1, 20, 1, 0, 1, col)
    col += 1
    add_term(4, 1, 21, 1, 1, 1, col)
    col += 1
    add_term(9, 1, 20, 1, 2, 1, col)
    col += 1
    add_term(9, 1, 21, 1, 3, 1, col)
    col += 1
    
    # Block 5: A_right * B22 (2x3 x 3x2) - rank 11
    # B22: B[2,2]=B22, B[2,3]=B23, B[3,2]=B32, B[3,3]=B33, B[4,2]=B42, B[4,3]=B43
    # V indices: B22: V[12], B23: V[13], B32: V[17], B33: V[18], B42: V[22], B43: V[23]
    # C result: C[0,2], C[0,3], C[1,2], C[1,3]
    # W indices: C[0,2]: W[4], C[0,3]: W[5], C[1,2]: W[6], C[1,3]: W[7]
    #
    # Split: A_r1 = [A02, A03; A12, A13], A_r2 = [A04; A14]
    #        B22a = [B22, B23; B32, B33] (2x2), B22b = [B42, B43] (1x2)
    
    # M1 = (A02+A13)(B22+B33)
    add_term(2, 1, 12, 1, 4, 1, col); add_term(8, 1, 18, 1, 4, 1, col)
    add_term(2, 1, 12, 1, 6, 1, col); add_term(8, 1, 18, 1, 6, 1, col)
    add_term(2, 1, 12, 1, 5, -1, col); add_term(8, 1, 18, 1, 5, -1, col)
    add_term(2, 1, 12, 1, 7, 1, col); add_term(8, 1, 18, 1, 7, 1, col)
    col += 1
    
    # M2 = (A12+A13)B22
    add_term(7, 1, 12, 1, 6, 1, col); add_term(8, 1, 12, 1, 6, 1, col)
    add_term(7, 1, 12, 1, 7, -1, col); add_term(8, 1, 12, 1, 7, -1, col)
    col += 1
    
    # M3 = A02(B23-B33)
    add_term(2, 1, 13, 1, 5, 1, col)
    add_term(2, 1, 18, -1, 5, 1, col)
    add_term(2, 1, 13, 1, 7, 1, col)
    add_term(2, 1, 18, -1, 7, 1, col)
    col += 1
    
    # M4 = A13(B32-B22)
    add_term(8, 1, 17, 1, 4, 1, col)
    add_term(8, 1, 12, -1, 4, 1, col)
    add_term(8, 1, 17, 1, 6, 1, col)
    add_term(8, 1, 12, -1, 6, 1, col)
    col += 1
    
    # M5 = (A02+A03)B33
    add_term(2, 1, 18, 1, 5, 1, col); add_term(3, 1, 18, 1, 5, 1, col)
    add_term(2, 1, 18, 1, 7, -1, col); add_term(3, 1, 18, 1, 7, -1, col)
    add_term(2, 1, 18, 1, 4, -1, col); add_term(3, 1, 18, 1, 4, -1, col)
    col += 1
    
    # M6 = (A12-A02)(B22+B23)
    add_term(7, 1, 12, 1, 7, 1, col); add_term(2, -1, 12, 1, 7, 1, col)
    add_term(7, 1, 13, 1, 7, 1, col); add_term(2, -1, 13, 1, 7, 1, col)
    col += 1
    
    # M7 = (A03-A13)(B32+B33)
    add_term(3, 1, 17, 1, 4, 1, col); add_term(8, -1, 17, 1, 4, 1, col)
    add_term(3, 1, 18, 1, 4, 1, col); add_term(8, -1, 18, 1, 4, 1, col)
    col += 1
    
    # A_r2 * B22b (2x1 x 1x2) - rank 4
    add_term(4, 1, 22, 1, 4, 1, col)
    col += 1
    add_term(4, 1, 23, 1, 5, 1, col)
    col += 1
    add_term(9, 1, 22, 1, 6, 1, col)
    col += 1
    add_term(9, 1, 23, 1, 7, 1, col)
    col += 1
    
    # Block 6: A_right * B23 (2x3 x 3x1) - rank 6
    # B23: B[2,4]=B24, B[3,4]=B34, B[4,4]=B44
    # V indices: B24: V[14], B34: V[19], B44: V[24]
    # C result: C[0,4], C[1,4]
    # W indices: C[0,4]: W[8], C[1,4]: W[9]
    #
    # Split: A_r1 = [A02, A03; A12, A13], A_r2 = [A04; A14]
    #        B23a = [B24; B34] (2x1), B23b = [B44] (1x1)
    
    # A_r1 * B23a (2x2 x 2x1) - rank 4
    # C[0,4] += A02*B24 + A03*B34
    # C[1,4] += A12*B24 + A13*B34
    add_term(2, 1, 14, 1, 8, 1, col)
    col += 1
    add_term(3, 1, 19, 1, 8, 1, col)
    col += 1
    add_term(7, 1, 14, 1, 9, 1, col)
    col += 1
    add_term(8, 1, 19, 1, 9, 1, col)
    col += 1
    
    # A_r2 * B23b (2x1 x 1x1) - rank 2
    # C[0,4] += A04*B44
    # C[1,4] += A14*B44
    add_term(4, 1, 24, 1, 8, 1, col)
    col += 1
    add_term(9, 1, 24, 1, 9, 1, col)
    col += 1
    
    # Convert to integers where possible
    for mat in [U, V, W]:
        for row in mat:
            for j in range(len(row)):
                if abs(row[j] - round(row[j])) < 1e-9:
                    row[j] = round(row[j])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
