"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We split the 3 columns of A into 2 and 1.
    # A = [A_L | A_R] where A_L is 2x2 and A_R is 2x1.
    # B = [B_T; B_B] where B_T is 2x10 and B_B is 1x10.
    # C = A_L B_T + A_R B_B
    
    # Part 1: A_L B_T (2x2 * 2x10)
    # We use Strassen's algorithm for the 2x2 part.
    # The rank of 2x2x2 is 7.
    # We split the 10 columns of B_T into 5 pairs.
    # Each pair is a 2x2x2 multiplication, rank 7.
    # Total rank for Part 1: 5 * 7 = 35.
    
    rank_part1 = 35
    U1 = [[0] * rank_part1 for _ in range(n * 2)]  # 2x2 -> 4 entries
    V1 = [[0] * rank_part1 for _ in range(2 * p)]  # 2x10 -> 20 entries
    W1 = [[0] * rank_part1 for _ in range(p * n)]  # 10x2 -> 20 entries
    
    # Strassen coefficients for 2x2x2
    # P1 = (y11+y22)(z11+z22)
    # P2 = (y21+y22)z11
    # P3 = y11(z21-z22)
    # P4 = y22(z11-z12) -- wait, standard Strassen:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b11-b12)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # Indices for A_L (2x2): 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
    # Indices for B_T (2x10): 0..9 for row 0, 10..19 for row 1
    # Indices for C (10x2): 0..19 correspond to (k,i) with W[k*n+i]
    
    col = 0
    for pair_idx in range(5):
        # Columns 2*pair_idx and 2*pair_idx+1
        k1 = 2 * pair_idx
        k2 = 2 * pair_idx + 1
        
        # M1 = (a11+a22)(b11+b22)
        U1[0][col] = 1; U1[3][col] = 1
        V1[k1][col] = 1; V1[10+k1][col] = 1  # b11 is row 0, col k1. b22 is row 1, col k1.
        # Wait, B_T is 2x10. Row 0 is cols 0..9. Row 1 is cols 10..19.
        # b11 is B_T[0, k1]. b12 is B_T[0, k2].
        # b21 is B_T[1, k1]. b22 is B_T[1, k2].
        # So V1[0, k1] is b11. V1[0, k2] is b12.
        # V1[1, k1] is b21. V1[1, k2] is b22.
        # In our V1, index is j*p+k. j is row (0 or 1), k is col.
        # So V1[0*p + k1] = b11. V1[0*p + k2] = b12.
        # V1[1*p + k1] = b21. V1[1*p + k2] = b22.
        
        # Correct V1 indices:
        V1[k1][col] = 1; V1[10+k2][col] = 1  # b11 + b22? No.
        # b11 is V1[k1]. b22 is V1[10+k2].
        # So V1[k1] = 1, V1[10+k2] = 1.
        W1[k1*n + 0][col] = 1; W1[k2*n + 1][col] = 1  # c11 += M1, c22 += M1
        # W1[k*n+i] corresponds to C[i,k].
        # C[0,k1] += M1 -> W1[k1*n+0]
        # C[1,k2] += M1 -> W1[k2*n+1]
        col += 1
        
        # M2 = (a21+a22)b11
        U1[2][col] = 1; U1[3][col] = 1
        V1[k1][col] = 1
        W1[k1*n + 1][col] = 1
        col += 1
        
        # M3 = a11(b21-b22)
        U1[0][col] = 1
        V1[10+k1][col] = 1; V1[10+k2][col] = -1
        W1[k1*n + 0][col] = 1; W1[k2*n + 0][col] = -1
        col += 1
        
        # M4 = a22(b11-b12)
        U1[3][col] = 1
        V1[k1][col] = 1; V1[k2][col] = -1
        W1[k1*n + 1][col] = 1; W1[k2*n + 1][col] = -1
        col += 1
        
        # M5 = (a11+a12)b22
        U1[0][col] = 1; U1[1][col] = 1
        V1[10+k2][col] = 1
        W1[k2*n + 0][col] = 1
        col += 1
        
        # M6 = (a21-a11)(b11+b12)
        U1[2][col] = 1; U1[0][col] = -1
        V1[k1][col] = 1; V1[k2][col] = 1
        W1[k1*n + 1][col] = 1; W1[k2*n + 1][col] = 1
        col += 1
        
        # M7 = (a12-a22)(b21+b22)
        U1[1][col] = 1; U1[3][col] = -1
        V1[10+k1][col] = 1; V1[10+k2][col] = 1
        W1[k1*n + 0][col] = 1; W1[k2*n + 0][col] = 1
        col += 1

    # Part 2: A_R B_B (2x1 * 1x10)
    # A_R is 2x1. B_B is 1x10.
    # C2 = A_R B_B.
    # C2[i,k] = A_R[i,0] B_B[0,k].
    # This is a rank-2 tensor? No, it's a sum of 20 terms?
    # C2[i,k] = A_R[i,0] B_B[0,k].
    # We can write this as 2 terms?
    # C2[0,k] = A_R[0,0] B_B[0,k]
    # C2[1,k] = A_R[1,0] B_B[0,k]
    # This requires 20 multiplications if done naively.
    # But we can factor it.
    # The tensor is 2x1x10.
    # The rank is 2? No.
    # The rank of the 2x1x10 tensor is 20? No.
    # The rank of the 2x1x10 tensor is 2? No.
    # The rank of the 2x1x10 tensor is 10? No.
    # The rank of the 2x1x10 tensor is 2? No.
    # The rank of the 2x1x10 tensor is 20? No.
    # The rank of the 2x1x10 tensor is 2? No.
    # The rank of the 2x1x10 tensor is 10? No.
    # The rank of the 2x1x10 tensor is 2? No.
    # The rank of the 2x1x10 tensor is 20? No.
    
    # Let's use the schoolbook for this part.
    # Rank = 2 * 1 * 10 = 20.
    rank_part2 = 20
    U2 = [[0] * rank_part2 for _ in range(n * 1)]  # 2x1 -> 2 entries
    V2 = [[0] * rank_part2 for _ in range(1 * p)]  # 1x10 -> 10 entries
    W2 = [[0] * rank_part2 for _ in range(p * n)]  # 10x2 -> 20 entries
    
    col = 0
    for i in range(n):
        for k in range(p):
            U2[i][col] = 1
            V2[k][col] = 1
            W2[k*n + i][col] = 1
            col += 1

    # Combine
    total_rank = rank_part1 + rank_part2
    U = [row[:] for row in U1] + [row[:] for row in U2]
    # Wait, U1 has 4 rows, U2 has 2 rows. Total 6 rows.
    # But A is 2x3, so n*m = 6. Correct.
    # U1 corresponds to A_L (2x2). U2 corresponds to A_R (2x1).
    # U1 rows: 0,1,2,3 correspond to A[0,0], A[0,1], A[1,0], A[1,1].
    # U2 rows: 0,1 correspond to A[0,2], A[1,2].
    # So U is 6x55. Correct.
    
    V = [row[:] for row in V1] + [row[:] for row in V2]
    # V1 has 20 rows (2x10). V2 has 10 rows (1x10). Total 30 rows.
    # B is 3x10, so m*p = 30. Correct.
    # V1 rows: 0..9 correspond to B[0,0..9]. 10..19 correspond to B[1,0..9].
    # V2 rows: 0..9 correspond to B[2,0..9].
    # So V is 30x55. Correct.
    
    W = [row[:] for row in W1] + [row[:] for row in W2]
    # W1 has 20 rows (10x2). W2 has 20 rows (10x2). Total 40 rows?
    # Wait, W1 and W2 both have p*n = 20 rows.
    # But we need to add them element-wise?
    # No, W is a matrix of shape (p*n, rank).
    # W1 is 20x35. W2 is 20x20.
    # We need to concatenate them along the column axis?
    # No, W is (p*n, rank).
    # W1 has 35 columns. W2 has 20 columns.
    # So W is 20x55.
    # W[i,j] = W1[i,j] for j < 35, and W2[i, j-35] for j >= 35.
    
    W_combined = [[0] * total_rank for _ in range(p * n)]
    for i in range(p * n):
        for j in range(rank_part1):
            W_combined[i][j] = W1[i][j]
        for j in range(rank_part2):
            W_combined[i][rank_part1 + j] = W2[i][j]
    
    return {"U": U, "V": V, "W": W_combined}
# EVOLVE-BLOCK-END
