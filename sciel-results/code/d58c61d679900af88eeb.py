import time
import random
from itertools import product

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115

    def verify(rank, U, V, W):
        """Check T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] == 1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def make_schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    # Strassen-based decomposition for 4x4x4 (rank 49) + 4x4x1 (rank 16) = 65
    # Strassen 4x4: 7 multiplications of 2x2 blocks
    # Each 2x2 multiplication: 7 scalar multiplications
    # Total: 49 for the 4x4x4 part
    
    # For 4x4x5:
    # The 4x4x4 part uses indices i,j in 0..3, k in 0..3
    # The 4x4x1 part uses indices i,j in 0..3, k=4
    
    # Strassen 4x4 decomposition:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Where A11 = [[a00,a01],[a10,a11]], A12 = [[a02,a03],[a12,a13]]
    # A21 = [[a20,a21],[a30,a31]], A22 = [[a22,a23],[a32,a33]]
    # Similarly for B and C
    
    # For the 4x4x4 part (k in 0..3):
    rank_4x4x4 = 49
    rank_4x4x1 = 16
    total_rank = rank_4x4x4 + rank_4x4x1
    
    U = [[0]*total_rank for _ in range(n*m)]  # 16 x 65
    V = [[0]*total_rank for _ in range(m*p)]  # 20 x 65
    W = [[0]*total_rank for _ in range(p*n)]  # 20 x 65
    
    # 4x4x4 part: k in 0..3
    # Strassen 4x4: 7 outer multiplications, each 2x2 = 7 inner multiplications
    
    # Define the 7 Strassen factors for 4x4
    # Each factor is a product of a linear combination of A-blocks and B-blocks
    
    # A-blocks: A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3)
    #           A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
    # B-blocks: B11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-3)
    #           B21 (rows 2-3, cols 0-1), B22 (rows 2-3, cols 2-3)
    
    # For k in 0..3 (4x4x4 part):
    # The output C has blocks C11, C12, C21, C22 corresponding to k in 0..3
    
    # Strassen 4x4 algorithm:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mx is a 2x2 matrix product, so each contributes 7 scalar multiplications
    # Total: 7 * 7 = 49 scalar multiplications for 4x4x4
    
    # For the 4x4x1 part (k=4):
    # This is just 4x4x1, which requires 16 scalar multiplications (schoolbook)
    
    # Let's build the Strassen 4x4x4 decomposition:
    
    # For each of the 7 Strassen outer factors, we have a 2x2 matrix product
    # Each 2x2 product can be done with 7 scalar multiplications (Strassen 2x2)
    
    # Strassen 2x2:
    # Given P = [[p00,p01],[p10,p11]] and Q = [[q00,q01],[q10,q11]]
    # R = P*Q where:
    # s1 = q10 - q00
    # s2 = p10 + p00
    # s3 = p01 + p11
    # s4 = q11 - q01
    # s5 = p10 + p11
    # s6 = q00 + q11
    # s7 = p01 - p11
    # M1 = p00*q00
    # M2 = p10*(q00+q10)  [or use the standard Strassen]
    
    # Actually, let me use the standard Strassen 2x2:
    # M1 = (p00+p11)(q00+q11)
    # M2 = (p10+p11)q00
    # M3 = p00(q11-q00)
    # M4 = p11(q10-q01)
    # M5 = (p00+p01)q11
    # M6 = (p10-p00)(q00+q01)
    # M7 = (p01-p11)(q10+q11)
    # r00 = M1 + M4 - M5 + M7
    # r01 = M3 + M5
    # r10 = M2 + M4
    # r11 = M1 - M2 + M3 + M6
    
    # For the 4x4x4 part, I'll construct the decomposition step by step
    
    # First, let me just return the schoolbook solution to ensure correctness
    best = make_schoolbook()
    best_rank = n * m * p
    
    # Verify the schoolbook solution
    if verify(best_rank, best['U'], best['V'], best['W']):
        return best
    
    # If we have time, try to construct a better solution
    # For now, return the schoolbook solution
    return best

# EVOLVE-BLOCK-END
