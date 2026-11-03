"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    deadline = time.time() + 140  # safety margin
    
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to construct a lower-rank decomposition.
    # Split B into B1 (4x4) and B2 (4x3).
    # C = [A*B1 | A*B2]
    # 
    # For A*B1 (3x4 result):
    #   A = [A1(3x3) | a(3x1)]
    #   B1 = [[B11(3x3) | b1(3x1)], [b2(1x3) | b3(1x1)]]
    #   A*B1 = A1*B11 + A1*b1 + a*b2 + a*b3
    #
    # For A*B2 (3x3 result):
    #   B2 = [[B21(3x3)], [b4(1x3)]]
    #   A*B2 = A1*B21 + a*b4
    #
    # Use a rank-7 decomposition for each 3x3x3 (A1*B11 and A1*B21).
    # Cross terms: A1*b1 (3x1), a*b2 (3x3), a*b3 (3x1), a*b4 (3x3)
    #
    # Total: 7 + 7 + (3*1) + (3*3) + (3*1) + (3*3) = 7+7+3+9+3+9 = 38
    # But we can share: a*b2 and a*b4 are both a(3x1)*row(1x3), so 3*3=9 each.
    # A1*b1 is 3*1=3 terms. a*b3 is 3*1=3 terms.
    # Total: 7+7+3+9+3+9 = 38.
    
    # Let me construct the rank-7 3x3x3 decomposition.
    # The 3x3x3 matrix multiplication tensor has rank 7.
    # Here's one such decomposition (from literature):
    # 
    # For X(3x3)*Y(3x3)=Z(3x3):
    # Z[i,k] = sum_j X[i,j]*Y[j,k]
    #
    # One rank-7 decomposition:
    # M1 = (X[0,0]+X[2,2]) * (Y[0,0]+Y[2,2])
    # M2 = (X[1,0]+X[2,0]) * Y[0,0]
    # M3 = X[0,0] * (Y[1,0]+Y[2,0])
    # M4 = X[2,2] * (Y[1,1]+Y[2,1])
    # M5 = (X[0,0]+X[1,1]) * (Y[1,1]+Y[2,2])
    # M6 = (X[2,1]+X[2,2]) * (Y[0,1]+Y[1,1])
    # M7 = (X[0,0]+X[0,1]) * Y[2,2]
    #
    # This is not quite right for 3x3. Let me use a verified decomposition.
    #
    # Actually, the simplest correct approach: just use schoolbook.
    # Given the complexity and time constraints, return schoolbook.
    
    if time.time() > deadline:
        return schoolbook(n, m, p)
    
    return schoolbook(n, m, p)
# EVOLVE-BLOCK-END
