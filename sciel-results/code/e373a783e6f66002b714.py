"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Rank 4 decomposition
    rank = 4
    U = [[0] * rank for _ in range(n * m)]  # (8, 4)
    V = [[0] * rank for _ in range(m * p)]  # (16, 4)
    W = [[0] * rank for _ in range(p * n)]  # (8, 4)
    
    # For each a in {0, 1} (row of A) and c in {0, 1} (col block of B)
    # r = a * 2 + c
    for a in range(n):
        for c in range(2):
            r = a * 2 + c
            # U[i*m + j, r] = 1 for all j in {0, 1, 2, 3}
            # i = a
            for j in range(m):
                U[a * m + j][r] = 1
            
            # V[j*p + k, r] = 1 for all j in {0, 1, 2, 3} and k in {2c, 2c+1}
            for j in range(m):
                for k in [2*c, 2*c+1]:
                    V[j * p + k][r] = 1
            
            # W[k*n + i, r] = 1/4 for k in {2c, 2c+1} and i = a
            # Using 0.25 to get exact 1.0 sum (4 * 1 * 1 * 0.25 = 1)
            # 0.25 is a half-integer? No, 1/4 is not half-integer.
            # Task says "integer/half-integer entries".
            # 0.5 is half-integer. 0.25 is not.
            # So we need to adjust.
            # If we use 0.5, sum is 2.
            # We can scale U or V by 0.5.
            # Let U=0.5, V=1, W=0.5. Sum = 4 * 0.5 * 1 * 0.5 = 1.
            # 0.5 is a half-integer.
            
            for j in range(m):
                U[a * m + j][r] = 0.5
            for k in [2*c, 2*c+1]:
                W[k * n + a][r] = 0.5
            # V remains 1.
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(total - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
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
# EVOLVE-BLOCK-END
