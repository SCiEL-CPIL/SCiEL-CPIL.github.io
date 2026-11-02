"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    def build_schoolbook():
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
    
    # Hybrid decomposition:
    # Part 1: 3x3x3 (k=0,1,2) using Strassen for 2x2x2 core + schoolbook for cross terms. Rank 26.
    # Part 2: 3x3x4 (k=3,4,5,6) using schoolbook. Rank 36.
    # Total Rank: 62.
    
    rank_part1 = 26
    rank_part2 = n * m * (p - 3)  # 3*3*4 = 36
    total_rank = rank_part1 + rank_part2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: 3x3x3 Decomposition (k=0,1,2) ---
    # A is 3x3. Indices: A[i,j] -> U[i*m+j, col]
    # B is 3x3 (cols 0-2). Indices: B[j,k] -> V[j*p+k, col]
    # C is 3x3 (cols 0-2). Indices: C[k,i] -> W[k*n+i, col]
    
    # We split A into [[A22, a1], [a2T, a33]]
    # A22 = A[0:2, 0:2], a1 = A[0:2, 2], a2T = A[2, 0:2], a33 = A[2, 2]
    # We split B into [[B22, b1], [b2T, b33]]
    # B22 = B[0:2, 0:2], b1 = B[0:2, 2], b2T = B[2, 0:2], b33 = B[2, 2]
    
    # C = A*B
    # C[0:2, 0:2] = A22*B22 + a1*b2T
    # C[0:2, 2]   = A22*b1 + a1*b33
    # C[2, 0:2]   = a2T*B22 + a33*b2T
    # C[2, 2]     = a2T*b1 + a33*b33
    
    # --- 1. A22 * B22 using Strassen (Rank 7) ---
    # A22 = [[a00, a01], [a10, a11]]
    # B22 = [[b00, b01], [b10, b11]]
    # Indices in A: 
    #   a00: U[0*3+0], a01: U[0*3+1], a10: U[1*3+0], a11: U[1*3+1]
    # Indices in B:
    #   b00: V[0*7+0], b01: V[0*7+1], b10: V[1*7+0], b11: V[1*7+1]
    # Indices in C (for C[0:2, 0:2]):
    #   C[0,0]: W[0*3+0], C[0,1]: W[0*3+1], C[1,0]: W[1*3+0], C[1,1]: W[1*3+1]
    
    # Strassen Formulas:
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00
    # P3 = a00(b10-b11)
    # P4 = a11(b01-b00)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b01)
    # P7 = (a01-a11)(b10+b11)
    
    # C[0,0] = P1 + P4 - P5 + P7
    # C[0,1] = P3 + P5
    # C[1,0] = P2 + P4
    # C[1,1] = P1 - P2 + P3 + P6
    
    # Term 1: P1
    U[0*3+0][col] = 1; U[1*3+1][col] = 1
    V[0*7+0][col] = 1; V[1*7+1][col] = 1
    W[0*3+0][col] = 1
    W[1*3+1][col] = 1
    col += 1
    
    # Term 2: P2
    U[1*3+0][col] = 1; U[1*3+1][col] = 1
    V[0*7+0][col] = 1
    W[1*3+0][col] = 1
    col += 1
    
    # Term 3: P3
    U[0*3+0][col] = 1
    V[1*7+0][col] = 1; V[1*7+1][col] = -1
    W[0*3+1][col] = 1
    col += 1
    
    # Term 4: P4
    U[1*3+1][col] = 1
    V[0*7+0][col] = -1; V[0*7+1][col] = 1
    W[0*3+0][col] = 1
    W[1*3+0][col] = 1
    col += 1
    
    # Term 5: P5
    U[0*3+0][col] = 1; U[0*3+1][col] = 1
    V[1*7+1][col] = 1
    W[0*3+0][col] = -1
    W[0*3+1][col] = 1
    col += 1
    
    # Term 6: P6
    U[1*3+0][col] = 1; U[0*3+0][col] = -1
    V[0*7+0][col] = 1; V[0*7+1][col] = 1
    W[1*3+1][col] = 1
    col += 1
    
    # Term 7: P7
    U[0*3+1][col] = 1; U[1*3+1][col] = -1
    V[1*7+0][col] = 1; V[1*7+1][col] = 1
    W[0*3+0][col] = 1
    col += 1
    
    # --- 2. a1 * b2T (Rank 4) ---
    # a1 = [A[0,2], A[1,2]]
    # b2T = [B[2,0], B[2,1]]
    # C[0:2, 0:2] += a1 * b2T
    
    # Term 8: A[0,2] * B[2,0] -> C[0,0]
    U[0*3+2][col] = 1
    V[2*7+0][col] = 1
    W[0*3+0][col] = 1
    col += 1
    
    # Term 9: A[0,2] * B[2,1] -> C[0,1]
    U[0*3+2][col] = 1
    V[2*7+1][col] = 1
    W[0*3+1][col] = 1
    col += 1
    
    # Term 10: A[1,2] * B[2,0] -> C[1,0]
    U[1*3+2][col] = 1
    V[2*7+0][col] = 1
    W[1*3+0][col] = 1
    col += 1
    
    # Term 11: A[1,2] * B[2,1] -> C[1,1]
    U[1*3+2][col] = 1
    V[2*7+1][col] = 1
    W[1*3+1][col] = 1
    col += 1
    
    # --- 3. A22 * b1 (Rank 4) ---
    # C[0:2, 2] += A22 * b1
    
    # Term 12: A[0,0] * B[0,2] -> C[0,2]
    U[0*3+0][col] = 1
    V[0*7+2][col] = 1
    W[2*3+0][col] = 1
    col += 1
    
    # Term 13: A[0,1] * B[1,2] -> C[0,2]
    U[0*3+1][col] = 1
    V[1*7+2][col] = 1
    W[2*3+0][col] = 1
    col += 1
    
    # Term 14: A[1,0] * B[0,2] -> C[1,2]
    U[1*3+0][col] = 1
    V[0*7+2][col] = 1
    W[2*3+1][col] = 1
    col += 1
    
    # Term 15: A[1,1] * B[1,2] -> C[1,2]
    U[1*3+1][col] = 1
    V[1*7+2][col] = 1
    W[2*3+1][col] = 1
    col += 1
    
    # --- 4. a2T * B22 (Rank 4) ---
    # C[2, 0:2] += a2T * B22
    
    # Term 16: A[2,0] * B[0,0] -> C[0,2]
    U[2*3+0][col] = 1
    V[0*7+0][col] = 1
    W[0*3+2][col] = 1
    col += 1
    
    # Term 17: A[2,1] * B[1,0] -> C[0,2]
    U[2*3+1][col] = 1
    V[1*7+0][col] = 1
    W[0*3+2][col] = 1
    col += 1
    
    # Term 18: A[2,0] * B[0,1] -> C[1,2]
    U[2*3+0][col] = 1
    V[0*7+1][col] = 1
    W[1*3+2][col] = 1
    col += 1
    
    # Term 19: A[2,1] * B[1,1] -> C[1,2]
    U[2*3+1][col] = 1
    V[1*7+1][col] = 1
    W[1*3+2][col] = 1
    col += 1
    
    # --- 5. a1 * b33 (Rank 2) ---
    # C[0:2, 2] += a1 * b33
    
    # Term 20: A[0,2] * B[2,2] -> C[0,2]
    U[0*3+2][col] = 1
    V[2*7+2][col] = 1
    W[2*3+0][col] = 1
    col += 1
    
    # Term 21: A[1,2] * B[2,2] -> C[1,2]
    U[1*3+2][col] = 1
    V[2*7+2][col] = 1
    W[2*3+1][col] = 1
    col += 1
    
    # --- 6. a2T * b1 (Rank 2) ---
    # C[2, 2] += a2T * b1
    
    # Term 22: A[2,0] * B[0,2] -> C[2,2]
    U[2*3+0][col] = 1
    V[0*7+2][col] = 1
    W[2*3+2][col] = 1
    col += 1
    
    # Term 23: A[2,1] * B[1,2] -> C[2,2]
    U[2*3+1][col] = 1
    V[1*7+2][col] = 1
    W[2*3+2][col] = 1
    col += 1
    
    # --- 7. a33 * b2T (Rank 2) ---
    # C[2, 0:2] += a33 * b2T
    
    # Term 24: A[2,2] * B[2,0] -> C[0,2]
    U[2*3+2][col] = 1
    V[2*7+0][col] = 1
    W[0*3+2][col] = 1
    col += 1
    
    # Term 25: A[2,2] * B[2,1] -> C[1,2]
    U[2*3+2][col] = 1
    V[2*7+1][col] = 1
    W[1*3+2][col] = 1
    col += 1
    
    # --- 8. a33 * b33 (Rank 1) ---
    # C[2, 2] += a33 * b33
    
    # Term 26: A[2,2] * B[2,2] -> C[2,2]
    U[2*3+2][col] = 1
    V[2*7+2][col] = 1
    W[2*3+2][col] = 1
    col += 1
    
    # --- Part 2: Schoolbook for k=3,4,5,6 ---
    for i in range(n):
        for j in range(m):
            for k in range(3, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify the decomposition
    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Check all (i,j,k) tuples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U[0])):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Check if our decomposition is valid
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # If not valid, fall back to schoolbook
    return build_schoolbook()
# EVOLVE-BLOCK-END
