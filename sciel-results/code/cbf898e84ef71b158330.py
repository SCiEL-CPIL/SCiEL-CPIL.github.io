"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    
    # Deadline guard: return schoolbook if we run out of time
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin
    
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
    
    # Try to find a lower-rank decomposition using optimization
    # Strategy: use the 23-rank 3x3x3 decomposition for the first 3 columns
    # and schoolbook for the remaining 4 columns
    # Total: 23 + 36 = 59
    
    # The 23-rank decomposition for 3x3x3 matrix multiplication
    # This is a known result (Bini et al., 1979)
    # We'll encode it as a list of (u, v, w) triples where u, v, w are 9-vectors
    
    # For 3x3x3, the 23-rank decomposition coefficients:
    # Each term has u (9 entries for A), v (9 entries for B), w (9 entries for C)
    # Indices: A[i,j] -> i*3+j, B[j,k] -> j*3+k, C[k,i] -> k*3+i
    
    # Known 23-rank decomposition (one standard form):
    # We use the following 23 terms (u, v, w) where each is a 9-vector
    # This is based on the standard 23-rank algorithm
    
    # Term 1: (1,0,0, 0,1,0, 0,0,1) for A; (0,1,0, 1,0,0, 0,0,1) for B; (1,0,0, 0,1,0, 0,0,1) for C
    # Actually, let me use a verified set of 23 coefficients
    
    # For safety, I'll use a simpler approach: Strassen-based block decomposition
    # which gives rank 26 for 3x3x3, then 36 for 3x3x4, total 62
    
    # Strassen for 2x2x2 block within 3x3x3:
    # A = [[A22, a1], [a2T, a33]] where A22 is 2x2, a1 is 2x1, a2T is 1x2, a33 is 1x1
    # B = [[B22, b1], [b2T, b33]] where B22 is 2x2, b1 is 2x1, b2T is 1x2, b33 is 1x1
    
    # C11 = A22*B22 + a1*b2T  (2x2 = 2x2*2x2 + 2x1*1x2)
    # C12 = A22*b1 + a1*b33   (2x1 = 2x2*2x1 + 2x1*1x1)
    # C21 = a2T*B22 + a33*b2T (1x2 = 1x2*2x2 + 1x1*1x2)
    # C22 = a2T*b1 + a33*b33  (1x1 = 1x2*2x1 + 1x1*1x1)
    
    # Strassen 2x2x2 (rank 7):
    # P1 = (a11+a22)(b11+b22)
    # P2 = (a21+a22)b11
    # P3 = a11(b21-b22)
    # P4 = a22(b12-b11)
    # P5 = (a11+a12)b22
    # P6 = (a21-a11)(b11+b12)
    # P7 = (a12-a22)(b21+b22)
    # C11 = P1+P4-P5+P7
    # C12 = P3+P5
    # C21 = P2+P4
    # C22 = P1-P2+P3+P6
    
    # For the 2x2x2 part:
    # U entries for A22: (1,1,0,0), (0,0,1,1), (1,0,0,0), (0,1,0,0), (1,1,0,0), (0,0,1,-1), (0,0,0,0)
    # Wait, let me be more careful.
    
    # A22 = [[a00, a01], [a10, a11]]
    # B22 = [[b00, b01], [b10, b11]]
    
    # P1: U_A = [1,1,0,0], V_B = [1,1,1,1] (flattened: b00+b10 for first row, b01+b11 for second? No)
    # Actually for 2x2: A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # P1 = (a00+a11)(b00+b11)
    #   U_A = [1,0,0,1], V_B = [1,0,0,1], W_C = [1,0,0,1]
    # P2 = (a10+a11)b00
    #   U_A = [0,0,1,1], V_B = [1,0,0,0], W_C = [0,0,1,0]
    # P3 = a00(b10-b11)
    #   U_A = [1,0,0,0], V_B = [0,0,1,-1], W_C = [0,1,0,0]
    # P4 = a11(b01-b00)
    #   U_A = [0,0,0,1], V_B = [-1,1,0,0], W_C = [0,0,0,1]
    # P5 = (a00+a01)b11
    #   U_A = [1,1,0,0], V_B = [0,0,0,1], W_C = [0,0,0,1]
    # P6 = (a10-a00)(b00+b01)
    #   U_A = [-1,0,1,0], V_B = [1,1,0,0], W_C = [0,0,1,1]
    # P7 = (a01-a11)(b10+b11)
    #   U_A = [0,1,0,-1], V_B = [0,0,1,1], W_C = [1,0,0,0]
    
    # C11 = P1+P4-P5+P7
    # C12 = P3+P5
    # C21 = P2+P4
    # C22 = P1-P2+P3+P6
    
    # Now for the full 3x3x3 decomposition with Strassen on the top-left 2x2:
    # A = [[A22, a1], [a2T, a33]]
    # B = [[B22, b1], [b2T, b33]]
    
    # The 7 Strassen terms give us C11 (2x2)
    # Then we need:
    # a1*b2T: 2x1 * 1x2 = 2x2, rank 4
    # A22*b1: 2x2 * 2x1 = 2x1, rank 4
    # a2T*B22: 1x2 * 2x2 = 1x2, rank 4
    # a1*b33: 2x1 * 1x1 = 2x1, rank 2
    # a2T*b1: 1x2 * 2x1 = 1x1, rank 2
    # a33*b2T: 1x1 * 1x2 = 1x2, rank 2
    # a33*b33: 1x1 * 1x1 = 1x1, rank 1
    
    # Total: 7 + 4 + 4 + 4 + 2 + 2 + 2 + 1 = 26
    
    # Let's build this 26-rank decomposition for the first 3 columns of B
    # and schoolbook for the last 4 columns
    
    rank_part1 = 26
    rank_part2 = n * m * (p - 3)  # 3*3*4 = 36
    total_rank = rank_part1 + rank_part2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 3x3x3 with rank 26 (Strassen-based)
    # Indices for A: i*m+j where i,j in 0..2
    # Indices for B: j*p+k where j in 0..2, k in 0..2 (for this part)
    # Indices for C: k*n+i where k in 0..2, i in 0..2
    
    # Strassen 2x2x2 terms (7 terms)
    # A22 is A[0:2, 0:2], B22 is B[0:2, 0:3] (first 3 columns)
    # Wait, B is 3x7, so B22 should be B[0:2, 0:3]? No, B22 is 2x2.
    
    # Let me reconsider. For the 3x3x3 subproblem:
    # A is 3x3, B is 3x3 (first 3 columns of the full B), C is 3x3
    # A22 = A[0:2, 0:2], a1 = A[0:2, 2], a2T = A[2, 0:2], a33 = A[2, 2]
    # B22 = B[0:2, 0:2], b1 = B[0:2, 2], b2T = B[2, 0:2], b33 = B[2, 2]
    
    # Strassen terms for A22*B22:
    # P1: (a00+a11)(b00+b11)
    U[0*3+0][col] = 1; U[1*3+1][col] = 1
    V[0*7+0][col] = 1; V[1*7+1][col] = 1
    W[0*3+0][col] = 1; W[1*3+1][col] = 1
    col += 1
    
    # P2: (a10+a11)b00
    U[1*3+0][col] = 1; U[1*3+1][col] = 1
    V[0*7+0][col] = 1
    W[0*3+1][col] = 1
    col += 1
    
    # P3: a00(b10-b11)
    U[0*3+0][col] = 1
    V[1*7+0][col] = 1; V[1*7+1][col] = -1
    W[1*3+0][col] = 1
    col += 1
    
    # P4: a11(b01-b00)
    U[1*3+1][col] = 1
    V[0*7+0][col] = -1; V[0*7+1][col] = 1
    W[1*3+1][col] = 1
    col += 1
    
    # P5: (a00+a01)b11
    U[0*3+0][col] = 1; U[0*3+1][col] = 1
    V[1*7+1][col] = 1
    W[1*3+1][col] = 1
    col += 1
    
    # P6: (a10-a00)(b00+b01)
    U[0*3+0][col] = -1; U[1*3+0][col] = 1
    V[0*7+0][col] = 1; V[0*7+1][col] = 1
    W[1*3+0][col] = 1; W[1*3+1][col] = 1
    col += 1
    
    # P7: (a01-a11)(b10+b11)
    U[0*3+1][col] = 1; U[1*3+1][col] = -1
    V[1*7+0][col] = 1; V[1*7+1][col] = 1
    W[0*3+0][col] = 1
    col += 1
    
    # Now a1*b2T: 2x1 * 1x2 = 2x2, rank 4
    # a1 = A[0:2, 2], b2T = B[2, 0:2]
    # C11 += a1*b2T
    # Term 8: A[0,2]*B[2,0]
    U[0*3+2][col] = 1
    V[2*7+0][col] = 1
    W[0*3+0][col] = 1
    col += 1
    
    # Term 9: A[0,2]*B[2,1]
    U[0*3+2][col] = 1
    V[2*7+1][col] = 1
    W[1*3+0][col] = 1
    col += 1
    
    # Term 10: A[1,2]*B[2,0]
    U[1*3+2][col] = 1
    V[2*7+0][col] = 1
    W[0*3+1][col] = 1
    col += 1
    
    # Term 11: A[1,2]*B[2,1]
    U[1*3+2][col] = 1
    V[2*7+1][col] = 1
    W[1*3+1][col] = 1
    col += 1
    
    # A22*b1: 2x2 * 2x1 = 2x1, rank 4
    # C12 = A22*b1 + a1*b33
    # Term 12: A[0,0]*B[0,2]
    U[0*3+0][col] = 1
    V[0*7+2][col] = 1
    W[2*3+0][col] = 1
    col += 1
    
    # Term 13: A[0,1]*B[1,2]
    U[0*3+1][col] = 1
    V[1*7+2][col] = 1
    W[2*3+0][col] = 1
    col += 1
    
    # Term 14: A[1,0]*B[0,2]
    U[1*3+0][col] = 1
    V[0*7+2][col] = 1
    W[2*3+1][col] = 1
    col += 1
    
    # Term 15: A[1,1]*B[1,2]
    U[1*3+1][col] = 1
    V[1*7+2][col] = 1
    W[2*3+1][col] = 1
    col += 1
    
    # a2T*B22: 1x2 * 2x2 = 1x2, rank 4
    # C21 = a2T*B22 + a33*b2T
    # Term 16: A[2,0]*B[0,0]
    U[2*3+0][col] = 1
    V[0*7+0][col] = 1
    W[0*3+2][col] = 1
    col += 1
    
    # Term 17: A[2,1]*B[1,0]
    U[2*3+1][col] = 1
    V[1*7+0][col] = 1
    W[0*3+2][col] = 1
    col += 1
    
    # Term 18: A[2,0]*B[0,1]
    U[2*3+0][col] = 1
    V[0*7+1][col] = 1
    W[1*3+2][col] = 1
    col += 1
    
    # Term 19: A[2,1]*B[1,1]
    U[2*3+1][col] = 1
    V[1*7+1][col] = 1
    W[1*3+2][col] = 1
    col += 1
    
    # a1*b33: 2x1 * 1x1 = 2x1, rank 2
    # Term 20: A[0,2]*B[2,2]
    U[0*3+2][col] = 1
    V[2*7+2][col] = 1
    W[2*3+0][col] = 1
    col += 1
    
    # Term 21: A[1,2]*B[2,2]
    U[1*3+2][col] = 1
    V[2*7+2][col] = 1
    W[2*3+1][col] = 1
    col += 1
    
    # a2T*b1: 1x2 * 2x1 = 1x1, rank 2
    # Term 22: A[2,0]*B[0,2]
    U[2*3+0][col] = 1
    V[0*7+2][col] = 1
    W[2*3+2][col] = 1
    col += 1
    
    # Term 23: A[2,1]*B[1,2]
    U[2*3+1][col] = 1
    V[1*7+2][col] = 1
    W[2*3+2][col] = 1
    col += 1
    
    # a33*b2T: 1x1 * 1x2 = 1x2, rank 2
    # Term 24: A[2,2]*B[2,0]
    U[2*3+2][col] = 1
    V[2*7+0][col] = 1
    W[0*3+2][col] = 1
    col += 1
    
    # Term 25: A[2,2]*B[2,1]
    U[2*3+2][col] = 1
    V[2*7+1][col] = 1
    W[1*3+2][col] = 1
    col += 1
    
    # a33*b33: 1x1 * 1x1 = 1x1, rank 1
    # Term 26: A[2,2]*B[2,2]
    U[2*3+2][col] = 1
    V[2*7+2][col] = 1
    W[2*3+2][col] = 1
    col += 1
    
    # Part 2: Schoolbook for columns 3-6 of B
    for i in range(n):
        for j in range(m):
            for k in range(3, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Safety: ensure we haven't exceeded deadline
    if time.time() > deadline:
        return build_schoolbook()
    
    # Verify the decomposition
    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
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
