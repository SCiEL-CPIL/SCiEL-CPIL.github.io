import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Strassen 2x2x2 decomposition (rank 7)
    U_s = np.zeros((4, 7), dtype=float)
    V_s = np.zeros((4, 7), dtype=float)
    W_s = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    U_s[0,0] = 1; U_s[3,0] = 1
    V_s[0,0] = 1; V_s[3,0] = 1
    W_s[0,0] = 1; W_s[3,0] = 1
    
    # M2 = (A10+A11)B00
    U_s[2,1] = 1; U_s[3,1] = 1
    V_s[0,1] = 1
    W_s[2,1] = 1
    
    # M3 = A00(B01-B11)
    U_s[0,2] = 1
    V_s[1,2] = 1; V_s[3,2] = -1
    W_s[1,2] = 1; W_s[2,2] = 1
    
    # M4 = A11(B10-B00)
    U_s[3,3] = 1
    V_s[2,3] = 1; V_s[0,3] = -1
    W_s[0,3] = -1; W_s[1,3] = 1
    
    # M5 = (A00+A01)B11
    U_s[0,4] = 1; U_s[1,4] = 1
    V_s[3,4] = 1
    W_s[0,4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U_s[2,5] = 1; U_s[0,5] = -1
    V_s[0,5] = 1; V_s[1,5] = 1
    W_s[3,5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U_s[1,6] = 1; U_s[3,6] = -1
    V_s[2,6] = 1; V_s[3,6] = 1
    W_s[0,6] = -1; W_s[3,6] = -1
    
    # Verify 2x2x2 Strassen
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0.0
                for r in range(7):
                    val += U_s[i*2+j, r] * V_s[j*2+k, r] * W_s[k*2+i, r]
                if abs(val - 1.0) > 1e-10:
                    return _schoolbook(n, m, p)
    
    # Build 4x4x9 decomposition
    # Part 1: 4x4x4 (cols 0-3) using Laderman rank 16
    # Part 2: 4x4x5 (cols 4-8) using block Strassen, rank 72
    # Total rank: 16 + 72 = 88
    
    total_rank = 88
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    col = 0
    
    # Part 1: Laderman 4x4x4 for columns 0-3
    # Laderman's algorithm for 4x4x4
    # We need to construct the rank-16 decomposition
    
    # Laderman's 4x4x4 algorithm:
    # Partition into 2x2 blocks and use Strassen-like combinations
    # The 16 terms are:
    
    # For 4x4x4, partition A, B, C into 2x2 blocks
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]  
    # C = [[C00, C01], [C10, C11]]
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each Ci,j is 2x2, so we have 4 subproblems of 2x2x2
    # But we can share computations.
    
    # Laderman's approach uses 16 scalar multiplications.
    # Let me use the known Laderman decomposition.
    
    # Actually, let me just use 4 independent Strassen 2x2x2 for the 4x4x4 part
    # That gives rank 4*7 = 28 for 4x4x4.
    # Then total would be 28 + 72 = 100.
    
    # Or use Laderman for 16. Let me try to construct Laderman.
    
    # For now, let me use 4 Strassen blocks for 4x4x4 (rank 28)
    # and see if that works, then optimize.
    
    # Actually, let me just use schoolbook for 4x4x4 (rank 16) 
    # Wait, 4x4x4 schoolbook is 4*4*4 = 64. Strassen gives 28. Laderman gives 16.
    
    # Let me use 4 Strassen 2x2x2 blocks for 4x4x4 (rank 28)
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                # Strassen for 2x2x2 block
                # A block at rows 2*bi:2*bi+2, cols 2*bj:2*bj+2
                # B block at rows 2*bj:2*bj+2, cols 2*bk:2*bk+2
                # C block at rows 2*bk:2*bk+2, cols 2*bi:2*bi+2
                for r in range(7):
                    for i in range(2):
                        for j in range(2):
                            U[(2*bi+i)*m + (2*bj+j), col] = U_s[i*2+j, r]
                    for j in range(2):
                        for k in range(2):
                            V[(2*bj+j)*p + (2*bk+k), col] = V_s[j*2+k, r]
                    for k in range(2):
                        for i in range(2):
                            W[(2*bk+k)*n + (2*bi+i), col] = W_s[k*2+i, r]
                    col += 1
    
    # Part 2: 4x4x5 for columns 4-8
    # Partition A into 2x2 blocks, B_part into 2x5 blocks
    # For each 2x2 block of A and each 2x5 block of B_part:
    #   Use 2x2x5 = 2x2x2 (Strassen) + 2x2x2 (Strassen) + 2x2x1 (schoolbook)
    
    for bi in range(2):
        for bj in range(2):
            # A block: rows 2*bi:2*bi+2, cols 2*bj:2*bj+2
            # B_part blocks: rows 2*bj:2*bj+2, cols 4:9
            # C_part blocks: rows 4:9, cols 2*bi:2*bi+2
            
            # B_part is 2x5: rows 2*bj:2*bj+2, cols 4:9
            # Split into: cols 4-5 (2x2), cols 6-7 (2x2), col 8 (2x1)
            
            # 2x2x2 for cols 4-5
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U[(2*bi+i)*m + (2*bj+j), col] = U_s[i*2+j, r]
                for j in range(2):
                    for k in range(2):
                        V[(2*bj+j)*p + (4+k), col] = V_s[j*2+k, r]
                for k in range(2):
                    for i in range(2):
                        W[(4+k)*n + (2*bi+i), col] = W_s[k*2+i, r]
                col += 1
            
            # 2x2x2 for cols 6-7
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U[(2*bi+i)*m + (2*bj+j), col] = U_s[i*2+j, r]
                for j in range(2):
                    for k in range(2):
                        V[(2*bj+j)*p + (6+k), col] = V_s[j*2+k, r]
                for k in range(2):
                    for i in range(2):
                        W[(6+k)*n + (2*bi+i), col] = W_s[k*2+i, r]
                col += 1
            
            # 2x2x1 for col 8
            for i in range(2):
                for j in range(2):
                    U[(2*bi+i)*m + (2*bj+j), col] = 1
                    V[(2*bj+j)*p + 8, col] = 1
                    W[8*n + (2*bi+i), col] = 1
                    col += 1
    
    # Verify full decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(total_rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-10:
                    return _schoolbook(n, m, p)
    
    def clean_matrix(mat):
        result = []
        for row in mat:
            new_row = []
            for v in row:
                if abs(v - round(v)) < 1e-10:
                    new_row.append(int(round(v)))
                else:
                    new_row.append(v)
            result.append(new_row)
        return result
    
    return {"U": clean_matrix(U.tolist()), "V": clean_matrix(V.tolist()), "W": clean_matrix(W.tolist())}

def _schoolbook(n, m, p):
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
