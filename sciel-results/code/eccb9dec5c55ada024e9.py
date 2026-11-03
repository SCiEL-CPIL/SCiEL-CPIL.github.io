"""Rewritten program for 4x5x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor entries
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries.append((a, b, c))
    
    # Schoolbook fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[c][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try block-based Strassen approach
    # Partition: n=4=2+2, m=5=3+2, p=6=3+3
    # 2x2x2 blocks: 8 blocks
    # For 2x2x2: Strassen-2 rank 7
    # For 2x3x3: schoolbook rank 18
    # For 3x2x3: schoolbook rank 18
    # For 3x3x3: Strassen-3 rank 7
    
    def strassen2_decomp():
        """2x2x2 Strassen, rank 7"""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # M1=(A11+A22)(B11+B22): C11+=M1, C22+=M1
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        # M2=(A21+A22)B21: C21+=M2
        U[2][1]=1; U[3][1]=1; V[2][1]=1; W[2][1]=1
        # M3=A11(B12-B22): C12+=M3
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1
        # M4=A22(B21-B11): C11+=M4, C21+=M4
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        # M5=(A11+A12)B22: C11-=M5, C12+=M5
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        # M6=(A21-A11)(B11+B12): C22+=M6
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        # M7=(A12-A22)(B21+B22): C11+=M7, C22+=M7
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1; W[3][6]=1
        return U, V, W
    
    def strassen3_decomp():
        """3x3x3 schoolbook, rank 27"""
        U = [[0]*27 for _ in range(9)]
        V = [[0]*27 for _ in range(9)]
        W = [[0]*27 for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
        return U, V, W
    
    # Try partition n=2+2, m=3+2, p=3+3
    # Blocks: (2,3,3),(2,3,3),(2,2,3),(2,2,3),(2,3,3),(2,3,3),(2,2,3),(2,2,3)
    # Ranks: 18,18,12,12,18,18,12,12 = total 120. Not better.
    
    # Try n=3+1, m=3+2, p=3+3
    # Blocks: (3,3,3)=27, (3,3,3)=27, (3,2,3)=18, (3,2,3)=18,
    #         (1,3,3)=9, (1,3,3)=9, (1,2,3)=6, (1,2,3)=6
    # Total: 27+27+18+18+9+9+6+6 = 120
    
    # Build this decomposition
    total_rank = 120
    U_full = [[0] * total_rank for _ in range(nm)]
    V_full = [[0] * total_rank for _ in range(mp)]
    W_full = [[0] * total_rank for _ in range(pn)]
    
    col = 0
    # Block (0,0,0): n=3,m=3,p=3 -> schoolbook
    U3, V3, W3 = strassen3_decomp()
    for r in range(27):
        for i in range(3):
            for j in range(3):
                U_full[i*m+j][col+r] = U3[i*3+j][r]
        for j in range(3):
            for k in range(3):
                V_full[j*p+k][col+r] = V3[j*3+k][r]
        for k in range(3):
            for i in range(3):
                W_full[k*n+i][col+r] = W3[k*3+i][r]
    col += 27
    
    # Block (0,0,1): n=3,m=3,p=3 -> schoolbook
    U3, V3, W3 = strassen3_decomp()
    for r in range(27):
        for i in range(3):
            for j in range(3):
                U_full[i*m+j][col+r] = U3[i*3+j][r]
        for j in range(3):
            for k in range(3):
                V_full[j*p+(k+3)][col+r] = V3[j*3+k][r]
        for k in range(3):
            for i in range(3):
                W_full[(k+3)*n+i][col+r] = W3[k*3+i][r]
    col += 27
    
    # Block (0,1,0): n=3,m=2,p=3 -> schoolbook
    for i in range(3):
        for j in range(2):
            for k in range(3):
                U_full[i*m+(j+3)][col] = 1
                V_full[(j+3)*p+k][col] = 1
                W_full[k*n+i][col] = 1
                col += 1
    col += 0  # col should be 7+7+18=32
    
    # Block (0,1,1): n=3,m=2,p=3 -> schoolbook
    for i in range(3):
        for j in range(2):
            for k in range(3):
                U_full[i*m+(j+3)][col] = 1
                V_full[(j+3)*p+(k+3)][col] = 1
                W_full[(k+3)*n+i][col] = 1
                col += 1
    # col should be 54+18=72
    
    # Block (1,0,0): n=1,m=3,p=3 -> schoolbook
    for i in range(1):
        for j in range(3):
            for k in range(3):
                U_full[(i+3)*m+j][col] = 1
                V_full[j*p+k][col] = 1
                W_full[k*n+(i+3)][col] = 1
                col += 1
    # col should be 72+9=81
    
    # Block (1,0,1): n=1,m=3,p=3 -> schoolbook
    for i in range(1):
        for j in range(3):
            for k in range(3):
                U_full[(i+3)*m+j][col] = 1
                V_full[j*p+(k+3)][col] = 1
                W_full[(k+3)*n+(i+3)][col] = 1
                col += 1
    # col should be 81+9=90
    
    # Block (1,1,0): n=1,m=2,p=3 -> schoolbook
    for i in range(1):
        for j in range(2):
            for k in range(3):
                U_full[(i+3)*m+(j+3)][col] = 1
                V_full[(j+3)*p+k][col] = 1
                W_full[k*n+(i+3)][col] = 1
                col += 1
    # col should be 90+6=96
    
    # Block (1,1,1): n=1,m=2,p=3 -> schoolbook
    for i in range(1):
        for j in range(2):
            for k in range(3):
                U_full[(i+3)*m+(j+3)][col] = 1
                V_full[(j+3)*p+(k+3)][col] = 1
                W_full[(k+3)*n+(i+3)][col] = 1
                col += 1
    # col should be 96+6=102
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(total_rank):
                    s += U_full[i*m+j][r] * V_full[j*p+k][r] * W_full[k*n+i][r]
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break
        if not valid:
            break
    
    if valid:
        return {"U": U_full, "V": V_full, "W": W_full}
    
    # The 80-rank solution failed verification (Strassen-3 was incorrect).
    # Return the schoolbook solution.
    return best_result
# EVOLVE-BLOCK-END
