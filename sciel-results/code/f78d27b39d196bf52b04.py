import numpy as np

def construct():
    n, m, p = 2, 2, 15
    # Rank calculation: 7 blocks of 2x2x2 (rank 7) + 1 block of 2x2x1 (rank 4)
    # 15 = 7 * 2 + 1
    rank = 7 * 7 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen's algorithm for 2x2x2
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00
    # P3 = a00(b01-b11)
    # P4 = a11(b10-b00)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b01)
    # P7 = (a01-a11)(b10+b11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    for b in range(7):
        k_start = 2 * b
        
        # P1
        U[0][col] += 1; U[3][col] += 1
        V[0 * p + k_start][col] += 1; V[1 * p + k_start + 1][col] += 1
        W[k_start * n + 0][col] += 1; W[(k_start+1) * n + 1][col] += 1
        col += 1
        
        # P2
        U[2][col] += 1; U[3][col] += 1
        V[0 * p + k_start][col] += 1
        W[k_start * n + 1][col] += 1
        col += 1
        
        # P3
        U[0][col] += 1
        V[0 * p + k_start + 1][col] += 1; V[1 * p + k_start + 1][col] -= 1
        W[k_start * n + 0][col] += 1; W[(k_start+1) * n + 1][col] += 1
        col += 1
        
        # P4
        U[3][col] += 1
        V[1 * p + k_start][col] += 1; V[0 * p + k_start][col] -= 1
        W[k_start * n + 0][col] += 1; W[(k_start+1) * n + 1][col] += 1
        col += 1
        
        # P5
        U[0][col] += 1; U[1][col] += 1
        V[1 * p + k_start + 1][col] += 1
        W[k_start * n + 0][col] -= 1; W[(k_start+1) * n + 1][col] += 1
        col += 1
        
        # P6
        U[2][col] += 1; U[0][col] -= 1
        V[0 * p + k_start][col] += 1; V[0 * p + k_start + 1][col] += 1
        W[(k_start+1) * n + 1][col] += 1
        col += 1
        
        # P7
        U[1][col] += 1; U[3][col] -= 1
        V[1 * p + k_start][col] += 1; V[1 * p + k_start + 1][col] += 1
        W[k_start * n + 0][col] += 1; W[(k_start+1) * n + 1][col] += 1
        col += 1
        
    # Last block: 2x2x1 schoolbook
    k_last = 14
    
    # Q1 = a00 * b0,k
    U[0][col] += 1
    V[0 * p + k_last][col] += 1
    W[k_last * n + 0][col] += 1
    col += 1
    
    # Q2 = a01 * b1,k
    U[1][col] += 1
    V[1 * p + k_last][col] += 1
    W[k_last * n + 0][col] += 1
    col += 1
    
    # Q3 = a10 * b0,k
    U[2][col] += 1
    V[0 * p + k_last][col] += 1
    W[k_last * n + 1][col] += 1
    col += 1
    
    # Q4 = a11 * b1,k
    U[3][col] += 1
    V[1 * p + k_last][col] += 1
    W[k_last * n + 1][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
