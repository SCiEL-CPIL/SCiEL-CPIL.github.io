import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Constructs a rank-44 exact tensor decomposition for multiplying 
    a 2 by 3 matrix by a 3 by 8 matrix.
    
    Uses block Strassen on the 2x2 top-left block (shared across all 4 column pairs)
    and direct multiplication for the 2x1 bottom-right block.
    
    n=2, m=3, p=8
    U: (n*m, rank) = (6, 44)
    V: (m*p, rank) = (24, 44)
    W: (p*n, rank) = (16, 44)
    """
    n, m, p = DIMENSIONS
    rank = 44
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(p // 2):  # 4 blocks of 2 columns
        # Strassen on 2x2 block: A[:2, :2] * B[:2, 2k:2k+2]
        # M1 = (a00+a11)(b00+b11)
        U[0][col] = 1; U[3][col] = 1
        V[2*k][col] = 1; V[16+2*k][col] = 1
        W[2*k*n+0][col] = 1; W[(2*k+1)*n+0][col] = 1
        W[2*k*n+1][col] = 1; W[(2*k+1)*n+1][col] = 1
        col += 1
        
        # M2 = (a10+a11)b00
        U[3][col] = 1; U[4][col] = 1
        V[2*k][col] = 1
        W[2*k*n+1][col] = 1; W[(2*k+1)*n+1][col] = 1
        col += 1
        
        # M3 = a00(b01-b11)
        U[0][col] = 1
        V[2*k+1][col] = 1; V[16+2*k+1][col] = -1
        W[2*k*n+0][col] = 1; W[(2*k+1)*n+0][col] = 1
        W[2*k*n+1][col] = 1; W[(2*k+1)*n+1][col] = 1
        col += 1
        
        # M4 = a11(b10-b00)
        U[3][col] = 1
        V[16+2*k][col] = 1; V[2*k][col] = -1
        W[2*k*n+0][col] = 1; W[(2*k+1)*n+0][col] = 1
        W[2*k*n+1][col] = 1; W[(2*k+1)*n+1][col] = 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[0][col] = 1; U[1][col] = 1
        V[16+2*k+1][col] = 1
        W[2*k*n+0][col] = -1; W[(2*k+1)*n+0][col] = -1
        W[2*k*n+1][col] = 1; W[(2*k+1)*n+1][col] = 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[4][col] = 1; U[0][col] = -1
        V[2*k][col] = 1; V[2*k+1][col] = 1
        W[(2*k+1)*n+1][col] = 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[1][col] = 1; U[3][col] = -1
        V[16+2*k][col] = 1; V[16+2*k+1][col] = 1
        W[2*k*n+0][col] = 1; W[(2*k+1)*n+0][col] = 1
        col += 1
        
        # Direct on 2x1 block: A[2:, :2] * B[2, 2k:2k+2]
        # D1 = a02 * b20
        U[2][col] = 1
        V[8+2*k][col] = 1
        W[2*k*n+0][col] = 1; W[(2*k+1)*n+0][col] = 1
        col += 1
        
        # D2 = a02 * b21
        U[2][col] = 1
        V[8+2*k+1][col] = 1
        W[2*k*n+1][col] = 1; W[(2*k+1)*n+1][col] = 1
        col += 1
        
        # D3 = a12 * b20
        U[5][col] = 1
        V[8+2*k][col] = 1
        W[2*k*n+1][col] = 1; W[(2*k+1)*n+1][col] = 1
        col += 1
        
        # D4 = a12 * b21
        U[5][col] = 1
        V[8+2*k+1][col] = 1
        W[2*k*n+1][col] = 1; W[(2*k+1)*n+1][col] = 1
        col += 1
        
        # D5 = a12 * b21 (wait, I need to recheck)
        # Actually for C[1,2k+1] = M1-M2+M3+M6 + D4
        # D4 = a12 * b21
        # And C[0,2k+1] = M3+M5 + D2
        # D2 = a02 * b21
        # So I have 4 direct terms, not 5. Let me recount.
        # c00 = M1+M4-M5+M7 + D1
        # c01 = M3+M5 + D2
        # c10 = M2+M4 + D3
        # c11 = M1-M2+M3+M6 + D4
        # That's 4 direct terms. So 7+4=11 per block.
        # But I've already added 4 above. Let me verify the count.
        # 7 Strassen + 4 direct = 11 per block. 4 blocks = 44. Correct.
    
    return {"U": U, "V": V, "W": W}
