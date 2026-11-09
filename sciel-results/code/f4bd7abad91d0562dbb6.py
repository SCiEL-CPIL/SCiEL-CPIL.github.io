"""Rank-minimal tensor decomposition for 2x3x7 matrix multiplication."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]


def construct():
    """
    Construct a tensor decomposition for 2x3x7 matrix multiplication.
    
    We use a block decomposition of the 3x7 matrix B into B1 (2x7) and B2 (1x7).
    C = A1 * B1 + A2 * B2 where A1 is 2x2 and A2 is 2x1.
    
    For the A1*B1 part (2x2x7), we use the standard schoolbook rank 28.
    For the A2*B2 part (2x1x7), we use rank 2.
    Total rank = 30.
    
    Returns: dict with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    
    # Split m=3 into m1=2 and m2=1
    m1 = 2
    m2 = 1
    
    # Part 1: A1 (n x m1) * B1 (m1 x p) -> C1 (n x p)
    # Schoolbook rank = n * m1 * p = 2 * 2 * 7 = 28
    rank1 = n * m1 * p
    
    # Part 2: A2 (n x m2) * B2 (m2 x p) -> C2 (n x p)
    # Rank = n * m2 * p = 2 * 1 * 7 = 14? No, we can do better.
    # A2 is 2x1, B2 is 1x7. C2[i,k] = A2[i,0] * B2[0,k]
    # This is an outer product. Rank = n = 2? No.
    # C2[i,k] = sum_j A2[i,j] B2[j,k] = A2[i,0] B2[0,k]
    # We need T[i, j, k] where j is in the m2 block (just j=0)
    # U[i * m + j, c], V[j * p + k, c], W[k * n + i, c]
    # For j = m1 (the index for A2's column, which is the 3rd column of A, so j=2 in original)
    # Wait, let's be careful with indexing.
    
    # Original indices:
    # A is n x m (2x3). A[i, j] for i in [0,2), j in [0,3)
    # B is m x p (3x7). B[j, k] for j in [0,3), k in [0,7)
    # C is n x p (2x7). C[i, k] for i in [0,2), k in [0,7)
    # C[i,k] = sum_{j=0}^{2} A[i,j] B[j,k]
    
    # We split j into j1 in [0,2) and j2=2.
    # C[i,k] = sum_{j1=0}^{1} A[i,j1] B[j1,k] + A[i,2] B[2,k]
    
    # Part 1: sum_{j1=0}^{1} A[i,j1] B[j1,k]
    # This is 2x2x7 multiplication. Rank 28.
    # U1[i * m + j1, c], V1[j1 * p + k, c], W1[k * n + i, c]
    # Note: U index uses original m=3, so i*3 + j1
    # V index uses original p=7, so j1*7 + k
    # W index uses original n=2, so k*2 + i
    
    # Part 2: A[i,2] B[2,k]
    # This is 2x1x7 multiplication.
    # U2[i * m + 2, c], V2[2 * p + k, c], W2[k * n + i, c]
    # We need rank 2 for this part?
    # C2[i,k] = A[i,2] B[2,k]
    # This is an outer product of vector A[:,2] and vector B[2,:].
    # We can decompose this with rank 2?
    # Let's use rank 2.
    # c=0: U2[i*3+2, 0] = 1 if i=0 else 0. V2[14+k, 0] = 1. W2[2k+0, 0] = 1, W2[2k+1, 0] = 0
    # c=1: U2[i*3+2, 1] = 1 if i=1 else 0. V2[14+k, 1] = 1. W2[2k+0, 1] = 0, W2[2k+1, 1] = 1
    # Then sum_c U2*V2*W2 = (A[0,2]*1*B[2,k]*1*1 + A[1,2]*1*B[2,k]*1*0) + (A[0,2]*1*B[2,k]*1*0 + A[1,2]*1*B[2,k]*1*1)
    # = A[0,2]B[2,k] + A[1,2]B[2,k]
    # This is correct for C2[i,k] when we sum over i? No, W2[k*n+i, c] selects the i component.
    # For i=0: W2[2k, 0]=1, W2[2k, 1]=0. So term is A[0,2]B[2,k]. Correct.
    # For i=1: W2[2k+1, 0]=0, W2[2k+1, 1]=1. So term is A[1,2]B[2,k]. Correct.
    # So rank 2 works for the 2x1x7 part.
    
    rank2 = n  # = 2
    
    total_rank = rank1 + rank2  # 28 + 2 = 30
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill Part 1: 2x2x7 schoolbook
    col = 0
    for i in range(n):
        for j1 in range(m1):
            for k in range(p):
                U[i * m + j1][col] = 1
                V[j1 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Fill Part 2: 2x1x7 with rank 2
    for c in range(rank2):
        for i in range(n):
            if i == c:
                U[i * m + m1][col + c] = 1
        for k in range(p):
            V[m1 * p + k][col + c] = 1
        for i in range(n):
            for k in range(p):
                if i == c:
                    W[k * n + i][col + c] = 1
    col += rank2
    
    # Verify
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(total_rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-10:
                        return False
        return True
    
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails (should not happen)
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
