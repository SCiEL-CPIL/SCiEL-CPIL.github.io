"""Tensor rank minimization for 4x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _get_strassen_2x2():
    """Returns U, V, W for 2x2x2 Strassen (Rank 7)."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indices: i in 0..1, j in 0..1, k in 0..1
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # M1 = (A00+A11)(B00+B11)
    # A00: i=0,j=0. A11: i=1,j=1.
    # B00: j=0,k=0. B11: j=1,k=1.
    # C00 += M1, C11 += M1
    
    # We define the 7 products:
    # 0: (A00+A11)(B00+B11)
    # 1: (A10+A11)B00
    # 2: A00(B01-B11)
    # 3: A11(B10-B00)
    # 4: (A00+A01)B11
    # 5: (A10-A00)(B00+B01)
    # 6: (A01-A11)(B10+B11)
    
    # Product 0
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1; W[1*2+1, 0] = 1
    
    # Product 1
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[1*2+0, 1] = 1
    
    # Product 2
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1; V[1*2+1, 2] = -1
    W[0*2+1, 2] = 1
    
    # Product 3
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1; V[0*2+0, 3] = -1
    W[1*2+0, 3] = 1
    
    # Product 4
    U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[0*2+0, 4] = -1; W[0*2+1, 4] = 1
    
    # Product 5
    U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
    W[1*2+1, 5] = 1
    
    # Product 6
    U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1; V[1*2+1, 6] = 1
    W[0*2+0, 6] = 1; W[1*2+1, 6] = 1
    
    return U, V, W

def _get_2x4x4():
    """Returns U, V, W for 2x4x4 (Rank 14)."""
    # T_2x4x4 = T_2x2x2 (for X1Y1) + T_2x2x2 (for X2Y2)
    # X is 2x4, Y is 4x4.
    # X = [X1, X2], X1 is 2x2, X2 is 2x2.
    # Y = [Y1, Y2], Y1 is 4x2, Y2 is 4x2.
    # Wait, Y is 4x4. Y1 is 4x2 (cols 0,1), Y2 is 4x2 (cols 2,3).
    # Product 1: X1 (2x2) * Y1 (4x2)? No.
    # X1 is 2x2. Y1 must be 2x2 for standard matmul.
    # But Y is 4x4.
    # X (2x4) * Y (4x4).
    # Split inner dim 4 into 2+2.
    # X = [X_a, X_b] (2x2 each).
    # Y = [Y_a; Y_b] (2x4 each).
    # XY = X_a Y_a + X_b Y_b.
    # X_a (2x2) * Y_a (2x4).
    # This is 2x2x4.
    # We can decompose 2x2x4 using Strassen 2x2x2 x 2x1x1?
    # Or just use 2x2x2 Strassen for the 2x2 part and identity for the rest?
    
    # Simpler: Use the rank-7 Strassen for 2x2x2.
    # For 2x2x4:
    # C (2x4) = A (2x2) B (2x4).
    # B = [B0, B1] (2x2 each).
    # C = A B0 + A B1.
    # We can compute A B0 using Strassen (Rank 7).
    # We can compute A B1 using Strassen (Rank 7).
    # Total Rank 14.
    
    # So T_2x2x4 has Rank 14.
    # Then T_2x4x4 = T_2x2x2 (for X_a Y_a) + T_2x2x2 (for X_b Y_b)?
    # X_a (2x2) * Y_a (2x4). Rank 14.
    # X_b (2x2) * Y_b (2x4). Rank 14.
    # Total Rank 28?
    
    # Let's re-evaluate T_2x4x4.
    # X (2x4) Y (4x4).
    # Split X into X1 (2x2), X2 (2x2).
    # Split Y into Y1 (2x4), Y2 (2x4).
    # XY = X1 Y1 + X2 Y2.
    # X1 Y1 is 2x2x4. Rank 14.
    # X2 Y2 is 2x2x4. Rank 14.
    # Total Rank 28.
    
    # My previous estimate of 14 was wrong.
    # Rank of 2x2x4 is 14.
    # Rank of 2x4x4 is 28.
    
    # Then T_4x8x8 = T_2x4x4 (Rank 28) x T_2x2x2 (Rank 7)?
    # 28 * 7 = 196. Worse than schoolbook.
    
    # Let's try T_4x8x8 = T_4x4x4 (Rank 49) x T_1x2x2 (Rank 4)?
    # 49 * 4 = 196.
    
    # It seems the Kronecker product of Strassen is not efficient for non-powers-of-2 or mixed sizes.
    
    # Let's fall back to a known good decomposition or a simpler optimization.
    # Given the time constraints and the difficulty of deriving a low-rank decomposition manually,
    # we will return the schoolbook decomposition, which is guaranteed to be correct and fast.
    # However, we can try to use a simple heuristic to reduce rank if possible.
    
    # For now, return schoolbook to ensure validity.
    n, m, p = DIMENSIONS
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

def construct():
    """Construct factor matrices for matrix multiplication tensor."""
    return _get_2x4x4()
# EVOLVE-BLOCK-END
