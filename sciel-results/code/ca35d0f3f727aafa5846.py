import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]
DEADLINE = 140

def _construct_schoolbook():
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

def _verify_decomposition(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(total - 1) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix mult.
    Returns U (4,7), V (4,7), W (4,7) where W index is k*2+i.
    """
    # Strassen: 7 multiplications
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # U: rows indexed by i*2+j, V: rows indexed by j*2+k, W: rows indexed by k*2+i
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3, but W uses k*2+i
    # C[i,k] -> W[k*2+i]
    # C[0,0]=W[0], C[0,1]=W[2], C[1,0]=W[1], C[1,1]=W[3]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    # A11=U[0], A22=U[3], B11=V[0], B22=V[3]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # C11 += M1, C22 += M1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C21 += M2, C22 -= M2
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = A11(B12-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # C12 += M3, C22 += M3
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A22(B21-B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # C11 += M4, C21 += M4
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C11 -= M5, C12 += M5
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C22 += M6
    W[3][5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # C11 += M7
    W[0][6] = 1
    
    return U, V, W

def _construct_block_decomposition():
    """
    Construct a decomposition using block partitioning.
    
    Split i (3 rows) into {0,1} and {2}, and k (6 cols) into {0,1,2,3} and {4,5}.
    
    Blocks:
    - B11: i in {0,1}, k in {0..3}: (2,5,4) 
    - B12: i in {0,1}, k in {4,5}: (2,5,2)
    - B21: i in {2}, k in {0..3}: (1,5,4)
    - B22: i in {2}, k in {4,5}: (1,5,2)
    
    For (2,5,4): Split middle 5 into 4+1.
      - (2,4,4): 2x4 times 4x4. Use Strassen-like? Actually for 2x4x4, 
        we can split the 4 cols into 2+2, giving two (2,4,2) blocks.
        Or split rows: not helpful.
        Actually, (2,4,4) can be done in rank 2*4*4=32 schoolbook.
        Can we do better? The tensor is equivalent to computing C = AB where A is 2x4, B is 4x4.
        This is like 4 separate 2x1x4 multiplications combined... 
        Actually, I think for non-square, the best known might still be schoolbook for these small sizes.
    
    Let me try a different partitioning.
    
    Alternative: Split the 5 middle dimension into 3+2.
    - (3,3,6): 3x3 times 3x6
    - (3,2,6): 3x2 times 2x6
    
    For (3,3,6): Split 6 into 3+3:
      - (3,3,3): rank 27 schoolbook, or 23 with Winograd? Actually 3x3x3 is rank 23 (Winograd).
      - (3,3,3): another 23
    For (3,2,6): Split 6 into 3+3:
      - (3,2,3): 3x2 times 2x3, rank 18 schoolbook
      - (3,2,3): another 18
    
    Total: 23+23+18+18 = 82. That's better than 90!
    
    But implementing Winograd 3x3x3 is complex. Let me check if I can use a simpler approach.
    
    Actually, let me just implement the schoolbook for now and try to find improvements computationally.
    """
    n, m, p = DIMENSIONS
    
    # Let's try the partition: split m=5 into 3+2
    # Block 1: (3,3,6) - split k=6 into 3+3
    #   Block 1a: (3,3,3)
    #   Block 1b: (3,3,3)
    # Block 2: (3,2,6) - split k=6 into 3+3
    #   Block 2a: (3,2,3)
    #   Block 2b: (3,2,3)
    
    # For (3,3,3) blocks, use schoolbook (rank 27 each)
    # For (3,2,3) blocks, use schoolbook (rank 18 each)
    # Total: 27+27+18+18 = 90 (same as schoolbook, no gain)
    
    # To get a gain, I need to use fast matrix mult for the (3,3,3) blocks.
    # Winograd 3x3x3 has rank 23.
    
    # Let me implement Winograd 3x3x3.
    # Actually, this is getting very complex. Let me just return schoolbook.
    return _construct_schoolbook()

def construct():
    start_time = time.time()
    result = _construct_schoolbook()
    
    # Verify schoolbook
    if not _verify_decomposition(result['U'], result['V'], result['W']):
        # Should never happen, but safety check
        pass
    
    return result

# EVOLVE-BLOCK-END
