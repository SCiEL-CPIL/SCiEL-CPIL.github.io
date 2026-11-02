import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition."""
    if not U or not V or not W:
        return False
    r = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for row in U:
        if len(row) != r: return False
    for row in V:
        if len(row) != r: return False
    for row in W:
        if len(row) != r: return False
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(r))
                if val != 1:
                    return False
    return True

def _strassen_2x2():
    """
    Return the Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)*(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (c+d)*e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = a*(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = d*(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (a+b)*h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (c-a)*(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (b-d)*(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _construct_rank34(n, m, p):
    """
    Construct a rank-34 decomposition for 3x3 * 3x4.
    Uses Strassen for the 2x2 blocks in the top-left 2x2 of A and 
    the corresponding 2x2 blocks of B.
    """
    r = 34
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    col = 0
    
    # 1. Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    sU, sV, sW = _strassen_2x2()
    for c in range(7):
        for a_idx in range(4):
            i, j = divmod(a_idx, 2)
            U[i*m+j][col+c] = sU[a_idx][c]
        for b_idx in range(4):
            j, k = divmod(b_idx, 2)
            V[j*p+k][col+c] = sV[b_idx][c]
        for c_idx in range(4):
            k, i = divmod(c_idx, 2)
            W[k*n+i][col+c] = sW[c_idx][c]
    col += 7
    
    # 2. Strassen for A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    for c in range(7):
        for a_idx in range(4):
            i, j = divmod(a_idx, 2)
            U[i*m+j][col+c] = sU[a_idx][c]
        for b_idx in range(4):
            j, k_local = divmod(b_idx, 2)
            k = k_local + 2
            V[j*p+k][col+c] = sV[b_idx][c]
        for c_idx in range(4):
            k_local, i = divmod(c_idx, 2)
            k = k_local + 2
            W[k*n+i][col+c] = sW[c_idx][c]
    col += 7
    
    # 3. A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    for i in range(2):
        for k in range(2):
            U[i*m+2][col] = 1
            V[2*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1
            
    # 4. A[0:2, 2] * B[2, 2:4] -> C[0:2, 2:4]
    for i in range(2):
        for k in range(2, 4):
            U[i*m+2][col] = 1
            V[2*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1
            
    # 5. A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    for j in range(2):
        for k in range(2):
            U[2*m+j][col] = 1
            V[j*p+k][col] = 1
            W[k*n+2][col] = 1
            col += 1
            
    # 6. A[2, 2] * B[2, 0:2] -> C[2, 0:2]
    for k in range(2):
        U[2*m+2][col] = 1
        V[2*p+k][col] = 1
        W[k*n+2][col] = 1
        col += 1
        
    # 7. A[2, 0:2] * B[0:2, 2:4] -> C[2, 2:4]
    for j in range(2):
        for k in range(2, 4):
            U[2*m+j][col] = 1
            V[j*p+k][col] = 1
            W[k*n+2][col] = 1
            col += 1
            
    # 8. A[2, 2] * B[2, 2:4] -> C[2, 2:4]
    for k in range(2, 4):
        U[2*m+2][col] = 1
        V[2*p+k][col] = 1
        W[k*n+2][col] = 1
        col += 1
        
    if col != r:
        raise ValueError(f"Rank mismatch: expected {r}, got {col}")
        
    return U, V, W

def _construct_rank22(n, m, p):
    """
    Construct a rank-22 decomposition for 3x3 * 3x4.
    Uses the known rank-21 algorithm for 3x3 * 3x3 (Winograd)
    and appends a rank-1 schoolbook term for the 4th column of B
    combined with the 3x3 result? No, that's not right.
    
    Wait, the rank of 3x3*3x4 is at most R(3,3,3) + 9 = 30?
    Actually, R(3,3,4) is conjectured to be 22?
    Let's try to construct a rank-22 decomposition based on known results.
    One known rank-22 algorithm for 3x3x4 exists.
    However, coding it from scratch is error-prone.
    
    Let's try a different strategy:
    Can we reuse the rank-21 3x3x3 algorithm for the first 3 columns of B,
    and then handle the 4th column?
    The 4th column of C is A * B[:, 3].
    This is a 3x3 * 3x1 multiplication.
    The rank of 3x3*3x1 is 9 (schoolbook).
    So total rank = 21 + 9 = 30.
    
    Is there a better way?
    Maybe we can combine the 4th column with the existing rank-21 terms?
    If we can embed the 4th column into the rank-21 decomposition,
    we might reduce the rank.
    
    Let's try to construct a rank-22 decomposition.
    According to literature, R(3,3,4) <= 22.
    Let's try to find or derive such a decomposition.
    
    Actually, let's try a simpler improvement first:
    Can we reduce the rank of the "tail" terms in the rank-34 construction?
    In the rank-34 construction, we had:
    - 7 + 7 = 14 terms for the top-left 2x2 blocks.
    - 2 * 2 = 4 terms for A[0:2,2] * B[2,0:2]
    - 2 * 2 = 4 terms for A[0:2,2] * B[2,2:4]
    - 2 * 2 = 4 terms for A[2,0:2] * B[0:2,0:2]
    - 1 * 2 = 2 terms for A[2,2] * B[2,0:2]
    - 2 * 2 = 4 terms for A[2,0:2] * B[0:2,2:4]
    - 1 * 2 = 2 terms for A[2,2] * B[2,2:4]
    Total = 14 + 4 + 4 + 4 + 2 + 4 + 2 = 34.
    
    Can we use Strassen-like tricks for the remaining 1x3 and 3x1 blocks?
    For example, for A[0:2,2] * B[2,0:2], this is a 2x1 * 1x2 multiplication.
    The result is a 2x2 matrix.
    The rank of this "sub-multiplication" is 4 (schoolbook).
    Can we do it in rank 3? No, R(2,1,2) = 4.
    
    What about A[2,0:2] * B[0:2,0:2]? This is 1x2 * 2x2.
    The result is 1x2.
    The rank of this is 4 (schoolbook).
    Can we do it in rank 3? No, R(1,2,2) = 4.
    
    So the rank-34 construction seems optimal for that block partition.
    
    Let's try the rank-22 construction from the literature.
    The rank-22 algorithm for 3x3x4 is non-trivial.
    However, we can try to use the rank-21 3x3x3 algorithm and see if we can
    "extend" it to 3x3x4 with only 1 additional term? No, that's impossible.
    
    Let's try to implement the rank-21 3x3x3 algorithm (Winograd) and
    then add 9 schoolbook terms for the 4th column, giving rank 30.
    This is better than 34.
    """
    # Winograd's rank-21 algorithm for 3x3 * 3x3
    # The coefficients are complex. Let's use a known representation.
    # I will hardcode the U, V, W for 3x3x3 rank 21.
    
    U_21 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    # This is just identity. I need the actual Winograd coefficients.
    # Since I don't have them memorized perfectly, I will fallback to a simpler improvement.
    # Let's try to reduce the rank from 34 to 30 by using the rank-21 3x3x3 block.
    # But coding the rank-21 algorithm correctly is risky.
    
    # Instead, let's try to optimize the rank-34 construction.
    # Is there a way to share terms between the blocks?
    # For example, the term for A[0,0] * B[0,0] appears in multiple places?
    # No, each term in the rank-34 construction is for a specific (i,j,k) triple
    # in the "tail" part, or a specific Strassen term in the "head" part.
    
    # Let's try a different partition:
    # Use Strassen for A[0:2, 0:2] * B[0:2, 0:4] (rank 14)
    # Use Strassen for A[0:2, 0:2] * B[2, 0:4] (rank ?)
    # A[0:2,0:2] is 2x2, B[2,0:4] is 1x4.
    # This is 2x1x4. Rank is 8.
    # Total so far: 14 + 8 = 22.
    # Remaining:
    # A[2,0:2] * B[0:4,0:4] (1x2 * 4x4 -> 1x4). Rank 8.
    # A[2,2] * B[0:4,0:4] (1x1 * 4x4 -> 1x4). Rank 4.
    # A[0:2,2] * B[0:4,0:4] (2x1 * 4x4 -> 2x4). Rank 8.
    # Total: 22 + 8 + 4 + 8 = 42. Worse.
    
    # Let's stick with rank 34. It is a known improvement over 36.
    # I will return rank 34.
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try rank 34
    try:
        U, V, W = _construct_rank34(n, m, p)
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
