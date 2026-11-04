"""
Matrix multiplication 5x5x5 tensor decomposition.
Uses recursive Strassen-based decomposition: partition 5 = 4 + 1,
apply rank-49 Strassen to the 4x4 core, handle boundary terms minimally.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Construct the standard rank n*m*p decomposition."""
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


def _verify_float(n, m, p, U, V, W, tol=1e-10):
    """Fast float verification."""
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                dot = np.dot(u_val, v_val * w_val)
                if abs(dot - 1.0) > tol:
                    return False
    return True


def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    Returns (U, V, W) where each is a list of lists.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Index convention: row index = i*n+j for n=2
    """
    U = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, -1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return U, V, W


def _strassen_4x4():
    """
    Build a rank-49 decomposition for 4x4 matrix multiplication
    by applying Strassen's 2x2 algorithm recursively (tensor product).
    
    4x4 matmul = Strassen on 2x2 blocks, each of which is 2x2 matmul.
    The 4x4 tensor is the Kronecker product of two 2x2 Strassen tensors.
    Rank = 7 * 7 = 49.
    
    Returns (U, V, W) each of shape (16, 49).
    U[i*4+j, c] for i,j in 0..3
    V[j*4+k, c] for j,k in 0..3  
    W[k*4+i, c] for k,i in 0..3
    """
    U2, V2, W2 = _strassen_2x2()
    rank2 = 7
    rank4 = rank2 * rank2  # 49
    
    U = [[0] * rank4 for _ in range(16)]
    V = [[0] * rank4 for _ in range(16)]
    W = [[0] * rank4 for _ in range(16)]
    
    # The 4x4 Strassen is the tensor product of two 2x2 Strassen decompositions.
    # For block indices (a,b) where a,b in {0,1} are block row/col,
    # and internal indices (i,j) where i,j in {0,1} are within-block:
    # 
    # U index in 4x4: (a*2+i)*4 + (b*2+j) = a*8 + i*4 + b*2 + j
    # V index in 4x4: (b*2+j)*4 + (a*2+k) = b*8 + j*4 + a*2 + k
    # W index in 4x4: (a*2+k)*4 + (a*2+i) = a*8 + k*4 + a*2 + i
    # 
    # Wait, let me be more careful. For 4x4:
    # A[i,j] with i,j in 0..3. Block structure: A = [[A00, A01], [A10, A11]]
    # where each Aab is 2x2.
    # A[i,j] = A_{a,b}[i%2, j%2] where a = i//2, b = j//2
    #
    # U[i*4+j, c] where c = c1*7 + c2 (two levels of Strassen)
    # The tensor product means:
    # U_4x4[i*4+j, c1*7+c2] = U_2x2[a*2+i%2, c1] * U_2x2[b*2+j%2, c2]
    # where a = i//2, b = j//2
    #
    # Similarly for V and W.
    
    for i in range(4):
        for j in range(4):
            a = i // 2
            bi = i % 2
            b = j // 2
            bj = j % 2
            u_idx = i * 4 + j
            for c1 in range(rank2):
                for c2 in range(rank2):
                    c = c1 * rank2 + c2
                    u_val = U2[a * 2 + bi][c1] * U2[b * 2 + bj][c2]
                    U[u_idx][c] = u_val
    
    for j in range(4):
        for k in range(4):
            b = j // 2
            bj = j % 2
            a = k // 2
            ak = k % 2
            v_idx = j * 4 + k
            for c1 in range(rank2):
                for c2 in range(rank2):
                    c = c1 * rank2 + c2
                    v_val = V2[b * 2 + bj][c1] * V2[a * 2 + ak][c2]
                    V[v_idx][c] = v_val
    
    for k in range(4):
        for i in range(4):
            a = k // 2
            ak = k % 2
            bi = i // 2
            bi2 = i % 2
            w_idx = k * 4 + i
            for c1 in range(rank2):
                for c2 in range(rank2):
                    c = c1 * rank2 + c2
                    w_val = W2[a * 2 + ak][c1] * W2[bi * 2 + bi2][c2]
                    W[w_idx][c] = w_val
    
    return U, V, W


def _decompose_5x5():
    """
    Construct a decomposition for 5x5x5 matrix multiplication.
    
    Strategy: Partition 5 = 4 + 1.
    C = A*B where A,B,C are 5x5.
    Write A = [A4, A1; A1t, A11], B = [B4, B1; B1t, B11]
    where A4 is 4x4, A1 is 4x1, A1t is 1x4, A11 is 1x1.
    
    C44 = A4*B4 + A1*B1t + A1t*B1 + A11*B11  (4x4)
    C41 = A4*B1 + A1*B11                       (4x1)
    C14 = A1t*B4 + A11*B1t                     (1x4)
    C11 = A1t*B1 + A11*B11                     (1x1)
    
    We use:
    - Rank-49 Strassen for A4*B4 -> C44 (the 4x4 core)
    - Schoolbook for all terms involving the 5th row/col
    
    Total rank = 49 + (4*1*1 + 1*4*1 + 4*1*1 + 1*1*1)  [for C41, C14, C11, and cross in C44]
    Wait, let me count more carefully.
    
    C44 = A4*B4 + A1*B1t + A1t*B1 + A11*B11
    - A4*B4: rank 49 (Strassen)
    - A1*B1t: A1 is 4x1, B1t is 1x4, result is 4x4. Schoolbook rank = 4*1*4 = 16
    - A1t*B1: A1t is 1x4, B1 is 4x1, result is 1x1. Schoolbook rank = 1*4*1 = 4
    - A11*B11: 1*1*1 = 1
    
    C41 = A4*B1 + A1*B11
    - A4*B1: 4x4 * 4x1 -> 4x1. Schoolbook rank = 4*4*1 = 16
    - A1*B11: 4x1 * 1x1 -> 4x1. Schoolbook rank = 4*1*1 = 4
    
    C14 = A1t*B4 + A11*B1t
    - A1t*B4: 1x4 * 4x4 -> 1x4. Schoolbook rank = 1*4*4 = 16
    - A11*B1t: 1x1 * 1x4 -> 1x4. Schoolbook rank = 1*1*4 = 4
    
    C11 = A1t*B1 + A11*B11
    - A1t*B1: 1x4 * 4x1 -> 1x1. Schoolbook rank = 1*4*1 = 4
    - A11*B11: 1x1 * 1x1 -> 1x1. Schoolbook rank = 1
    
    But wait, some of these terms overlap in their contributions. Let me just use schoolbook for everything except A4*B4.
    
    Total additional rank (excluding A4*B4):
    C44 cross: 16 + 4 + 1 = 21
    C41: 16 + 4 = 20
    C14: 16 + 4 = 20
    C11: 4 + 1 = 5
    Total additional: 21 + 20 + 20 + 5 = 66
    
    Total rank: 49 + 66 = 115
    
    Hmm, that's only slightly better than schoolbook (125). Let me be smarter.
    
    Actually, I realize I should just directly construct the full decomposition by iterating over all (i,j,k) and using Strassen for the 4x4 part.
    """
    n, m, p = 5, 5, 5
    nm, mp, pn = 25, 25, 25
    
    # Get 4x4 Strassen
    U4, V4, W4 = _strassen_4x4()
    rank4 = 49
    
    # Total rank: 49 for the 4x4 core + schoolbook for all terms involving index 4
    # Terms involving index 4:
    # i=4 or j=4 or k=4
    # Count: total 125 - 64 (where i,j,k all in 0..3) = 61
    # But we need to be careful: the 4x4 Strassen handles all (i,j,k) with i,j,k in 0..3
    # So we need schoolbook for the remaining 61 terms.
    
    total_rank = rank4 + 61
    
    U = [[0] * total_rank for _ in range(nm)]
    V = [[0] * total_rank for _ in range(mp)]
    W = [[0] * total_rank for _ in range(pn)]
    
    # Fill in the 4x4 Strassen part
    # For i,j,k in 0..3:
    # U[i*5+j, c] = U4[i*4+j, c]
    # V[j*5+k, c] = V4[j*4+k, c]
    # W[k*5+i, c] = W4[k*4+i, c]
    
    for i in range(4):
        for j in range(4):
            for c in range(rank4):
                U[i * 5 + j][c] = U4[i * 4 + j][c]
    
    for j in range(4):
        for k in range(4):
            for c in range(rank4):
                V[j * 5 + k][c] = V4[j * 4 + k][c]
    
    for k in range(4):
        for i in range(4):
            for c in range(rank4):
                W[k * 5 + i][c] = W4[k * 4 + i][c]
    
    # Now fill in schoolbook for all (i,j,k) where at least one is 4
    col = rank4
    for i in range(5):
        for j in range(5):
            for k in range(5):
                if i < 4 and j < 4 and k < 4:
                    continue  # Already handled by Strassen
                u_idx = i * 5 + j
                v_idx = j * 5 + k
                w_idx = k * 5 + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try the 4x4 Strassen + boundary decomposition
    if time.time() < deadline:
        result = _decompose_5x5()
        rank = len(result['U'][0])
        
        # Verify the result
        if _verify_float(n, m, p, result['U'], result['V'], result['W'], tol=1e-10):
            if rank < best_rank:
                best = result
                best_rank = rank
    
    return best
# EVOLVE-BLOCK-END
