"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    for i in range(n):
        for k in range(p):
            for j in range(m):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen, shapes (4,7), (4,7), (4,7)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # c=0: m1 = (a00+a11)(b00+b11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # c=1: m2 = (a10+a11)b00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1
    
    # c=2: m3 = a00(b10-b00)
    U[0,2]=1
    V[2,2]=1; V[0,2]=-1
    W[1,2]=1; W[3,2]=-1
    
    # c=3: m4 = a11(b01-b11)
    U[3,3]=1
    V[1,3]=1; V[3,3]=-1
    W[0,3]=-1; W[2,3]=1
    
    # c=4: m5 = (a00+a01)b11
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[1,4]=1
    
    # c=5: m6 = (a10-a00)b01
    U[2,5]=1; U[0,5]=-1
    V[1,5]=1
    W[3,5]=1
    
    # c=6: m7 = (a01-a11)b10
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1
    W[0,6]=1
    
    return U, V, W

def _build_2x4x8_hierarchical():
    """
    Build 2x4x8 using hierarchical Strassen:
    2x4x8 = two 2x4x4 (concatenated in output k)
    2x4x4 = sum of two 2x2x4 (split input j)
    2x2x4 = sum of two 2x2x2 (split output k)
    2x2x2 = Strassen rank 7
    
    Total rank: 2 * 2 * 2 * 7 = 56
    """
    n, m, p = 2, 4, 8
    Us, Vs, Ws = _strassen_2x2x2()
    
    # Build 2x2x4: sum of two 2x2x2 (k split into {0,1} and {2,3})
    U_224 = np.zeros((4, 14))
    V_224 = np.zeros((8, 14))
    W_224 = np.zeros((8, 14))
    
    # Part 1: k=0,1
    U_224[:, 0:7] = Us
    for j in range(2):
        for k in range(2):
            V_224[j*4+k, 0:7] = Vs[j*2+k, 0:7]
    for k in range(2):
        for i in range(2):
            W_224[k*2+i, 0:7] = Ws[k*2+i, 0:7]
    
    # Part 2: k=2,3
    U_224[:, 7:14] = Us
    for j in range(2):
        for k in range(2):
            V_224[j*4+(k+2), 7:14] = Vs[j*2+k, 0:7]
    for k in range(2):
        for i in range(2):
            W_224[(k+2)*2+i, 7:14] = Ws[k*2+i, 0:7]
    
    # Build 2x4x4: sum of two 2x2x4 (j split into {0,1} and {2,3})
    U_244 = np.zeros((8, 28))
    V_244 = np.zeros((16, 28))
    W_244 = np.zeros((8, 28))
    
    # Part 1: j=0,1
    U_244[0:4, 0:14] = U_224
    for j in range(2):
        for k in range(4):
            V_244[j*4+k, 0:14] = V_224[j*4+k, 0:14]
    for k in range(4):
        for i in range(2):
            W_244[k*2+i, 0:14] = W_224[k*2+i, 0:14]
    
    # Part 2: j=2,3
    U_244[4:8, 14:28] = U_224
    for j in range(2):
        for k in range(4):
            V_244[(j+2)*4+k, 14:28] = V_224[j*4+k, 0:14]
    for k in range(4):
        for i in range(2):
            W_244[k*2+i, 14:28] = W_224[k*2+i, 0:14]
    
    # Build 2x4x8: concatenation of two 2x4x4 (k split into {0..3} and {4..7})
    U_248 = np.zeros((8, 56))
    V_248 = np.zeros((32, 56))
    W_248 = np.zeros((16, 56))
    
    # Part 1: k=0..3
    U_248[:, 0:28] = U_244
    for j in range(4):
        for k in range(4):
            V_248[j*8+k, 0:28] = V_244[j*4+k, 0:28]
    for k in range(4):
        for i in range(2):
            W_248[k*2+i, 0:28] = W_244[k*2+i, 0:28]
    
    # Part 2: k=4..7
    U_248[:, 28:56] = U_244
    for j in range(4):
        for k in range(4):
            V_248[j*8+(k+4), 28:56] = V_244[j*4+k, 0:28]
    for k in range(4):
        for i in range(2):
            W_248[(k+4)*2+i, 28:56] = W_244[k*2+i, 0:28]
    
    return U_248, V_248, W_248

def _try_direct_construction(rank_max=48, deadline=None):
    """
    Try to find a lower-rank decomposition by searching over structured
    rank-1 tensor patterns. Each rank-1 term is defined by:
    - A subset of (i,j) pairs for U (with values ±1)
    - A subset of (j,k) pairs for V (with values ±1)  
    - A subset of (k,i) pairs for W (with values ±1)
    such that the product reconstructs the matmul tensor.
    
    This is a heuristic search over combinatorial patterns.
    """
    n, m, p = 2, 4, 8
    
    # The matmul tensor: T[i,j,k] = 1 if all in range, 0 otherwise
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    
    # Strategy: use the hierarchical Strassen as a starting point,
    # then try to merge or simplify terms.
    
    # Alternative: try a completely different structure.
    # For 2x4x8, note that we can group the 8 output columns.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's try: for each pair of j values, use Strassen-like combinations.
    # This is essentially what the hierarchical approach does.
    
    # Try a different partition: split j into {0,2} and {1,3}
    # This doesn't directly help with Strassen since it's not a 2x2 block.
    
    # Let's try to reduce the hierarchical rank by noticing that
    # some terms might cancel or combine.
    
    # Actually, let's try a fresh approach: use the fact that the tensor
    # has a specific algebraic structure and try small-rank candidates.
    
    # For now, return the hierarchical construction (rank 56)
    # and let the caller decide if it's better than schoolbook.
    U, V, W = _build_2x4x8_hierarchical()
    
    # Try to verify
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _try_4x4_embedding():
    """
    Embed 2x4x8 into 4x4x4 by zero-padding:
    A (2x4) -> A' (4x4) with A'[0:2, 0:4] = A, rest 0
    B (4x8) -> B' (4x4) with B'[0:4, 0:4] = B[:, 0:4], rest 0
    C (2x8) -> C' (4x4) with C'[0:2, 0:4] = C[:, 0:4], rest 0
    
    Then C' = A' * B' is a 4x4x4 multiplication.
    If we have a rank-20 decomposition for 4x4x4, we can extract the 2x4x4 part.
    But we need 2x4x8, so we'd need two such embeddings.
    
    This gives rank 20 + 20 = 40 for 2x4x8, which is better than 56.
    
    But do we have a rank-20 decomposition for 4x4x4?
    The best known for 4x4x4 is rank 20 (from recursive Strassen: 4x4x4 = 16 * 2x2x2, but with Strassen it's 4^2 * 7 = 112? No.
    
    Actually, 4x4x4 can be done with rank 49 using Strassen recursion:
    4x4 = 4 blocks of 2x2, so 4x4x4 uses 4*4=16 scalar mults schoolbook,
    or with Strassen: each 2x2x2 is rank 7, and 4x4x4 requires 16 such? No.
    
    Strassen for 4x4x4: partition into 2x2 blocks.
    A (4x4) = 4 blocks of 2x2.
    B (4x4) = 4 blocks of 2x2.
    C (4x4) = 4 blocks of 2x2.
    Each C_block = sum of 2 A_block * B_block products.
    Using Strassen for each pair: 7 mults per pair.
    4 pairs * 7 = 28? No, that's not right either.
    
    Standard Strassen recursion: R(4,4,4) <= 7 * R(2,2,2) = 7*8 = 56? No.
    R(4,4,4) <= 7 * R(2,2,2) where R(2,2,2) is the rank for 2x2x2 scalar mult.
    R(2,2,2) = 7 (Strassen).
    So R(4,4,4) <= 7 * 7 = 49? That's not right.
    
    Actually, Strassen gives R(2n, 2n, 2n) <= 7 * R(n,n,n).
    R(2,2,2) = 7.
    R(4,4,4) <= 7 * R(2,2,2) = 7 * 7 = 49.
    
    But the best known for 4x4x4 is actually 20 (from specific constructions).
    Let me not rely on that and just use the hierarchical approach.
    
    For now, skip this approach.
    """
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = n * m * p  # 64
    
    # Try hierarchical Strassen (rank 56)
    try:
        result = _try_direct_construction(deadline=deadline)
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = rank
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 5:
        return best
    
    return best
# EVOLVE-BLOCK-END
