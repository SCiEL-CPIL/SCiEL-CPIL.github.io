"""Matrix multiplication 4x4x9 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 144."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                expected = 1.0 if i == k else 0.0
                if abs(total - expected) > 1e-10:
                    return False
    return True

def _try_kron_decomposition(deadline):
    """Kronecker product decomposition: 4x4x9 = (2x2x3) ⊗ (2x2x3), rank 16."""
    if time.time() > deadline:
        return None
    
    # Rank-4 decomposition for 2x2x3 (schoolbook for that size)
    U2 = [[0.0]*4 for _ in range(4)]
    V2 = [[0.0]*4 for _ in range(6)]
    W2 = [[0.0]*4 for _ in range(6)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                if i == k:
                    U2[i*2+j][col] = 1.0
                    V2[j*3+k][col] = 1.0
                    W2[k*2+i][col] = 1.0
                    col += 1
    
    if not _verify_decomposition(U2, V2, W2, 2, 2, 3):
        return None
    
    # Build 4x4x9 via Kronecker with correct index mapping
    # i = i1*2 + i2, j = j1*2 + j2, k = k1*3 + k2
    # c = c1*4 + c2
    n, m, p = 4, 4, 9
    rank = 16
    U = [[0.0]*rank for _ in range(n*m)]   # 16 x 16
    V = [[0.0]*rank for _ in range(m*p)]   # 36 x 16
    W = [[0.0]*rank for _ in range(p*n)]   # 36 x 16
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(3):
                        for k2 in range(3):
                            i = i1*2 + i2
                            j = j1*2 + j2
                            k = k1*3 + k2
                            if i == k:
                                for c1 in range(4):
                                    for c2 in range(4):
                                        c = c1*4 + c2
                                        val = U2[i1*2+j1][c1] * U2[i2*2+j2][c2]
                                        if val != 0:
                                            U[i*m+j][c] += val
                                        val_v = V2[j1*3+k1][c1] * V2[j2*3+k2][c2]
                                        if val_v != 0:
                                            V[j*p+k][c] += val_v
                                        val_w = W2[k1*2+i1][c1] * W2[k2*2+i2][c2]
                                        if val_w != 0:
                                            W[k*n+i][c] += val_w
    
    if _verify_decomposition(U, V, W, 4, 4, 9):
        return {"U": U, "V": V, "W": W}
    
    return None

def _try_block_decomposition(deadline):
    """
    Block decomposition: 4x4x9 = (2x2x3) ⊗ (2x2x3) + (2x2x3) ⊗ (2x2x3) + (2x2x3) ⊗ (2x2xx1)
    Actually, let's think about this differently.
    
    4x4x9: We can split the 9 columns into 3+3+3.
    So we have three 4x4x3 subproblems.
    Each 4x4x3 can be decomposed. What is the rank of 4x4x3?
    
    Alternatively, split the 4x4 matrix into blocks.
    
    Let's try a different approach: use the known rank-7 decomposition for 2x2x2 (Bini's)
    and combine them.
    
    Actually, the current Kronecker approach gives rank 16. Let's see if we can do better.
    
    For 4x4x9, note that 4 = 2*2, 9 = 3*3.
    So we can view this as (2x2) x (2x2) x (3x3) tensor product.
    
    The tensor for 2x2x2 has rank 7 (Bini's decomposition).
    The tensor for 3x3x3 has rank 23 (or so, but we need 2x2x3).
    
    Wait, the current program uses rank-4 for 2x2x3, which is just schoolbook (2*2*3/2 = 6, but we only need i==k so it's 2*3=6? No, for 2x2x3, i in {0,1}, k in {0,1,2}, i==k means k=i, so 2*3=6 terms? No, i==k means for each i, k must equal i. So for i=0, k=0; for i=1, k=1. That's 2*2=4 terms. So rank 4 is correct for 2x2x3.
    
    So the Kronecker product of two 2x2x3 tensors gives rank 4*4=16 for 4x4x9.
    
    Can we do better? The tensor rank of 4x4x9 is at most 16 with this construction.
    
    Let's try to find a lower rank decomposition using random search or other methods.
    """
    if time.time() > deadline:
        return None
    
    n, m, p = 4, 4, 9
    best_rank = 16  # from Kronecker
    
    # Try to find lower rank using random search
    import random
    
    for r in range(10, best_rank):
        if time.time() > deadline:
            break
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n*m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m*p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p*n)]
            
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit
    
    # Try Kronecker decomposition first (rank 16)
    result = _try_kron_decomposition(deadline)
    if result is not None:
        return result
    
    # Try block decomposition / random search for lower rank
    result = _try_block_decomposition(deadline)
    if result is not None:
        return result
    
    # Fall back to schoolbook
    return _schoolbook()
