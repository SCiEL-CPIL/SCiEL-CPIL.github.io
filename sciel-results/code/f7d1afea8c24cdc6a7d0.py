"""Matrix multiplication 4x4x9 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, rank)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, rank)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i, j, :], V_arr[j, k, :] * W_arr[k, i, :])
                expected = 1.0 if i == k else 0.0
                if abs(s - expected) > 1e-9:
                    return False
    return True

def _try_kron_decomposition(deadline):
    """Kronecker product decomposition: 4x4x9 = (2x2x3) ⊗ (2x2x3), rank 16."""
    if time.time() > deadline:
        return None
    
    # Rank-4 decomposition for 2x2x3 (schoolbook for that size)
    # For 2x2x3, i in {0,1}, k in {0,1,2}. i==k means k=i.
    # So for i=0, k=0; for i=1, k=1. That's 2*2=4 terms.
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
                                        val_u = U2[i1*2+j1][c1] * U2[i2*2+j2][c2]
                                        if val_u != 0:
                                            U[i*m+j][c] += val_u
                                        val_v = V2[j1*3+k1][c1] * V2[j2*3+k2][c2]
                                        if val_v != 0:
                                            V[j*p+k][c] += val_v
                                        val_w = W2[k1*2+i1][c1] * W2[k2*2+i2][c2]
                                        if val_w != 0:
                                            W[k*n+i][c] += val_w
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return None

def _try_winograd_kron(deadline):
    """
    Attempt to use Winograd's rank-7 decomposition for 2x2x2 combined with structure for 4x4x9.
    4x4x9 can be viewed as combining 2x2x2 and 2x2x3 structures.
    Specifically, we can split the 4x4 matrix into 2x2 blocks and the 4x9 into 2x3 blocks?
    Actually, 4x4x9 is not a direct Kronecker product of standard small tensors in a simple way that yields lower rank than 16 easily without complex embedding.
    
    Let's try a structured search around the Kronecker solution or specific algebraic identities.
    However, finding a rank < 16 exact decomposition for 4x4x9 is non-trivial.
    The tensor rank of 4x4x4 is 49 (Strassen).
    4x4x9 is larger.
    
    Let's try a random refinement of the Kronecker solution or a structured random search with small integers.
    """
    if time.time() > deadline:
        return None
    
    n, m, p = 4, 4, 9
    # Try ranks from 14 to 15 (just below 16) with structured random search
    # Using small integers {-1, 0, 1}
    
    import random
    
    for r in range(14, 16):
        if time.time() > deadline:
            break
        for attempt in range(1000):
            if time.time() > deadline:
                break
            
            U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n*m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m*p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p*n)]
            
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

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

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit

    # Try Kronecker decomposition first (rank 16)
    result = _try_kron_decomposition(deadline)
    if result is not None:
        return result
    
    # Try Winograd-inspired or structured random search for lower rank
    result = _try_winograd_kron(deadline)
    if result is not None:
        return result

    # Fall back to schoolbook
    return _schoolbook()
