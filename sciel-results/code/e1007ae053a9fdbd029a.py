import time
import numpy as np

DIMENSIONS = [2, 5, 5]
_DEADLINE = 135.0

def _remaining():
    return _DEADLINE - time.time()

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _strassen_2x2x2(offset_i, offset_j, offset_k):
    """
    Returns U, V, W contributions for a 2x2x2 block starting at indices offset_i, offset_j, offset_k.
    The block covers i in [offset_i, offset_i+1], j in [offset_j, offset_j+1], k in [offset_k, offset_k+1].
    """
    U = []
    V = []
    W = []
    
    # Strassen 7 multiplications
    # M1 = A11 * (B11 + B22)
    # M2 = (A11 + A21) * B22
    # M3 = A21 * (B11 + B12)
    # M4 = A12 * (B21 + B22)
    # M5 = (A11 + A12) * B11
    # M6 = (A21 + A12) * (B11 + B22)
    # M7 = (A11 + A21) * (B21 + B12)
    
    # Indices:
    # A11 is (i=0, j=0) relative to offset
    # A21 is (i=1, j=0)
    # A12 is (i=0, j=1)
    # A22 is (i=1, j=1) -- not used in Strassen directly? 
    # Wait, Strassen computes C = AB.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to the tensor decomposition.
    # The tensor T[i,j,k] = 1.
    # We want sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    
    # Let's define the 7 terms.
    # Each term c has U, V, W vectors.
    
    # Term 1: M1 = A11 * (B11 + B22)
    # U[0,0] = 1, U[1,0] = 0, U[0,1] = 0, U[1,1] = 0
    # V[0,0] = 1, V[0,1] = 0, V[1,0] = 1, V[1,1] = 0  (B11 + B22)
    # W[0,0] = 1, W[0,1] = 0, W[1,0] = 0, W[1,1] = 1 (Contributes to C11 and C22)
    # Actually, W depends on k and i.
    # M1 contributes to C11 (i=0, k=0) and C22 (i=1, k=1).
    # So W[0,0] = 1, W[1,1] = 1.
    
    # Let's build the matrices.
    # U is 2x2, V is 2x2, W is 2x2 (relative to block).
    
    terms = []
    
    # 1
    u = [[1, 0], [0, 0]]
    v = [[1, 0], [1, 0]] # B11 + B22? No. B is 2x2. B11 is (0,0), B22 is (1,1).
    # V[j,k]. j is row of A, k is col of B?
    # Standard: A is n x m, B is m x p.
    # Here 2x2x2. A is 2x2, B is 2x2.
    # V[j,k] where j is index of A (col), k is index of B (row).
    # B11 is B[0,0]. B22 is B[1,1].
    # So V[0,0] = 1, V[1,1] = 1.
    v = [[1, 0], [0, 1]]
    w = [[1, 0], [0, 1]] # W[k,i]. k is row of B, i is row of A.
    # M1 contributes to C[0,0] and C[1,1].
    # C[0,0] uses i=0, k=0. W[0,0]=1.
    # C[1,1] uses i=1, k=1. W[1,1]=1.
    terms.append((u, v, w))
    
    # 2: M2 = (A11 + A21) * B22
    u = [[1, 0], [1, 0]] # A11 + A21
    v = [[0, 0], [0, 1]] # B22
    w = [[0, 0], [0, 1]] # Contributes to C21 (i=1, k=0) and C22 (i=1, k=1)?
    # C21 = M2 + M4. i=1, k=0.
    # C22 = M1 - M2 + M3 + M6. i=1, k=1.
    # So W[0,1] = 1, W[1,1] = 1.
    w = [[0, 1], [0, 1]]
    terms.append((u, v, w))
    
    # 3: M3 = A21 * (B11 + B12)
    u = [[0, 0], [1, 0]] # A21
    v = [[1, 1], [0, 0]] # B11 + B12
    w = [[0, 1], [1, 1]] # Contributes to C12 (i=0, k=1) and C22 (i=1, k=1)?
    # C12 = M3 + M5. i=0, k=1.
    # C22 = ... M3 ... i=1, k=1.
    # So W[1,0] = 1, W[1,1] = 1.
    w = [[0, 1], [1, 1]]
    terms.append((u, v, w))
    
    # 4: M4 = A12 * (B21 + B22)
    u = [[0, 1], [0, 0]] # A12
    v = [[0, 0], [1, 1]] # B21 + B22
    w = [[0, 1], [0, 1]] # Contributes to C11 (i=0, k=0) and C21 (i=1, k=0)?
    # C11 = ... + M4 ... i=0, k=0.
    # C21 = M2 + M4. i=1, k=0.
    # So W[0,0] = 1, W[0,1] = 1.
    w = [[1, 1], [0, 0]]
    terms.append((u, v, w))
    
    # 5: M5 = (A11 + A12) * B11
    u = [[1, 1], [0, 0]] # A11 + A12
    v = [[1, 0], [0, 0]] # B11
    w = [[0, 1], [1, 0]] # Contributes to C11 (i=0, k=0) and C12 (i=0, k=1)?
    # C11 = ... - M5 ... i=0, k=0.
    # C12 = ... + M5 ... i=0, k=1.
    # So W[0,0] = -1, W[1,0] = 1.
    w = [[-1, 1], [0, 0]]
    terms.append((u, v, w))
    
    # 6: M6 = (A21 + A12) * (B11 + B22)
    u = [[0, 1], [1, 0]] # A21 + A12
    v = [[1, 0], [0, 1]] # B11 + B22
    w = [[0, 0], [0, 1]] # Contributes to C22 (i=1, k=1)?
    # C22 = ... + M6 ... i=1, k=1.
    # So W[1,1] = 1.
    w = [[0, 0], [0, 1]]
    terms.append((u, v, w))
    
    # 7: M7 = (A11 + A21) * (B21 + B12)
    u = [[1, 0], [1, 0]] # A11 + A21
    v = [[0, 1], [1, 0]] # B21 + B12
    w = [[0, 1], [1, 1]] # Contributes to C11 (i=0, k=0) and C22 (i=1, k=1)?
    # C11 = ... + M7 ... i=0, k=0.
    # C22 = ... M7 ... i=1, k=1? No, C22 doesn't have M7.
    # C11 = M1 + M4 - M5 + M7.
    # So W[0,0] = 1.
    w = [[1, 0], [0, 0]]
    terms.append((u, v, w))
    
    return terms

def _construct():
    n, m, p = DIMENSIONS
    # Rank 17 construction
    # 2x4x4 block (indices j=0..3, k=0..3)
    # Split into two 2x2x2 blocks: (j=0..1, k=0..1) and (j=2..3, k=2..3)
    
    all_terms = []
    
    # Block 1: j=0,1; k=0,1
    terms1 = _strassen_2x2x2(0, 0, 0)
    all_terms.extend(terms1)
    
    # Block 2: j=2,3; k=2,3
    terms2 = _strassen_2x2x2(0, 2, 2)
    all_terms.extend(terms2)
    
    # Now we need to handle the rest of the 2x5x5 tensor.
    # The above covers i=0,1; j=0..3; k=0..3.
    # Missing:
    # 1. j=4, k=0..3 (A[:,4] * B[0..3,:]) -> 2x1x4 block
    # 2. j=0..3, k=4 (A[:,:4] * B[4,:]) -> 2x4x1 block
    # 3. j=4, k=4 (A[:,4] * B[4,:]) -> 2x1x1 block
    
    # Term for 2x1x4 (j=4, k=0..3)
    # U[i,4,c] V[4,k,c] W[k,i,c]
    # We can use 1 term.
    # U[i,4] = 1 for all i.
    # V[4,k] = 1 for all k.
    # W[k,i] = 1 for all k,i.
    # This gives 1 for all i,k.
    u = [[0]*5 for _ in range(2)]
    v = [[0]*5 for _ in range(5)]
    w = [[0]*2 for _ in range(5)]
    
    u[0][4] = 1
    u[1][4] = 1
    for k in range(4):
        v[4][k] = 1
        for i in range(2):
            w[k][i] = 1
            
    all_terms.append((u, v, w))
    
    # Term for 2x4x1 (j=0..3, k=4)
    # U[i,j,c] V[j,4,c] W[4,i,c]
    # Use 1 term.
    u = [[0]*5 for _ in range(2)]
    v = [[0]*5 for _ in range(5)]
    w = [[0]*2 for _ in range(5)]
    
    for i in range(2):
        for j in range(4):
            u[i][j] = 1
    for j in range(4):
        v[j][4] = 1
    for i in range(2):
        w[4][i] = 1
        
    all_terms.append((u, v, w))
    
    # Term for 2x1x1 (j=4, k=4)
    # U[i,4,c] V[4,4,c] W[4,i,c]
    # Use 1 term.
    u = [[0]*5 for _ in range(2)]
    v = [[0]*5 for _ in range(5)]
    w = [[0]*2 for _ in range(5)]
    
    u[0][4] = 1
    u[1][4] = 1
    v[4][4] = 1
    w[4][0] = 1
    w[4][1] = 1
    
    all_terms.append((u, v, w))
    
    # Now we have a list of (U, V, W) terms.
    # Each U is 2x5, V is 5x5, W is 5x2.
    # We need to stack them into matrices of shape (10, R), (25, R), (10, R).
    
    R = len(all_terms)
    U = [[0]*R for _ in range(10)]
    V = [[0]*R for _ in range(25)]
    W = [[0]*R for _ in range(10)]
    
    for c, (u, v, w) in enumerate(all_terms):
        for i in range(2):
            for j in range(5):
                U[i*5 + j][c] = u[i][j]
        for j in range(5):
            for k in range(5):
                V[j*5 + k][c] = v[j][k]
        for k in range(5):
            for i in range(2):
                W[k*2 + i][c] = w[k][i]
                
    return {"U": U, "V": V, "W": W}

def construct():
    try:
        result = _construct()
        n, m, p = DIMENSIONS
        if _verify(n, m, p, result['U'], result['V'], result['W']):
            return result
        else:
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][c] = 1
                        V[j * p + k][c] = 1
                        W[k * n + i][c] = 1
                        c += 1
            return {"U": U, "V": V, "W": W}
    except Exception:
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
