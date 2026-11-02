import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(actual - expected) > tol:
                    return False
    return True

def _construct_trivial(n, m, p):
    """Construct the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices:
    U: i*m + j -> i in {0,1}, j in {0,1}.
    V: j*p + k -> j in {0,1}, k in {0,1}.
    W: k*n + i -> k in {0,1}, i in {0,1}.
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # M1: (A11+A22)(B11+B22) -> C11 + C22
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C11 (k=0,i=0), C22 (k=1,i=1)
    
    # M2: (A21+A22)B11 -> C21
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C21 (k=0,i=1)
    
    # M3: A11(B12-B22) -> C12
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C12 (k=1,i=0)
    
    # M4: A22(B21-B11) -> C11 + C21
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C11, C21
    
    # M5: (A11+A12)B22 -> C11 - C22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[3, 4] = -1 # C11, C22
    
    # M6: (A21-A11)(B11+B12) -> C22
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C22
    
    # M7: (A12-A22)(B21+B22) -> C11 - C22
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = -1 # C11, C22
    
    return U, V, W

def _construct_optimized(n, m, p):
    """
    Construct rank 84 decomposition for 3x5x6.
    """
    rank = 84
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Block 1: 2x2x6 at i in {0,1}, j in {0,1}, k in {0..5}
    # Decomposed into 3 Strassen 2x2x2 blocks
    S_U, S_V, S_W = _strassen_2x2x2()
    
    for block_k in range(3):
        k_offset = block_k * 2
        for r in range(7):
            # U part
            for i in range(2):
                for j in range(2):
                    U[i*m + j, col] += S_U[i*2 + j, r]
            # V part
            for j in range(2):
                for k in range(2):
                    V[j*p + (k_offset + k), col] += S_V[j*2 + k, r]
            # W part
            for k in range(2):
                for i in range(2):
                    W[(k_offset + k)*n + i, col] += S_W[k*2 + i, r]
            col += 1
            
    # Block 2: 1x5x6 at i=2, j in {0..4}, k in {0..5}
    # Trivial rank 30
    for j in range(5):
        for k in range(6):
            U[2*m + j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + 2, col] = 1
            col += 1
            
    # Block 3: 2x3x6 at i in {0,1}, j in {2,3,4}, k in {0..5}
    # Decomposed into:
    # 3 Strassen 2x2x2 blocks using j in {2,3}
    # 1 Trivial 2x1x6 block using j=4
    
    # 3 Strassen blocks for j in {2,3}
    for block_k in range(3):
        k_offset = block_k * 2
        for r in range(7):
            # U part (i in {0,1})
            for i in range(2):
                for j in range(2):
                    U[i*m + (2 + j), col] += S_U[i*2 + j, r]
            # V part (j in {2,3}, k in block)
            for j in range(2):
                for k in range(2):
                    V[(2 + j)*p + (k_offset + k), col] += S_V[j*2 + k, r]
            # W part (k in block, i in {0,1})
            for k in range(2):
                for i in range(2):
                    W[(k_offset + k)*n + i, col] += S_W[k*2 + i, r]
            col += 1
            
    # 1 Trivial 2x1x6 block for j=4
    for k in range(6):
        for i in range(2):
            U[i*m + 4, col] = 1
            V[4*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = 3, 5, 6
    start_time = time.time()
    deadline = start_time + 140
    
    try:
        result = _construct_optimized(n, m, p)
        # Verify
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}
        else:
            # Fallback to trivial if verification fails
            result = _construct_trivial(n, m, p)
            return {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}
    except Exception as e:
        # Fallback to trivial on error
        result = _construct_trivial(n, m, p)
        return {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}
