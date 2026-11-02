import numpy as np
import time

DIMENSIONS = [2, 2, 8]

def verify_decomposition(U, V, W, n=2, m=2, p=8):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Test with random matrices
    for _ in range(3):
        A = np.random.randint(-5, 6, (n, m)).astype(float)
        B = np.random.randint(-5, 6, (m, p)).astype(float)
        C_expected = A @ B
        
        # Compute using decomposition
        C_computed = np.zeros((n, p), dtype=float)
        for t in range(rank):
            a_val = np.dot(U[:, t], A.flatten())
            b_val = np.dot(V[:, t], B.flatten())
            c_contrib = a_val * b_val * W[:, t]
            
            # Convert c_contrib (index k*n+i) to matrix
            # c_contrib[i + k*n] = C[i, k]
            # Reshape to (n, p)
            C_computed += c_contrib.reshape((n, p))
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    
    return True

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Start with Strassen-based 28-rank decomposition
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen products for 2x2 matrix multiplication
    products = [
        {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
        {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): -1}},
        {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}},
        {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
        {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
        {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
        {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1}},
    ]
    
    for q in range(p // 2):  # 4 groups
        k1 = 2 * q
        for r, prod in enumerate(products):
            t = q * 7 + r  # term index
            
            # Fill U: A coefficients
            for (i, j), coeff in prod['A'].items():
                U[i * m + j, t] = coeff
            
            # Fill V: B coefficients
            for (j, k_sub), coeff in prod['B'].items():
                k_global = k1 + k_sub
                V[j * p + k_global, t] = coeff
            
            # Fill W: C contributions
            for (i, k_sub), coeff in prod['C'].items():
                k_global = k1 + k_sub
                W[k_global * n + i, t] = coeff

    # Verify the decomposition
    if not verify_decomposition(U.tolist(), V.tolist(), W.tolist()):
        # Fallback to schoolbook if Strassen fails
        rank = 32
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t = i * m * p + j * p + k
                    U[i * m + j, t] = 1
                    V[j * p + k, t] = 1
                    W[k * n + i, t] = 1
    
    # Try to reduce rank by looking for linear dependencies
    if time.time() < deadline - 10:
        for t_remove in range(rank - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            
            U_test = np.delete(U, t_remove, axis=1)
            V_test = np.delete(V, t_remove, axis=1)
            W_test = np.delete(W, t_remove, axis=1)
            
            if verify_decomposition(U_test.tolist(), V_test.tolist(), W_test.tolist()):
                U = U_test
                V = V_test
                W = W_test
                rank -= 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
