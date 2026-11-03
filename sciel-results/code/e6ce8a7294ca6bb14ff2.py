import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 6]

def _schoolbook():
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

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    T_reconstructed = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_reconstructed += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                if T_reconstructed[a,b,c] != 1:
                    return False
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                is_valid = (j_a == j_b) and (k_b == k_c) and (i_a == i_c)
                if is_valid:
                    if T_reconstructed[a,b,c] != 1:
                        return False
                else:
                    if T_reconstructed[a,b,c] != 0:
                        return False
    return True

def _find_2x3x3_rank9():
    n, m, p = 2, 3, 3
    target_rank = 9
    U_shape = n*m
    V_shape = m*p
    W_shape = p*n
    
    # Random search
    # This is a heuristic.
    start_time = time.time()
    deadline = start_time + 10 # 10 seconds for this search
    
    while time.time() < deadline:
        # Random initialization with small integers
        U = np.random.randint(-2, 3, (U_shape, target_rank))
        V = np.random.randint(-2, 3, (V_shape, target_rank))
        W = np.random.randint(-2, 3, (W_shape, target_rank))
        
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
            
    return None

def construct():
    # Try to find rank 18 for 2x3x6
    # By finding rank 9 for 2x3x3 and duplicating
    
    result_2x3x3 = _find_2x3x3_rank9()
    
    if result_2x3x3:
        U_9, V_9, W_9 = result_2x3x3
        # Duplicate for 2x3x6
        n, m, p = 2, 3, 6
        rank = 18
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # First 9 terms for B1 (cols 0-2)
        for r in range(9):
            for i in range(n*m):
                U[i][r] = U_9[i][r]
            for j in range(m):
                for k in range(3):
                    V[j*6 + k][r] = V_9[j*3 + k][r]
            for k in range(3):
                for i in range(n):
                    W[k*2 + i][r] = W_9[k*2 + i][r]
                    
        # Next 9 terms for B2 (cols 3-5)
        for r in range(9):
            for i in range(n*m):
                U[i][9+r] = U_9[i][r]
            for j in range(m):
                for k in range(3):
                    V[j*6 + (3+k)][9+r] = V_9[j*3 + k][r]
            for k in range(3):
                for i in range(n):
                    W[(3+k)*2 + i][9+r] = W_9[k*2 + i][r]
                    
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
            
    return _schoolbook()
