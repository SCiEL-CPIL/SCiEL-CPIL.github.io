import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def find_3x3x2_decomposition(timeout=110):
    n, m, p = 3, 3, 2
    target_rank = 13
    start_time = time.time()
    
    for attempt in range(5000):
        if time.time() - start_time > timeout:
            break
            
        U = np.random.randint(-2, 3, (n*m, target_rank))
        V = np.random.randint(-2, 3, (m*p, target_rank))
        W = np.random.randint(-2, 3, (p*n, target_rank))
        
        # Efficient check
        # T[i, k, j] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We want T[i, k, j] == 1 for all i, k, j
        
        # Reshape to allow matrix multiplication
        # U is 9x13, V is 6x13, W is 6x13
        # This is not a simple matrix multiplication.
        
        # Iterate over j
        valid = True
        for j in range(m):
            # U_j is 3x13 (rows i)
            # V_j is 2x13 (rows k)
            # W_j is 6x13? No, W depends on k and i.
            # W[k*n + i, r]
            
            # Let's compute the tensor T for this j
            # T[i, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            
            # We can compute this for all i, k
            for i in range(n):
                for k in range(p):
                    val = 0
                    for r in range(target_rank):
                        val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    if val != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
                
        if valid:
            return U, V, W
            
    return None

def construct():
    sub_U, sub_V, sub_W = find_3x3x2_decomposition(timeout=110)
    
    if sub_U is None:
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
    
    n, m, p = DIMENSIONS
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for b in range(3):
        offset = b * 13
        for r in range(13):
            for i in range(n):
                for j in range(m):
                    U[i*m + j][offset + r] = int(sub_U[i*m + j, r])
            
            for j in range(m):
                for k_sub in range(2):
                    k_full = 2*b + k_sub
                    V[j*p + k_full][offset + r] = int(sub_V[j*2 + k_sub, r])
            
            for k_sub in range(2):
                for i in range(n):
                    k_full = 2*b + k_sub
                    W[k_full*n + i][offset + r] = int(sub_W[k_sub*3 + i, r])
                    
    return {"U": U, "V": V, "W": W}
