import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Build the target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # We will construct the decomposition for 3x3x3 and 3x3x2 parts separately
    # and then concatenate them.
    
    # Part 1: 3x3x3 (k=0,1,2)
    # We use the current Strassen-based decomposition with rank 26.
    # If we can find a better one, we will use it.
    
    def add_strassen_block(i_start, j_start, k_start, U, V, W, current_col):
        def idx_A(i, j): return (i_start + i) * m + (j_start + j)
        def idx_B(j, k): return (j_start + j) * p + (k_start + k)
        def idx_C(k, i): return (k_start + k) * n + (i_start + i)
        
        U_block = np.zeros((n*m, 7))
        V_block = np.zeros((m*p, 7))
        W_block = np.zeros((p*n, 7))
        
        # M1
        U_block[idx_A(0,0), 0] = 1; U_block[idx_A(1,1), 0] = 1
        V_block[idx_B(0,0), 0] = 1; V_block[idx_B(1,1), 0] = 1
        W_block[idx_C(0,0), 0] = 1; W_block[idx_C(1,1), 0] = 1
        
        # M2
        U_block[idx_A(1,0), 1] = 1; U_block[idx_A(1,1), 1] = 1
        V_block[idx_B(0,0), 1] = 1
        W_block[idx_C(0,1), 1] = 1; W_block[idx_C(1,1), 1] = -1
        
        # M3
        U_block[idx_A(0,0), 2] = 1
        V_block[idx_B(1,0), 2] = 1; V_block[idx_B(1,1), 2] = -1
        W_block[idx_C(0,1), 2] = 1; W_block[idx_C(1,1), 2] = 1
        
        # M4
        U_block[idx_A(1,1), 3] = 1
        V_block[idx_B(0,1), 3] = 1; V_block[idx_B(0,0), 3] = -1
        W_block[idx_C(0,0), 3] = 1; W_block[idx_C(0,1), 3] = 1
        
        # M5
        U_block[idx_A(0,0), 4] = 1; U_block[idx_A(0,1), 4] = 1
        V_block[idx_B(1,1), 4] = 1
        W_block[idx_C(0,0), 4] = -1; W_block[idx_C(0,1), 4] = 1
        
        # M6
        U_block[idx_A(1,1), 5] = 1; U_block[idx_A(0,0), 5] = -1
        V_block[idx_B(0,0), 5] = 1; V_block[idx_B(1,0), 5] = 1
        W_block[idx_C(1,1), 5] = 1
        
        # M7
        U_block[idx_A(0,1), 6] = 1; U_block[idx_A(1,1), 6] = -1
        V_block[idx_B(1,1), 6] = 1; V_block[idx_B(0,1), 6] = 1
        W_block[idx_C(0,0), 6] = 1
        
        U[:, current_col:current_col+7] = U_block
        V[:, current_col:current_col+7] = V_block
        W[:, current_col:current_col+7] = W_block
        
        return current_col + 7

    # Part 1: 3x3x3
    U1 = np.zeros((n*m, 26))
    V1 = np.zeros((m*p, 26))
    W1 = np.zeros((p*n, 26))
    col1 = 0
    col1 = add_strassen_block(0, 0, 0, U1, V1, W1, col1)
    
    for i in range(n):
        for j in range(m):
            for k in range(3):
                if i in [0,1] and j in [0,1] and k in [0,1]:
                    continue
                U1[i*m+j, col1] = 1
                V1[j*p+k, col1] = 1
                W1[k*n+i, col1] = 1
                col1 += 1
                
    # Part 2: 3x3x2 (k=3,4)
    U2 = np.zeros((n*m, 17))
    V2 = np.zeros((m*p, 17))
    W2 = np.zeros((p*n, 17))
    col2 = 0
    col2 = add_strassen_block(0, 0, 3, U2, V2, W2, col2)
    
    for i in range(n):
        for j in range(m):
            for k in range(3, 5):
                if i in [0,1] and j in [0,1] and (k-3) in [0,1]:
                    continue
                U2[i*m+j, col2] = 1
                V2[j*p+k, col2] = 1
                W2[k*n+i, col2] = 1
                col2 += 1
                
    # Combine
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    
    # Verify
    approx = np.zeros_like(T)
    for r in range(U.shape[1]):
        approx += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
    if not np.allclose(T, approx, atol=1e-10):
        # Fallback to schoolbook
        U_sch = np.zeros((n*m, 45))
        V_sch = np.zeros((m*p, 45))
        W_sch = np.zeros((p*n, 45))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sch[i*m+j, col] = 1
                    V_sch[j*p+k, col] = 1
                    W_sch[k*n+i, col] = 1
                    col += 1
        return {"U": U_sch.tolist(), "V": V_sch.tolist(), "W": W_sch.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
