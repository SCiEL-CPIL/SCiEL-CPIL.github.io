import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank 60 as fallback
    rank_fallback = n * m * p
    U_fb = [[0] * rank_fallback for _ in range(n * m)]
    V_fb = [[0] * rank_fallback for _ in range(m * p)]
    W_fb = [[0] * rank_fallback for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    # Rank-55 decomposition:
    # Part 1: j in {0,1}, 2x2x10, split into 5 blocks of 2x2x2, schoolbook rank 8 each = 40
    # Part 2: j=2, 2x1x10, schoolbook rank 20
    # Total: 60, no improvement.
    
    # To get rank 55, I need to use Strassen for the 2x2x2 blocks.
    # Let me implement Strassen correctly.
    
    rank_55 = 55
    U = [[0] * rank_55 for _ in range(n * m)]
    V = [[0] * rank_55 for _ in range(m * p)]
    W = [[0] * rank_55 for _ in range(p * n)]
    
    col = 0
    
    # Strassen's 7 rank-1 terms for 2x2x2 matrix multiplication
    # U indices: i*2+j, i,j in {0,1} -> 0,1,2,3
    # V indices: j*2+k, j,k in {0,1} -> 0,1,2,3
    # W indices: k*2+i, k,i in {0,1} -> 0,1,2,3
    
    strassen_terms = [
        # (U, V, W)
        ([1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),  # M3
        ([0, 0, 0, 1], [1, -1, 0, 0], [0, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]),  # M5
        ([0, 0, 1, -1], [1, 0, 1, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, -1, 0], [0, 1, 0, 1], [1, 0, 0, 0]),  # M7
    ]
    
    # For each 2x2x2 block (i in {0,1}, j in {0,1}, k in {k0,k1})
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        for u_term, v_term, w_term in strassen_terms:
            # U: u_term[i*2+j] for i,j in {0,1}
            # Map to global U index: i*m + j = i*3 + j
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = u_term[i * 2 + j]
            
            # V: v_term[j*2+k] for j,k in {0,1}
            # Map to global V index: j*p + k
            for j in range(2):
                for k in range(2):
                    V[j * p + (k0 + k)][col] = v_term[j * 2 + k]
            
            # W: w_term[k*2+i] for k,i in {0,1}
            # Map to global W index: k*n + i
            for k in range(2):
                for i in range(2):
                    W[(k0 + k) * n + i][col] = w_term[k * 2 + i]
            
            col += 1
    
    # Part 2: j=2, 2x1x10, schoolbook rank 20
    for i in range(2):
        for k in range(10):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Verify the decomposition
    def verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": U_fb, "V": V_fb, "W": W_fb}
