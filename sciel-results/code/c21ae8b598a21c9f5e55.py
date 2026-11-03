import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # beta[s][kA][lA]
    beta = [
        [[1, 0], [0, 1]],  # M0
        [[0, 0], [1, 1]],  # M1
        [[1, 0], [0, 0]],  # M2
        [[0, 0], [0, 1]],  # M3
        [[1, 1], [0, 0]],  # M4
        [[-1, 0], [1, 0]], # M5
        [[0, 1], [0, -1]]  # M6
    ]
    
    # gamma[s][kB][lB]
    gamma = [
        [[1, 0], [0, 1]],  # M0
        [[1, 0], [0, 0]],  # M1
        [[0, 0], [1, -1]], # M2
        [[-1, 1], [0, 0]], # M3
        [[0, 0], [0, 1]],  # M4
        [[1, 1], [0, 0]],  # M5
        [[0, 0], [1, 1]]   # M6
    ]
    
    # alpha[s][kC][lC]
    alpha = [
        [[1, 0], [0, 1]],  # M0 -> C00, C11
        [[0, 0], [1, -1]], # M1 -> C10, C11
        [[0, 1], [0, 1]],  # M2 -> C01, C11
        [[1, 0], [1, 0]],  # M3 -> C00, C10
        [[-1, 1], [0, 0]], # M4 -> C00, C01
        [[0, 0], [0, 1]],  # M5 -> C11
        [[1, 0], [0, 0]]   # M6 -> C00
    ]
    
    col = 0
    for s in range(7):
        for r in range(3):
            for c in range(3):
                for d in range(3):
                    # Update U
                    for kA in range(2):
                        for lA in range(2):
                            val = beta[s][kA][lA]
                            if val != 0:
                                row_u = (kA * 3 + r) * 6 + (lA * 3 + d)
                                U[row_u][col] = val
                                
                    # Update V
                    for kB in range(2):
                        for lB in range(2):
                            val = gamma[s][kB][lB]
                            if val != 0:
                                row_v = (kB * 3 + d) * 6 + (lB * 3 + c)
                                V[row_v][col] = val
                                
                    # Update W
                    for kC in range(2):
                        for lC in range(2):
                            val = alpha[s][kC][lC]
                            if val != 0:
                                row_w = (lC * 3 + c) * 6 + (kC * 3 + r)
                                W[row_w][col] = val
                                
                    col += 1
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time taken: {end - start:.2f}s")
    # Simple verification
    n, m, p = 6, 6, 6
    R = len(res["U"][0])
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Random test
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    
    # Flatten A and B
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    # Calculate C
    C = A @ B
    C_flat = C.flatten()
    
    # Calculate using decomposition
    # C_flat[i*p + j] = sum_r W[j*n + i, r] * U[i*m + k, r] * V[k*p + j, r]
    # This is effectively:
    # C = (W^T @ diag(U^T @ ... ))? No.
    # C_{i,j} = sum_k sum_r U_{i,k}^{flat} V_{k,j}^{flat} ... 
    # Let's just do the tensor contraction.
    # U is (n*m, R), V is (m*p, R), W is (p*n, R)
    # C_{i,j} = sum_{k=0}^{m-1} sum_{r=0}^{R-1} U[i*m+k, r] V[k*p+j, r] W[j*n+i, r]
    
    C_calc = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            val = 0
            for k in range(m):
                for r in range(R):
                    val += U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
            C_calc[i, j] = val
            
    err = np.max(np.abs(C - C_calc))
    print(f"Max error: {err}")
    if err < 1e-9:
        print("Decomposition is valid.")
    else:
        print("Decomposition is INVALID.")
