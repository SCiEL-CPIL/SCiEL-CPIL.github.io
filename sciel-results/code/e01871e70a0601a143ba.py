import time
import numpy as np

DIMENSIONS = [2, 2, 8]

def _verify_decomposition(U, V, W, n, m, p):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    # Use einsum for speed
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                expected[a, b, c] = 1
    
    return np.allclose(T, expected, atol=1e-9)

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # 100s guard
    
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen-like decomposition for 4 blocks of 2x2
    # Block b corresponds to columns 2b and 2b+1 of B
    for b in range(4):
        base_col = b * 7
        
        # Indices for V (m*p = 16)
        # Block b has columns 2b, 2b+1
        # V index = j*p + k
        # j=0: 0*8 + 2b = 2b
        # j=0: 0*8 + 2b+1 = 2b+1
        # j=1: 1*8 + 2b = 8+2b
        # j=1: 1*8 + 2b+1 = 8+2b+1
        v00 = 2*b
        v01 = 2*b+1
        v10 = 8+2*b
        v11 = 8+2*b+1
        
        # Indices for W (p*n = 16)
        # W index = k*n + i
        # k=2b: 2b*2 + 0 = 4b
        # k=2b: 2b*2 + 1 = 4b+1
        # k=2b+1: (2b+1)*2 + 0 = 4b+2
        # k=2b+1: (2b+1)*2 + 1 = 4b+3
        w00 = 4*b
        w10 = 4*b+1
        w01 = 4*b+2
        w11 = 4*b+3
        
        # Term 0
        U[0][base_col] = 1
        U[3][base_col] = 1
        V[v00][base_col] = 1
        V[v11][base_col] = 1
        W[w00][base_col] = 1
        W[w11][base_col] = 1
        
        # Term 1
        U[2][base_col+1] = 1
        U[3][base_col+1] = 1
        V[v00][base_col+1] = 1
        W[w10][base_col+1] = 1
        W[w11][base_col+1] = -1
        
        # Term 2
        U[0][base_col+2] = 1
        V[v01][base_col+2] = 1
        V[v11][base_col+2] = -1
        W[w01][base_col+2] = 1
        W[w11][base_col+2] = 1
        
        # Term 3
        U[3][base_col+3] = 1
        V[v10][base_col+3] = 1
        V[v00][base_col+3] = -1
        W[w00][base_col+3] = 1
        W[w10][base_col+3] = 1
        
        # Term 4
        U[0][base_col+4] = 1
        U[1][base_col+4] = 1
        V[v11][base_col+4] = 1
        W[w00][base_col+4] = -1
        W[w01][base_col+4] = 1
        
        # Term 5
        U[2][base_col+5] = 1
        U[0][base_col+5] = -1
        V[v00][base_col+5] = 1
        V[v01][base_col+5] = 1
        W[w11][base_col+5] = 1
        
        # Term 6
        U[1][base_col+6] = 1
        U[3][base_col+6] = -1
        V[v10][base_col+6] = 1
        V[v11][base_col+6] = 1
        W[w00][base_col+6] = 1
        
        if time.time() > deadline:
            break
            
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if Strassen fails (should not happen)
        rank = 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][idx] = 1
                    V[b][idx] = 1
                    W[c][idx] = 1
                    idx += 1
        # Re-verify fallback
        if not _verify_decomposition(U, V, W, n, m, p):
            raise ValueError("Failed to find valid decomposition")

    return {"U": U, "V": V, "W": W}
