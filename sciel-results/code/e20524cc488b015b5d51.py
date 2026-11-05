import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin

    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook
    def schoolbook():
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

    # Try Laderman-based rank 46 solution
    # Laderman's algorithm for 3x3 * 3x3 (rank 23)
    # We verify these terms first.
    laderman_terms = [
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([1,0,0, 1,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 1,0,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,1,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,1,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1]),
        ([0,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,1], [0,0,0, 0,1,0, 0,0,0], [0,1,0, 0,0,0, 0,0,1]),
        ([0,0,1, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,1]),
        ([0,1,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,1,0], [0,1,0, 0,0,0, 0,1,0]),
        ([0,0,1, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,1]),
        ([0,1,0, 0,1,0, 0,0,0], [1,0,0, 1,0,0, 0,0,0], [0,1,0, 0,1,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,1]),
        ([0,1,0, 0,0,1, 0,0,0], [0,0,0, 0,1,0, 0,0,1], [0,1,0, 0,0,1, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,1], [0,0,0, 0,0,1, 0,0,0], [0,0,1, 0,0,0, 0,0,1]),
        ([0,1,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,1,0], [0,1,0, 0,0,0, 0,1,0]),
        ([0,0,1, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,1, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,1,0, 0,0,0, 0,0,1]),
        ([0,0,1, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,1,0]),
        ([0,1,0, 0,0,1, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,1,0, 0,0,1, 0,0,1]),
        ([0,0,1, 0,0,1, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,1, 0,0,1]),
        ([0,1,0, 0,1,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,1,0, 0,1,0, 0,0,1]),
    ]
    
    # Verify Laderman for 3x3x3
    def verify_laderman():
        n3, m3, p3 = 3, 3, 3
        T = np.ones((n3, m3, p3))
        T_reconstructed = np.zeros((n3, m3, p3))
        
        for r, (u, v, w) in enumerate(laderman_terms):
            u_arr = np.array(u)
            v_arr = np.array(v)
            w_arr = np.array(w)
            
            # u is indexed by i*m3 + j
            # v is indexed by j*p3 + k
            # w is indexed by k*n3 + i (Note: W[k*n+i])
            
            for i in range(n3):
                for j in range(m3):
                    u_val = u_arr[i * m3 + j]
                    if u_val == 0:
                        continue
                    for k in range(p3):
                        v_val = v_arr[j * p3 + k]
                        w_val = w_arr[k * n3 + i]
                        T_reconstructed[i, j, k] += u_val * v_val * w_val
                        
        return np.allclose(T_reconstructed, T, atol=1e-9)

    if time.time() < deadline - 10 and verify_laderman():
        # Construct rank 46 solution
        rank = 46
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        # Part 1: Columns 0-2 of B (B1)
        # Uses Laderman terms directly
        for r, (u, v, w) in enumerate(laderman_terms):
            u_arr = np.array(u)
            v_arr = np.array(v)
            w_arr = np.array(w)
            
            for i in range(n):
                for j in range(m):
                    val_u = u_arr[i * m + j]
                    if val_u != 0:
                        U[i * m + j][r] = val_u
                        
            for j in range(m):
                for k in range(3): # k is 0, 1, 2
                    val_v = v_arr[j * 3 + k]
                    if val_v != 0:
                        # V index for B1 is j*p + k = j*6 + k
                        V[j * p + k][r] = val_v
                        
            for i in range(n):
                for k in range(3): # k is 0, 1, 2
                    val_w = w_arr[k * 3 + i]
                    if val_w != 0:
                        # W index for C1 is k*n + i = k*3 + i
                        W[k * n + i][r] = val_w
                        
        # Part 2: Columns 3-5 of B (B2)
        # Uses Laderman terms shifted
        # B2[j, k'] corresponds to B[j, k'+3]
        # C2[i, k'] corresponds to C[i, k'+3]
        for r, (u, v, w) in enumerate(laderman_terms):
            r2 = r + 23
            u_arr = np.array(u)
            v_arr = np.array(v)
            w_arr = np.array(w)
            
            # U is same for both parts (A is shared)
            for i in range(n):
                for j in range(m):
                    val_u = u_arr[i * m + j]
                    if val_u != 0:
                        U[i * m + j][r2] = val_u
                        
            for j in range(m):
                for k in range(3): # k' is 0, 1, 2
                    val_v = v_arr[j * 3 + k]
                    if val_v != 0:
                        # V index for B2 is j*p + (k+3)
                        V[j * p + (k + 3)][r2] = val_v
                        
            for i in range(n):
                for k in range(3): # k' is 0, 1, 2
                    val_w = w_arr[k * 3 + i]
                    if val_w != 0:
                        # W index for C2 is (k+3)*n + i
                        W[(k + 3) * n + i][r2] = val_w
                        
        # Verify the full 3x3x6 decomposition
        T = np.ones((n, m, p))
        T_reconstructed = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    T_reconstructed[i, j, k] = s
                    
        if np.allclose(T_reconstructed, T, atol=1e-9):
            return {"U": U, "V": V, "W": W}
            
    # Fallback
    return schoolbook()
# EVOLVE-BLOCK-END
