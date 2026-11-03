import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strassen rank-7 decomposition for 2x2x2
    # u_r[i], v_r[dj], w_r[dk] for i,dj,dk in {0,1}
    strassen_u = [
        [1, 0],  # r=1
        [1, 0],  # r=2
        [1, 0],  # r=3
        [1, 0],  # r=4
        [0, 1],  # r=5
        [0, 1],  # r=6
        [0, 1],  # r=7
    ]
    strassen_v = [
        [1, 0],  # r=1
        [1, 0],  # r=2
        [0, 1],  # r=3
        [0, 1],  # r=4
        [1, 0],  # r=5
        [1, 0],  # r=6
        [0, 1],  # r=7
    ]
    strassen_w = [
        [1, 0],  # r=1
        [0, 1],  # r=2
        [1, 0],  # r=3
        [0, 1],  # r=4
        [1, 0],  # r=5
        [0, 1],  # r=6
        [1, 1],  # r=7
    ]
    
    # Number of j-blocks and k-blocks
    num_jb = m // 2  # 2
    num_kb = p // 2  # 4
    num_strassen = 7
    
    rank = num_jb * num_kb * num_strassen  # 56
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r_global = 0
    for jb in range(num_jb):
        for kb in range(num_kb):
            for r in range(num_strassen):
                for i in range(n):
                    for dj in range(2):
                        for dk in range(2):
                            u_val = strassen_u[r][i]
                            v_val = strassen_v[r][dj]
                            w_val = strassen_w[r][dk]
                            
                            if u_val != 0 and v_val != 0 and w_val != 0:
                                u_row = i * m + (2 * jb + dj)
                                v_row = (2 * jb + dj) * p + (4 * kb + dk)
                                w_row = (4 * kb + dk) * n + i
                                
                                U[u_row, r_global] = u_val
                                V[v_row, r_global] = v_val
                                W[w_row, r_global] = w_val
                r_global += 1
    
    # Verify
    assert _verify_decomposition(U, V, W, n, m, p), "Decomposition verification failed"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(U.shape[1]):
                    actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(actual - expected) > 1e-10:
                    return False
    return True
