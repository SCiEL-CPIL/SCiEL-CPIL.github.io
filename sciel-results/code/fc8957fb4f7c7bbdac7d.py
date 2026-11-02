import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    n, m, p = 2, 4, 4
    t_start = time.time()
    deadline = 110.0  # safety margin before 120s kill

    # We use 4 Strassen blocks of rank 7
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(u_map, v_map, w_map, base_col):
        # M1: (X00+X11)*(Y00+Y11)
        U[u_map[(0,0)]][base_col + 0] += 1
        U[u_map[(1,1)]][base_col + 0] += 1
        V[v_map[(0,0)]][base_col + 0] += 1
        V[v_map[(1,1)]][base_col + 0] += 1
        W[w_map[(0,0)]][base_col + 0] += 1
        W[w_map[(1,1)]][base_col + 0] += 1
        
        # M2: (X10+X11)*Y00
        U[u_map[(1,0)]][base_col + 1] += 1
        U[u_map[(1,1)]][base_col + 1] += 1
        V[v_map[(0,0)]][base_col + 1] += 1
        W[w_map[(1,0)]][base_col + 1] += 1
        W[w_map[(1,1)]][base_col + 1] -= 1
        
        # M3: X00*(Y01-Y11)
        U[u_map[(0,0)]][base_col + 2] += 1
        V[v_map[(0,1)]][base_col + 2] += 1
        V[v_map[(1,1)]][base_col + 2] -= 1
        W[w_map[(0,1)]][base_col + 2] += 1
        W[w_map[(1,1)]][base_col + 2] += 1
        
        # M4: X11*(Y10-Y00)
        U[u_map[(1,1)]][base_col + 3] += 1
        V[v_map[(1,0)]][base_col + 3] += 1
        V[v_map[(0,0)]][base_col + 3] -= 1
        W[w_map[(0,0)]][base_col + 3] += 1
        W[w_map[(1,0)]][base_col + 3] += 1
        
        # M5: (X00+X01)*Y11
        U[u_map[(0,0)]][base_col + 4] += 1
        U[u_map[(0,1)]][base_col + 4] += 1
        V[v_map[(1,1)]][base_col + 4] += 1
        W[w_map[(0,0)]][base_col + 4] -= 1
        W[w_map[(0,1)]][base_col + 4] += 1
        
        # M6: (X10-X00)*(Y00+Y01)
        U[u_map[(1,0)]][base_col + 5] += 1
        U[u_map[(0,0)]][base_col + 5] -= 1
        V[v_map[(0,0)]][base_col + 5] += 1
        V[v_map[(0,1)]][base_col + 5] += 1
        W[w_map[(1,1)]][base_col + 5] += 1
        
        # M7: (X01-X11)*(Y10+Y11)
        U[u_map[(0,1)]][base_col + 6] += 1
        U[u_map[(1,1)]][base_col + 6] -= 1
        V[v_map[(1,0)]][base_col + 6] += 1
        V[v_map[(1,1)]][base_col + 6] += 1
        W[w_map[(0,0)]][base_col + 6] += 1

    # Instance 1: A1 * B11 -> C_left (cols 0,1)
    u1 = {(0,0): 0, (0,1): 1, (1,0): 4, (1,1): 5}
    v1 = {(0,0): 0, (0,1): 1, (1,0): 4, (1,1): 5}
    w1 = {(0,0): 0, (0,1): 2, (1,0): 1, (1,1): 3}
    add_strassen_block(u1, v1, w1, 0)
    
    # Instance 2: A2 * B21 -> C_left (cols 0,1)
    u2 = {(0,0): 2, (0,1): 3, (1,0): 6, (1,1): 7}
    v2 = {(0,0): 8, (0,1): 9, (1,0): 12, (1,1): 13}
    w2 = w1
    add_strassen_block(u2, v2, w2, 7)
    
    # Instance 3: A1 * B12 -> C_right (cols 2,3)
    u3 = u1
    v3 = {(0,0): 2, (0,1): 3, (1,0): 6, (1,1): 7}
    w3 = {(0,0): 4, (0,1): 6, (1,0): 5, (1,1): 7}
    add_strassen_block(u3, v3, w3, 14)
    
    # Instance 4: A2 * B22 -> C_right (cols 2,3)
    u4 = u2
    v4 = {(0,0): 10, (0,1): 11, (1,0): 14, (1,1): 15}
    w4 = w3
    add_strassen_block(u4, v4, w4, 21)

    # Verify correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Target tensor: T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Decomposed tensor
    T_decomp = np.zeros_like(T)
    for r in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    if not np.allclose(T, T_decomp, atol=1e-10):
        # Fallback to naive rank-32
        rank_naive = n * m * p
        U_naive = [[0] * rank_naive for _ in range(n * m)]
        V_naive = [[0] * rank_naive for _ in range(m * p)]
        W_naive = [[0] * rank_naive for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_naive[i*m+j][r] = 1
                    V_naive[j*p+k][r] = 1
                    W_naive[k*n+i][r] = 1
        return {"U": U_naive, "V": V_naive, "W": W_naive}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
