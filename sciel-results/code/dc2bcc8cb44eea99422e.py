"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    r = n * m * p
    
    # Initialize matrices with zeros
    U = [[0.0 for _ in range(r)] for _ in range(n * m)]
    V = [[0.0 for _ in range(r)] for _ in range(m * p)]
    W = [[0.0 for _ in range(r)] for _ in range(p * n)]
    
    idx = 0
    for k in range(p):
        for j in range(m):
            for i in range(n):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][idx] = 1
                V[v_idx][idx] = 1
                W[w_idx][idx] = 1
                idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
