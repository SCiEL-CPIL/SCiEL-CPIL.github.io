import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Strategy: Use 6 blocks of 2x2x2 Strassen decomposition.
    # This is the standard optimal known decomposition for this specific problem structure 
    # given the constraints, yielding rank 42.
    
    num_blocks = p // 2  # 6 blocks
    rank = num_blocks * 7  # 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(num_blocks):
        if time.time() > deadline:
            break
        base = t * 7
        
        v_b00 = 0 * p + 2 * t
        v_b01 = 0 * p + 2 * t + 1
        v_b10 = 1 * p + 2 * t
        v_b11 = 1 * p + 2 * t + 1
        
        w_c00 = (2 * t) * n + 0
        w_c01 = (2 * t + 1) * n + 0
        w_c10 = (2 * t) * n + 1
        w_c11 = (2 * t + 1) * n + 1
        
        u_a00 = 0
        u_a01 = 1
        u_a10 = 2
        u_a11 = 3
        
        c = base + 0
        U[u_a00][c] = 1; U[u_a11][c] = 1
        V[v_b00][c] = 1; V[v_b11][c] = 1
        W[w_c00][c] = 1; W[w_c11][c] = 1
        
        c = base + 1
        U[u_a10][c] = 1; U[u_a11][c] = 1
        V[v_b00][c] = 1
        W[w_c10][c] = 1; W[w_c11][c] = -1
        
        c = base + 2
        U[u_a00][c] = 1
        V[v_b01][c] = 1; V[v_b11][c] = -1
        W[w_c01][c] = 1; W[w_c11][c] = 1
        
        c = base + 3
        U[u_a11][c] = 1
        V[v_b10][c] = 1; V[v_b00][c] = -1
        W[w_c00][c] = 1; W[w_c10][c] = 1
        
        c = base + 4
        U[u_a00][c] = 1; U[u_a01][c] = 1
        V[v_b11][c] = 1
        W[w_c00][c] = -1; W[w_c01][c] = 1
        
        c = base + 5
        U[u_a10][c] = 1; U[u_a00][c] = -1
        V[v_b00][c] = 1; V[v_b01][c] = 1
        W[w_c11][c] = 1
        
        c = base + 6
        U[u_a01][c] = 1; U[u_a11][c] = -1
        V[v_b10][c] = 1; V[v_b11][c] = 1
        W[w_c00][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
