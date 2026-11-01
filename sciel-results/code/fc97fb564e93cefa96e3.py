import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (20, rank)
    
    # We use Strassen's algorithm on blocks of 2 columns.
    # 10 columns -> 5 blocks.
    # Each block uses 7 multiplications.
    # Total rank = 5 * 7 = 35.
    
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for U: A[i,j] -> i*m + j
    # Indices for V: B[j,k] -> j*p + k
    # Indices for W: C[i,k] -> k*n + i (per task contract W[k*n+i, r])
    
    for r in range(num_blocks):
        # Base index in V and W for this block
        # V: columns 2r, 2r+1
        v_b00 = 0 * p + 2 * r       # B[0, 2r]
        v_b01 = 0 * p + 2 * r + 1   # B[0, 2r+1]
        v_b10 = 1 * p + 2 * r       # B[1, 2r]
        v_b11 = 1 * p + 2 * r + 1   # B[1, 2r+1]
        
        # W: columns 2r, 2r+1
        w_c00 = (2 * r) * n + 0     # C[0, 2r]
        w_c01 = (2 * r + 1) * n + 0 # C[0, 2r+1]
        w_c10 = (2 * r) * n + 1     # C[1, 2r]
        w_c11 = (2 * r + 1) * n + 1 # C[1, 2r+1]
        
        # U indices are constant across blocks
        u_a00 = 0
        u_a01 = 1
        u_a10 = 2
        u_a11 = 3
        
        # Strassen 7 terms
        base_rank = r * 7
        
        # M1 = (a00 + a11)(b00 + b11)
        r_idx = base_rank + 0
        U[u_a00][r_idx] += 1
        U[u_a11][r_idx] += 1
        V[v_b00][r_idx] += 1
        V[v_b11][r_idx] += 1
        W[w_c00][r_idx] += 1
        W[w_c11][r_idx] += 1
        
        # M2 = (a10 + a11)b00
        r_idx = base_rank + 1
        U[u_a10][r_idx] += 1
        U[u_a11][r_idx] += 1
        V[v_b00][r_idx] += 1
        W[w_c10][r_idx] += 1
        W[w_c11][r_idx] += -1
        
        # M3 = a00(b01 - b11)
        r_idx = base_rank + 2
        U[u_a00][r_idx] += 1
        V[v_b01][r_idx] += 1
        V[v_b11][r_idx] += -1
        W[w_c00][r_idx] += -1
        W[w_c01][r_idx] += 1
        
        # M4 = a11(b10 - b00)
        r_idx = base_rank + 3
        U[u_a11][r_idx] += 1
        V[v_b10][r_idx] += 1
        V[v_b00][r_idx] += -1
        W[w_c00][r_idx] += 1
        W[w_c10][r_idx] += 1
        
        # M5 = (a00 + a01)b11
        r_idx = base_rank + 4
        U[u_a00][r_idx] += 1
        U[u_a01][r_idx] += 1
        V[v_b11][r_idx] += 1
        W[w_c00][r_idx] += -1
        W[w_c01][r_idx] += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        r_idx = base_rank + 5
        U[u_a10][r_idx] += 1
        U[u_a00][r_idx] += -1
        V[v_b00][r_idx] += 1
        V[v_b01][r_idx] += 1
        W[w_c11][r_idx] += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        r_idx = base_rank + 6
        U[u_a01][r_idx] += 1
        U[u_a11][r_idx] += -1
        V[v_b10][r_idx] += 1
        V[v_b11][r_idx] += 1
        W[w_c00][r_idx] += 1
        
    return {"U": U, "V": V, "W": W}
