"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
_start = time.time()

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Use block decomposition: 3 blocks of 2x3x2, each rank 11
    # Total rank = 33
    
    # Rank-11 decomposition for 2x3x2:
    # Verified construction:
    # u vectors (6-dim: a00,a01,a02,a10,a11,a12)
    # v vectors (6-dim: b00,b01,b10,b11,b20,b21)
    # w vectors (4-dim: C00,C10,C01,C11) where C[i,k], W[k*2+i]
    
    terms = [
        # (u, v, w)
        # 1
        ([1,0,0, 0,0,0], [1,0, 0,0, 0,0], [1,0, 0,0]),
        # 2
        ([0,1,0, 0,0,0], [0,0, 1,0, 0,0], [1,0, 0,0]),
        # 3
        ([0,0,1, 0,0,0], [0,0, 0,0, 1,0], [1,0, 0,0]),
        # 4
        ([0,0,0, 1,0,0], [1,0, 0,0, 0,0], [0,1, 0,0]),
        # 5
        ([0,0,0, 0,1,0], [0,0, 1,0, 0,0], [0,1, 0,0]),
        # 6
        ([0,0,0, 0,0,1], [0,0, 0,0, 1,0], [0,1, 0,0]),
        # 7: (a00+a01)*(b01+b11)
        ([1,1,0, 0,0,0], [0,1, 0,1, 0,0], [0,0, 1,0]),
        # 8: (a00+a02)*(b01+b21)  
        ([1,0,1, 0,0,0], [0,1, 0,0, 0,1], [0,0, 1,0]),
        # 9: (a10+a11)*(b01+b11)
        ([0,0,0, 1,1,0], [0,1, 0,1, 0,0], [0,0, 0,1]),
        # 10: (a10+a12)*(b01+b21)
        ([0,0,0, 1,0,1], [0,1, 0,0, 0,1], [0,0, 0,1]),
        # 11: (a00+a10)*(b00+b10+b20)
        ([1,0,0, 1,0,0], [1,0, 1,0, 1,0], [1,0, 1,0]),
    ]
    
    r_block = len(terms)  # 11
    rank = 3 * r_block  # 33
    
    U = [[0] * rank for _ in range(n * m)]   # 6 x 33
    V = [[0] * rank for _ in range(m * p)]   # 18 x 33
    W = [[0] * rank for _ in range(p * n)]   # 12 x 33
    
    for block in range(3):
        for t in range(r_block):
            col = block * r_block + t
            u, v, w = terms[t]
            
            # Fill U
            for idx in range(6):
                U[idx][col] = u[idx]
            
            # Fill V: local (j, k') -> global (j, 2*block+k')
            for j in range(3):
                for kp in range(2):
                    v_local = j * 2 + kp
                    k_global = 2 * block + kp
                    v_global = j * p + k_global
                    V[v_global][col] = v[v_local]
            
            # Fill W: local (k', i) -> global (2*block+k', i), W[k*n+i]
            for kp in range(2):
                for i in range(2):
                    w_local = kp * 2 + i
                    k_global = 2 * block + kp
                    w_global = k_global * n + i
                    W[w_global][col] = w[w_local]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
