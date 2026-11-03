import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # List of (u, v, w) for Strassen 2x2x2
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 1]), # M2
        ([1, 0, 0, 0], [0, 1, -1, 0], [0, 0, 1, 1]), # M3
        ([0, 0, 0, 1], [-1, 1, 0, 0], [1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
    ]
    
    # Function to map local 2x2 indices to global 4x4x4 indices, then to 4x4x9
    def map_2x2_to_4x4x9(u_local, v_local, w_local, br_a, bc_a, br_b, bc_b, br_c, bc_c):
        u_global = np.zeros(16)
        v_global = np.zeros(36)
        w_global = np.zeros(36)
        
        for i in range(2):
            for j in range(2):
                u_global[(2*br_a + i) * 4 + (2*bc_a + j)] = u_local[i*2 + j]
                
        for j in range(2):
            for k in range(2):
                # B is 4x9. We use cols 0-3.
                # v index: j*9 + k
                v_global[(2*br_b + j) * 9 + (2*bc_b + k)] = v_local[j*2 + k]
                
        for k in range(2):
            for i in range(2):
                # C is 4x9. We use cols 0-3.
                # w index: k*4 + i
                w_global[(2*bc_c + k) * 4 + (2*br_c + i)] = w_local[k*2 + i]
                
        return u_global, v_global, w_global

    block_products = [
        (0, 0, 0, 0, 0, 0),
        (0, 1, 1, 0, 0, 0),
        (0, 0, 0, 1, 0, 1),
        (0, 1, 1, 1, 0, 1),
        (1, 0, 0, 0, 1, 0),
        (1, 1, 1, 0, 1, 0),
        (1, 0, 0, 1, 1, 1),
        (1, 1, 1, 1, 1, 1)
    ]
    
    tensors = []
    
    for br_a, bc_a, br_b, bc_b, br_c, bc_c in block_products:
        for u_l, v_l, w_l in strassen_2x2:
            u_g, v_g, w_g = map_2x2_to_4x4x9(u_l, v_l, w_l, br_a, bc_a, br_b, bc_b, br_c, bc_c)
            tensors.append((u_g, v_g, w_g))
            
    for i in range(4):
        for j in range(4):
            u_g = np.zeros(16)
            v_g = np.zeros(36)
            w_g = np.zeros(36)
            
            u_g[i*4 + j] = 1.0
            v_g[j*9 + 8] = 1.0
            w_g[8*4 + i] = 1.0
            
            tensors.append((u_g, v_g, w_g))
            
    rank = len(tensors)
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for r, (u_g, v_g, w_g) in enumerate(tensors):
        U[:, r] = u_g
        V[:, r] = v_g
        W[:, r] = w_g
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
