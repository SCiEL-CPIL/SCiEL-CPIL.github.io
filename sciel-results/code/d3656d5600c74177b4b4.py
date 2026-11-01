import time

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 21 solution
    U_block, V_block, W_block = get_strassen_block()
    
    U = [[0] * 21 for _ in range(n * m)]
    V = [[0] * 21 for _ in range(m * p)]
    W = [[0] * 21 for _ in range(p * n)]
    
    for b in range(3):
        # U is same for all blocks
        for i in range(4):
            for c in range(7):
                U[i][b*7 + c] = U_block[i][c]
                
        # V mapping
        # Local j, k -> Global j*p + k
        # Local k is 0, 1. Global k is 2b, 2b+1.
        for j in range(2):
            for k_local in range(2):
                k_global = 2*b + k_local
                v_row_local = j * 2 + k_local
                v_row_global = j * p + k_global
                for c in range(7):
                    V[v_row_global][b*7 + c] = V_block[v_row_local][c]
                    
        # W mapping
        # Local k, i -> Global k*n + i
        # Local k is 0, 1. Global k is 2b, 2b+1.
        # Local index l = k*2 + i.
        # Global index = k_global * n + i.
        for l_local in range(4):
            k_local = l_local // 2
            i_local = l_local % 2
            k_global = 2*b + k_local
            w_row_global = k_global * n + i_local
            for c in range(7):
                W[w_row_global][b*7 + c] = W_block[l_local][c]
                
    return {"U": U, "V": V, "W": W}

def get_strassen_block():
    # ... (implementation from above)
    pass
