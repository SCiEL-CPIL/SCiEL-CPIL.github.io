import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 4, 4, 6
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Top-level Strassen Coefficients
    alpha_top = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    beta_top = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    recon_top = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    # Inner Strassen Coefficients (for 2x2 x 2x2)
    alpha_in = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    beta_in = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    recon_in = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    col = 0
    
    for s_t in range(7):
        if time.time() > deadline - 5: break
        
        # Part 1: Inner Strassen (2x2 x 2x2)
        for s_i in range(7):
            if time.time() > deadline - 5: break
            
            # U
            for a_t in range(4):
                if alpha_top[s_t][a_t] == 0: continue
                rt, ct = divmod(a_t, 2)
                for a_i in range(4):
                    if alpha_in[s_i][a_i] == 0: continue
                    ri, ci = divmod(a_i, 2)
                    u_idx = (rt*2 + ri)*4 + (ct*2 + ci)
                    U[u_idx, col] += alpha_top[s_t][a_t] * alpha_in[s_i][a_i]
            
            # V
            for b_t in range(4):
                if beta_top[s_t][b_t] == 0: continue
                rt, ct = divmod(b_t, 2)
                for b_i in range(4):
                    if beta_in[s_i][b_i] == 0: continue
                    ri, ci = divmod(b_i, 2)
                    # B is 4x6. Top block ct is 0 or 1.
                    # Inner block ci is 0 or 1 (cols 0,1 of the 2x3 block).
                    # Global col = ct*3 + ci
                    v_idx = (rt*2 + ri)*6 + (ct*3 + ci)
                    V[v_idx, col] += beta_top[s_t][b_t] * beta_in[s_i][b_i]
            
            # W
            for a_b, b_b, g_t in recon_top[s_t]:
                rt_a, ct_a = divmod(a_b, 2)
                rt_b, ct_b = divmod(b_b, 2)
                for a_sb, b_sb, g_i in recon_in[s_i]:
                    ri_a, ci_a = divmod(a_sb, 2)
                    ri_b, ci_b = divmod(b_sb, 2)
                    # C is 4x6.
                    # Row: rt_a*2 + ri_a
                    # Col: ct_b*3 + ci_b (cols 0,1 of the 2x3 result block)
                    w_idx = (rt_a*2 + ri_a)*4 + (ct_b*3 + ci_b)
                    W[w_idx, col] += g_t * g_i
            
            col += 1
        
        # Part 2: Schoolbook (2x2 x 2x1)
        # Iterates over the 4 elements of the 2x2 A block.
        # Each element A[ri, ci] is multiplied by B[ci, 2] (the 3rd column of the 2x3 block).
        for ri in range(2):
            for ci in range(2):
                if time.time() > deadline - 5: break
                
                # U: Pick A[ri, ci]
                for a_t in range(4):
                    if alpha_top[s_t][a_t] == 0: continue
                    rt, ct = divmod(a_t, 2)
                    # A element is at (rt*2+ri, ct*2+ci)
                    u_idx = (rt*2 + ri)*4 + (ct*2 + ci)
                    U[u_idx, col] += alpha_top[s_t][a_t]
                
                # V: Pick B[ci, 2]
                # The row of B is ci (matching col of A).
                # The col of B is 2 (the 3rd col of the 2x3 block).
                for b_t in range(4):
                    if beta_top[s_t][b_t] == 0: continue
                    rt, ct = divmod(b_t, 2)
                    # B element is at (rt*2+ci, ct*3+2)
                    v_idx = (rt*2 + ci)*6 + (ct*3 + 2)
                    V[v_idx, col] += beta_top[s_t][b_t]
                
                # W: Add to C[ri, 2]
                # The result of A[ri, ci] * B[ci, 2] goes to C[ri, 2].
                for a_b, b_b, g_t in recon_top[s_t]:
                    rt_a, ct_a = divmod(a_b, 2)
                    rt_b, ct_b = divmod(b_b, 2)
                    # The 2x2 A block is at a_b. The 2x1 B block is at b_b.
                    # The specific element A[ri, ci] is within a_b.
                    # The specific element B[ci, 2] is within b_b.
                    # The result C[ri, 2] is within the result block at a_b (row) and b_b (col).
                    # Wait, the reconstruction recon_top gives the contribution to the 2x2 C block.
                    # The 2x2 C block is formed by the 2x2 A block and 2x2 B block.
                    # But here we are doing 2x2 * 2x1.
                    # The result is a 2x1 vector.
                    # This vector is the 3rd column of the 2x3 C block.
                    # So for a given top step s_t, the result block is at (a_b, b_b) in the 2x2 grid of 2x3 blocks.
                    # The row index within the 2x3 block is ri.
                    # The col index within the 2x3 block is 2.
                    w_idx = (rt_a*2 + ri)*4 + (ct_b*3 + 2)
                    W[w_idx, col] += g_t
                
                col += 1

    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
