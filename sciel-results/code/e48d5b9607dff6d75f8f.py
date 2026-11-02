import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0

    n, m, p = 2, 5, 6
    
    U_cols = []
    V_cols = []
    W_cols = []

    def add_rank_step(u_pairs, v_pairs, w_pairs):
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        for idx, val in u_pairs:
            u_col[idx] = val
        for idx, val in v_pairs:
            v_col[idx] = val
        for idx, val in w_pairs:
            w_col[idx] = val
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def add_strassen_block(j1, j2, k1, k2):
        add_rank_step([(u_idx(0,j1), 1), (u_idx(1,j2), 1)], 
                     [(v_idx(j1,k1), 1), (v_idx(j2,k2), 1)], 
                     [(w_idx(k1,0), 1), (w_idx(k2,1), 1)])
        add_rank_step([(u_idx(1,j1), 1), (u_idx(1,j2), 1)], 
                     [(v_idx(j1,k1), 1)], 
                     [(w_idx(k1,1), 1), (w_idx(k2,1), -1)])
        add_rank_step([(u_idx(0,j1), 1)], 
                     [(v_idx(j1,k2), 1), (v_idx(j2,k2), -1)], 
                     [(w_idx(k2,0), 1), (w_idx(k2,1), 1)])
        add_rank_step([(u_idx(1,j2), 1)], 
                     [(v_idx(j2,k1), 1), (v_idx(j1,k1), -1)], 
                     [(w_idx(k1,0), 1), (w_idx(k1,1), 1)])
        add_rank_step([(u_idx(0,j1), 1), (u_idx(0,j2), 1)], 
                     [(v_idx(j2,k2), 1)], 
                     [(w_idx(k1,0), -1), (w_idx(k2,0), 1)])
        add_rank_step([(u_idx(1,j1), 1), (u_idx(0,j1), -1)], 
                     [(v_idx(j1,k1), 1), (v_idx(j1,k2), 1)], 
                     [(w_idx(k2,1), 1)])
        add_rank_step([(u_idx(0,j2), 1), (u_idx(1,j2), -1)], 
                     [(v_idx(j2,k1), 1), (v_idx(j2,k2), 1)], 
                     [(w_idx(k1,0), 1)])

    def add_2x3x2_block(j1, j2, j3, k1, k2):
        # 2 rows x 3 cols (j) x 2 cols (k)
        # Use Strassen on (j1,j2) x (k1,k2) for rank 7
        # Plus additional rank 3 for j3 contributions
        # Actually, let me use a direct rank-10 decomposition
        
        # For 2x3x2: A is 2x3, B is 3x2, C is 2x2
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
        
        # Use Strassen on j1,j2 with k1,k2 (7 steps)
        add_strassen_block(j1, j2, k1, k2)
        
        # Now add j3 contributions: need to add A[0,2]*B[2,0], A[0,2]*B[2,1], A[1,2]*B[2,0], A[1,2]*B[2,1]
        # But we can't just add 4 more since that gives 11.
        # We need to modify the Strassen steps to incorporate j3.
        
        # Alternative: use a different rank-10 decomposition
        # Let me use:
        # m1 = (A00+A11)(B00+B11) -> contributes to C00, C11
        # m2 = (A10+A11)B00 -> C10
        # m3 = A00(B01-B11) -> C01
        # m4 = A11(B10-B00) -> C10
        # m5 = (A00+A01)B11 -> C00
        # m6 = (A10-A00)(B00+B01) -> C01
        # m7 = (A01-A11)(B10+B11) -> C00
        # These 7 handle j1,j2. Now for j3:
        # We need to add A02*B20 to C00, A02*B21 to C01, A12*B20 to C10, A12*B21 to C11
        # That's 4 more terms = 11 total. Not optimal.
        
        # Let me try a different approach: use rank-10 directly
        # I'll look for a known decomposition or derive one.
        
        # Actually, let me just use the 7+4=11 approach for now and see if I can do better.
        # For now, add the 4 extra terms for j3:
        add_rank_step([(u_idx(0,j3), 1)], [(v_idx(j3,k1), 1)], [(w_idx(k1,0), 1)])
        add_rank_step([(u_idx(0,j3), 1)], [(v_idx(j3,k2), 1)], [(w_idx(k2,0), 1)])
        add_rank_step([(u_idx(1,j3), 1)], [(v_idx(j3,k1), 1)], [(w_idx(k1,1), 1)])
        add_rank_step([(u_idx(1,j3), 1)], [(v_idx(j3,k2), 1)], [(w_idx(k2,1), 1)])

    def add_2x2x3_block(j1, j2, k1, k2, k3):
        # 2 rows x 2 cols (j) x 3 cols (k)
        # Use Strassen on (j1,j2) x (k1,k2) for rank 7
        # Plus 4 more for k3
        add_strassen_block(j1, j2, k1, k2)
        add_rank_step([(u_idx(0,j1), 1)], [(v_idx(j1,k3), 1)], [(w_idx(k3,0), 1)])
        add_rank_step([(u_idx(0,j2), 1)], [(v_idx(j2,k3), 1)], [(w_idx(k3,0), 1)])
        add_rank_step([(u_idx(1,j1), 1)], [(v_idx(j1,k3), 1)], [(w_idx(k3,1), 1)])
        add_rank_step([(u_idx(1,j2), 1)], [(v_idx(j2,k3), 1)], [(w_idx(k3,1), 1)])

    # Split j into (0,1,2) and (3,4), k into (0,1), (2,3), (4,5)
    # 2x3x2 blocks: 3 of them, each rank 11 (7+4) = 33
    # 2x2x2 blocks: 3 of them, each rank 7 = 21
    # Total: 54. Same as before!
    
    # Let me try: j into (0,1,2) and (3,4), k into (0,1,2) and (3,4,5)
    # 2x3x3 blocks: 2 of them
    # 2x2x3 blocks: 2 of them
    # 2x3x2 blocks: 2 of them
    # 2x2x2 blocks: 1 of them
    
    # This is getting complex. Let me just verify the current 54-rank solution works
    # and try a slightly different partition.
    
    # Try: j = (0,1,2), (3,4); k = (0,1), (2,3,4), (5)
    # This is asymmetric and messy.
    
    # Let me just go with the original 54-rank approach but verify it's correct.
    
    j_groups = [(0, 1), (2, 3), (4,)]
    k_groups = [(0, 1), (2, 3), (4, 5)]
    
    for j_range in j_groups:
        for k_range in k_groups:
            if len(j_range) == 2 and len(k_range) == 2:
                add_strassen_block(j_range[0], j_range[1], k_range[0], k_range[1])
            elif len(j_range) == 1 and len(k_range) == 2:
                for j in j_range:
                    for k in k_range:
                        for i in range(n):
                            add_rank_step([(u_idx(i,j), 1)], [(v_idx(j,k), 1)], [(w_idx(k,i), 1)])
            elif len(j_range) == 2 and len(k_range) == 1:
                for j in j_range:
                    for k in k_range:
                        for i in range(n):
                            add_rank_step([(u_idx(i,j), 1)], [(v_idx(j,k), 1)], [(w_idx(k,i), 1)])
            else:
                for j in j_range:
                    for k in k_range:
                        for i in range(n):
                            add_rank_step([(u_idx(i,j), 1)], [(v_idx(j,k), 1)], [(w_idx(k,i), 1)])

    rank = len(U_cols)
    U = [[U_cols[r][i] for r in range(rank)] for i in range(n * m)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(m * p)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(p * n)]

    return {"U": U, "V": V, "W": W}
