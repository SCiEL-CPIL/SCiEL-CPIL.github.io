import time

def construct():
    # Matrix multiplication 2x5 by 5x6
    # Output W[k*n+i, r] where n=2, p=6
    # We use the known valid rank 54 decomposition
    # Structure:
    # Block 1: 2x2x6 (3 Strassen 2x2x2) -> Rank 21
    # Block 2: 2x3x6 (2 Strassen 2x2x2 + 2 Schoolbook 2x1x2 per 2x3x2) -> Rank 33
    
    # Total: 54

    n, m, p = 2, 5, 6
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank_step(u_pairs, v_pairs, w_pairs):
        nonlocal col
        if col >= rank:
            return False
        for idx, val in u_pairs:
            U[idx][col] = val
        for idx, val in v_pairs:
            V[idx][col] = val
        for idx, val in w_pairs:
            W[idx][col] = val
        col += 1
        return True

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def add_strassen(j1, j2, k1, k2):
        # Rank 7 for 2x2x2
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

    def add_schoolbook_2x1x2(j, k1, k2):
        # Rank 4 for 2x1x2
        add_rank_step([(u_idx(0,j), 1)], [(v_idx(j,k1), 1)], [(w_idx(k1,0), 1)])
        add_rank_step([(u_idx(0,j), 1)], [(v_idx(j,k2), 1)], [(w_idx(k2,0), 1)])
        add_rank_step([(u_idx(1,j), 1)], [(v_idx(j,k1), 1)], [(w_idx(k1,1), 1)])
        add_rank_step([(u_idx(1,j), 1)], [(v_idx(j,k2), 1)], [(w_idx(k2,1), 1)])

    # Execute the decomposition
    # 1. j=0,1; k=0,1
    add_strassen(0, 1, 0, 1)
    
    # 2. j=0,1; k=2,3
    add_strassen(0, 1, 2, 3)
    
    # 3. j=0,1; k=4,5
    add_strassen(0, 1, 4, 5)
    
    # 4. j=2,3; k=0,1
    add_strassen(2, 3, 0, 1)
    
    # 5. j=4; k=0,1
    add_schoolbook_2x1x2(4, 0, 1)
    
    # 6. j=2,3; k=2,3
    add_strassen(2, 3, 2, 3)
    
    # 7. j=4; k=2,3
    add_schoolbook_2x1x2(4, 2, 3)
    
    # 8. j=2,3; k=4,5
    add_strassen(2, 3, 4, 5)
    
    # 9. j=4; k=4,5
    add_schoolbook_2x1x2(4, 4, 5)

    return {"U": U, "V": V, "W": W}
