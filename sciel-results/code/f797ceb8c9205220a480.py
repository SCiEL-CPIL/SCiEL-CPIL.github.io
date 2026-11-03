import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=7
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))
    
    def strassen_222(i1, j1, k1):
        # M1 = (A[i1,j1]+A[i1+1,j1+1]) * (B[j1,k1]+B[j1+1,k1+1])
        add_term(
            {u_idx(i1,j1): 1, u_idx(i1+1,j1+1): 1},
            {v_idx(j1,k1): 1, v_idx(j1+1,k1+1): 1},
            {w_idx(k1,i1): 1, w_idx(k1+1,i1+1): 1}
        )
        # M2 = (A[i1+1,j1]+A[i1+1,j1+1]) * B[j1,k1]
        add_term(
            {u_idx(i1+1,j1): 1, u_idx(i1+1,j1+1): 1},
            {v_idx(j1,k1): 1},
            {w_idx(k1,i1+1): 1}
        )
        # M3 = A[i1,j1] * (B[j1,k1+1]-B[j1+1,k1+1])
        add_term(
            {u_idx(i1,j1): 1},
            {v_idx(j1,k1+1): 1, v_idx(j1+1,k1+1): -1},
            {w_idx(k1+1,i1): 1}
        )
        # M4 = A[i1+1,j1+1] * (B[j1+1,k1]-B[j1,k1])
        add_term(
            {u_idx(i1+1,j1+1): 1},
            {v_idx(j1+1,k1): 1, v_idx(j1,k1): -1},
            {w_idx(k1,i1): 1, w_idx(k1,i1+1): 1}
        )
        # M5 = (A[i1,j1]+A[i1,j1+1]) * B[j1+1,k1+1]
        add_term(
            {u_idx(i1,j1): 1, u_idx(i1,j1+1): 1},
            {v_idx(j1+1,k1+1): 1},
            {w_idx(k1,i1): -1, w_idx(k1+1,i1): 1, w_idx(k1+1,i1+1): -1}
        )
        # M6 = (A[i1+1,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k1+1])
        add_term(
            {u_idx(i1+1,j1): 1, u_idx(i1,j1): -1},
            {v_idx(j1,k1): 1, v_idx(j1,k1+1): 1},
            {w_idx(k1+1,i1+1): 1}
        )
        # M7 = (A[i1,j1+1]-A[i1+1,j1+1]) * (B[j1+1,k1]+B[j1+1,k1+1])
        add_term(
            {u_idx(i1,j1+1): 1, u_idx(i1+1,j1+1): -1},
            {v_idx(j1+1,k1): 1, v_idx(j1+1,k1+1): 1},
            {w_idx(k1,i1): 1}
        )
    
    def schoolbook_221(i1, j1, k1):
        add_term({u_idx(i1,j1): 1}, {v_idx(j1,k1): 1}, {w_idx(k1,i1): 1})
        add_term({u_idx(i1,j1+1): 1}, {v_idx(j1+1,k1): 1}, {w_idx(k1,i1): 1})
        add_term({u_idx(i1+1,j1): 1}, {v_idx(j1,k1): 1}, {w_idx(k1,i1+1): 1})
        add_term({u_idx(i1+1,j1+1): 1}, {v_idx(j1+1,k1): 1}, {w_idx(k1,i1+1): 1})
    
    def schoolbook_147():
        # A[i=2, j=0..3] * B[j=0..3, k=0..6]
        for j in range(4):
            for k in range(7):
                add_term({u_idx(2,j): 1}, {v_idx(j,k): 1}, {w_idx(k,2): 1})
    
    # 2x4x7 part: i=0,1
    # Split j into 0,1 and 2,3
    # For each 2x2x7, split k into (0,1), (2,3), (4,5) using Strassen, and k=6 using schoolbook 2x2x1
    
    # First 2x2x7: i=0,1; j=0,1
    strassen_222(0, 0, 0)  # k=0,1
    strassen_222(0, 0, 2)  # k=2,3
    strassen_222(0, 0, 4)  # k=4,5
    schoolbook_221(0, 0, 6)  # k=6
    
    # Second 2x2x7: i=0,1; j=2,3
    strassen_222(0, 2, 0)  # k=0,1
    strassen_222(0, 2, 2)  # k=2,3
    strassen_222(0, 2, 4)  # k=4,5
    schoolbook_221(0, 2, 6)  # k=6
    
    # Wait, strassen_222 takes (i1, j1, k1) where i1 is the starting row of A, j1 is the starting col of A / row of B, k1 is the starting col of B.
    # For the second block, j1 should be 2, and i1 should still be 0 (since we're using i=0,1).
    # But strassen_222 uses i1 and i1+1 for A rows, and j1, j1+1 for A cols and B rows.
    # So for the second block: i1=0, j1=2, k1=0,2,4.
    # That's correct.
    
    # 1x4x7 part: i=2
    schoolbook_147()
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_entries, v_entries, w_entries) in enumerate(terms):
        for idx, val in u_entries.items():
            U[idx][r] = val
        for idx, val in v_entries.items():
            V[idx][r] = val
        for idx, val in w_entries.items():
            W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
