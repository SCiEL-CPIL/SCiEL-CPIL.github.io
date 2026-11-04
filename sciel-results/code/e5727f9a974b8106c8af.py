import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Universe of non-zero entries
    universe = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                universe.add((u_idx, v_idx, w_idx))
                
    candidates = []
    
    # Candidate Type 1: Lines for fixed (i, j)
    for i in range(n):
        for j in range(m):
            u_vec = np.zeros(n*m)
            u_vec[i*m + j] = 1
            v_vec = np.zeros(m*p)
            for k in range(p):
                v_vec[j*p + k] = 1
            w_vec = np.zeros(p*n)
            for k in range(p):
                w_vec[k*n + i] = 1
            candidates.append((u_vec, v_vec, w_vec))
            
    # Candidate Type 2: Strassen terms for 2x2x2 blocks
    # We need to define the 7 Strassen terms for a block starting at (i0, j0, k0)
    # This requires n>=2, m>=2, p>=2.
    
    def add_strassen_block(i0, j0, k0):
        # Local indices 0,1
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We define the vectors for A, B, C parts.
        # A indices: (i, j) -> (i0+i, j0+j)
        # B indices: (j, k) -> (j0+j, k0+k)
        # C indices: (k, i) -> (k0+k, i0+i)
        
        def a_idx(i, j): return (i0+i)*m + (j0+j)
        def b_idx(j, k): return (j0+j)*p + (k0+k)
        def c_idx(k, i): return (k0+k)*n + (i0+i)
        
        terms = [
            # M1
            (
                {a_idx(0,0): 1, a_idx(1,1): 1},
                {b_idx(0,0): 1, b_idx(1,1): 1},
                {c_idx(0,0): 1, c_idx(0,1): 1, c_idx(1,0): 1, c_idx(1,1): 1} # This is wrong, C is sum of M's
            ),
        ]
        # The C vector for each M term is not just the block, it's the contribution to C.
        # C = M1 + M4 - M5 + M7, M3+M5, M2+M4, M1-M2+M3+M6
        # So for M1, the C vector has:
        # c(0,0): +1
        # c(1,1): +1
        # c(0,1): 0
        # c(1,0): 0
        # Wait, the C combination is:
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # So for M1:
        # w_vec has c(0,0): 1, c(1,1): 1.
        # For M2:
        # w_vec has c(1,0): 1, c(1,1): -1.
        # Etc.
        
        w_combos = {
            0: {c_idx(0,0): 1, c_idx(1,1): 1}, # M1
            1: {c_idx(1,0): 1, c_idx(1,1): -1}, # M2
            2: {c_idx(0,1): 1, c_idx(1,1): 1},  # M3
            3: {c_idx(0,0): 1, c_idx(1,0): 1},  # M4
            4: {c_idx(0,0): -1, c_idx(0,1): 1}, # M5
            5: {c_idx(1,1): 1},                  # M6
            6: {c_idx(0,0): 1, c_idx(1,1): 1},   # M7
        }
        
        a_coefs = [
            {a_idx(0,0): 1, a_idx(1,1): 1}, # M1
            {a_idx(1,0): 1, a_idx(1,1): 1}, # M2
            {a_idx(0,0): 1},                # M3
            {a_idx(1,1): 1},                # M4
            {a_idx(0,0): 1, a_idx(0,1): 1}, # M5
            {a_idx(1,0): 1, a_idx(0,0): -1},# M6
            {a_idx(0,1): 1, a_idx(1,1): -1},# M7
        ]
        
        b_coefs = [
            {b_idx(0,0): 1, b_idx(1,1): 1}, # M1
            {b_idx(0,0): 1},                # M2
            {b_idx(0,1): 1, b_idx(1,1): -1},# M3
            {b_idx(1,0): 1, b_idx(0,0): -1},# M4
            {b_idx(1,1): 1},                # M5
            {b_idx(0,0): 1, b_idx(0,1): 1}, # M6
            {b_idx(1,0): 1, b_idx(1,1): 1}, # M7
        ]
        
        for t in range(7):
            u_vec = np.zeros(n*m)
            for idx, val in a_coefs[t].items():
                u_vec[idx] = val
            v_vec = np.zeros(m*p)
            for idx, val in b_coefs[t].items():
                v_vec[idx] = val
            w_vec = np.zeros(p*n)
            for idx, val in w_combos[t].items():
                w_vec[idx] = val
            candidates.append((u_vec, v_vec, w_vec))
            
    for i0 in range(0, n, 2):
        for j0 in range(0, m, 2):
            for k0 in range(0, p, 2):
                if i0+1 < n and j0+1 < m and k0+1 < p:
                    add_strassen_block(i0, j0, k0)
                    
    # Greedy Set Cover
    covered = set()
    U_cols = []
    V_cols = []
    W_cols = []
    
    def get_cover(u, v, w):
        cover = set()
        for u_idx in range(n*m):
            if u[u_idx] != 0:
                for v_idx in range(m*p):
                    if v[v_idx] != 0:
                        for w_idx in range(p*n):
                            if w[w_idx] != 0:
                                if (u_idx, v_idx, w_idx) in universe:
                                    cover.add((u_idx, v_idx, w_idx))
        return cover

    candidate_covers = []
    for c in candidates:
        cover = get_cover(c[0], c[1], c[2])
        if cover:
            candidate_covers.append((cover, c))
            
    candidate_covers.sort(key=lambda x: len(x[0]), reverse=True)
    
    while len(covered) < len(universe):
        best_cover = None
        best_cand = None
        best_gain = 0
        
        for cover, cand in candidate_covers:
            gain = len(cover - covered)
            if gain > best_gain:
                best_gain = gain
                best_cover = cover
                best_cand = cand
                
        if best_gain == 0:
            break
            
        covered.update(best_cover)
        U_cols.append(best_cand[0])
        V_cols.append(best_cand[1])
        W_cols.append(best_cand[2])
        
    if U_cols:
        U = np.array(U_cols).T
        V = np.array(V_cols).T
        W = np.array(W_cols).T
    else:
        U = np.zeros((n*m, 0))
        V = np.zeros((m*p, 0))
        W = np.zeros((p*n, 0))
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
