import numpy as np

def construct():
    n, m, p = 2, 2, 11
    # Rank 39: 5 * Strassen(7) + 1 * Rank4(4)
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_indices, v_indices, w_indices, value):
        nonlocal col
        for i in u_indices:
            U[i][col] += value
        for j in v_indices:
            V[j][col] += value
        for k in w_indices:
            W[k][col] += value
        col += 1

    # Strassen decomposition for 2x2 by 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[C00, C01], [C10, C11]]
    
    # We map local A indices to global U indices
    # U: 0:a, 1:b, 2:c, 3:d
    
    for block in range(5):
        # Local B columns are 2*block and 2*block+1
        # Global V indices:
        # e = b[0, 2b] -> V[0*11 + 2b]
        # f = b[0, 2b+1] -> V[0*11 + 2b+1]
        # g = b[1, 2b] -> V[1*11 + 2b]
        # h = b[1, 2b+1] -> V[1*11 + 2b+1]
        
        e_idx = 0 * 11 + 2 * block
        f_idx = 0 * 11 + 2 * block + 1
        g_idx = 1 * 11 + 2 * block
        h_idx = 1 * 11 + 2 * block + 1
        
        # Global W indices:
        # C00 = C[2b, 0] -> W[2b*2 + 0]
        # C01 = C[2b+1, 0] -> W[(2b+1)*2 + 0]
        # C10 = C[2b, 1] -> W[2b*2 + 1]
        # C11 = C[2b+1, 1] -> W[(2b+1)*2 + 1]
        
        c00_idx = 2 * block * 2 + 0
        c01_idx = (2 * block + 1) * 2 + 0
        c10_idx = 2 * block * 2 + 1
        c11_idx = (2 * block + 1) * 2 + 1
        
        # M1 = (a+d)(e+h) -> C00 + C11
        # U: a, d. V: e, h. W: C00, C11.
        add_term([0, 3], [e_idx, h_idx], [c00_idx, c11_idx], 1)
        
        # M2 = (c+d)e -> C10 - C11
        # U: c, d. V: e. W: C10, -C11.
        # Note: add_term adds value to all. We need to handle signs.
        # Let's modify add_term to take a list of (index, coeff) or just pass value and handle signs in indices?
        # Easier: call add_term multiple times or modify to accept sparse dicts.
        # Let's rewrite add_term to accept lists of (idx, val).
        
    # Redefine add_term for flexibility
    col = 0
    def add_term_sparse(u_sp, v_sp, w_sp):
        nonlocal col
        for i, val in u_sp:
            U[i][col] += val
        for j, val in v_sp:
            V[j][col] += val
        for k, val in w_sp:
            W[k][col] += val
        col += 1

    for block in range(5):
        e_idx = 0 * 11 + 2 * block
        f_idx = 0 * 11 + 2 * block + 1
        g_idx = 1 * 11 + 2 * block
        h_idx = 1 * 11 + 2 * block + 1
        
        c00_idx = 2 * block * 2 + 0
        c01_idx = (2 * block + 1) * 2 + 0
        c10_idx = 2 * block * 2 + 1
        c11_idx = (2 * block + 1) * 2 + 1
        
        # M1 = (a+d)(e+h) -> C00 + C11
        add_term_sparse([(0,1), (3,1)], [(e_idx,1), (h_idx,1)], [(c00_idx,1), (c11_idx,1)])
        
        # M2 = (c+d)e -> C10 - C11
        add_term_sparse([(2,1), (3,1)], [(e_idx,1)], [(c10_idx,1), (c11_idx,-1)])
        
        # M3 = a(f-g) -> C01 + C11
        add_term_sparse([(0,1)], [(f_idx,1), (g_idx,-1)], [(c01_idx,1), (c11_idx,1)])
        
        # M4 = d(g-e) -> C00 + C10
        add_term_sparse([(3,1)], [(g_idx,1), (e_idx,-1)], [(c00_idx,1), (c10_idx,1)])
        
        # M5 = (a+b)h -> -C00 + C01
        add_term_sparse([(0,1), (1,1)], [(h_idx,1)], [(c00_idx,-1), (c01_idx,1)])
        
        # M6 = (c-a)(e+f) -> C11
        add_term_sparse([(2,1), (0,-1)], [(e_idx,1), (f_idx,1)], [(c11_idx,1)])
        
        # M7 = (b-d)(g+h) -> C00
        add_term_sparse([(1,1), (3,-1)], [(g_idx,1), (h_idx,1)], [(c00_idx,1)])

    # Rank 4 block for column 10
    # C0 = a*b0 + c*b1
    # C1 = d*b0 + b*b1
    # U: 0:a, 1:b, 2:c, 3:d
    # V: 10:b0, 21:b1
    # W: 20:C0, 21:C1
    
    b0_idx = 10
    b1_idx = 21
    c0_idx = 20
    c1_idx = 21
    
    # T1 = a * b0 * C0
    add_term_sparse([(0,1)], [(b0_idx,1)], [(c0_idx,1)])
    # T2 = c * b1 * C0
    add_term_sparse([(2,1)], [(b1_idx,1)], [(c0_idx,1)])
    # T3 = d * b0 * C1
    add_term_sparse([(3,1)], [(b0_idx,1)], [(c1_idx,1)])
    # T4 = b * b1 * C1
    add_term_sparse([(1,1)], [(b1_idx,1)], [(c1_idx,1)])

    return {"U": U, "V": V, "W": W}
