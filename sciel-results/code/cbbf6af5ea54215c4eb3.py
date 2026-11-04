def construct():
    n, m, p = 3, 4, 4
    
    terms = []  # list of (U_dict, V_dict, W_dict)
    
    # Strassen 2x2x2 terms
    def strassen_2x2x2(i0, j0, k0):
        a, b = i0, i0+1
        c, d = j0, j0+1
        e, f = k0, k0+1
        
        # Each term: (U_entries, V_entries, W_entries)
        # U_entries: dict from (i,j) to coeff
        # V_entries: dict from (j,k) to coeff
        # W_entries: dict from (k,i) to coeff
        
        s = [
            ({(a,c):1, (b,d):1}, {(c,e):1, (d,f):1}, {(e,a):1, (f,b):1}),
            ({(b,c):1, (b,d):1}, {(c,e):1}, {(e,b):1}),
            ({(a,c):1}, {(c,f):1, (d,f):-1}, {(f,a):1}),
            ({(b,d):1}, {(d,e):1, (c,e):-1}, {(e,b):1, (f,b):1}),
            ({(a,c):1, (a,d):1}, {(d,f):1}, {(e,a):-1, (f,a):1}),
            ({(b,c):1, (a,c):-1}, {(c,e):1, (d,e):1}, {(f,b):1}),
            ({(a,d):1, (b,d):-1}, {(c,f):1, (d,f):1}, {(e,a):1}),
        ]
        return s
    
    # Schoolbook terms for a single (i,j,k)
    def schoolbook_term(i, j, k):
        return [({(i,j):1}, {(j,k):1}, {(k,i):1})]
    
    # Build all terms
    all_terms = []
    
    # For i=0,1: use Strassen on 2x2x2 blocks
    for j0 in range(0, m, 2):
        for k0 in range(0, p, 2):
            all_terms.extend(strassen_2x2x2(0, j0, k0))
            # Also need to handle i=1 separately? No, Strassen handles both i=0 and i=1
    
    # Wait, the Strassen 2x2x2 handles i∈{0,1} for a given (j0,k0) block.
    # But I also need to handle the case where i=2.
    
    # For i=2: schoolbook
    for j in range(m):
        for k in range(p):
            all_terms.extend(schoolbook_term(2, j, k))
    
    # Now, the Strassen terms for i∈{0,1} cover all (i,j,k) with i∈{0,1}.
    # Let me verify: for j0=0, k0=0: covers i∈{0,1}, j∈{0,1}, k∈{0,1}
    # for j0=0, k0=2: covers i∈{0,1}, j∈{0,1}, k∈{2,3}
    # for j0=2, k0=0: covers i∈{0,1}, j∈{2,3}, k∈{0,1}
    # for j0=2, k0=2: covers i∈{0,1}, j∈{2,3}, k∈{2,3}
    # Total: 4 blocks * 7 terms = 28 terms for i∈{0,1}
    # Plus 16 schoolbook terms for i=2
    # Total: 44 terms
    
    rank = len(all_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(all_terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val
    
    return {"U": U, "V": V, "W": W}
