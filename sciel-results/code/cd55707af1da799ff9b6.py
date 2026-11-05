"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Partition sizes
    # Using [3, 2] for n, m, p
    splits = [3, 2]
    
    # Precompute starts
    starts_n = [0, 3]
    starts_m = [0, 3]
    starts_p = [0, 3]
    
    # Rank calculation
    # We will generate terms and count them
    # For now, we just build the matrices
    
    # Helper to add a term
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    # Note: In the prompt, U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # The indices in the prompt's schoolbook:
    # U[i * m + j]
    # V[j * p + k]
    # W[k * n + i]
    
    # We need to map local block indices to global indices
    # Block (bi, bk, bj) corresponds to:
    # i in [starts_n[bi], starts_n[bi+1])
    # k in [starts_m[bk], starts_m[bk+1])  (Inner index)
    # j in [starts_p[bj], starts_p[bj+1])
    
    # Wait, A is n x m. B is m x p.
    # A indices: i (0..n-1), k (0..m-1)
    # B indices: k (0..m-1), j (0..p-1)
    # C indices: i (0..n-1), j (0..p-1) -> W index j*n + i
    
    # Block sizes:
    # n splits: [3, 2]
    # m splits: [3, 2]
    # p splits: [3, 2]
    
    rank = 0
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        U_terms.append((u_idx, u_val))
        V_terms.append((v_idx, v_val))
        W_terms.append((w_idx, w_val))
        
    # Iterate over blocks
    for bi in range(2):
        for bk in range(2):
            for bj in range(2):
                ni = splits[bi]
                mk = splits[bk]
                pj = splits[bj]
                
                # Determine if Strassen applies
                # Strassen for 2x2x2
                if ni == 2 and mk == 2 and pj == 2:
                    # 7 terms
                    # Local indices i, k, j in 0..1
                    # U local: i * mk + k
                    # V local: k * pj + j
                    # W local: j * ni + i
                    # Global indices:
                    # i_g = starts_n[bi] + i
                    # k_g = starts_m[bk] + k
                    # j_g = starts_p[bj] + j
                    # U_g = i_g * m + k_g
                    # V_g = k_g * p + j_g
                    # W_g = j_g * n + i_g
                    
                    # Strassen coefficients
                    # P1: (A00+A11)(B00+B11)
                    # u: (0,0)=1, (1,1)=1
                    # v: (0,0)=1, (1,1)=1
                    # w: (0,0)=1, (1,1)=1
                    # P2: (A10+A11)B00
                    # u: (1,0)=1, (1,1)=1
                    # v: (0,0)=1
                    # w: (0,1)=1, (1,1)=-1
                    # P3: A00(B01-B11)
                    # u: (0,0)=1
                    # v: (0,1)=1, (1,1)=-1
                    # w: (0,1)=1, (1,1)=1
                    # P4: A11(B10-B00)
                    # u: (1,1)=1
                    # v: (1,0)=1, (0,0)=-1
                    # w: (0,0)=1, (0,1)=1
                    # P5: (A00+A01)B11
                    # u: (0,0)=1, (0,1)=1
                    # v: (1,1)=1
                    # w: (0,0)=-1, (0,1)=1
                    # P6: (A10-A00)(B00+B01)
                    # u: (1,0)=1, (0,0)=-1
                    # v: (0,0)=1, (0,1)=1
                    # w: (1,1)=1
                    # P7: (A01-A11)(B10+B11)
                    # u: (0,1)=1, (1,1)=-1
                    # v: (1,0)=1, (1,1)=1
                    # w: (0,0)=1
                    
                    strassen_terms = [
                        # (u_coeffs, v_coeffs, w_coeffs)
                        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
                        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
                        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
                        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
                        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
                        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
                        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
                    ]
                    
                    for u_c, v_c, w_c in strassen_terms:
                        for (i_l, k_l), u_val in u_c.items():
                            # For each non-zero U, we need to match V and W
                            # This is not how it works.
                            # Each term r has a full U matrix, V matrix, W matrix.
                            # We add the term to the global matrices.
                            pass
                            
                    # Correct approach:
                    for u_c, v_c, w_c in strassen_terms:
                        for i_l in range(2):
                            for k_l in range(2):
                                u_val = u_c.get((i_l, k_l), 0)
                                if u_val != 0:
                                    i_g = starts_n[bi] + i_l
                                    k_g = starts_m[bk] + k_l
                                    u_idx = i_g * m + k_g
                                    U_terms.append((u_idx, u_val))
                                    
                                    # We need to add V and W for the same term r
                                    # But U, V, W are separate lists.
                                    # We need to group them by term index.
                                    pass
                                    
                    # Let's restructure to build U, V, W directly
                    pass
                else:
                    # Schoolbook
                    pass
                    
    # This structure is getting messy. I will build U, V, W as lists of dicts or sparse arrays.
    # Given the small size, dense is fine.
    
    # Re-implementing cleanly
    U = [[0] * 100 for _ in range(n * m)] # Max rank estimate
    V = [[0] * 100 for _ in range(m * p)]
    W = [[0] * 100 for _ in range(p * n)]
    
    rank = 0
    
    def add_rank_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal rank
        U[u_idx][rank] = u_val
        V[v_idx][rank] = v_val
        W[w_idx][rank] = w_val
        rank += 1
        
    for bi in range(2):
        for bk in range(2):
            for bj in range(2):
                ni = splits[bi]
                mk = splits[bk]
                pj = splits[bj]
                
                if ni == 2 and mk == 2 and pj == 2:
                    # Strassen
                    strassen_terms = [
                        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
                        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
                        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
                        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
                        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
                        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
                        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
                    ]
                    for u_c, v_c, w_c in strassen_terms:
                        for i_l in range(2):
                            for k_l in range(2):
                                u_val = u_c.get((i_l, k_l), 0)
                                if u_val != 0:
                                    i_g = starts_n[bi] + i_l
                                    k_g = starts_m[bk] + k_l
                                    u_idx = i_g * m + k_g
                                    U[u_idx][rank] = u_val
                                    
                        for k_l in range(2):
                            for j_l in range(2):
                                v_val = v_c.get((k_l, j_l), 0)
                                if v_val != 0:
                                    k_g = starts_m[bk] + k_l
                                    j_g = starts_p[bj] + j_l
                                    v_idx = k_g * p + j_g
                                    V[v_idx][rank] = v_val
                                    
                        for j_l in range(2):
                            for i_l in range(2):
                                w_val = w_c.get((j_l, i_l), 0)
                                if w_val != 0:
                                    j_g = starts_p[bj] + j_l
                                    i_g = starts_n[bi] + i_l
                                    w_idx = j_g * n + i_g
                                    W[w_idx][rank] = w_val
                                    
                        rank += 1
                else:
                    # Schoolbook
                    for i_l in range(ni):
                        for k_l in range(mk):
                            for j_l in range(pj):
                                i_g = starts_n[bi] + i_l
                                k_g = starts_m[bk] + k_l
                                j_g = starts_p[bj] + j_l
                                
                                u_idx = i_g * m + k_g
                                v_idx = k_g * p + j_g
                                w_idx = j_g * n + i_g
                                
                                U[u_idx][rank] = 1
                                V[v_idx][rank] = 1
                                W[w_idx][rank] = 1
                                rank += 1
                                
    # Trim to actual rank
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
