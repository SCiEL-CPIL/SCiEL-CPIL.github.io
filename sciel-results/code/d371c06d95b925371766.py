# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    
    Uses GQ-style decomposition with block Strassen.
    Base rank: 36.
    Attempts to reduce rank by removing 1 or 2 terms via linear system solving.
    """
    import time
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for safety
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    rank_terms = []
    
    def add_strassen_2x2x2(a_offset, b_offset, c_offset, sign=1):
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset
        
        terms = [
            ({(i0, j0): 1, (i0+1, j0+1): 1}, 
             {(j0, k0): 1, (j0+1, k0+1): 1}, 
             {(i0, k0): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0): 1, (i0+1, j0+1): 1}, 
             {(j0, k0): 1}, 
             {(i0+1, k0): 1, (i0+1, k0+1): -1}),
            ({(i0, j0): 1}, 
             {(j0, k0+1): 1, (j0+1, k0+1): -1}, 
             {(i0, k0+1): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0+1): 1}, 
             {(j0+1, k0): 1, (j0, k0): -1}, 
             {(i0, k0): 1, (i0+1, k0): 1}),
            ({(i0, j0): 1, (i0, j0+1): 1}, 
             {(j0+1, k0+1): 1}, 
             {(i0, k0): -1, (i0, k0+1): 1}),
            ({(i0+1, j0): 1, (i0, j0): -1}, 
             {(j0, k0): 1, (j0, k0+1): 1}, 
             {(i0+1, k0+1): 1}),
            ({(i0, j0+1): 1, (i0+1, j0+1): -1}, 
             {(j0+1, k0): 1, (j0+1, k0+1): 1}, 
             {(i0, k0): 1}),
        ]
        
        for u, v, w in terms:
            u_scaled = {k: val * sign for k, val in u.items()}
            v_scaled = {k: val * sign for k, val in v.items()}
            w_scaled = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_scaled, v_scaled, w_scaled))
            
    def add_schoolbook_2x2x1(a_offset, b_offset, c_offset, sign=1):
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset
        
        terms = [
            ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
            ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
            ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
            ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
        ]
        
        for u, v, w in terms:
            u_scaled = {k: val * sign for k, val in u.items()}
            v_scaled = {k: val * sign for k, val in v.items()}
            w_scaled = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_scaled, v_scaled, w_scaled))
            
    add_strassen_2x2x2((0, 0), (0, 0), (0, 0))
    add_strassen_2x2x2((0, 0), (0, 2), (0, 2))
    add_schoolbook_2x2x1((0, 0), (0, 4), (0, 4))
    
    add_strassen_2x2x2((0, 2), (2, 0), (0, 0))
    add_strassen_2x2x2((0, 2), (2, 2), (0, 2))
    add_schoolbook_2x2x1((0, 2), (2, 4), (0, 4))
    
    base_rank = len(rank_terms)
    
    def build_matrices(terms):
        r = len(terms)
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j][col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k][col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i][col] = coeff
        return U, V, W
        
    def verify_decomposition(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U[0])):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True
        
    from fractions import Fraction
    
    def try_remove_terms(terms, remove_indices):
        """Try to find coefficients for terms excluding remove_indices."""
        if time.time() > deadline:
            return None
            
        r = len(terms) - len(remove_indices)
        remaining = [terms[i] for i in range(len(terms)) if i not in remove_indices]
        if not remaining:
            return None
            
        U_rem, V_rem, W_rem = build_matrices(remaining)
        
        num_eq = n * m * p
        M = [[Fraction(0)] * r for _ in range(num_eq)]
        b = [Fraction(1)] * num_eq
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for col in range(r):
                        M[eq_idx][col] = Fraction(U_rem[i * m + j][col]) * Fraction(V_rem[j * p + k][col]) * Fraction(W_rem[k * n + i][col])
                        
        aug = [row[:] + [b[i]] for i, row in enumerate(M)]
        
        rank_found = 0
        pivot_cols = []
        for col in range(r):
            if rank_found >= num_eq:
                break
            pivot = None
            for row in range(rank_found, num_eq):
                if aug[row][col] != 0:
                    pivot = row
                    break
            if pivot is None:
                continue
            aug[rank_found], aug[pivot] = aug[pivot], aug[rank_found]
            pivot_val = aug[rank_found][col]
            for j in range(col, r + 1):
                aug[rank_found][j] = aug[rank_found][j] / pivot_val
            for row in range(num_eq):
                if row != rank_found and aug[row][col] != 0:
                    factor = aug[row][col]
                    for j in range(col, r + 1):
                        aug[row][j] -= factor * aug[rank_found][j]
            pivot_cols.append(col)
            rank_found += 1
            
        for row in range(rank_found, num_eq):
            if any(aug[row][j] != 0 for j in range(r)) and aug[row][r] != 0:
                return None
                
        alpha = [Fraction(0)] * r
        for col in range(r):
            if col in pivot_cols:
                row_idx = pivot_cols.index(col)
                alpha[col] = aug[row_idx][r]
                
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for col in range(r):
                        s += alpha[col] * Fraction(U_rem[i * m + j][col]) * Fraction(V_rem[j * p + k][col]) * Fraction(W_rem[k * n + i][col])
                    if s != 1:
                        return None
                        
        valid = True
        for col in range(r):
            a = alpha[col]
            if a == 0:
                continue
            for (i, j), coeff in remaining[col][0].items():
                val = a * Fraction(coeff)
                if (2 * val).denominator != 1:
                    valid = False
                    break
            if not valid:
                break
                
        if not valid:
            return None
            
        U_final = [[0] * r for _ in range(n * m)]
        V_final = [[0] * r for _ in range(m * p)]
        W_final = [[0] * r for _ in range(p * n)]
        
        for col in range(r):
            a = alpha[col]
            u_dict, v_dict, w_dict = remaining[col]
            for (i, j), coeff in u_dict.items():
                U_final[i * m + j][col] = float(a * Fraction(coeff))
            for (j, k), coeff in v_dict.items():
                V_final[j * p + k][col] = float(coeff)
            for (i, k), coeff in w_dict.items():
                W_final[k * n + i][col] = float(coeff)
                
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for col in range(r):
                        s += U_final[i * m + j][col] * V_final[j * p + k][col] * W_final[k * n + i][col]
                    if abs(s - 1) > 1e-9:
                        return None
                        
        return U_final, V_final, W_final

    best_result = None
    best_rank = base_rank
    
    # Try removing 1 term
    for t in range(base_rank):
        if time.time() > deadline:
            break
        result = try_remove_terms(rank_terms, [t])
        if result is not None:
            U_final, V_final, W_final = result
            if len(U_final[0]) < best_rank:
                best_rank = len(U_final[0])
                best_result = (U_final, V_final, W_final)
                
    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
        
    # Try removing 2 terms
    # Limit the number of pairs to check to avoid timeout
    import itertools
    pairs_checked = 0
    max_pairs = 100 # Tune this based on time
    
    for t1 in range(base_rank):
        if time.time() > deadline:
            break
        for t2 in range(t1 + 1, base_rank):
            if time.time() > deadline:
                break
            if pairs_checked >= max_pairs:
                break
            result = try_remove_terms(rank_terms, [t1, t2])
            pairs_checked += 1
            if result is not None:
                U_final, V_final, W_final = result
                if len(U_final[0]) < best_rank:
                    best_rank = len(U_final[0])
                    best_result = (U_final, V_final, W_final)
                    break
        if best_result is not None and best_rank < base_rank - 1:
            break
            
    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
        
    U, V, W = build_matrices(rank_terms)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
