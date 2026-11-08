import numpy as np
import time

# EVOLVE-BLOCK-START
_n, _m, _p = 4, 4, 4

def _build_decomposition():
    """Build the rank-49 Strassen-based decomposition, then reduce rank by
    exploiting linear dependencies among the rank-1 terms."""
    _n, _m, _p = 4, 4, 4
    rank = 49

    U = [[0] * rank for _ in range(_n * _m)]
    V = [[0] * rank for _ in range(_m * _p)]
    W = [[0] * rank for _ in range(_p * _n)]

    def get_U_index(row, col):
        return row * _m + col

    def get_V_index(row, col):
        return row * _p + col

    def get_W_index(row, col):
        return col * _n + row

    def a_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)

    def b_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)

    def c_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)

    L_terms = []
    R_terms = []
    contrib_terms = []

    # Term 0: P1 = (A11 + A22)(B11 + B22)
    L0 = [[{} for _ in range(2)] for _ in range(2)]
    R0 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            ar, ac = a_elem(0, 0, r, c)
            dr, dc = a_elem(1, 1, r, c)
            L0[r][c] = {(ar, ac): 1, (dr, dc): 1}
            er, ec = b_elem(0, 0, r, c)
            hr, hc = b_elem(1, 1, r, c)
            R0[r][c] = {(er, ec): 1, (hr, hc): 1}
    contrib0 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(0, 0, r, c)
            contrib0[(r, c)] = [(cr, cc, 1)]
            cr2, cc2 = c_elem(1, 1, r, c)
            contrib0[(r, c)].append((cr2, cc2, 1))
    L_terms.append(L0)
    R_terms.append(R0)
    contrib_terms.append(contrib0)

    # Term 1: P2 = (A21 + A22)B11
    L1 = [[{} for _ in range(2)] for _ in range(2)]
    R1 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            ar, ac = a_elem(1, 0, r, c)
            dr, dc = a_elem(1, 1, r, c)
            L1[r][c] = {(ar, ac): 1, (dr, dc): 1}
            er, ec = b_elem(0, 0, r, c)
            R1[r][c] = {(er, ec): 1}
    contrib1 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(1, 0, r, c)
            contrib1[(r, c)] = [(cr, cc, 1)]
            cr2, cc2 = c_elem(1, 1, r, c)
            contrib1[(r, c)].append((cr2, cc2, -1))
    L_terms.append(L1)
    R_terms.append(R1)
    contrib_terms.append(contrib1)

    # Term 2: P3 = A11(B12 - B22)
    L2 = [[{} for _ in range(2)] for _ in range(2)]
    R2 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            ar, ac = a_elem(0, 0, r, c)
            L2[r][c] = {(ar, ac): 1}
            fr, fc = b_elem(0, 1, r, c)
            hr, hc = b_elem(1, 1, r, c)
            R2[r][c] = {(fr, fc): 1, (hr, hc): -1}
    contrib2 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(0, 1, r, c)
            contrib2[(r, c)] = [(cr, cc, 1)]
            cr2, cc2 = c_elem(1, 1, r, c)
            contrib2[(r, c)].append((cr2, cc2, 1))
    L_terms.append(L2)
    R_terms.append(R2)
    contrib_terms.append(contrib2)

    # Term 3: P4 = A22(B21 - B11)
    L3 = [[{} for _ in range(2)] for _ in range(2)]
    R3 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            dr, dc = a_elem(1, 1, r, c)
            L3[r][c] = {(dr, dc): 1}
            gr, gc = b_elem(1, 0, r, c)
            er, ec = b_elem(0, 0, r, c)
            R3[r][c] = {(gr, gc): 1, (er, ec): -1}
    contrib3 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(0, 0, r, c)
            contrib3[(r, c)] = [(cr, cc, 1)]
            cr2, cc2 = c_elem(1, 0, r, c)
            contrib3[(r, c)].append((cr2, cc2, 1))
    L_terms.append(L3)
    R_terms.append(R3)
    contrib_terms.append(contrib3)

    # Term 4: P5 = (A11 + A12)B22
    L4 = [[{} for _ in range(2)] for _ in range(2)]
    R4 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            ar, ac = a_elem(0, 0, r, c)
            br, bc = a_elem(0, 1, r, c)
            L4[r][c] = {(ar, ac): 1, (br, bc): 1}
            hr, hc = b_elem(1, 1, r, c)
            R4[r][c] = {(hr, hc): 1}
    contrib4 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(0, 0, r, c)
            contrib4[(r, c)] = [(cr, cc, -1)]
            cr2, cc2 = c_elem(0, 1, r, c)
            contrib4[(r, c)].append((cr2, cc2, 1))
    L_terms.append(L4)
    R_terms.append(R4)
    contrib_terms.append(contrib4)

    # Term 5: P6 = (A21 - A11)(B11 + B12)
    L5 = [[{} for _ in range(2)] for _ in range(2)]
    R5 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            cr, cc = a_elem(1, 0, r, c)
            ar, ac = a_elem(0, 0, r, c)
            L5[r][c] = {(cr, cc): 1, (ar, ac): -1}
            er, ec = b_elem(0, 0, r, c)
            fr, fc = b_elem(0, 1, r, c)
            R5[r][c] = {(er, ec): 1, (fr, fc): 1}
    contrib5 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(1, 1, r, c)
            contrib5[(r, c)] = [(cr, cc, 1)]
    L_terms.append(L5)
    R_terms.append(R5)
    contrib_terms.append(contrib5)

    # Term 6: P7 = (A12 - A22)(B21 + B22)
    L6 = [[{} for _ in range(2)] for _ in range(2)]
    R6 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            br, bc = a_elem(0, 1, r, c)
            dr, dc = a_elem(1, 1, r, c)
            L6[r][c] = {(br, bc): 1, (dr, dc): -1}
            gr, gc = b_elem(1, 0, r, c)
            hr, hc = b_elem(1, 1, r, c)
            R6[r][c] = {(gr, gc): 1, (hr, hc): 1}
    contrib6 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = c_elem(0, 0, r, c)
            contrib6[(r, c)] = [(cr, cc, 1)]
    L_terms.append(L6)
    R_terms.append(R6)
    contrib_terms.append(contrib6)

    sub_contrib = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)]
    ]

    # Store per-term factor data for later reconstruction
    term_factors = []  # list of (U_sparse, V_sparse, W_sparse) where each is dict {flat_idx: coeff}

    for t in range(7):
        L_t = L_terms[t]
        R_t = R_terms[t]
        contrib_t = contrib_terms[t]

        a = L_t[0][0]
        b = L_t[0][1]
        c = L_t[1][0]
        d = L_t[1][1]
        e = R_t[0][0]
        f = R_t[0][1]
        g = R_t[1][0]
        h = R_t[1][1]

        sub_products = []
        x1 = {}
        for key, coeff in a.items():
            x1[key] = x1.get(key, 0) + coeff
        for key, coeff in d.items():
            x1[key] = x1.get(key, 0) + coeff
        y1 = {}
        for key, coeff in e.items():
            y1[key] = y1.get(key, 0) + coeff
        for key, coeff in h.items():
            y1[key] = y1.get(key, 0) + coeff
        sub_products.append((x1, y1))

        x2 = {}
        for key, coeff in c.items():
            x2[key] = x2.get(key, 0) + coeff
        for key, coeff in d.items():
            x2[key] = x2.get(key, 0) + coeff
        y2 = dict(e)
        sub_products.append((x2, y2))

        x3 = dict(a)
        y3 = {}
        for key, coeff in f.items():
            y3[key] = y3.get(key, 0) + coeff
        for key, coeff in h.items():
            y3[key] = y3.get(key, 0) - coeff
        sub_products.append((x3, y3))

        x4 = dict(d)
        y4 = {}
        for key, coeff in g.items():
            y4[key] = y4.get(key, 0) + coeff
        for key, coeff in e.items():
            y4[key] = y4.get(key, 0) - coeff
        sub_products.append((x4, y4))

        x5 = {}
        for key, coeff in a.items():
            x5[key] = x5.get(key, 0) + coeff
        for key, coeff in b.items():
            x5[key] = x5.get(key, 0) + coeff
        y5 = dict(h)
        sub_products.append((x5, y5))

        x6 = {}
        for key, coeff in c.items():
            x6[key] = x6.get(key, 0) + coeff
        for key, coeff in a.items():
            x6[key] = x6.get(key, 0) - coeff
        y6 = {}
        for key, coeff in e.items():
            y6[key] = y6.get(key, 0) + coeff
        for key, coeff in f.items():
            y6[key] = y6.get(key, 0) + coeff
        sub_products.append((x6, y6))

        x7 = {}
        for key, coeff in b.items():
            x7[key] = x7.get(key, 0) + coeff
        for key, coeff in d.items():
            x7[key] = x7.get(key, 0) - coeff
        y7 = {}
        for key, coeff in g.items():
            y7[key] = y7.get(key, 0) + coeff
        for key, coeff in h.items():
            y7[key] = y7.get(key, 0) + coeff
        sub_products.append((x7, y7))

        for k in range(7):
            x_k, y_k = sub_products[k]
            rank_idx = t * 7 + k

            U_sparse = {}
            V_sparse = {}
            W_sparse = {}

            for (row, col), coeff in x_k.items():
                if coeff != 0:
                    u_idx = get_U_index(row, col)
                    U_sparse[u_idx] = U_sparse.get(u_idx, 0) + coeff
                    U[u_idx][rank_idx] += coeff

            for (row, col), coeff in y_k.items():
                if coeff != 0:
                    v_idx = get_V_index(row, col)
                    V_sparse[v_idx] = V_sparse.get(v_idx, 0) + coeff
                    V[v_idx][rank_idx] += coeff

            for (pr, pc, sub_coeff) in sub_contrib[k]:
                if (pr, pc) in contrib_t:
                    for (out_row, out_col, block_coeff) in contrib_t[(pr, pc)]:
                        w_idx = get_W_index(out_row, out_col)
                        W_sparse[w_idx] = W_sparse.get(w_idx, 0) + sub_coeff * block_coeff
                        W[w_idx][rank_idx] += sub_coeff * block_coeff

            term_factors.append((U_sparse, V_sparse, W_sparse))

    # Now check for linear dependencies among the 49 rank-1 tensors
    # Build the 4D tensor: T_full[idx_u, idx_v, idx_w, term]
    # We want to find the minimum number of terms that span the same tensor.
    #
    # The tensor T = sum_r T_r where T_r is rank-1 (outer product of U[:,r], V[:,r], W[:,r])
    # We need sum_r T_r = T_target (the matmul tensor)
    # If some T_r is a linear combination of others, we can eliminate it.
    #
    # Approach: flatten each T_r into a vector of length 16*16*16 = 4096
    # Then find the rank of the 49 vectors. If rank < 49, find a basis subset.
    
    # Build flattened tensors
    T_flat = np.zeros((4096, 49), dtype=np.float64)
    for r in range(49):
        Us = term_factors[r][0]
        Vs = term_factors[r][1]
        Ws = term_factors[r][2]
        for (ui, uc) in Us.items():
            for (vi, vc) in Vs.items():
                for (wi, wc) in Ws.items():
                    idx = ui * 256 + vi * 16 + wi
                    T_flat[idx, r] += uc * vc * wc

    # Compute rank via SVD
    try:
        U_svd, s_vals, Vt = np.linalg.svd(T_flat, full_matrices=False)
        tol = 1e-10
        rank_found = int(np.sum(s_vals > tol))
    except np.linalg.LinAlgError:
        rank_found = 49

    if rank_found >= 49:
        return U, V, W

    # Find a minimal spanning subset of the 49 terms
    # Use greedy: pick terms in order of largest SVD contribution
    # Actually, simpler: use the SVD left singular vectors to identify independent directions
    # Then greedily select terms that maximize span.
    
    # Greedy selection: start with empty set, add the term that maximizes rank increase
    selected = []
    current_rank = 0
    remaining = list(range(49))
    
    while current_rank < rank_found and remaining:
        best_idx = None
        best_rank = current_rank
        for idx in remaining:
            test_vecs = [T_flat[:, i] for i in selected + [idx]]
            test_mat = np.column_stack(test_vecs)
            try:
                s_test = np.linalg.svd(test_mat, full_matrices=False, compute_uv=False)
                r_test = int(np.sum(s_test > tol))
            except np.linalg.LinAlgError:
                r_test = len(selected) + 1
            if r_test > best_rank:
                best_rank = r_test
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        current_rank = best_rank
    
    new_rank = len(selected)
    if new_rank >= 49:
        return U, V, W

    # Rebuild U, V, W with only the selected terms
    U_new = [[0] * new_rank for _ in range(_n * _m)]
    V_new = [[0] * new_rank for _ in range(_m * _p)]
    W_new = [[0] * new_rank for _ in range(_p * _n)]

    for new_r, old_r in enumerate(selected):
        Us, Vs, Ws = term_factors[old_r]
        for ui, uc in Us.items():
            U_new[ui][new_r] = uc
        for vi, vc in Vs.items():
            V_new[vi][new_r] = vc
        for wi, wc in Ws.items():
            W_new[wi][new_r] = wc

    # Verify the decomposition is still exact
    # Check: sum_r U_new[:,r] * V_new[:,r] * W_new[:,r] should equal original T
    T_check = np.zeros((16, 16, 16), dtype=np.float64)
    for r in range(new_rank):
        for ui in range(16):
            if U_new[ui][r] != 0:
                for vi in range(16):
                    if V_new[vi][r] != 0:
                        for wi in range(16):
                            if W_new[wi][r] != 0:
                                T_check[ui, vi, wi] += U_new[ui][r] * V_new[vi][r] * W_new[wi][r]
    
    T_orig = np.zeros((16, 16, 16), dtype=np.float64)
    for r in range(49):
        Us, Vs, Ws = term_factors[r]
        for ui, uc in Us.items():
            for vi, vc in Vs.items():
                for wi, wc in Ws.items():
                    T_orig[ui, vi, wi] += uc * vc * wc
    
    if np.max(np.abs(T_check - T_orig)) > 1e-8:
        return U, V, W
    
    return U_new, V_new, W_new

# Build once at import time
_U, _V, _W = _build_decomposition()

def construct():
    """
    Rank-reduced decomposition for 4x4 matrix multiplication using recursive Strassen
    with linear dependency elimination.
    Returns precomputed factor matrices.
    """
    return {"U": _U, "V": _V, "W": _W}
# EVOLVE-BLOCK-END
