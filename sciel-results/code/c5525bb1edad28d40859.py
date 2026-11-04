import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    deadline = time.time() + 110.0
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W decompose the matrix multiplication tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                ui = i * m + j
                for k in range(p):
                    vk = j * p + k
                    wk = k * n + i
                    total = 0.0
                    for r in range(rank):
                        total += U[ui][r] * V[vk][r] * W[wk][r]
                    if abs(total - 1.0) > tol:
                        return False
        return True

    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    def add_strassen_2x2x2(terms, r1, c1, k1):
        """Add Strassen terms for a 2x2x2 block starting at (r1,c1,k1)."""
        A00 = r1 * m + c1
        A01 = r1 * m + (c1 + 1)
        A10 = (r1 + 1) * m + c1
        A11 = (r1 + 1) * m + (c1 + 1)
        
        B00 = c1 * p + k1
        B01 = c1 * p + (k1 + 1)
        B10 = (c1 + 1) * p + k1
        B11 = (c1 + 1) * p + (k1 + 1)
        
        C00_w = k1 * n + r1
        C01_w = (k1 + 1) * n + r1
        C10_w = k1 * n + (r1 + 1)
        C11_w = (k1 + 1) * n + (r1 + 1)
        
        terms.append(({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00_w: 1, C11_w: 1}))
        terms.append(({A10: 1, A11: 1}, {B00: 1}, {C10_w: 1, C11_w: -1}))
        terms.append(({A00: 1}, {B01: 1, B11: -1}, {C01_w: 1, C11_w: 1}))
        terms.append(({A11: 1}, {B10: 1, B00: -1}, {C00_w: 1, C10_w: 1}))
        terms.append(({A00: 1, A01: 1}, {B11: 1}, {C00_w: -1, C01_w: 1}))
        terms.append(({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11_w: 1}))
        terms.append(({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00_w: 1}))
    
    def build_block_decomposition(partition):
        """Build a block decomposition with Strassen for 2x2x2 blocks."""
        terms = []
        blocks = partition
        
        for r1, r2 in blocks:
            for c1, c2 in blocks:
                for k1, k2 in blocks:
                    ar = r2 - r1
                    ac = c2 - c1
                    bc = k2 - k1
                    
                    if ar == 2 and ac == 2 and bc == 2:
                        add_strassen_2x2x2(terms, r1, c1, k1)
                    else:
                        for ii in range(r1, r2):
                            for jj in range(c1, c2):
                                for kk in range(k1, k2):
                                    u_idx = ii * m + jj
                                    v_idx = jj * p + kk
                                    w_idx = kk * n + ii
                                    terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
        
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for r in range(rank):
            U_dict, V_dict, W_dict = terms[r]
            for idx, val in U_dict.items():
                U[idx][r] = val
            for idx, val in V_dict.items():
                V[idx][r] = val
            for idx, val in W_dict.items():
                W[idx][r] = val
        
        return U, V, W, terms

    # Try different block partitions
    partitions = [
        [(0, 2), (2, 4), (4, 5)],  # [2,2,1]
        [(0, 4), (4, 5)],          # [4,1]
        [(0, 3), (3, 5)],          # [3,2]
    ]
    
    best_U, best_V, best_W = get_schoolbook()
    best_rank = len(best_U[0])
    
    for partition in partitions:
        if time.time() > deadline - 20.0:
            break
        U, V, W, terms = build_block_decomposition(partition)
        rank = len(U[0])
        if verify(U, V, W):
            if rank < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = rank

    # New structural approach: iterative greedy residual minimization
    # Start from the best block decomposition and try to reduce rank by
    # finding linear combinations of terms that can be replaced by a single
    # rank-1 term with smaller support.
    
    if best_rank < n * m * p and time.time() < deadline - 15.0:
        # Rebuild terms from best decomposition
        terms = []
        for r in range(best_rank):
            U_dict = {}
            V_dict = {}
            W_dict = {}
            for idx in range(n * m):
                if best_U[idx][r] != 0:
                    U_dict[idx] = best_U[idx][r]
            for idx in range(m * p):
                if best_V[idx][r] != 0:
                    V_dict[idx] = best_V[idx][r]
            for idx in range(p * n):
                if best_W[idx][r] != 0:
                    W_dict[idx] = best_W[idx][r]
            terms.append((U_dict, V_dict, W_dict))
        
        # Greedy merge: try to replace pairs of terms with a single term
        def try_merge_pair(i, j, terms_list):
            """Try to merge terms i and j into a single term."""
            U1, V1, W1 = terms_list[i]
            U2, V2, W2 = terms_list[j]
            
            # Try coefficients: (a, b) such that a*term1 + b*term2 is rank-1
            for a, b in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                U_comb = {}
                V_comb = {}
                W_comb = {}
                
                for idx, val in U1.items():
                    U_comb[idx] = U_comb.get(idx, 0) + a * val
                for idx, val in U2.items():
                    U_comb[idx] = U_comb.get(idx, 0) + b * val
                
                for idx, val in V1.items():
                    V_comb[idx] = V_comb.get(idx, 0) + a * val
                for idx, val in V2.items():
                    V_comb[idx] = V_comb.get(idx, 0) + b * val
                
                for idx, val in W1.items():
                    W_comb[idx] = W_comb.get(idx, 0) + a * val
                for idx, val in W2.items():
                    W_comb[idx] = W_comb.get(idx, 0) + b * val
                
                # Remove zeros
                U_comb = {k: v for k, v in U_comb.items() if abs(v) > 1e-12}
                V_comb = {k: v for k, v in V_comb.items() if abs(v) > 1e-12}
                W_comb = {k: v for k, v in W_comb.items() if abs(v) > 1e-12}
                
                if not U_comb or not V_comb or not W_comb:
                    continue
                
                # Check if this is a valid rank-1 term
                u_keys = set(U_comb.keys())
                v_keys = set(V_comb.keys())
                w_keys = set(W_comb.keys())
                
                u_list = list(u_keys)
                v_list = list(v_keys)
                w_list = list(w_keys)
                
                rank1 = True
                for u1 in u_list:
                    for v1 in v_list:
                        for w1 in w_list:
                            for u2 in u_list:
                                for v2 in v_list:
                                    for w2 in w_list:
                                        lhs = U_comb.get(u1,0) * V_comb.get(v1,0) * W_comb.get(w1,0) * \
                                              U_comb.get(u2,0) * V_comb.get(v2,0) * W_comb.get(w2,0)
                                        rhs = U_comb.get(u1,0) * V_comb.get(v2,0) * W_comb.get(w1,0) * \
                                              U_comb.get(u2,0) * V_comb.get(v1,0) * W_comb.get(w2,0)
                                        if abs(lhs - rhs) > 1e-9:
                                            rank1 = False
                                            break
                                    if not rank1:
                                        break
                                if not rank1:
                                    break
                            if not rank1:
                                break
                        if not rank1:
                            break
                    if not rank1:
                        break
                if rank1:
                    return (U_comb, V_comb, W_comb)
            return None
        
        # Iteratively try to merge pairs
        merged = True
        while merged and time.time() < deadline - 5.0:
            merged = False
            new_terms = list(terms)
            for i in range(len(new_terms)):
                if time.time() > deadline - 5.0:
                    break
                for j in range(i + 1, len(new_terms)):
                    if time.time() > deadline - 5.0:
                        break
                    result = try_merge_pair(i, j, new_terms)
                    if result is not None:
                        new_terms = [t for idx, t in enumerate(new_terms) if idx != i and idx != j]
                        new_terms.append(result)
                        terms = new_terms
                        merged = True
                        break
                if merged:
                    break
        
        # Rebuild matrices
        rank = len(terms)
        if rank < best_rank:
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            for r in range(rank):
                U_dict, V_dict, W_dict = terms[r]
                for idx, val in U_dict.items():
                    U[idx][r] = val
                for idx, val in V_dict.items():
                    V[idx][r] = val
                for idx, val in W_dict.items():
                    W[idx][r] = val
            if verify(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
