"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def verify_numpy(U, V, W, rank):
        """Verify decomposition using numpy for speed."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i*m+j, :]
                    v_row = V_np[j*p+k, :]
                    w_row = W_np[k*n+i, :]
                    val = np.sum(u_row * v_row * w_row)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def verify_sparse(U, V, W, rank):
        """Verify decomposition using sparse representation for speed."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_r = i*m + j
                    v_r = j*p + k
                    w_r = k*n + i
                    total = 0.0
                    for c in range(rank):
                        uv = U[u_r][c] * V[v_r][c]
                        if uv != 0:
                            total += uv * W[w_r][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
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
        return {"U": U, "V": V, "W": W}

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (U_pattern, V_pattern, W_pattern) where patterns are
        dicts mapping (local_i, local_j) -> value for U, etc.
        """
        terms = []
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

    def build_strassen_decomposition(i_groups, j_groups, k_groups):
        """
        Build a decomposition using Strassen on 2x2x2 blocks and schoolbook elsewhere.
        """
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7
                    else:
                        r = si * sj * sk
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        strassen_terms = strassen_2x2x2_terms()
        
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                for (u_pat, v_pat, w_pat) in strassen_terms:
                    for (li, lj), val_u in u_pat.items():
                        gi, gj = ig[li], jg[lj]
                        U[gi * m + gj][col] = val_u
                    for (lj, lk), val_v in v_pat.items():
                        gj, gk = jg[lj], kg[lk]
                        V[gj * p + gk][col] = val_v
                    for (lk, li), val_w in w_pat.items():
                        gk, gi = kg[lk], ig[li]
                        W[gk * n + gi][col] = val_w
                    col += 1
            else:
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    def get_all_splits(dim):
        """Get all ways to split a dimension into groups of size 1 or 2."""
        splits = []
        def backtrack(remaining, current):
            if not remaining:
                splits.append(list(current))
                return
            if len(remaining) >= 2:
                current.append([remaining[0], remaining[1]])
                backtrack(remaining[2:], current)
                current.pop()
            current.append([remaining[0]])
            backtrack(remaining[1:], current)
            current.pop()
        backtrack(list(range(dim)), [])
        return splits

    def try_improved_decomposition(current_rank):
        """
        Try to find a lower-rank decomposition by trying different groupings.
        """
        if time.time() > deadline:
            return None
        
        best_result = None
        best_rank = current_rank
        
        # Generate all possible splits
        i_splits = get_all_splits(n)
        j_splits = get_all_splits(m)
        k_splits = get_all_splits(p)
        
        for i_g in i_splits:
            for j_g in j_splits:
                for k_g in k_splits:
                    if time.time() > deadline:
                        return best_result
                    
                    # Calculate rank for this grouping
                    total_rank = 0
                    for ig in i_g:
                        for jg in j_g:
                            for kg in k_g:
                                si, sj, sk = len(ig), len(jg), len(kg)
                                if si == 2 and sj == 2 and sk == 2:
                                    total_rank += 7
                                else:
                                    total_rank += si * sj * sk
                    
                    if total_rank >= best_rank:
                        continue
                    
                    # Build the decomposition
                    result = build_strassen_decomposition(i_g, j_g, k_g)
                    
                    # Verify
                    if verify_sparse(result["U"], result["V"], result["W"], total_rank):
                        best_result = {"U": result["U"], "V": result["V"], "W": result["W"]}
                        best_rank = total_rank
        
        return best_result

    def random_search_improvement(current_rank, max_attempts=200):
        """
        Try random modifications to find lower-rank decompositions.
        This uses a more sophisticated approach: start with a valid decomposition
        and try to merge or simplify terms.
        """
        if time.time() > deadline:
            return None
        
        best_result = None
        best_rank = current_rank
        
        # Start with the best block decomposition
        i_splits = get_all_splits(n)
        j_splits = get_all_splits(m)
        k_splits = get_all_splits(p)
        
        # Find the best block decomposition first
        best_block = None
        best_block_rank = current_rank
        
        for i_g in i_splits:
            for j_g in j_splits:
                for k_g in k_splits:
                    total_rank = 0
                    for ig in i_g:
                        for jg in j_g:
                            for kg in k_g:
                                si, sj, sk = len(ig), len(jg), len(kg)
                                if si == 2 and sj == 2 and sk == 2:
                                    total_rank += 7
                                else:
                                    total_rank += si * sj * sk
                    if total_rank < best_block_rank:
                        result = build_strassen_decomposition(i_g, j_g, k_g)
                        if verify_sparse(result["U"], result["V"], result["W"], total_rank):
                            best_block = result
                            best_block_rank = total_rank
        
        if best_block is None:
            return None
        
        U = [row[:] for row in best_block["U"]]
        V = [row[:] for row in best_block["V"]]
        W = [row[:] for row in best_block["W"]]
        rank = best_block["rank"]
        
        # Try to reduce rank by zeroing out random entries
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            U_try = [row[:] for row in U]
            V_try = [row[:] for row in V]
            W_try = [row[:] for row in W]
            
            # Randomly zero out some entries
            for _ in range(3):
                mat = random.choice([U_try, V_try, W_try])
                r = random.randint(0, len(mat) - 1)
                c = random.randint(0, rank - 1)
                mat[r][c] = 0
            
            if verify_sparse(U_try, V_try, W_try, rank):
                # Check if we can reduce the actual rank
                # Find columns that are all zero
                active_cols = []
                for c in range(rank):
                    col_nonzero = False
                    for r in range(len(U_try)):
                        if U_try[r][c] != 0:
                            col_nonzero = True
                            break
                    if not col_nonzero:
                        for r in range(len(V_try)):
                            if V_try[r][c] != 0:
                                col_nonzero = True
                                break
                    if not col_nonzero:
                        for r in range(len(W_try)):
                            if W_try[r][c] != 0:
                                col_nonzero = True
                                break
                    if col_nonzero:
                        active_cols.append(c)
                
                if len(active_cols) < best_rank:
                    # Compress to active columns
                    U_new = [[U_try[r][c] for c in active_cols] for r in range(len(U_try))]
                    V_new = [[V_try[r][c] for c in active_cols] for r in range(len(V_try))]
                    W_new = [[W_try[r][c] for c in active_cols] for r in range(len(W_try))]
                    new_rank = len(active_cols)
                    
                    if verify_sparse(U_new, V_new, W_new, new_rank):
                        best_result = {"U": U_new, "V": V_new, "W": W_new}
                        best_rank = new_rank
                        U = U_new
                        V = V_new
                        W = W_new
                        rank = new_rank
        
        return best_result

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Try Strassen-based decomposition with the default grouping
    i_groups_default = [[0, 1], [2]]
    j_groups_default = [[0, 1], [2, 3], [4]]
    k_groups_default = [[0, 1], [2, 3], [4, 5]]
    
    strassen_result = build_strassen_decomposition(i_groups_default, j_groups_default, k_groups_default)
    strassen_rank = strassen_result.pop("rank")
    
    if verify_sparse(strassen_result["U"], strassen_result["V"], strassen_result["W"], strassen_rank):
        if strassen_rank < best_rank:
            best_result = strassen_result
            best_rank = strassen_rank

    # Try to improve further with different groupings
    if time.time() < deadline - 10:
        improved = try_improved_decomposition(best_rank)
        if improved is not None:
            improved_rank = len(improved["U"][0])
            if improved_rank < best_rank:
                best_result = improved
                best_rank = improved_rank

    # Try random search for further improvements
    if time.time() < deadline - 5:
        random_improved = random_search_improvement(best_rank)
        if random_improved is not None:
            random_rank = len(random_improved["U"][0])
            if random_rank < best_rank:
                best_result = random_improved
                best_rank = random_rank

    return best_result
# EVOLVE-BLOCK-END
