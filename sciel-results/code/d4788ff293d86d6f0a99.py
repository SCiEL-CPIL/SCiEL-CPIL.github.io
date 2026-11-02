import time
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 5, 6]

def _make_zero_matrices(n, m, p, rank):
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    return U, V, W

def _add_strassen_block(U, V, W, col,
                        i1, i2, j1, j2, k1, k2,
                        n, m, p):
    # Strassen factors for a 2x2x2 block
    # factor 0: (A00+A11)*(B00+B11)
    U[i1*m+j1][col+0] += 1
    U[i2*m+j2][col+0] += 1
    V[j1*p+k1][col+0] += 1
    V[j2*p+k2][col+0] += 1
    W[k1*n+i1][col+0] += 1
    W[k1*n+i2][col+0] += -1
    W[k2*n+i1][col+0] += -1
    W[k2*n+i2][col+0] += -1

    # factor 1: (A10+A11)*B00
    U[i2*m+j1][col+1] += 1
    U[i2*m+j2][col+1] += 1
    V[j1*p+k1][col+1] += 1
    W[k2*n+i1][col+1] += 1
    W[k2*n+i2][col+1] += 1

    # factor 2: A00*(B01-B11)
    U[i1*m+j1][col+2] += 1
    V[j1*p+k2][col+2] += 1
    V[j2*p+k2][col+2] += -1
    W[k1*n+i2][col+2] += 1
    W[k2*n+i2][col+2] += 1

    # factor 3: A11*(B10-B00)
    U[i2*m+j2][col+3] += 1
    V[j2*p+k1][col+3] += 1
    V[j1*p+k1][col+3] += -1
    W[k1*n+i1][col+3] += 1

    # factor 4: (A00+A01)*B11
    U[i1*m+j1][col+4] += 1
    U[i1*m+j2][col+4] += 1
    V[j2*p+k2][col+4] += 1
    W[k1*n+i1][col+4] += -1
    W[k1*n+i2][col+4] += -1

    # factor 5: (A10-A00)*(B00+B01)
    U[i2*m+j1][col+5] += 1
    U[i1*m+j1][col+5] += -1
    V[j1*p+k1][col+5] += 1
    V[j1*p+k2][col+5] += 1
    W[k2*n+i1][col+5] += 1
    W[k2*n+i2][col+5] += 1

    # factor 6: (A01-A11)*(B10+B11)
    U[i1*m+j2][col+6] += 1
    U[i2*m+j2][col+6] += -1
    V[j2*p+k1][col+6] += 1
    V[j2*p+k2][col+6] += 1
    W[k1*n+i1][col+6] += -1
    W[k2*n+i1][col+6] += -1

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    nm, mp, pn = n*m, m*p, p*n
    
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_est = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
    max_err = float(np.max(np.abs(T_est - T)))
    return max_err < tol, max_err

def _verify_half_integer(U, V, W, tol=1e-12):
    for arr_name, arr in [('U', U), ('V', V), ('W', W)]:
        arr_f = np.array(arr, dtype=float)
        doubled = arr_f * 2
        rounded = np.round(doubled)
        if not np.allclose(doubled, rounded, atol=tol):
            return False
    return True

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U, V, W = _make_zero_matrices(n, m, p, rank)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W, rank

def _find_max_disjoint_strassen_blocks(n, m, p, deadline=None):
    all_blocks = []
    for i1 in range(n-1):
        for i2 in range(i1+1, n):
            for j1 in range(m-1):
                for j2 in range(j1+1, m):
                    for k1 in range(p-1):
                        for k2 in range(k1+1, p):
                            all_blocks.append((i1, i2, j1, j2, k1, k2))
    
    num_blocks = len(all_blocks)
    
    # Precompute compatibility: two blocks are compatible if they are NOT
    # (sharing i AND sharing j AND sharing k)
    compat = [[True]*num_blocks for _ in range(num_blocks)]
    for a in range(num_blocks):
        i1a, i2a, j1a, j2a, k1a, k2a = all_blocks[a]
        i_set_a = {i1a, i2a}
        j_set_a = {j1a, j2a}
        k_set_a = {k1a, k2a}
        for b in range(a+1, num_blocks):
            i1b, i2b, j1b, j2b, k1b, k2b = all_blocks[b]
            i_set_b = {i1b, i2b}
            j_set_b = {j1b, j2b}
            k_set_b = {k1b, k2b}
            if (i_set_a & i_set_b) and (j_set_a & j_set_b) and (k_set_a & k_set_b):
                compat[a][b] = False
                compat[b][a] = False
    
    # Use greedy + local search for maximum independent set
    # First, greedy by degree
    degrees = [sum(1 for j in range(num_blocks) if not compat[i][j]) for i in range(num_blocks)]
    
    # Greedy: pick block with most conflicts first, then remove it and its incompatibles
    remaining = set(range(num_blocks))
    selected = []
    
    while remaining:
        # Pick the block with highest degree among remaining
        best = max(remaining, key=lambda x: sum(1 for y in remaining if not compat[x][y]))
        selected.append(best)
        # Remove best and all blocks incompatible with best
        to_remove = {best}
        for y in remaining:
            if not compat[best][y]:
                to_remove.add(y)
        remaining -= to_remove
    
    greedy_size = len(selected)
    
    # Try to improve with local search
    if deadline is not None and time.time() > deadline - 5:
        return [all_blocks[i] for i in selected]
    
    # Local search: try to add blocks that were removed
    removed_blocks = [b for b in range(num_blocks) if b not in selected]
    improved = True
    while improved and (deadline is None or time.time() < deadline - 5):
        improved = False
        selected_set = set(selected)
        for rb in removed_blocks:
            if rb in selected_set:
                continue
            # Check if rb is compatible with all selected
            if all(compat[rb][s] for s in selected):
                selected.append(rb)
                selected_set.add(rb)
                improved = True
                break
    
    # Try swaps: remove one selected, add two removed
    if deadline is None or time.time() < deadline - 5:
        for idx in range(len(selected)):
            if deadline is not None and time.time() > deadline - 5:
                break
            trial = selected[:idx] + selected[idx+1:]
            trial_set = set(trial)
            # Try to add as many removed blocks as possible
            added = 0
            for rb in removed_blocks:
                if rb in trial_set:
                    continue
                if all(compat[rb][s] for s in trial):
                    trial.append(rb)
                    trial_set.add(rb)
                    added += 1
            if len(trial) > len(selected):
                selected = trial
                improved = True
    
    return [all_blocks[i] for i in selected]

def _build_rank_optimized(n, m, p, deadline=None):
    blocks = _find_max_disjoint_strassen_blocks(n, m, p, deadline=deadline)
    num_blocks = len(blocks)
    
    total_terms = n * m * p
    terms_covered = num_blocks * 8
    remaining_terms = total_terms - terms_covered
    rank = num_blocks * 7 + remaining_terms
    
    U, V, W = _make_zero_matrices(n, m, p, rank)
    
    covered = set()
    for block in blocks:
        i1, i2, j1, j2, k1, k2 = block
        for i in [i1, i2]:
            for j in [j1, j2]:
                for k in [k1, k2]:
                    covered.add((i, j, k))
    
    col = 0
    for block in blocks:
        i1, i2, j1, j2, k1, k2 = block
        _add_strassen_block(U, V, W, col,
                            i1=i1, i2=i2, j1=j1, j2=j2, k1=k1, k2=k2,
                            n=n, m=m, p=p)
        col += 7
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    assert col == rank, f"Expected col={rank}, got col={col}"
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    
    U_opt, V_opt, W_opt, rank_opt = _build_rank_optimized(n, m, p, deadline=deadline)
    
    valid_opt, max_err_opt = _verify_decomposition(U_opt, V_opt, W_opt, n, m, p)
    
    if valid_opt and _verify_half_integer(U_opt, V_opt, W_opt):
        return {"U": U_opt, "V": V_opt, "W": W_opt}
    
    U_sb, V_sb, W_sb, rank_sb = _build_schoolbook(n, m, p)
    return {"U": U_sb, "V": V_sb, "W": W_sb}
