import time
import numpy as np

DEADLINE = 145.0

def construct():
    start_time = time.time()
    n, m, p = 2, 5, 6
    total_entries = n * m * p  # 60

    def time_ok():
        return time.time() - start_time < DEADLINE

    # ---- 1) Build a pool of structured rank-1 tensors (candidate contributions) ----
    # A rank-1 tensor is (u_vec, v_vec, w_vec) with contribution c[i,j,k] = u[i,j]*v[j,k]*w[k,i]
    candidates = []  # list of (u_vec, v_vec, w_vec, contrib_vec)

    def add_candidate(u_sparse, v_sparse, w_sparse):
        u_vec = np.zeros(n * m)
        v_vec = np.zeros(m * p)
        w_vec = np.zeros(p * n)
        for (i, j), val in u_sparse.items():
            u_vec[i * m + j] = val
        for (j, k), val in v_sparse.items():
            v_vec[j * p + k] = val
        for (k, i), val in w_sparse.items():
            w_vec[k * n + i] = val
        contrib = np.zeros(total_entries)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    contrib[i * m * p + j * p + k] = u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
        if np.any(contrib):
            candidates.append((u_vec, v_vec, w_vec, contrib))

    # Type 1: standard unit tensors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_candidate({(i,j): 1}, {(j,k): 1}, {(k,i): 1})

    # Type 2: Strassen 2x2x2 blocks
    def strassen_terms(io, jo, ko):
        terms = []
        terms.append(({(io,jo): 1, (io+1,jo+1): 1},
                      {(jo,ko): 1, (jo+1,ko+1): 1},
                      {(ko,io): 1, (ko+1,io+1): 1}))
        terms.append(({(io+1,jo): 1, (io+1,jo+1): 1},
                      {(jo,ko): 1},
                      {(ko,io+1): 1, (ko+1,io+1): -1}))
        terms.append(({(io,jo): 1},
                      {(jo,ko+1): 1, (jo+1,ko+1): -1},
                      {(ko+1,io): 1, (ko+1,io+1): 1}))
        terms.append(({(io+1,jo+1): 1},
                      {(jo+1,ko): 1, (jo,ko): -1},
                      {(ko,io): 1, (ko,io+1): 1}))
        terms.append(({(io,jo): 1, (io,jo+1): 1},
                      {(jo+1,ko+1): 1},
                      {(ko,io): -1, (ko+1,io): 1}))
        terms.append(({(io+1,jo): 1, (io,jo): -1},
                      {(jo,ko): 1, (jo,ko+1): 1},
                      {(ko+1,io+1): 1}))
        terms.append(({(io,jo+1): 1, (io+1,jo+1): -1},
                      {(jo+1,ko): 1, (jo+1,ko+1): 1},
                      {(ko,io): 1}))
        return terms

    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 1):
                for u_c, v_c, w_c in strassen_terms(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)

    # Type 3: structured 2x3x3 sign-pattern candidates
    for io in range(n - 1):
        for jo in range(m - 2):
            for ko in range(p - 2):
                for i_pat in [[1,1],[1,-1],[-1,1]]:
                    for j_pat in [[1,1,1],[1,1,-1],[1,-1,1],[-1,1,1]]:
                        for k_pat in [[1,1,1],[1,1,-1],[1,-1,1],[-1,1,1]]:
                            u_sparse = {}
                            v_sparse = {}
                            w_sparse = {}
                            for ii in range(2):
                                u_sparse[(io+ii, jo)] = i_pat[ii]
                                u_sparse[(io+ii, jo+1)] = i_pat[ii]
                                u_sparse[(io+ii, jo+2)] = i_pat[ii]
                            for jj in range(3):
                                v_sparse[(jo+jj, ko)] = j_pat[jj]
                                v_sparse[(jo+jj, ko+1)] = j_pat[jj]
                                v_sparse[(jo+jj, ko+2)] = j_pat[jj]
                            for kk in range(3):
                                w_sparse[(ko+kk, io)] = k_pat[kk]
                                w_sparse[(ko+kk, io+1)] = k_pat[kk]
                            add_candidate(u_sparse, v_sparse, w_sparse)

    # Type 4: diagonal-like patterns
    for io in range(n - 1):
        for jo in range(m - 2):
            for ko in range(p - 2):
                for shift in range(3):
                    u_s = {}
                    v_s = {}
                    w_s = {}
                    for ii in range(2):
                        for jj in range(3):
                            if (jj - ii) % 3 == shift:
                                u_s[(io+ii, jo+jj)] = 1
                    for jj in range(3):
                        for kk in range(3):
                            if (kk - jj) % 3 == shift:
                                v_s[(jo+jj, ko+kk)] = 1
                    for kk in range(3):
                        for ii in range(2):
                            if (ii - kk) % 3 == shift:
                                w_s[(ko+kk, io+ii)] = 1
                    add_candidate(u_s, v_s, w_s)

    # Type 5: S3 permutation-based structured candidates on 2x2x2
    S3_elements = [
        [0, 1, 2],
        [1, 0, 2],
        [0, 2, 1],
        [2, 1, 0],
        [1, 2, 0],
        [2, 0, 1],
    ]
    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 1):
                for perm in S3_elements:
                    u_s = {}; v_s = {}; w_s = {}
                    if perm == [0, 1, 2]:
                        u_s = {(io, jo): 1, (io+1, jo+1): 1}
                        v_s = {(jo, ko): 1, (jo+1, ko+1): 1}
                        w_s = {(ko, io): 1, (ko+1, io+1): 1}
                    elif perm == [1, 0, 2]:
                        u_s = {(io, jo+1): 1, (io+1, jo): 1}
                        v_s = {(jo, ko): 1, (jo+1, ko+1): 1}
                        w_s = {(ko, io+1): 1, (ko+1, io): 1}
                    elif perm == [0, 2, 1]:
                        u_s = {(io, jo): 1, (io+1, jo+1): 1}
                        v_s = {(jo, ko+1): 1, (jo+1, ko): 1}
                        w_s = {(ko, io): 1, (ko+1, io+1): 1}
                    elif perm == [2, 1, 0]:
                        u_s = {(io, jo+1): 1, (io+1, jo): 1}
                        v_s = {(jo, ko+1): 1, (jo+1, ko): 1}
                        w_s = {(ko, io+1): 1, (ko+1, io): 1}
                    elif perm == [1, 2, 0]:
                        u_s = {(io, jo+1): 1, (io+1, jo): 1}
                        v_s = {(jo+1, ko): 1, (jo, ko+1): 1}
                        w_s = {(ko, io): 1, (ko+1, io+1): 1}
                    elif perm == [2, 0, 1]:
                        u_s = {(io, jo+1): 1, (io+1, jo): 1}
                        v_s = {(jo+1, ko): 1, (jo, ko+1): 1}
                        w_s = {(ko+1, io): 1, (ko, io+1): 1}
                    add_candidate(u_s, v_s, w_s)

    # Type 6: 2x3x2 structured blocks
    for io in range(n - 1):
        for jo in range(m - 2):
            for ko in range(p - 1):
                for perm in S3_elements:
                    u_s = {}; v_s = {}; w_s = {}
                    for jj in range(3):
                        u_s[(io, jo+jj)] = 1
                        u_s[(io+1, jo+jj)] = 1
                    for jj in range(3):
                        for kk in range(2):
                            if perm[jj] == kk:
                                v_s[(jo+jj, ko+kk)] = 1
                    for kk in range(2):
                        for ii in range(2):
                            w_s[(ko+kk, io+ii)] = 1
                    add_candidate(u_s, v_s, w_s)

    num_candidates = len(candidates)
    # Precompute candidate contribution matrix and norms
    cand_matrix = np.zeros((total_entries, num_candidates))
    norms = np.zeros(num_candidates)
    for idx in range(num_candidates):
        vec = candidates[idx][3]
        cand_matrix[:, idx] = vec
        norms[idx] = np.dot(vec, vec)

    T_target = np.ones(total_entries)

    # ---- 2) Greedy selection to get an initial (possibly over-complete) decomposition ----
    residual = T_target.copy()
    selected_indices = []
    selected_coeffs = []

    max_iter = 200
    for _ in range(max_iter):
        if not time_ok():
            break
        dots = cand_matrix.T @ residual
        coeffs = np.zeros(num_candidates)
        valid = norms > 1e-12
        coeffs[valid] = dots[valid] / norms[valid]
        reductions = dots * coeffs
        reductions[~valid] = -1e18
        best_idx = int(np.argmax(reductions))
        best_reduction = reductions[best_idx]
        best_coeff = coeffs[best_idx]
        if best_reduction < 1e-9:
            break
        selected_indices.append(best_idx)
        selected_coeffs.append(best_coeff)
        residual -= best_coeff * candidates[best_idx][3]
        if np.linalg.norm(residual) < 1e-9:
            break

    # ---- 3) Random restarts with randomized candidate ordering to escape local minima ----
    best_rank = len(selected_indices)
    best_solution = (list(selected_indices), list(selected_coeffs))
    if np.linalg.norm(residual) < 1e-9:
        pass  # record current as best

    # Try a few random restarts with shuffled candidate priority
    rng = np.random.default_rng(42)
    num_restarts = 5
    for restart in range(num_restarts):
        if not time_ok():
            break
        residual_r = T_target.copy()
        sel_r = []
        coeff_r = []
        # Perturb: randomly skip some top candidates
        skip_set = set(rng.choice(num_candidates, size=min(5, num_candidates), replace=False).tolist())
        for _ in range(max_iter):
            if not time_ok():
                break
            dots = cand_matrix.T @ residual_r
            coeffs = np.zeros(num_candidates)
            valid = norms > 1e-12
            coeffs[valid] = dots[valid] / norms[valid]
            reductions = dots * coeffs
            reductions[~valid] = -1e18
            for s in skip_set:
                reductions[s] = -1e18
            best_idx = int(np.argmax(reductions))
            best_reduction = reductions[best_idx]
            best_coeff = coeffs[best_idx]
            if best_reduction < 1e-9:
                break
            sel_r.append(best_idx)
            coeff_r.append(best_coeff)
            residual_r -= best_coeff * candidates[best_idx][3]
            if np.linalg.norm(residual_r) < 1e-9:
                break
        if len(sel_r) < best_rank and np.linalg.norm(residual_r) < 1e-9:
            best_rank = len(sel_r)
            best_solution = (sel_r, coeff_r)

    # ---- 4) Pruning: iteratively remove redundant terms ----
    def try_prune(indices, coeffs):
        changed = True
        while changed and time_ok():
            changed = False
            for trial in range(len(indices)):
                if not time_ok():
                    break
                test_residual = T_target.copy()
                for r, idx in enumerate(indices):
                    if r != trial:
                        test_residual -= coeffs[r] * candidates[idx][3]
                if np.linalg.norm(test_residual) < 1e-8:
                    indices.pop(trial)
                    coeffs.pop(trial)
                    changed = True
                    break
        return indices, coeffs

    best_indices, best_coeffs = try_prune(list(best_solution[0]), list(best_solution[1]))
    best_rank = len(best_indices)

    # ---- 5) Local search: swap one selected term for an unselected one to reduce rank ----
    if time_ok() and best_rank > 10:
        for _ in range(3):  # a few rounds of local search
            if not time_ok():
                break
            improved = False
            for pos in range(len(best_indices)):
                if not time_ok():
                    break
                # Try removing best_indices[pos] and see if we can cover the gap
                removed_idx = best_indices[pos]
                removed_coeff = best_coeffs[pos]
                # Compute residual after removing this term
                res_without = T_target.copy()
                for r, idx in enumerate(best_indices):
                    if r != pos:
                        res_without -= best_coeffs[r] * candidates[idx][3]
                # Try to fill the gap with a single new term
                dots = cand_matrix.T @ res_without
                valid = norms > 1e-12
                coeffs_new = np.zeros(num_candidates)
                coeffs_new[valid] = dots[valid] / norms[valid]
                reductions = dots * coeffs_new
                reductions[~valid] = -1e18
                # Exclude already-selected terms
                for r, idx in enumerate(best_indices):
                    if r != pos:
                        reductions[idx] = -1e18
                best_new = int(np.argmax(reductions))
                best_new_coeff = coeffs_new[best_new]
                # Check if adding this new term makes the residual small
                test_res = res_without - best_new_coeff * candidates[best_new][3]
                if np.linalg.norm(test_res) < 1e-8:
                    # Swap: replace removed_idx with best_new
                    best_indices[pos] = best_new
                    best_coeffs[pos] = best_new_coeff
                    improved = True
                    break
            if not improved:
                break

    # ---- 6) Verify and return ----
    if len(best_indices) <= 60:
        rank_found = len(best_indices)
        U = np.zeros((n * m, rank_found))
        V = np.zeros((m * p, rank_found))
        W = np.zeros((p * n, rank_found))
        for r, idx in enumerate(best_indices):
            coeff = best_coeffs[r]
            u_vec, v_vec, w_vec, _ = candidates[idx]
            U[:, r] = u_vec * coeff
            V[:, r] = v_vec
            W[:, r] = w_vec

        # Verify
        A_test = np.random.randint(-10, 10, (n, m))
        B_test = np.random.randint(-10, 10, (m, p))
        C_test = A_test @ B_test
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0
                for j in range(m):
                    for r in range(rank_found):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_calc[i, k] = val

        if np.allclose(C_test, C_calc, atol=1e-5):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: 54-rank block decomposition
    rank_new = 54
    U_new = np.zeros((n * m, rank_new))
    V_new = np.zeros((m * p, rank_new))
    W_new = np.zeros((p * n, rank_new))
    current_rank = 0

    def add_strassen_block(i_off, j_off, k_off):
        nonlocal current_rank
        U_new[(i_off)*m + (j_off), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = -1
        current_rank += 1
        U_new[(i_off)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = -1
        W_new[(k_off+1)*n + (i_off), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = -1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        U_new[(i_off)*m + (j_off), current_rank] = 1
        U_new[(i_off)*m + (j_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = -1
        W_new[(k_off+1)*n + (i_off), current_rank] = 1
        current_rank += 1
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        U_new[(i_off)*m + (j_off), current_rank] = -1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        V_new[(j_off)*p + (k_off+1), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        U_new[(i_off)*m + (j_off+1), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = -1
        V_new[(j_off+1)*p + (k_off), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        current_rank += 1

    def add_standard_block(i_off, j_off, k_off):
        nonlocal current_rank
        U_new[(i_off)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        current_rank += 1
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        current_rank += 1

    # Use 2x2x6 blocks: for each pair of rows (i,i+1) and column pair (k,k+1),
    # we can use Strassen on 2x2x2 blocks. For 2x5x6:
    # Split into: rows {0,1}, cols {0,1,2,3,4,5}
    # Use Strassen blocks at (0,0,0), (0,0,2), (0,0,4) for j=0,1
    # Standard for j=2 (2x6 = 12)
    # Strassen blocks at (0,3,0), (0,3,2), (0,3,4) for j=3,4
    # Total: 7*3 + 12 + 7*3 = 21 + 12 + 21 = 54
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    for k in range(6):
        add_standard_block(0, 2, k)
    add_strassen_block(0, 3, 0)
    add_strassen_block(0, 3, 2)
    add_strassen_block(0, 3, 4)

    # Verify the decomposition
    A_test = np.random.randint(-10, 10, (n, m))
    B_test = np.random.randint(-10, 10, (m, p))
    C_test = A_test @ B_test
    C_calc = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0
            for j in range(m):
                for r in range(rank_new):
                    val += U_new[i*m+j, r] * V_new[j*p+k, r] * W_new[k*n+i, r]
            C_calc[i, k] = val

    if np.allclose(C_test, C_calc, atol=1e-5):
        return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}

    # Final fallback: naive rank 60
    U_fallback = np.zeros((n * m, 60))
    V_fallback = np.zeros((m * p, 60))
    W_fallback = np.zeros((p * n, 60))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i*m+j, idx] = 1
                V_fallback[j*p+k, idx] = 1
                W_fallback[k*n+i, idx] = 1
                idx += 1
    return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}
