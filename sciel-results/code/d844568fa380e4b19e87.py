import time
import numpy as np

DEADLINE = 115.0

def construct():
    start_time = time.time()
    n, m, p = 2, 5, 6
    total_entries = n * m * p  # 60
    
    # Build a large pool of candidate rank-1 tensors
    # A rank-1 tensor is (u_vec, v_vec, w_vec) where:
    #   u_vec has length n*m, v_vec has length m*p, w_vec has length p*n
    # and the contribution to the (i,j,k) entry is u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    candidates = []  # list of (u_vec, v_vec, w_vec, contrib_vec)
    
    def add_candidate(u_sparse, v_sparse, w_sparse):
        """Add a candidate defined by sparse dicts."""
        u_vec = np.zeros(n * m)
        v_vec = np.zeros(m * p)
        w_vec = np.zeros(p * n)
        
        for (i, j), val in u_sparse.items():
            u_vec[i * m + j] = val
        for (j, k), val in v_sparse.items():
            v_vec[j * p + k] = val
        for (k, i), val in w_sparse.items():
            w_vec[k * n + i] = val
        
        # Compute contribution vector
        contrib = np.zeros(total_entries)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    contrib[i * m * p + j * p + k] = u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
        
        # Only add if non-zero
        if np.any(contrib):
            candidates.append((u_vec, v_vec, w_vec, contrib))
    
    # Type 1: Standard unit tensors
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
    
    # Type 3: Generate diverse structured candidates for wider blocks
    # For 2x3x3 blocks, generate all combinations of +/-1 patterns
    for io in range(n - 1):
        for jo in range(m - 2):
            for ko in range(p - 2):
                # Generate candidates with various sign patterns
                # Pattern: sum of A entries times sum of B entries times sum of W entries
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
    
    # Type 4: Diagonal-like patterns
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
    
    # Now use greedy to find a good decomposition
    T_target = np.ones(total_entries)
    residual = T_target.copy()
    
    selected_indices = []
    selected_coeffs = []
    
    num_candidates = len(candidates)
    
    # Precompute norms of all candidates
    norms = np.zeros(num_candidates)
    for idx in range(num_candidates):
        vec = candidates[idx][3]
        norms[idx] = np.dot(vec, vec)
    
    # Build a matrix of all candidate contributions for faster dot products
    # cand_matrix: shape (total_entries, num_candidates)
    cand_matrix = np.zeros((total_entries, num_candidates))
    for idx in range(num_candidates):
        cand_matrix[:, idx] = candidates[idx][3]
    
    max_iter = 150
    for _ in range(max_iter):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Vectorized: compute all dot products at once
        dots = cand_matrix.T @ residual  # shape (num_candidates,)
        coeffs = np.zeros(num_candidates)
        coeffs[norms > 1e-12] = dots[norms > 1e-12] / norms[norms > 1e-12]
        reductions = dots * coeffs
        
        # Mask out invalid
        reductions[norms < 1e-12] = -1e18
        
        best_idx = int(np.argmax(reductions))
        best_reduction = reductions[best_idx]
        best_coeff = coeffs[best_idx]
        
        if best_idx == -1 or best_reduction < 1e-9:
            break
        
        selected_indices.append(best_idx)
        selected_coeffs.append(best_coeff)
        residual -= best_coeff * candidates[best_idx][3]
        
        if np.linalg.norm(residual) < 1e-9:
            break
    
    # Refinement pass: try to reduce rank by checking if any selected term
    # can be removed and its contribution absorbed by others
    if len(selected_indices) > 10:
        for trial_remove in range(len(selected_indices)):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Temporarily remove this term and see if residual is still small
            test_residual = T_target.copy()
            for r, idx in enumerate(selected_indices):
                if r == trial_remove:
                    continue
                test_residual -= selected_coeffs[r] * candidates[idx][3]
            
            if np.linalg.norm(test_residual) < 1e-8:
                # This term is redundant, remove it
                selected_indices.pop(trial_remove)
                selected_coeffs.pop(trial_remove)
                break
    
    # Check if we got an exact solution
    if np.linalg.norm(residual) < 1e-9 and len(selected_indices) <= 60:
        rank_found = len(selected_indices)
        
        U = np.zeros((n * m, rank_found))
        V = np.zeros((m * p, rank_found))
        W = np.zeros((p * n, rank_found))
        
        for r, idx in enumerate(selected_indices):
            coeff = selected_coeffs[r]
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
    
    # Fallback: return the known 54-rank solution
    n, m, p = 2, 5, 6
    rank_new = 54
    U_new = np.zeros((n * m, rank_new))
    V_new = np.zeros((m * p, rank_new))
    W_new = np.zeros((p * n, rank_new))
    
    current_rank = 0
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal current_rank
        # r=0: M1
        U_new[(i_off)*m + (j_off), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        # r=1: M2
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = -1
        current_rank += 1
        # r=2: M3
        U_new[(i_off)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = -1
        W_new[(k_off+1)*n + (i_off), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        # r=3: M4
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = -1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        # r=4: M5
        U_new[(i_off)*m + (j_off), current_rank] = 1
        U_new[(i_off)*m + (j_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = -1
        W_new[(k_off+1)*n + (i_off), current_rank] = 1
        current_rank += 1
        # r=5: M6
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        U_new[(i_off)*m + (j_off), current_rank] = -1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        V_new[(j_off)*p + (k_off+1), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        # r=6: M7
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

    # Part 1: A[:, 0:2] * B[0:2, :]
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # Part 2a: A[:, 2:3] * B[2:3, :]
    for k in range(6):
        add_standard_block(0, 2, k)
        
    # Part 2b: A[:, 3:5] * B[3:5, :]
    add_strassen_block(0, 3, 0)
    add_strassen_block(0, 3, 2)
    add_strassen_block(0, 3, 4)
    
    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
