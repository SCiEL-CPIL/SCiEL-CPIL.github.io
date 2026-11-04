"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # First, build the structured decomposition as a baseline (rank 84)
    n_block_ranges = [(0, 2), (2, 3)]
    m_block_ranges = [(0, 2), (2, 4), (4, 5)]
    p_block_ranges = [(0, 2), (2, 4), (4, 6)]
    
    terms = []
    
    def add_strassen(a_rows, c_cols, d_cols):
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0, d1 = d_cols
        
        def U(la, lc): return a_rows[la]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[la]
        
        terms.append((
            [(U(0,0), 1), (U(1,1), 1)],
            [(V(0,0), 1), (V(1,1), 1)],
            [(W(0,0), 1), (W(1,1), 1)]
        ))
        terms.append((
            [(U(1,0), 1), (U(1,1), 1)],
            [(V(0,0), 1)],
            [(W(0,1), 1), (W(1,1), -1)]
        ))
        terms.append((
            [(U(0,0), 1)],
            [(V(0,1), 1), (V(1,1), -1)],
            [(W(1,0), 1), (W(1,1), 1)]
        ))
        terms.append((
            [(U(1,1), 1)],
            [(V(1,0), 1), (V(0,0), -1)],
            [(W(0,0), 1), (W(0,1), 1)]
        ))
        terms.append((
            [(U(0,0), 1), (U(0,1), 1)],
            [(V(1,1), 1)],
            [(W(0,0), -1), (W(1,0), 1)]
        ))
        terms.append((
            [(U(1,0), 1), (U(0,0), -1)],
            [(V(0,0), 1), (V(0,1), 1)],
            [(W(1,1), 1)]
        ))
        terms.append((
            [(U(0,1), 1), (U(1,1), -1)],
            [(V(1,0), 1), (V(1,1), 1)],
            [(W(0,0), 1)]
        ))
    
    def add_schoolbook_2x2x1(a_rows, c_cols, d_cols):
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0 = d_cols[0]
        def U(la, lc): return a_rows[la]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[0]
        def W(ld, la): return d_cols[0]*n + a_rows[la]
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,1), 1)], [(V(1,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(1,0), 1)], [(V(0,0), 1)], [(W(0,1), 1)]))
        terms.append(([(U(1,1), 1)], [(V(1,0), 1)], [(W(0,1), 1)]))
    
    def add_schoolbook_2x1x2(a_rows, c_cols, d_cols):
        a0, a1 = a_rows
        c0 = c_cols[0]
        d0, d1 = d_cols
        def U(la, lc): return a_rows[la]*m + c_cols[0]
        def V(lc, ld): return c_cols[0]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[la]
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,0), 1)], [(V(0,1), 1)], [(W(1,0), 1)]))
        terms.append(([(U(1,0), 1)], [(V(0,0), 1)], [(W(0,1), 1)]))
        terms.append(([(U(1,0), 1)], [(V(0,1), 1)], [(W(1,1), 1)]))
    
    def add_schoolbook_1x2x2(a_rows, c_cols, d_cols):
        a0 = a_rows[0]
        c0, c1 = c_cols
        d0, d1 = d_cols
        def U(la, lc): return a_rows[0]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[0]
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,1), 1)], [(V(1,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,0), 1)], [(V(0,1), 1)], [(W(1,0), 1)]))
        terms.append(([(U(0,1), 1)], [(V(1,1), 1)], [(W(1,0), 1)]))
    
    def add_schoolbook_1x1x2(a_rows, c_cols, d_cols):
        a0 = a_rows[0]
        c0 = c_cols[0]
        d0, d1 = d_cols
        def U(la, lc): return a_rows[0]*m + c_cols[0]
        def V(lc, ld): return c_cols[0]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[0]
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,0), 1)], [(V(0,1), 1)], [(W(1,0), 1)]))
    
    for ni, (ns, ne) in enumerate(n_block_ranges):
        for mi, (ms, me) in enumerate(m_block_ranges):
            for pi, (ps, pe) in enumerate(p_block_ranges):
                n_size = ne - ns
                m_size = me - ms
                p_size = pe - ps
                a_rows = list(range(ns, ne))
                c_cols = list(range(ms, me))
                d_cols = list(range(ps, pe))
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    add_strassen(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 2 and p_size == 1:
                    add_schoolbook_2x2x1(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 1 and p_size == 2:
                    add_schoolbook_2x1x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 2 and p_size == 2:
                    add_schoolbook_1x2x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 1 and p_size == 2:
                    add_schoolbook_1x1x2(a_rows, c_cols, d_cols)
                else:
                    raise ValueError(f"Unhandled block: {n_size}x{m_size}x{p_size}")
    
    rank_structured = len(terms)
    U_struct = [[0] * rank_structured for _ in range(n * m)]
    V_struct = [[0] * rank_structured for _ in range(m * p)]
    W_struct = [[0] * rank_structured for _ in range(p * n)]
    
    for ci, (u_entries, v_entries, w_entries) in enumerate(terms):
        for row, coeff in u_entries:
            U_struct[row][ci] += coeff
        for row, coeff in v_entries:
            V_struct[row][ci] += coeff
        for row, coeff in w_entries:
            W_struct[row][ci] += coeff
    
    U_arr = np.array(U_struct, dtype=np.float64)
    V_arr = np.array(V_struct, dtype=np.float64)
    W_arr = np.array(W_struct, dtype=np.float64)
    
    # Fast verification
    def verify_fast(U_mat, V_mat, W_mat, n, m, p):
        r = U_mat.shape[1]
        if r == 0:
            return False
        U_r = U_mat.reshape(n, m, r)
        V_r = V_mat.reshape(m, p, r)
        W_r = W_mat.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, 1.0, atol=1e-9)
    
    if not verify_fast(U_arr, V_arr, W_arr, n, m, p):
        return {"U": U_struct, "V": V_struct, "W": W_struct}
    
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    best_rank = rank_structured
    
    # Greedy single-column elimination on structured decomposition
    for attempt in range(best_rank):
        if time.time() > deadline:
            break
        found = False
        for col in range(best_rank):
            if time.time() > deadline:
                break
            if best_rank <= 1:
                break
            U_test = np.delete(best_U, col, axis=1)
            V_test = np.delete(best_V, col, axis=1)
            W_test = np.delete(best_W, col, axis=1)
            if verify_fast(U_test, V_test, W_test, n, m, p):
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank -= 1
                found = True
                break
        if not found:
            break
    
    # Now try randomized algebraic rank-reduction search
    # Strategy: start with dense random half-integer factors and use QR pivoting
    # to identify and eliminate linearly dependent columns
    
    rng = np.random.default_rng(42)
    
    def try_random_dense_init(base_rank, num_trials=20):
        """Try random dense initializations and use QR to find minimal rank."""
        nonlocal best_U, best_V, best_W, best_rank
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            # Start with a dense random initialization at base_rank
            # Use small values to keep numerical stability
            scale = 1.0
            U_rand = rng.standard_normal((n*m, base_rank)) * scale
            V_rand = rng.standard_normal((m*p, base_rank)) * scale
            W_rand = rng.standard_normal((p*n, base_rank)) * scale
            
            # Scale to make the tensor close to identity
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # We want T[i,j,k] = 1 for all i,j,k
            
            # First, check if this random init is close to valid
            if not verify_fast(U_rand, V_rand, W_rand, n, m, p):
                # Try to fix by adjusting one factor
                # This is heuristic; we'll use the structured result as fallback
                continue
            
            # Use QR decomposition with column pivoting to identify
            # which columns are linearly independent
            # We'll work with the combined factor structure
            
            # For rank reduction, we need to find columns that can be removed
            # while preserving the tensor identity
            # Use QR pivoting on each factor separately
            
            # Actually, let's use a different approach:
            # Start from the structured decomposition and try to find
            # a lower-rank equivalent via random perturbations + projection
            
            pass
    
    # Better approach: Use the structured decomposition and try to find
    # a lower-rank representation via random column combinations
    
    # The key insight: if we can find a rank-r decomposition where r < current_rank,
    # we can express the current factors as linear combinations of fewer columns
    
    # Try: for each possible target rank r from best_rank-1 down to some minimum,
    # attempt to find a rank-r decomposition by solving a linear system
    
    def attempt_rank_reduction(target_rank, max_iter=100):
        """Attempt to find a rank-target_rank decomposition."""
        nonlocal best_U, best_V, best_W, best_rank
        if target_rank >= best_rank:
            return False
        
        # Initialize from current best
        U_curr = best_U.copy()
        V_curr = best_V.copy()
        W_curr = best_W.copy()
        curr_rank = best_rank
        
        # Try to eliminate (curr_rank - target_rank) columns
        cols_to_eliminate = curr_rank - target_rank
        
        # Use iterative approach: at each step, try to merge two columns
        # into one by finding a linear combination that preserves the tensor
        
        for _ in range(cols_to_eliminate):
            if time.time() > deadline:
                return False
            
            # Try all pairs of columns to see if we can merge them
            merged = False
            for c1 in range(curr_rank):
                if time.time() > deadline:
                    return False
                for c2 in range(c1 + 1, curr_rank):
                    if time.time() > deadline:
                        return False
                    
                    # Try to find alpha, beta such that replacing columns c1,c2
                    # with a single column alpha*U[:,c1] + beta*U[:,c2], etc.
                    # preserves the tensor
                    # This is a nonlinear problem, so we use a heuristic
                    
                    # Heuristic: try alpha=1, beta=0 (just remove c2)
                    # or alpha=0, beta=1 (just remove c1)
                    # or alpha=1, beta=1 (merge)
                    
                    for alpha, beta in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                        # Create new factors with c2 replaced by alpha*U[:,c1] + beta*U[:,c2]
                        # and remove c1
                        U_new = U_curr.copy()
                        V_new = V_curr.copy()
                        W_new = W_curr.copy()
                        
                        # Replace column c2 with the combination
                        U_new[:, c2] = alpha * U_new[:, c1] + beta * U_new[:, c2]
                        V_new[:, c2] = alpha * V_new[:, c1] + beta * V_new[:, c2]
                        W_new[:, c2] = alpha * W_new[:, c1] + beta * W_new[:, c2]
                        
                        # Remove column c1
                        if c1 < c2:
                            U_new = np.delete(U_new, c1, axis=1)
                            V_new = np.delete(V_new, c1, axis=1)
                            W_new = np.delete(W_new, c1, axis=1)
                        else:
                            U_new = np.delete(U_new, c1, axis=1)
                            V_new = np.delete(V_new, c1, axis=1)
                            W_new = np.delete(W_new, c1, axis=1)
                        
                        if verify_fast(U_new, V_new, W_new, n, m, p):
                            best_U, best_V, best_W = U_new, V_new, W_new
                            curr_rank -= 1
                            merged = True
                            break
                    
                    if merged:
                        break
                
                if merged:
                    break
            
            if not merged:
                # Try random perturbations
                for _ in range(20):
                    if time.time() > deadline:
                        return False
                    
                    # Add small random perturbation to one column
                    col_to_perturb = rng.integers(0, curr_rank)
                    U_pert = best_U.copy()
                    V_pert = best_V.copy()
                    W_pert = best_W.copy()
                    
                    pert_scale = 0.1
                    U_pert[:, col_to_perturb] += rng.standard_normal(n*m) * pert_scale
                    V_pert[:, col_to_perturb] += rng.standard_normal(m*p) * pert_scale
                    W_pert[:, col_to_perturb] += rng.standard_normal(p*n) * pert_scale
                    
                    # Try to eliminate a different column
                    for col_to_remove in range(curr_rank):
                        if col_to_remove == col_to_perturb:
                            continue
                        U_test = np.delete(U_pert, col_to_remove, axis=1)
                        V_test = np.delete(V_pert, col_to_remove, axis=1)
                        W_test = np.delete(W_pert, col_to_remove, axis=1)
                        
                        if verify_fast(U_test, V_test, W_test, n, m, p):
                            best_U, best_V, best_W = U_test, V_test, W_test
                            curr_rank -= 1
                            merged = True
                            break
                    
                    if merged:
                        break
            
            if not merged:
                break
        
        best_rank = curr_rank
        return best_rank <= target_rank
    
    # Try to reduce rank below the structured result
    for target in range(best_rank - 1, max(20, best_rank - 10), -1):
        if time.time() > deadline:
            break
        if attempt_rank_reduction(target):
            break
    
    # Final cleanup: round to half-integers if possible
    U_final = best_U.tolist()
    V_final = best_V.tolist()
    W_final = best_W.tolist()
    
    for mat in [U_final, V_final, W_final]:
        for row in mat:
            for idx in range(len(row)):
                val = row[idx]
                rounded = round(val * 2) / 2.0
                if abs(val - rounded) < 1e-9:
                    row[idx] = rounded
                else:
                    row[idx] = round(val, 10)
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
