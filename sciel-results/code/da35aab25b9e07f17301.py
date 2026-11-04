"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Build the rank-42 Strassen decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        c = start_col
        # M1 = (a00+a11)(b00+b11)
        U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
        V[b_idx[0]][c] += 1; V[b_idx[3]][c] += 1
        W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
        # M2 = (a10+a11)b00
        U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
        V[b_idx[0]][c+1] += 1
        W[w_idx[2]][c+1] += 1; W[w_idx[3]][c+1] -= 1
        # M3 = a00(b01-b11)
        U[a_idx[0]][c+2] += 1
        V[b_idx[1]][c+2] += 1; V[b_idx[3]][c+2] -= 1
        W[w_idx[1]][c+2] += 1; W[w_idx[3]][c+2] += 1
        # M4 = a11(b10-b00)
        U[a_idx[3]][c+3] += 1
        V[b_idx[2]][c+3] += 1; V[b_idx[0]][c+3] -= 1
        W[w_idx[0]][c+3] += 1; W[w_idx[2]][c+3] += 1
        # M5 = (a00+a01)b11
        U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
        V[b_idx[3]][c+4] += 1
        W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1
        # M6 = (a10-a00)(b00+b01)
        U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
        V[b_idx[0]][c+5] += 1; V[b_idx[1]][c+5] += 1
        W[w_idx[3]][c+5] += 1
        # M7 = (a01-a11)(b10+b11)
        U[a_idx[1]][c+6] += 1; U[a_idx[3]][c+6] -= 1
        V[b_idx[2]][c+6] += 1; V[b_idx[3]][c+6] += 1
        W[w_idx[0]][c+6] += 1

    a_L = [0, 1, 4, 5]
    a_R = [2, 3, 6, 7]
    
    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    w_1 = [0, 2, 1, 3]
    
    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)
    
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]
    
    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)
    
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]
    
    add_strassen_block(a_L, b_3_T, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)
    
    # Convert to float arrays
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor matrix: column r = flattened (u_r ⊗ v_r ⊗ w_r)
    flat_terms = []
    for r in range(rank):
        term = np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r])
        flat_terms.append(term.flatten())
    M = np.array(flat_terms).T  # shape (2304, 42)
    
    def to_list(arr):
        result = []
        for row in arr:
            result.append([])
            for val in row:
                frac = round(val * 2) / 2
                if abs(frac - round(frac)) < 1e-10:
                    frac = int(round(frac))
                result[-1].append(frac)
        return result
    
    def verify_float(final_U, final_V, final_W, new_rank, tol=1e-7):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(new_rank):
                        total += final_U[i*m+j, r] * final_V[j*p+k, r] * final_W[k*n+i, r]
                    if abs(total - 1.0) > tol:
                        return False
        return True
    
    # Linear peeling algorithm:
    # Iteratively find rank-1 terms in the null space of the tensor
    # A null vector v in the null space of M means sum_r v_r * term_r = 0
    # This means we can eliminate one term by adjusting others
    
    best_result = None
    best_rank = rank
    
    def try_peeling(U_work, V_work, W_work, M_work, max_iter=10):
        """Try to peel rank-1 terms using null space analysis."""
        eliminated = set()
        U_cur = U_work.copy()
        V_cur = V_work.copy()
        W_cur = W_work.copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            remaining = [c for c in range(U_cur.shape[1]) if c not in eliminated]
            if len(remaining) <= 1:
                break
            
            # Get sub-matrix for remaining columns
            M_sub = M_work[:, remaining]
            
            try:
                _, s_vals, Vt = np.linalg.svd(M_sub, full_matrices=False)
            except np.linalg.LinAlgError:
                break
            
            tol = 1e-9
            eff_rank = int(np.sum(s_vals > tol))
            
            if eff_rank >= len(remaining):
                break
            
            # Get null space vectors
            null_vecs = Vt[eff_rank:, :]
            
            if null_vecs.shape[0] == 0:
                break
            
            # For each null vector, try to eliminate the column with largest coefficient
            best_elim = None
            best_score = 0
            
            for nv_idx in range(null_vecs.shape[0]):
                nv = null_vecs[nv_idx]
                for local_c, global_c in enumerate(remaining):
                    if abs(nv[local_c]) > best_score:
                        best_score = abs(nv[local_c])
                        best_elim = (nv_idx, global_c)
            
            if best_elim is None or best_score < 1e-8:
                break
            
            nv_idx, pivot_col = best_elim
            nv = null_vecs[nv_idx]
            local_pivot = remaining.index(pivot_col)
            pivot_val = nv[local_pivot]
            
            # Adjust all other columns to eliminate pivot_col
            for local_c, global_c in enumerate(remaining):
                if global_c == pivot_col:
                    continue
                coeff = -nv[local_c] / pivot_val
                if abs(coeff) > 1e-10:
                    U_cur[:, global_c] += coeff * U_cur[:, pivot_col]
                    V_cur[:, global_c] += coeff * V_cur[:, pivot_col]
                    W_cur[:, global_c] += coeff * W_cur[:, pivot_col]
            
            eliminated.add(pivot_col)
        
        if not eliminated:
            return None, rank
        
        keep = [c for c in range(U_cur.shape[1]) if c not in eliminated]
        new_rank = len(keep)
        final_U = U_cur[:, keep]
        final_V = V_cur[:, keep]
        final_W = W_cur[:, keep]
        
        if verify_float(final_U, final_V, final_W, new_rank):
            return {
                "U": to_list(final_U),
                "V": to_list(final_V),
                "W": to_list(final_W)
            }, new_rank
        return None, rank
    
    # Strategy 1: Basic peeling
    try:
        result, r = try_peeling(U_np, V_np, W_np, M, max_iter=15)
        if result is not None and r < best_rank:
            best_rank = r
            best_result = result
    except Exception:
        pass
    
    # Strategy 2: Randomized peeling with different column orderings
    try:
        if time.time() < deadline - 30:
            np.random.seed(42)
            for trial in range(30):
                if time.time() > deadline - 25:
                    break
                
                # Create a random permutation of columns
                perm = list(range(rank))
                np.random.shuffle(perm)
                
                # Reorder columns
                U_perm = U_np[:, perm]
                V_perm = V_np[:, perm]
                W_perm = W_np[:, perm]
                M_perm = M[:, perm]
                
                result, r = try_peeling(U_perm, V_perm, W_perm, M_perm, max_iter=10)
                if result is not None and r < best_rank:
                    best_rank = r
                    # Map back to original column ordering
                    # The result is already in permuted order, but that's fine
                    # since the evaluator just checks the decomposition
                    best_result = result
    except Exception:
        pass
    
    # Strategy 3: Targeted removal - try to remove specific pairs
    try:
        if time.time() < deadline - 15:
            # Try removing 1 term at a time
            for remove_idx in range(rank):
                if time.time() > deadline - 10:
                    break
                
                keep = [c for c in range(rank) if c != remove_idx]
                M_keep = M[:, keep]
                target = M[:, remove_idx]
                
                try:
                    d, _, _, _ = np.linalg.lstsq(M_keep, target, rcond=1e-10)
                    residual = np.linalg.norm(M_keep @ d - target)
                    
                    if residual < 1e-8:
                        new_U = U_np.copy()
                        new_V = V_np.copy()
                        new_W = W_np.copy()
                        
                        for local_idx, global_idx in enumerate(keep):
                            coeff = 1.0 + d[local_idx]
                            if abs(coeff) > 1e-10:
                                new_U[:, global_idx] *= coeff
                                new_V[:, global_idx] *= coeff
                                new_W[:, global_idx] *= coeff
                            else:
                                new_U[:, global_idx] = 0
                                new_V[:, global_idx] = 0
                                new_W[:, global_idx] = 0
                        
                        final_U = new_U[:, keep]
                        final_V = new_V[:, keep]
                        final_W = new_W[:, keep]
                        
                        if verify_float(final_U, final_V, final_W, len(keep)):
                            best_rank = len(keep)
                            best_result = {
                                "U": to_list(final_U),
                                "V": to_list(final_V),
                                "W": to_list(final_W)
                            }
                            break
                except Exception:
                    continue
    except Exception:
        pass
    
    if best_result is not None:
        return best_result
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
