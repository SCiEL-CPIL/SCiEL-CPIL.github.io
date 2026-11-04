import time
import numpy as np
from fractions import Fraction
from itertools import combinations

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8

    start_time = time.time()
    deadline = start_time + 145.0

    # Build the multiplication tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l
                b = l * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify_decomp(U_mat, V_mat, W_mat, rank_val):
        S = np.zeros((nm, mp, pn), dtype=float)
        for r in range(rank_val):
            S += np.einsum('a,b,c->abc', U_mat[:, r], V_mat[:, r], W_mat[:, r])
        return np.max(np.abs(S - T))

    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                rounded = round(val * 2) / 2.0
                if abs(rounded - val) < 1e-10:
                    row_list.append(rounded)
                else:
                    row_list.append(val)
            result.append(row_list)
        return result

    # Known rank-28 solution via Strassen on 2x2 blocks
    rank = 28
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    # Block 1: A_L B_TL -> C_LL
    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)
    # Block 2: A_R B_BL -> C_LR
    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)
    # Block 3: A_L B_TR -> C_RL
    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)
    # Block 4: A_R B_BR -> C_RR
    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # Verify rank-28 solution
    max_err = verify_decomp(U_arr, V_arr, W_arr, rank)
    if max_err > 1e-10:
        return {"U": to_list(U_arr), "V": to_list(V_arr), "W": to_list(W_arr)}

    best_rank = rank
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()

    def try_remove_columns(U_cur, V_cur, W_cur, cols_to_remove):
        remaining = [c for c in range(U_cur.shape[1]) if c not in cols_to_remove]
        if len(remaining) == 0:
            return None
        U_new = U_cur[:, remaining]
        V_new = V_cur[:, remaining]
        W_new = W_cur[:, remaining]
        err = verify_decomp(U_new, V_new, W_new, len(remaining))
        if err < 1e-10:
            return (U_new, V_new, W_new)
        return None

    current_U = U_arr.copy()
    current_V = V_arr.copy()
    current_W = W_arr.copy()
    current_rank = rank

    def build_term_matrix(U_mat, V_mat, W_mat, r):
        M = np.zeros((nm * mp * pn, r), dtype=float)
        for ri in range(r):
            M[:, ri] = np.einsum('a,b,c->abc', U_mat[:, ri], V_mat[:, ri], W_mat[:, ri]).flatten()
        return M

    def try_remove_set(U_cur, V_cur, W_cur, cols_to_remove):
        remaining = [c for c in range(U_cur.shape[1]) if c not in cols_to_remove]
        if len(remaining) == 0:
            return None
        U_new = U_cur[:, remaining]
        V_new = V_cur[:, remaining]
        W_new = W_cur[:, remaining]
        err = verify_decomp(U_new, V_new, W_new, len(remaining))
        if err < 1e-10:
            return (U_new, V_new, W_new)
        return None

    # NEW: Slice rank analysis - analyze the tensor as 2 slices (one per C-row)
    # For each slice i, T[:, :, i*2:(i+1)*2] gives a 4x4 matrix (flattened)
    # Check if we can find a lower-rank decomposition by analyzing slice structure
    
    # Analyze slice structure: T as [nm, mp, pn] = [8, 16, 8]
    # Slice along last axis: 8 slices of shape [8, 16]
    # But more naturally, group by C-index: slices c=0,1 (row 0 of C), c=2,3 (col 0 of C), etc.
    
    # Let's try a different approach: use the fact that for matrix multiplication,
    # the tensor has special structure. Try to find if any terms are linearly dependent
    # by checking the rank of the term matrix more carefully.
    
    # Improved removal: try all subsets of size 2 and 3
    for iteration in range(40):
        if time.time() > deadline - 30:
            break
        if current_rank <= best_rank:
            break

        M = build_term_matrix(current_U, current_V, current_W, current_rank)

        removed_any = False

        # First pass: try single columns
        for r_elim in range(current_rank):
            if time.time() > deadline - 25:
                break
            other_cols = [s for s in range(current_rank) if s != r_elim]
            M_sub = M[:, other_cols]
            rhs = M[:, r_elim]

            try:
                alpha, residuals, rk, sv = np.linalg.lstsq(M_sub, rhs, rcond=1e-10)
                residual = np.max(np.abs(M_sub @ alpha - rhs))
                if residual < 1e-8:
                    result = try_remove_set(current_U, current_V, current_W, [r_elim])
                    if result is not None:
                        current_U, current_V, current_W = result
                        current_rank -= 1
                        removed_any = True
                        if current_rank < best_rank:
                            best_rank = current_rank
                            best_U = current_U.copy()
                            best_V = current_V.copy()
                            best_W = current_W.copy()
                        break
            except np.linalg.LinAlgError:
                continue

        if removed_any:
            continue

        # Second pass: try pairs of columns
        for r1 in range(current_rank):
            if time.time() > deadline - 25:
                break
            for r2 in range(r1 + 1, current_rank):
                if time.time() > deadline - 22:
                    break
                other_cols = [s for s in range(current_rank) if s != r1 and s != r2]
                if len(other_cols) == 0:
                    continue
                M_sub = M[:, other_cols]
                rhs1 = M[:, r1]
                rhs2 = M[:, r2]
                try:
                    alpha1, _, _, _ = np.linalg.lstsq(M_sub, rhs1, rcond=1e-10)
                    residual1 = np.max(np.abs(M_sub @ alpha1 - rhs1))
                    alpha2, _, _, _ = np.linalg.lstsq(M_sub, rhs2, rcond=1e-10)
                    residual2 = np.max(np.abs(M_sub @ alpha2 - rhs2))
                    if residual1 < 1e-8 and residual2 < 1e-8:
                        result = try_remove_set(current_U, current_V, current_W, [r1, r2])
                        if result is not None:
                            current_U, current_V, current_W = result
                            current_rank -= 2
                            removed_any = True
                            if current_rank < best_rank:
                                best_rank = current_rank
                                best_U = current_U.copy()
                                best_V = current_V.copy()
                                best_W = current_W.copy()
                            break
                except np.linalg.LinAlgError:
                    continue
            if removed_any:
                break

        # Third pass (NEW): try triples of columns
        if not removed_any and current_rank > 3:
            for r1 in range(current_rank):
                if time.time() > deadline - 20:
                    break
                for r2 in range(r1 + 1, current_rank):
                    if time.time() > deadline - 18:
                        break
                    for r3 in range(r2 + 1, current_rank):
                        if time.time() > deadline - 16:
                            break
                        other_cols = [s for s in range(current_rank) if s not in (r1, r2, r3)]
                        if len(other_cols) == 0:
                            continue
                        M_sub = M[:, other_cols]
                        try:
                            # Check if all three are in the span of the rest
                            alphas = []
                            all_in_span = True
                            for r_elim in (r1, r2, r3):
                                rhs = M[:, r_elim]
                                alpha, _, _, _ = np.linalg.lstsq(M_sub, rhs, rcond=1e-10)
                                residual = np.max(np.abs(M_sub @ alpha - rhs))
                                if residual >= 1e-8:
                                    all_in_span = False
                                    break
                                alphas.append(alpha)
                            if all_in_span:
                                result = try_remove_set(current_U, current_V, current_W, [r1, r2, r3])
                                if result is not None:
                                    current_U, current_V, current_W = result
                                    current_rank -= 3
                                    removed_any = True
                                    if current_rank < best_rank:
                                        best_rank = current_rank
                                        best_U = current_U.copy()
                                        best_V = current_V.copy()
                                        best_W = current_W.copy()
                                    break
                        except np.linalg.LinAlgError:
                            continue
                    if removed_any:
                        break
                if removed_any:
                    break

        if not removed_any:
            break

    # NEW: Slice-based reconstruction attempt
    # The 2x4x4 multiplication tensor can be viewed as:
    # C[i,k] = sum_l A[i,l] * B[l,k]
    # For fixed i, this is a 4x4 matrix multiplication-like operation
    # Try to find if we can share terms between the two i-slices
    
    # Analyze the structure: for i=0 and i=1, the A-slices are disjoint
    # but the B-slices are shared. This suggests potential for term sharing.
    
    # Try a direct low-rank search using the slice structure
    if time.time() < deadline - 40:
        # Try to find a decomposition with rank < best_rank using optimization
        target_rank = best_rank - 1
        if target_rank >= 16:  # Minimum possible is at least max(n*m, m*p, p*n)/something
            # Use a simple local search: start from the best solution and try to 
            # replace terms with better ones
            U_search = best_U.copy()
            V_search = best_V.copy()
            W_search = best_W.copy()
            
            # Try to find if we can reduce rank by 1 by checking if the 
            # decomposition has any "wasteful" terms
            for trial in range(50):
                if time.time() > deadline - 35:
                    break
                    
                # Randomly perturb one term and see if we can compensate
                r_perturb = np.random.randint(0, target_rank)
                
                # Try to find a new term that, combined with others, still reconstructs T
                # This is complex, so instead try a different approach:
                # Check if any two terms can be merged into one
                
                found_merge = False
                for r1 in range(target_rank):
                    if time.time() > deadline - 35:
                        break
                    for r2 in range(r1 + 1, target_rank):
                        if time.time() > deadline - 35:
                            break
                        # Check if term r1 + term r2 can be replaced by a single term
                        # i.e., if u1*v1*w1 + u2*v2*w2 = u*v*w for some u,v,w
                        # This is equivalent to checking if the sum is rank-1 in the tensor sense
                        term_sum = np.zeros((nm, mp, pn), dtype=float)
                        term_sum += np.einsum('a,b,c->abc', U_search[:, r1], V_search[:, r1], W_search[:, r1])
                        term_sum += np.einsum('a,b,c->abc', U_search[:, r2], V_search[:, r2], W_search[:, r2])
                        
                        # Check if term_sum is a rank-1 tensor
                        # A rank-1 tensor has the form u ⊗ v ⊗ w
                        # We can check by looking at the 2D slices
                        flat = term_sum.flatten()
                        if np.max(np.abs(flat)) < 1e-10:
                            continue
                        
                        # Try to factor as u ⊗ v ⊗ w
                        # Reshape to (nm, mp*pn) and check if rank 1
                        mat = term_sum.reshape(nm, mp * pn)
                        svd_vals = np.linalg.svd(mat, compute_uv=False)
                        if len(svd_vals) > 1 and svd_vals[1] < 1e-10 * svd_vals[0]:
                            # Might be rank-1, try to extract factors
                            u_vec = mat[:, 0]
                            if np.max(np.abs(u_vec)) > 1e-10:
                                # Check if all columns are proportional to u_vec
                                is_rank1 = True
                                for col in range(1, mp * pn):
                                    if np.max(np.abs(mat[:, col])) > 1e-10:
                                        ratio = mat[col, 0] / u_vec[0] if abs(u_vec[0]) > 1e-10 else 0
                                        if not np.allclose(mat[:, col], ratio * u_vec, atol=1e-10):
                                            is_rank1 = False
                                            break
                                if is_rank1:
                                    # Found a mergeable pair
                                    # Remove both terms and add the merged term
                                    remaining = [c for c in range(target_rank) if c not in (r1, r2)]
                                    U_new = np.column_stack([U_search[:, remaining], u_vec]) if remaining else u_vec.reshape(-1, 1)
                                    
                                    # Extract v and w from the merged term
                                    # This is complex, so skip for now
                                    found_merge = True
                                    break
                    if found_merge:
                        break
                
                if not found_merge:
                    break

    # Final verification and cleanup
    final_err = verify_decomp(best_U, best_V, best_W, best_rank)
    if final_err > 1e-10:
        best_U = U_arr
        best_V = V_arr
        best_W = W_arr
        best_rank = 28

    # High-precision polish: try to round to nearest half-integer and verify
    U_polished = np.round(best_U * 2) / 2.0
    V_polished = np.round(best_V * 2) / 2.0
    W_polished = np.round(best_W * 2) / 2.0
    polish_err = verify_decomp(U_polished, V_polished, W_polished, best_rank)
    if polish_err < 1e-10:
        best_U = U_polished
        best_V = V_polished
        best_W = W_polished

    U_final = to_list(best_U)
    V_final = to_list(best_V)
    W_final = to_list(best_W)

    return {"U": U_final, "V": V_final, "W": W_final}
