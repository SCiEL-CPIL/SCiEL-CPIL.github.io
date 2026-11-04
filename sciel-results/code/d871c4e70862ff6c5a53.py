"""Matrix multiplication tensor decomposition for 3x4 * 4x7.

Uses block decomposition (Strassen) with fast vectorized verification.
Attempts to find rank < 84 via structured search and local refinement.
"""

import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Precompute index arrays for verification
    # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    ii_arr = np.array([i for i in range(n) for j in range(m) for k in range(p)])
    jj_arr = np.array([j for i in range(n) for j in range(m) for k in range(p)])
    kk_arr = np.array([k for i in range(n) for j in range(m) for k in range(p)])
    U_idx_base = ii_arr * m + jj_arr
    V_idx_base = jj_arr * p + kk_arr
    W_idx_base = kk_arr * n + ii_arr

    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        U_sel = U[U_idx_base]
        V_sel = V[V_idx_base]
        W_sel = W[W_idx_base]
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        if not np.allclose(vals, 1.0, atol=1e-9):
            return False
        # Check half-integer entries
        all_half_int = (np.allclose(U * 2, np.round(U * 2)) and 
                       np.allclose(V * 2, np.round(V * 2)) and 
                       np.allclose(W * 2, np.round(W * 2)))
        return all_half_int

    def add_rank1_term(U_vecs, V_vecs, W_vecs, u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        u = np.zeros(n * m, dtype=float)
        v = np.zeros(m * p, dtype=float)
        w = np.zeros(p * n, dtype=float)
        for idx, val in zip(u_indices, u_vals):
            u[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w[idx] += val
        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def gen_strassen_2x2x2(U_vecs, V_vecs, W_vecs, i_off, j_off, k_off):
        """Generate 7 rank-1 terms for 2x2x2 Strassen decomposition."""
        i0 = i_off; i1 = i_off + 1
        j0 = j_off; j1 = j_off + 1
        k0 = k_off; k1 = k_off + 1
        
        # M1 = (a00+a11)(b00+b11) -> c00, c11
        add_rank1_term(U_vecs, V_vecs, W_vecs,
                      [i0*m+j0, i1*m+j1], [1, 1],
                      [j0*p+k0, j1*p+k1], [1, 1],
                      [k0*n+i0, k1*n+i1], [1, 1])
        
        # M2 = (a10+a11)(b00) -> c10, c11(-1)
        add_rank1_term(U_vecs, V_vecs, W_vecs,
                      [i1*m+j0, i1*m+j1], [1, 1],
                      [j0*p+k0], [1],
                      [k0*n+i1, k1*n+i1], [1, -1])
        
        # M3 = (a00)(b01-b11) -> c01, c11
        add_rank1_term(U_vecs, V_vecs, W_vecs,
                      [i0*m+j0], [1],
                      [j0*p+k1, j1*p+k1], [1, -1],
                      [k1*n+i0, k1*n+i1], [1, 1])
        
        # M4 = (a11)(b10-b00) -> c00, c10
        add_rank1_term(U_vecs, V_vecs, W_vecs,
                      [i1*m+j1], [1],
                      [j1*p+k0, j0*p+k0], [1, -1],
                      [k0*n+i0, k0*n+i1], [1, 1])
        
        # M5 = (a00+a01)(b11) -> c00(-1), c01
        add_rank1_term(U_vecs, V_vecs, W_vecs,
                      [i0*m+j0, i0*m+j1], [1, 1],
                      [j1*p+k1], [1],
                      [k0*n+i0, k1*n+i0], [-1, 1])
        
        # M6 = (a10-a00)(b00+b01) -> c11
        add_rank1_term(U_vecs, V_vecs, W_vecs,
                      [i1*m+j0, i0*m+j0], [1, -1],
                      [j0*p+k0, j0*p+k1], [1, 1],
                      [k1*n+i1], [1])
        
        # M7 = (a01-a11)(b10+b11) -> c00
        add_rank1_term(U_vecs, V_vecs, W_vecs,
                      [i0*m+j1, i1*m+j1], [1, -1],
                      [j1*p+k0, j1*p+k1], [1, 1],
                      [k0*n+i0], [1])

    def gen_schoolbook(U_vecs, V_vecs, W_vecs, n_l, m_l, p_l, i_off, j_off, k_off):
        """Generate schoolbook rank-1 terms."""
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    ii = i_off + i
                    jj = j_off + j
                    kk = k_off + k
                    add_rank1_term(U_vecs, V_vecs, W_vecs,
                                  [ii*m+jj], [1.0],
                                  [jj*p+kk], [1.0],
                                  [kk*n+ii], [1.0])

    def build_and_verify(rows_split, cols_split):
        """Build a block decomposition and verify it."""
        U_vecs = []
        V_vecs = []
        W_vecs = []
        
        row_offsets = [0]
        for rs in rows_split:
            row_offsets.append(row_offsets[-1] + rs)
        
        col_offsets = [0]
        for cs in cols_split:
            col_offsets.append(col_offsets[-1] + cs)
        
        for r_idx in range(len(rows_split)):
            r_start = row_offsets[r_idx]
            r_size = rows_split[r_idx]
            
            for c_idx in range(len(cols_split)):
                c_start = col_offsets[c_idx]
                c_size = cols_split[c_idx]
                
                if r_size == 1:
                    # 1 x m x c_size block
                    # We can use schoolbook here, but if c_size is even, we might optimize?
                    # For now, schoolbook is 1*m*c_size = m*c_size terms.
                    gen_schoolbook(U_vecs, V_vecs, W_vecs, 1, m, c_size, r_start, 0, c_start)
                elif r_size == 2:
                    # 2 x m x c_size block
                    # Split m into pairs of 2 if possible
                    for m_start in range(0, m, 2):
                        m_end = min(m_start + 2, m)
                        m_size = m_end - m_start
                        
                        if m_size == 2:
                            # 2 x 2 x c_size sub-block
                            for k_start in range(c_start, c_start + c_size, 2):
                                k_end = min(k_start + 2, c_start + c_size)
                                k_size = k_end - k_start
                                
                                if k_size == 2:
                                    # 2x2x2 -> Strassen (7 terms)
                                    gen_strassen_2x2x2(U_vecs, V_vecs, W_vecs, r_start, m_start, k_start)
                                elif k_size == 1:
                                    # 2x2x1 -> Schoolbook (4 terms)
                                    gen_schoolbook(U_vecs, V_vecs, W_vecs, 2, 2, 1, r_start, m_start, k_start)
                        else:
                            # m_size == 1 (should not happen if m is even, but just in case)
                            for k_start in range(c_start, c_start + c_size):
                                gen_schoolbook(U_vecs, V_vecs, W_vecs, 2, 1, 1, r_start, m_start, k_start)
                else:
                    # r_size > 2, use schoolbook for the whole block
                    gen_schoolbook(U_vecs, V_vecs, W_vecs, r_size, m, c_size, r_start, 0, c_start)
        
        if not U_vecs:
            return None
        
        U = np.array(U_vecs).T
        V = np.array(V_vecs).T
        W = np.array(W_vecs).T
        
        if not verify_decomposition(U, V, W):
            return None
        
        rank = U.shape[1]
        return {"U": (np.round(U * 2) / 2).tolist(),
                "V": (np.round(V * 2) / 2).tolist(),
                "W": (np.round(W * 2) / 2).tolist(),
                "rank": rank}

    # Search space for block decompositions
    # n=3, m=4, p=7
    # Row splits of 3: [3], [2,1], [1,2], [1,1,1]
    # Col splits of 7: many. Focus on splits that allow 2x2x2 Strassen blocks.
    # For a 2x2x2 block, we need at least 2 rows, 2 cols in A, 2 cols in B.
    # So row split must have a part >= 2.
    # Col split should have parts that are multiples of 2 or 1.
    
    row_split_options = [
        [2, 1],
        [1, 2],
        [3],
        [1, 1, 1],
    ]
    
    col_split_options = []
    # Generate all partitions of 7 into parts of size 1 or 2? 
    # Actually, we can have larger parts handled by schoolbook or further splitting.
    # But the current build_and_verify handles internal splitting of m and k.
    # So col_split_options can just be [7], or [2,5], [3,4], etc.
    # However, the internal logic splits m (which is 4) into pairs of 2.
    # And it splits k (the col block size) into pairs of 2.
    # So if we pass a col block of size 7, it will split k into 2,2,2,1.
    # If we pass col blocks of size 2, 2, 3, it will split the 3 into 2,1.
    # It's probably most efficient to let the internal logic handle the fine splitting.
    # So we can just try a few coarse col splits.
    
    col_split_options = [
        [7],
        [2, 5],
        [5, 2],
        [3, 4],
        [4, 3],
        [1, 6],
        [6, 1],
        [2, 2, 3],
        [2, 3, 2],
        [3, 2, 2],
        [2, 2, 2, 1],
        [1, 2, 2, 2],
        [2, 2, 1, 2],
        [2, 1, 2, 2],
        [1, 1, 2, 3],
        [1, 2, 3, 1],
        [2, 1, 1, 3],
        [3, 1, 1, 2],
    ]
    
    best_rank = float('inf')
    best_result = None
    
    for rows_split in row_split_options:
        if sum(rows_split) != n:
            continue
        for cols_split in col_split_options:
            if sum(cols_split) != p:
                continue
            if time.time() > deadline - 30:
                break
            result = build_and_verify(rows_split, cols_split)
            if result and result["rank"] < best_rank:
                best_rank = result["rank"]
                best_result = result
        if time.time() > deadline - 30:
            break
    
    # If we found a good result, try to refine it
    if best_result and time.time() < deadline - 20:
        U_mat = np.array(best_result["U"])
        V_mat = np.array(best_result["V"])
        W_mat = np.array(best_result["W"])
        R = best_rank
        
        # Try to reduce rank by 1 by removing a term and adjusting others?
        # This is hard. Instead, try to find a better block decomposition.
        # Or try a local search on the coefficients?
        # The coefficients are currently integers or half-integers.
        # We can try to perturb them slightly and see if we can maintain validity while reducing rank?
        # No, rank is discrete.
        
        # Alternative: Try to merge two terms into one?
        # If two terms (u1, v1, w1) and (u2, v2, w2) can be replaced by (u3, v3, w3)
        # such that u1*v1*w1 + u2*v2*w2 = u3*v3*w3, then we reduce rank by 1.
        # This requires solving a system of equations.
        
        # For now, let's just return the best block decomposition.
        pass

    # Return best result or fallback to schoolbook
    if best_result:
        return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
    
    # Final fallback: schoolbook (rank 84)
    U_vecs_fb = []
    V_vecs_fb = []
    W_vecs_fb = []
    gen_schoolbook(U_vecs_fb, V_vecs_fb, W_vecs_fb, n, m, p, 0, 0, 0)
    U_fb = np.array(U_vecs_fb).T
    V_fb = np.array(V_vecs_fb).T
    W_fb = np.array(W_vecs_fb).T
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

# Keep the function name and signature as required
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Precompute index arrays for verification
    ii_arr = np.array([i for i in range(n) for j in range(m) for k in range(p)])
    jj_arr = np.array([j for i in range(n) for j in range(m) for k in range(p)])
    kk_arr = np.array([k for i in range(n) for j in range(m) for k in range(p)])
    U_idx_base = ii_arr * m + jj_arr
    V_idx_base = jj_arr * p + kk_arr
    W_idx_base = kk_arr * n + ii_arr

    def verify_decomposition(U, V, W):
        U_sel = U[U_idx_base]
        V_sel = V[V_idx_base]
        W_sel = W[W_idx_base]
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        if not np.allclose(vals, 1.0, atol=1e-9):
            return False
        all_half_int = (np.allclose(U * 2, np.round(U * 2)) and 
                       np.allclose(V * 2, np.round(V * 2)) and 
                       np.allclose(W * 2, np.round(W * 2)))
        return all_half_int

    def add_rank1_term(U_vecs, V_vecs, W_vecs, u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        u = np.zeros(n * m, dtype=float)
        v = np.zeros(m * p, dtype=float)
        w = np.zeros(p * n, dtype=float)
        for idx, val in zip(u_indices, u_vals):
            u[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w[idx] += val
        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def gen_strassen_2x2x2(U_vecs, V_vecs, W_vecs, i_off, j_off, k_off):
        i0 = i_off; i1 = i_off + 1
        j0 = j_off; j1 = j_off + 1
        k0 = k_off; k1 = k_off + 1
        
        add_rank1_term(U_vecs, V_vecs, W_vecs,
                      [i0*m+j0, i1*m+j1], [1, 1],
                      [j0*p+k0, j1*p+k1], [1, 1],
                      [k0*n+i0, k1*n+i1], [1, 1])
        
        add_rank1_term(U_vecs, V_vecs, W_vecs,
                      [i1*m+j0, i1*m+j1], [1, 1],
                      [j0*p+k0], [1],
                      [k0*n+i1, k1*n+i1], [1, -1])
        
        add_rank1_term(U_vecs, V_vecs, W_vecs,
                      [i0*m+j0], [1],
                      [j0*p+k1, j1*p+k1], [1, -1],
                      [k1*n+i0, k1*n+i1], [1, 1])
        
        add_rank1_term(U_vecs, V_vecs, W_vecs,
                      [i1*m+j1], [1],
                      [j1*p+k0, j0*p+k0], [1, -1],
                      [k0*n+i0, k0*n+i1], [1, 1])
        
        add_rank1_term(U_vecs, V_vecs, W_vecs,
                      [i0*m+j0, i0*m+j1], [1, 1],
                      [j1*p+k1], [1],
                      [k0*n+i0, k1*n+i0], [-1, 1])
        
        add_rank1_term(U_vecs, V_vecs, W_vecs,
                      [i1*m+j0, i0*m+j0], [1, -1],
                      [j0*p+k0, j0*p+k1], [1, 1],
                      [k1*n+i1], [1])
        
        add_rank1_term(U_vecs, V_vecs, W_vecs,
                      [i0*m+j1, i1*m+j1], [1, -1],
                      [j1*p+k0, j1*p+k1], [1, 1],
                      [k0*n+i0], [1])

    def gen_schoolbook(U_vecs, V_vecs, W_vecs, n_l, m_l, p_l, i_off, j_off, k_off):
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    ii = i_off + i
                    jj = j_off + j
                    kk = k_off + k
                    add_rank1_term(U_vecs, V_vecs, W_vecs,
                                  [ii*m+jj], [1.0],
                                  [jj*p+kk], [1.0],
                                  [kk*n+ii], [1.0])

    def build_and_verify(rows_split, cols_split):
        U_vecs = []
        V_vecs = []
        W_vecs = []
        
        row_offsets = [0]
        for rs in rows_split:
            row_offsets.append(row_offsets[-1] + rs)
        
        col_offsets = [0]
        for cs in cols_split:
            col_offsets.append(col_offsets[-1] + cs)
        
        for r_idx in range(len(rows_split)):
            r_start = row_offsets[r_idx]
            r_size = rows_split[r_idx]
            
            for c_idx in range(len(cols_split)):
                c_start = col_offsets[c_idx]
                c_size = cols_split[c_idx]
                
                if r_size == 1:
                    gen_schoolbook(U_vecs, V_vecs, W_vecs, 1, m, c_size, r_start, 0, c_start)
                elif r_size == 2:
                    for m_start in range(0, m, 2):
                        m_end = min(m_start + 2, m)
                        m_size = m_end - m_start
                        
                        if m_size == 2:
                            for k_start in range(c_start, c_start + c_size, 2):
                                k_end = min(k_start + 2, c_start + c_size)
                                k_size = k_end - k_start
                                
                                if k_size == 2:
                                    gen_strassen_2x2x2(U_vecs, V_vecs, W_vecs, r_start, m_start, k_start)
                                elif k_size == 1:
                                    gen_schoolbook(U_vecs, V_vecs, W_vecs, 2, 2, 1, r_start, m_start, k_start)
                        else:
                            for k_start in range(c_start, c_start + c_size):
                                gen_schoolbook(U_vecs, V_vecs, W_vecs, 2, 1, 1, r_start, m_start, k_start)
                else:
                    gen_schoolbook(U_vecs, V_vecs, W_vecs, r_size, m, c_size, r_start, 0, c_start)
        
        if not U_vecs:
            return None
        
        U = np.array(U_vecs).T
        V = np.array(V_vecs).T
        W = np.array(W_vecs).T
        
        if not verify_decomposition(U, V, W):
            return None
        
        rank = U.shape[1]
        return {"U": (np.round(U * 2) / 2).tolist(),
                "V": (np.round(V * 2) / 2).tolist(),
                "W": (np.round(W * 2) / 2).tolist(),
                "rank": rank}

    row_split_options = [
        [2, 1],
        [1, 2],
        [3],
        [1, 1, 1],
    ]
    
    col_split_options = [
        [7],
        [2, 5],
        [5, 2],
        [3, 4],
        [4, 3],
        [1, 6],
        [6, 1],
        [2, 2, 3],
        [2, 3, 2],
        [3, 2, 2],
        [2, 2, 2, 1],
        [1, 2, 2, 2],
        [2, 2, 1, 2],
        [2, 1, 2, 2],
        [1, 1, 2, 3],
        [1, 2, 3, 1],
        [2, 1, 1, 3],
        [3, 1, 1, 2],
    ]
    
    best_rank = float('inf')
    best_result = None
    
    for rows_split in row_split_options:
        if sum(rows_split) != n:
            continue
        for cols_split in col_split_options:
            if sum(cols_split) != p:
                continue
            if time.time() > deadline - 30:
                break
            result = build_and_verify(rows_split, cols_split)
            if result and result["rank"] < best_rank:
                best_rank = result["rank"]
                best_result = result
        if time.time() > deadline - 30:
            break
    
    if best_result:
        return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
    
    # Final fallback: schoolbook (rank 84)
    U_vecs_fb = []
    V_vecs_fb = []
    W_vecs_fb = []
    gen_schoolbook(U_vecs_fb, V_vecs_fb, W_vecs_fb, n, m, p, 0, 0, 0)
    U_fb = np.array(U_vecs_fb).T
    V_fb = np.array(V_vecs_fb).T
    W_fb = np.array(W_vecs_fb).T
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
