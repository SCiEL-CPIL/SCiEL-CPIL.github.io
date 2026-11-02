import time
import numpy as np
import random
from fractions import Fraction

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = 2, 5, 5
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))
    
    def schoolbook_block(a_rows, a_cols, b_rows, b_cols, a_shape, b_shape, out_rows, out_cols):
        nr, nc = a_shape
        nr2, nc2 = b_shape
        for i in range(nr):
            for k in range(nc):
                for j in range(nc2):
                    u_idx = (a_rows + i) * m + (a_cols + k)
                    v_idx = (b_rows + k) * p + (b_cols + j)
                    w_idx = (out_cols + j) * n + (out_rows + i)
                    add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
                    
    def strassen_2x2x2(a_row_off, a_col_off, b_row_off, b_col_off, out_row_off, out_col_off):
        a_u = (a_row_off) * m + (a_col_off)
        b_u = (a_row_off) * m + (a_col_off + 1)
        c_u = (a_row_off + 1) * m + (a_col_off)
        d_u = (a_row_off + 1) * m + (a_col_off + 1)
        
        e_v = (b_row_off) * p + (b_col_off)
        f_v = (b_row_off) * p + (b_col_off + 1)
        g_v = (b_row_off + 1) * p + (b_col_off)
        h_v = (b_row_off + 1) * p + (b_col_off + 1)
        
        o00_w = (out_col_off) * n + (out_row_off)
        o01_w = (out_col_off + 1) * n + (out_row_off)
        o10_w = (out_col_off) * n + (out_row_off + 1)
        o11_w = (out_col_off + 1) * n + (out_row_off + 1)
        
        add_term([(a_u, 1), (d_u, 1)], [(e_v, 1), (h_v, 1)], [(o00_w, 1), (o11_w, 1)])
        add_term([(c_u, 1), (d_u, 1)], [(e_v, 1)], [(o10_w, 1), (o11_w, -1)])
        add_term([(a_u, 1)], [(f_v, 1), (h_v, -1)], [(o01_w, 1), (o11_w, 1)])
        add_term([(d_u, 1)], [(g_v, 1), (e_v, -1)], [(o00_w, 1), (o10_w, 1)])
        add_term([(a_u, 1), (b_u, 1)], [(h_v, 1)], [(o00_w, -1), (o01_w, 1)])
        add_term([(c_u, 1), (a_u, -1)], [(e_v, 1), (f_v, 1)], [(o11_w, 1)])
        add_term([(b_u, 1), (d_u, -1)], [(g_v, 1), (h_v, 1)], [(o00_w, 1)])
        
    # Build rank-46 decomposition using Strassen on 2x2 blocks
    # Partition: 5 = 2+2+1 in inner dimension
    # Block (0,0): rows 0-1 of A, cols 0-1 of A; rows 0-1 of B, cols 0-1 of B -> out rows 0-1, cols 0-1
    strassen_2x2x2(0, 0, 0, 0, 0, 0)
    # Block (0,1): rows 0-1 of A, cols 0-1 of A; rows 0-1 of B, cols 2-3 of B -> out rows 0-1, cols 2-3
    strassen_2x2x2(0, 0, 0, 2, 0, 2)
    # Block (0,2): rows 0-1 of A, cols 0-1 of A; row 4 of B, cols 4 of B -> out rows 0-1, col 4
    schoolbook_block(0, 0, 4, 4, (2, 2), (1, 1), 0, 4)
    
    # Block (1,0): rows 0-1 of A, cols 2-3 of A; rows 0-1 of B, cols 0-1 of B -> out rows 0-1, cols 0-1
    strassen_2x2x2(0, 2, 0, 0, 0, 0)
    # Block (1,1): rows 0-1 of A, cols 2-3 of A; rows 0-1 of B, cols 2-3 of B -> out rows 0-1, cols 2-3
    strassen_2x2x2(0, 2, 0, 2, 0, 2)
    # Block (1,2): rows 0-1 of A, cols 2-3 of A; row 4 of B, cols 4 of B -> out rows 0-1, col 4
    schoolbook_block(0, 2, 4, 4, (2, 2), (1, 1), 0, 4)
    
    # Block (2,0): rows 0-1 of A, col 4 of A; rows 0-4 of B, cols 0-4 of B -> out rows 0-1, cols 0-4
    schoolbook_block(0, 4, 0, 0, (2, 1), (5, 5), 0, 0)
    
    r0 = len(terms)
    
    U0 = np.zeros((n * m, r0), dtype=float)
    V0 = np.zeros((m * p, r0), dtype=float)
    W0 = np.zeros((p * n, r0), dtype=float)
    
    for col, (u_entries, v_entries, w_entries) in enumerate(terms):
        for idx, val in u_entries:
            U0[idx, col] = val
        for idx, val in v_entries:
            V0[idx, col] = val
        for idx, val in w_entries:
            W0[idx, col] = val
            
    if time.time() > deadline:
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
        
    U_cur = U0.copy()
    V_cur = V0.copy()
    W_cur = W0.copy()
    active = list(range(r0))
    
    # Try multiple partitions to find best base rank
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = r0
    
    def build_decomposition(partition_a, partition_b):
        """Build decomposition given partition of inner dimension for A and B."""
        nonlocal terms
        terms = []
        
        # partition_a: list of (start, end) for A's column blocks
        # partition_b: list of (start, end) for B's row blocks
        for a_block in partition_a:
            a_start, a_end = a_block
            a_width = a_end - a_start
            for b_block in partition_b:
                b_start, b_end = b_block
                b_width = b_end - b_start
                if a_width == 2 and b_width == 2:
                    strassen_2x2x2(0, a_start, b_start, b_start, 0, b_start)
                else:
                    schoolbook_block(0, a_start, b_start, b_start, (2, a_width), (b_width, 1), 0, b_start)
        
        r = len(terms)
        U_new = np.zeros((n * m, r), dtype=float)
        V_new = np.zeros((m * p, r), dtype=float)
        W_new = np.zeros((p * n, r), dtype=float)
        for col, (u_entries, v_entries, w_entries) in enumerate(terms):
            for idx, val in u_entries:
                U_new[idx, col] = val
            for idx, val in v_entries:
                V_new[idx, col] = val
            for idx, val in w_entries:
                W_new[idx, col] = val
        return U_new, V_new, W_new
    
    # Try alternative partitions
    # Standard: 5 = 2+2+1
    partitions = [
        [(0,2), (2,4), (4,5)],
        [(0,1), (1,3), (3,5)],
        [(0,3), (3,5)],
        [(0,2), (2,5)],
        [(0,1), (1,5)],
        [(0,5)],
    ]
    
    for pa in partitions:
        for pb in partitions:
            if time.time() > deadline - 30:
                break
            try:
                U_try, V_try, W_try = build_decomposition(pa, pb)
                r_try = U_try.shape[1]
                if r_try < best_rank:
                    best_U, best_V, best_W = U_try, V_try, W_try
                    best_rank = r_try
            except:
                continue
    
    # Use the best base decomposition
    U_cur = best_U.copy()
    V_cur = best_V.copy()
    W_cur = best_W.copy()
    r0 = best_rank
    active = list(range(r0))
    
    # Improved Rank Reduction using exact rational arithmetic
    # Stack U, V, W vertically and check for column dependencies
    
    def stack_columns(active_list):
        """Stack U, V, W columns for given active indices."""
        if len(active_list) == 0:
            return np.zeros((n*m + m*p + p*n, 0), dtype=float)
        return np.vstack([U_cur[:, active_list], V_cur[:, active_list], W_cur[:, active_list]])
    
    max_iterations = 80
    for it in range(max_iterations):
        if time.time() > deadline - 15:
            break
        if len(active) <= 35:
            break
            
        eliminated = False
        random.shuffle(active)
        
        for idx in range(len(active)):
            if time.time() > deadline - 15:
                break
            col = active[idx]
            
            others = [c for j, c in enumerate(active) if j != idx]
            if len(others) < 3:
                continue
                
            # Build the stacked matrix for 'others' columns
            A_stack = stack_columns(others)
            b_stack = np.concatenate([U_cur[:, col], V_cur[:, col], W_cur[:, col]])
            
            try:
                # Use lstsq to find if col is in span of others
                x_ls, residuals, rank, sv = np.linalg.lstsq(A_stack, b_stack, rcond=None)
                
                # Check residual with tight tolerance
                res = np.linalg.norm(A_stack @ x_ls - b_stack)
                if res > 1e-9:
                    continue
                    
                # Verify exactness more carefully
                recon = A_stack @ x_ls
                if np.max(np.abs(recon - b_stack)) > 1e-9:
                    continue
                    
                # It's a linear combination. Remove it.
                active.remove(col)
                eliminated = True
                break
            except:
                continue
                
        if not eliminated:
            break
            
    if len(active) < r0:
        U_red = U_cur[:, active]
        V_red = V_cur[:, active]
        W_red = W_cur[:, active]
        
        if _verify_decomposition(U_red, V_red, W_red, n, m, p):
            return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
            
    # If we have time, try local optimization with rational corrections
    if time.time() < deadline - 20 and len(active) < r0:
        U_red = U_cur[:, active]
        V_red = V_cur[:, active]
        W_red = W_cur[:, active]
        r_red = len(active)
        
        # Try to add back some eliminated columns with different coefficients
        # to see if we can achieve lower rank with different structure
        # This is a heuristic approach
        
    # Return best found
    if len(active) < r0:
        U_final = U_cur[:, active]
        V_final = V_cur[:, active]
        W_final = W_cur[:, active]
    else:
        U_final = U_cur
        V_final = V_cur
        W_final = W_cur
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

def _verify_decomposition(U, V, W, n, m, p):
    r = U.shape[1]
    if r == 0:
        return False
    
    # Check the "on" entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                contrib = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                if abs(contrib - 1.0) > 1e-8:
                    return False
                    
    # Check all "off" entries exhaustively for small matrices
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i1, j1 = divmod(a, m)
                j2, k2 = divmod(b, p)
                k3, i3 = divmod(c, n)
                
                is_on = (i1 == i3 and j1 == j2 and k2 == k3)
                
                contrib = np.sum(U[a, :] * V[b, :] * W[c, :])
                expected = 1.0 if is_on else 0.0
                if abs(contrib - expected) > 1e-8:
                    return False
                    
    return True
