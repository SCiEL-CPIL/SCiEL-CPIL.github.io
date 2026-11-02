import time
import numpy as np
import random

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
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
        
    # Build rank-46 decomposition
    strassen_2x2x2(0, 0, 0, 0, 0, 0)
    strassen_2x2x2(0, 0, 0, 2, 0, 2)
    schoolbook_block(0, 0, 0, 4, (2, 2), (2, 1), 0, 4)
    
    strassen_2x2x2(0, 2, 2, 0, 0, 0)
    strassen_2x2x2(0, 2, 2, 2, 0, 2)
    schoolbook_block(0, 2, 2, 4, (2, 2), (2, 1), 0, 4)
    
    schoolbook_block(0, 4, 4, 0, (2, 1), (1, 5), 0, 0)
    
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
    
    # Improved Rank Reduction
    # Try to remove columns that are linear combinations of others
    # Use a sparsity-promoting approach
    
    max_iterations = 100
    for _ in range(max_iterations):
        if time.time() > deadline - 5:
            break
        if len(active) <= 40:
            break
            
        eliminated = False
        # Shuffle active to avoid bias
        random.shuffle(active)
        
        for idx in range(len(active)):
            col = active[idx]
            
            others = [c for j, c in enumerate(active) if j != idx]
            if len(others) < 3:
                continue
                
            U_others = U_cur[:, others]
            V_others = V_cur[:, others]
            W_others = W_cur[:, others]
            
            # Stack the matrices for joint least squares
            # We want x such that U_others x ~ U_col, V_others x ~ V_col, W_others x ~ W_col
            # This is an overdetermined system.
            
            try:
                # First, get the dense least squares solution
                # Concatenate vertically
                A_stack = np.vstack([U_others, V_others, W_others])
                b_stack = np.concatenate([U_cur[:, col], V_cur[:, col], W_cur[:, col]])
                
                x_ls, residuals, rank, sv = np.linalg.lstsq(A_stack, b_stack, rcond=None)
                
                # Check residual
                res = np.linalg.norm(A_stack @ x_ls - b_stack)
                if res > 1e-10:
                    continue
                    
                # Check if x_ls is sparse
                # If it is dense, try to sparsify it
                # We can try to zero out small components and re-solve
                x_sparse = x_ls.copy()
                for _ in range(5):
                    # Find the smallest non-zero component
                    mask = np.abs(x_sparse) > 1e-12
                    if np.sum(mask) <= 4:
                        break
                    # Zero out the smallest
                    min_idx = np.argmin(np.abs(x_sparse))
                    if np.abs(x_sparse[min_idx]) < 1e-6:
                        x_sparse[min_idx] = 0
                        # Re-solve with the mask
                        A_masked = A_stack[:, mask]
                        # This is getting complicated. Let's just check if the dense solution is already "sparse enough"
                        # or if we can find a sparse solution via a different method.
                        pass
                        break
                    else:
                        break
                        
                # For now, just use the dense solution if it's exact
                # Verify exactness
                if res < 1e-10:
                    # It is a linear combination. Remove it.
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
            
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

def _verify_decomposition(U, V, W, n, m, p):
    r = U.shape[1]
    
    # Check the "on" entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                contrib = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                if abs(contrib - 1.0) > 1e-10:
                    return False
                    
    # Check some "off" entries
    random.seed(42)
    for _ in range(500):
        a = random.randint(0, n*m - 1)
        b = random.randint(0, m*p - 1)
        c = random.randint(0, p*n - 1)
        
        i1, j1 = divmod(a, m)
        j2, k2 = divmod(b, p)
        k3, i3 = divmod(c, n)
        
        is_on = (i1 == i3 and j1 == j2 and k2 == k3)
        
        contrib = np.sum(U[a, :] * V[b, :] * W[c, :])
        expected = 1.0 if is_on else 0.0
        if abs(contrib - expected) > 1e-10:
            return False
            
    return True
