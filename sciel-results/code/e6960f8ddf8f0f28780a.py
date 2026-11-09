import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # We want to find a rank 24 decomposition.
    # We split the p=8 dimension into 4 groups of 2.
    # For each group, we use a rank 6 decomposition of 2x3x2.
    
    # First, find a rank 6 decomposition for 2x3x2.
    # T[i, j, k] = 1 for i in 0..1, j in 0..2, k in 0..1.
    # Flattened indices:
    # a = i*m + j = i*3 + j
    # b = j*p + k = j*2 + k
    # c = k*n + i = k*2 + i
    
    # We will use a simple search or hardcode if possible.
    # Let's try to find it using optimization.
    
    from scipy.optimize import minimize
    
    a_size = 2 * 3 # 6
    b_size = 3 * 2 # 6
    c_size = 2 * 2 # 4
    
    T = np.zeros((a_size, b_size, c_size))
    for i in range(2):
        for j in range(3):
            for k in range(2):
                a = i * 3 + j
                b = j * 2 + k
                c = k * 2 + i
                T[a, b, c] = 1.0
    
    rank_small = 6
    
    def _objective_small(x):
        U = x[:a_size*rank_small].reshape(a_size, rank_small)
        V = x[a_size*rank_small:a_size*rank_small+b_size*rank_small].reshape(b_size, rank_small)
        W = x[a_size*rank_small+b_size*rank_small:].reshape(c_size, rank_small)
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        residual = T - recon
        return np.sum(residual**2)
    
    # Try to find the small decomposition
    found_small = None
    for seed in range(10):
        np.random.seed(seed)
        x0 = np.random.randn(a_size*rank_small + b_size*rank_small + c_size*rank_small) * 0.1
        res = minimize(_objective_small, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
        if res.fun < 1e-8:
            U_small = res.x[:a_size*rank_small].reshape(a_size, rank_small)
            V_small = res.x[a_size*rank_small:a_size*rank_small+b_size*rank_small].reshape(b_size, rank_small)
            W_small = res.x[a_size*rank_small+b_size*rank_small:].reshape(c_size, rank_small)
            # Verify
            recon = np.einsum('ar,br,cr->abc', U_small, V_small, W_small)
            if np.max(np.abs(T - recon)) < 1e-8:
                found_small = (U_small, V_small, W_small)
                break
    
    if found_small is None:
        # Fallback to schoolbook for the whole thing
        return _schoolbook(n, m, p)
    
    U_small, V_small, W_small = found_small
    
    # Now assemble the rank 24 decomposition for 2x3x8.
    # The groups are k in {0,1}, {2,3}, {4,5}, {6,7}.
    # For group g (0..3), the local k index is k_local = k - 2g.
    # The global indices are:
    # a = i*3 + j (same as small)
    # b = j*8 + k = j*8 + 2g + k_local
    # c = k*2 + i = (2g + k_local)*2 + i = 4g + 2*k_local + i
    
    rank_total = 24
    U_total = np.zeros((n*m, rank_total))
    V_total = np.zeros((m*p, rank_total))
    W_total = np.zeros((p*n, rank_total))
    
    col_offset = 0
    for g in range(4):
        # Copy U_small
        U_total[:, col_offset:col_offset+rank_small] = U_small
        
        # Copy V_small with index shift
        # b_global = j*8 + 2g + k_local
        # b_small = j*2 + k_local
        # For each row r in V_small (which corresponds to j*2 + k_local),
        # the global row is j*8 + 2g + k_local.
        for r in range(b_size):
            j = r // 2
            k_local = r % 2
            b_global = j * 8 + 2 * g + k_local
            V_total[b_global, col_offset:col_offset+rank_small] = V_small[r, :]
        
        # Copy W_small with index shift
        # c_global = (2g + k_local)*2 + i = 4g + 2*k_local + i
        # c_small = k_local*2 + i
        for r in range(c_size):
            k_local = r // 2
            i = r % 2
            c_global = 4 * g + 2 * k_local + i
            W_total[c_global, col_offset:col_offset+rank_small] = W_small[r, :]
        
        col_offset += rank_small
    
    # Verify the full decomposition
    T_full = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_full[a, b, c] = 1.0
                
    recon_full = np.einsum('ar,br,cr->abc', U_total, V_total, W_total)
    if np.max(np.abs(T_full - recon_full)) > 1e-8:
        return _schoolbook(n, m, p)
        
    return {
        "U": _to_list(U_total),
        "V": _to_list(V_total),
        "W": _to_list(W_total)
    }

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {
        "U": _to_list(U),
        "V": _to_list(V),
        "W": _to_list(W)
    }

def _to_list(mat):
    result = []
    for row in mat:
        row_list = []
        for val in row:
            nearest_int = round(val)
            nearest_half = round(2 * val) / 2.0
            if abs(val - nearest_int) < 1e-6:
                row_list.append(int(nearest_int))
            elif abs(val - nearest_half) < 1e-6:
                row_list.append(nearest_half)
            else:
                row_list.append(float(val))
        result.append(row_list)
    return result
