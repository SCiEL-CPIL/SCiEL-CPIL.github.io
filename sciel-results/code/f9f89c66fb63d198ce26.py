import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 26, 26
    
    # Helper to index
    def A_idx(i, j, m=2): return i * m + j
    def B_idx(j, k, p_block): return j * p_block + k
    def C_idx(i, k, n_block=2): return k * n_block + i
    
    # --- Step 1: Try to find Rank 13 decomposition for 2x2x4 ---
    def find_2x2x4_decomp():
        # Dimensions for 2x2x4
        n_b, m_b, p_b = 2, 2, 4
        nm_b, mp_b, pn_b = n_b * m_b, m_b * p_b, p_b * n_b # 4, 8, 8
        R_b = 13
        
        # Target tensor entries
        target_entries = []
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    a = A_idx(i, j, m_b)
                    b = B_idx(j, k, p_b)
                    c = C_idx(i, k, n_b)
                    target_entries.append((a, b, c, 1.0))
        
        def loss(params):
            U = params[0:nm_b * R_b].reshape(nm_b, R_b)
            V = params[nm_b * R_b: nm_b * R_b + mp_b * R_b].reshape(mp_b, R_b)
            W = params[nm_b * R_b + mp_b * R_b:].reshape(pn_b, R_b)
            
            # Compute T
            T = np.einsum('ir,jr,kr->ijk', U, V, W)
            
            err = 0.0
            for a, b, c, val in target_entries:
                err += (T[a, b, c] - val) ** 2
            
            # Penalize non-zero entries (sparse structure)
            # We can just check the whole tensor against zeros
            # But iterating over all 256 entries is fine.
            # T shape (4, 8, 8) = 256 entries.
            # Target is sparse.
            # Create a target matrix
            T_target = np.zeros((nm_b, mp_b, pn_b))
            for a, b, c, val in target_entries:
                T_target[a, b, c] = val
            
            return np.sum((T - T_target) ** 2)
            
        # Initialize parameters
        # Use random initialization
        np.random.seed(42)
        params0 = np.random.randn(nm_b * R_b + mp_b * R_b + pn_b * R_b) * 0.1
        
        # Optimize
        try:
            res = minimize(loss, params0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6})
            if res.fun < 1e-6:
                params = res.x
                U = params[0:nm_b * R_b].reshape(nm_b, R_b)
                V = params[nm_b * R_b: nm_b * R_b + mp_b * R_b].reshape(mp_b, R_b)
                W = params[nm_b * R_b + mp_b * R_b:].reshape(pn_b, R_b)
                
                # Snap to half-integers
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                
                # Verify exactly
                T = np.einsum('ir,jr,kr->ijk', U, V, W)
                T_target = np.zeros((nm_b, mp_b, pn_b))
                for a, b, c, val in target_entries:
                    T_target[a, b, c] = val
                
                if np.allclose(T, T_target, atol=1e-9):
                    return U, V, W
        except Exception:
            pass
            
        return None

    U_224 = None
    V_224 = None
    W_224 = None
    
    if time.time() < deadline - 60:
        res_224 = find_2x2x4_decomp()
        if res_224:
            U_224, V_224, W_224 = res_224
            print("Found 2x2x4 rank 13 decomposition")
    
    # --- Step 2: Construct Full Solution ---
    
    # Initialize full matrices
    # If 2x2x4 found, use rank 43 (3 * 13 + 4)
    # Else, use rank 46 (6 * 7 + 4)
    
    if U_224 is not None:
        R_total = 3 * 13 + 4
        U = np.zeros((nm, R_total), dtype=float)
        V = np.zeros((mp, R_total), dtype=float)
        W = np.zeros((pn, R_total), dtype=float)
        
        # Block 0: k=0..3
        # U cols 0-12, V rows/cols shifted, W rows/cols shifted
        U[:, 0:13] = U_224
        # V for block 0: rows j*4+k. 
        # In full V (26 rows), rows 0..7 correspond to k=0..3?
        # V index: j*p + k. p=13.
        # So for k=0..3, indices are j*13+k.
        # j=0: 0,1,2,3. j=1: 13,14,15,16.
        # U_224 V part is 8x13. Rows 0..7 correspond to (j,k) for p_b=4.
        # We need to map V_224 rows to full V rows.
        for j in range(2):
            for k in range(4):
                v_idx_224 = j * 4 + k
                v_idx_full = j * 13 + k
                V[v_idx_full, 0:13] = V_224[v_idx_224, :]
                
        # W for block 0: indices k*2+i.
        # W_224 is 8x13. Rows 0..7 correspond to (k,i) for p_b=4, n_b=2.
        # Full W indices: k*2+i.
        for k in range(4):
            for i in range(2):
                w_idx_224 = k * 2 + i
                w_idx_full = k * 2 + i
                W[w_idx_full, 0:13] = W_224[w_idx_224, :]
                
        # Block 1: k=4..7
        col_offset = 13
        U[:, col_offset:col_offset+13] = U_224
        for j in range(2):
            for k in range(4):
                v_idx_224 = j * 4 + k
                v_idx_full = j * 13 + (k + 4)
                V[v_idx_full, col_offset:col_offset+13] = V_224[v_idx_224, :]
        for k in range(4):
            for i in range(2):
                w_idx_224 = k * 2 + i
                w_idx_full = (k + 4) * 2 + i
                W[w_idx_full, col_offset:col_offset+13] = W_224[w_idx_224, :]
                
        # Block 2: k=8..11
        col_offset = 26
        U[:, col_offset:col_offset+13] = U_224
        for j in range(2):
            for k in range(4):
                v_idx_224 = j * 4 + k
                v_idx_full = j * 13 + (k + 8)
                V[v_idx_full, col_offset:col_offset+13] = V_224[v_idx_224, :]
        for k in range(4):
            for i in range(2):
                w_idx_224 = k * 2 + i
                w_idx_full = (k + 8) * 2 + i
                W[w_idx_full, col_offset:col_offset+13] = W_224[w_idx_224, :]
                
        # Block 3: k=12 (2x2x1)
        col_offset = 39
        # Schoolbook for 2x2x1
        # U rows 0..3, V rows 12, 13+12=25, W rows 12*2, 12*2+1
        U[0, col_offset+0] = 1.0
        U[1, col_offset+1] = 1.0
        U[2, col_offset+2] = 1.0
        U[3, col_offset+3] = 1.0
        
        V[12, col_offset+0] = 1.0
        V[25, col_offset+1] = 1.0
        V[12, col_offset+2] = 1.0
        V[25, col_offset+3] = 1.0
        
        W[24, col_offset+0] = 1.0
        W[25, col_offset+1] = 1.0
        W[24, col_offset+2] = 1.0
        W[25, col_offset+3] = 1.0
        
        # Final Snap
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Verify
        T = np.einsum('ir,jr,kr->ijk', U, V, W)
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = A_idx(i, j)
                    b = B_idx(j, k)
                    c = C_idx(i, k)
                    if abs(T[a, b, c] - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        
        if valid:
            # Check zeros
            for i in range(nm):
                for j in range(mp):
                    for k in range(pn):
                        a_idx = i // 2
                        a_j = i % 2
                        b_j = j // 13
                        b_k = j % 13
                        c_k = k // 2
                        c_i = k % 2
                        if a_idx == c_i and a_j == b_j and b_k == c_k:
                            continue
                        if abs(T[i, j, k]) > 1e-9:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
        
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # --- Fallback: Rank 46 Solution ---
    # This is the logic from the previous program, kept for safety.
    
    R = 46
    U = np.zeros((nm, R), dtype=float)
    V = np.zeros((mp, R), dtype=float)
    W = np.zeros((pn, R), dtype=float)
    
    def add_strassen_block(k0, col_offset):
        prods = [
            ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
            ({(1,0):1,(1,1):1}, {(0,0):1}),
            ({(0,0):1}, {(0,1):1,(1,1):-1}),
            ({(1,1):1}, {(1,0):1,(0,0):-1}),
            ({(0,0):1,(0,1):1}, {(1,1):1}),
            ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}),
            ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}),
        ]
        for t, (a_coeff, b_coeff) in enumerate(prods):
            col = col_offset + t
            for (i, j), c in a_coeff.items():
                U[A_idx(i, j), col] += c
            for (j, kl), c in b_coeff.items():
                V[B_idx(j, k0 + kl), col] += c
        c_combos = [
            (0, k0,     [(0,1),(3,1),(4,-1),(6,1)]),
            (0, k0+1,   [(2,1),(4,1)]),
            (1, k0,     [(1,1),(3,1)]),
            (1, k0+1,   [(0,1),(1,-1),(2,1),(5,1)]),
        ]
        for (i, k, terms) in c_combos:
            for (t, coeff) in terms:
                W[C_idx(i, k), col_offset + t] += coeff

    def add_single_block(k, col_offset):
        prods = [
            ((0,0),(0,k)),
            ((0,1),(1,k)),
            ((1,0),(0,k)),
            ((1,1),(1,k)),
        ]
        for t, ((i,j),(jj,kk)) in enumerate(prods):
            col = col_offset + t
            U[A_idx(i, j), col] += 1
            V[B_idx(jj, kk), col] += 1
        W[C_idx(0, k), col_offset+0] += 1
        W[C_idx(0, k), col_offset+1] += 1
        W[C_idx(1, k), col_offset+2] += 1
        W[C_idx(1, k), col_offset+3] += 1

    col = 0
    for k0 in range(0, 12, 2):
        add_strassen_block(k0, col)
        col += 7
    add_single_block(12, col)
    col += 4
    assert col == R, f"col={col}, R={R}"
    
    # Verify fallback
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = A_idx(i, j)
                b = B_idx(j, k)
                c = C_idx(i, k)
                if abs(T[a, b, c] - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if valid:
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Final fallback: Schoolbook
    R0 = n * m * p
    U0 = np.zeros((nm, R0), dtype=float)
    V0 = np.zeros((mp, R0), dtype=float)
    W0 = np.zeros((pn, R0), dtype=float)
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col_idx] = 1.0
                V0[j * p + k, col_idx] = 1.0
                W0[k * n + i, col_idx] = 1.0
                col_idx += 1
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

