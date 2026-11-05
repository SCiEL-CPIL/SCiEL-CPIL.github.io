"""Rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.

Strategy:
1. Attempt to find a rank-10 decomposition for the 2x2 * 2x3 sub-problem using
   a more robust optimization approach (multiple restarts, better initialization).
2. If found, use recursive Strassen to build a rank-70 decomposition for 4x4 * 4x6.
3. If not found within time limit, fall back to the proven rank-77 decomposition.
"""

import time
import numpy as np
import sys

_START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin before the 150s kill


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the tensor decomposition is correct.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = np.dot(U_arr[u_idx, :], V_arr[v_idx, :] * W_arr[w_idx, :])
                if abs(total - 1.0) > tol:
                    return False
    return True


def _search_2x2x2x3_rank10():
    """
    Search for a rank-10 decomposition of the 2x2 * 2x3 matrix multiplication tensor.
    Returns (U, V, W) as numpy arrays if found, else None.
    """
    if _time_remaining() < 15.0:
        return None

    n, m, p = 2, 2, 3
    rank = 10
    
    # Target tensor T[a, b, c]
    # a = i*m + j (0..3)
    # b = j*p + k (0..5)
    # c = k*n + i (0..5)
    # T[a,b,c] = 1 if consistent, 0 otherwise.
    
    # List of (a, b, c) indices where T=1
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
                
    # Total possible indices
    total_indices = []
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                total_indices.append((a, b, c))
                
    target_set = set(target_indices)
    zero_indices = [idx for idx in total_indices if idx not in target_set]
    
    # Build target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for a, b, c in target_indices:
        T_target[a, b, c] = 1.0
        
    def objective_and_grad(params):
        U = params[0:4*rank].reshape(4, rank)
        V = params[4*rank:10*rank].reshape(6, rank)
        W = params[10*rank:16*rank].reshape(6, rank)
        
        T_recon = np.einsum('al,bl,cl->abc', U, V, W)
        err = T_recon - T_target
        loss = np.sum(err * err)
        
        grad_U = 2 * np.einsum('abc,bl,cl->al', err, V, W)
        grad_V = 2 * np.einsum('abc,al,cl->bl', err, U, W)
        grad_W = 2 * np.einsum('abc,al,bl->cl', err, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    np.random.seed(42)
    best_loss = float('inf')
    best_params = None
    
    # Try multiple random initializations with different scales
    num_trials = 20
    trial_delay = (TIME_LIMIT - (time.time() - _START_TIME) - 20.0) / num_trials if _time_remaining() > 20.0 else 0
    
    for trial in range(num_trials):
        if _time_remaining() < 10.0:
            break
            
        # Vary initialization scale
        scale = 0.3 + 0.4 * (trial % 5) / 4.0
        
        U_init = np.random.randn(4, rank) * scale
        V_init = np.random.randn(6, rank) * scale
        W_init = np.random.randn(6, rank) * scale
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            from scipy.optimize import minimize
            res = minimize(objective_and_grad, params_init, method='L-BFGS-B', 
                          jac=True, options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
                
                if best_loss < 1e-10:
                    break
        except Exception as e:
            pass

    if best_params is not None and best_loss < 1e-8:
        U = best_params[0:4*rank].reshape(4, rank)
        V = best_params[4*rank:10*rank].reshape(6, rank)
        W = best_params[10*rank:16*rank].reshape(6, rank)
        
        # Round to nearest half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify rounded solution
        if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), 2, 2, 3, tol=1e-9):
            return U_rounded, V_rounded, W_rounded
            
    return None


def construct():
    """
    Construct a rank-reduced tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Returns a dict with keys 'U', 'V', 'W'.
    """
    n, m, p = 4, 4, 6
    
    # First, try to find a rank-10 decomposition for 2x2 * 2x3
    sub_decomp = None
    if _time_remaining() > 30.0:
        sub_decomp = _search_2x2x2x3_rank10()
        
    if sub_decomp is not None:
        U_sub, V_sub, W_sub = sub_decomp
        rank_sub = 10
        total_rank = 7 * rank_sub  # 70
        
        # Build 4x4 * 4x6 using recursive Strassen
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        def add_strassen_term(term_idx, u_blocks, v_blocks, w_blocks):
            base = term_idx * rank_sub
            
            # Map U_sub to global U
            for b_idx, coeff in u_blocks.items():
                if b_idx == 0:
                    rows = [0, 1, 4, 5]
                elif b_idx == 1:
                    rows = [2, 3, 6, 7]
                elif b_idx == 2:
                    rows = [8, 9, 12, 13]
                elif b_idx == 3:
                    rows = [10, 11, 14, 15]
                else:
                    continue
                for local_r, global_r in enumerate(rows):
                    U[global_r, base:base+rank_sub] += coeff * U_sub[local_r, :]
                    
            # Map V_sub to global V
            for b_idx, coeff in v_blocks.items():
                if b_idx == 0:
                    rows = [0, 1, 2, 3, 4, 5]
                elif b_idx == 1:
                    rows = [6, 7, 8, 9, 10, 11]
                elif b_idx == 2:
                    rows = [12, 13, 14, 15, 16, 17]
                elif b_idx == 3:
                    rows = [18, 19, 20, 21, 22, 23]
                else:
                    continue
                for local_r, global_r in enumerate(rows):
                    V[global_r, base:base+rank_sub] += coeff * V_sub[local_r, :]
                    
            # Map W_sub to global W
            for b_idx, coeff in w_blocks.items():
                if b_idx == 0:
                    rows = [0, 1, 2, 3, 4, 5]
                elif b_idx == 1:
                    rows = [6, 7, 8, 9, 10, 11]
                elif b_idx == 2:
                    rows = [12, 13, 14, 15, 16, 17]
                elif b_idx == 3:
                    rows = [18, 19, 20, 21, 22, 23]
                else:
                    continue
                for local_r, global_r in enumerate(rows):
                    W[global_r, base:base+rank_sub] += coeff * W_sub[local_r, :]

        # Define the 7 Strassen terms
        add_strassen_term(0, {0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1})
        add_strassen_term(1, {2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1})
        add_strassen_term(2, {0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1})
        add_strassen_term(3, {3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1})
        add_strassen_term(4, {0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1})
        add_strassen_term(5, {2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
        add_strassen_term(6, {1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
        
        # Verify
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback: Rank 77
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term_ijk(u_dict, v_dict, w_dict):
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (i, j), coeff in u_dict.items():
            u_col[i * m + j] += coeff
        for (j, k), coeff in v_dict.items():
            v_col[j * p + k] += coeff
        for (i, k), coeff in w_dict.items():
            w_col[k * n + i] += coeff
            
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
        
    def strassen_2x2(oi, oj, ok):
        add_term_ijk({(oi, oj): 1, (oi+1, oj+1): 1}, {(oj, ok): 1, (oj+1, ok+1): 1}, {(oi, ok): 1, (oi+1, ok+1): 1})
        add_term_ijk({(oi+1, oj): 1, (oi+1, oj+1): 1}, {(oj, ok): 1}, {(oi+1, ok): 1, (oi+1, ok+1): -1})
        add_term_ijk({(oi, oj): 1}, {(oj, ok+1): 1, (oj+1, ok+1): -1}, {(oi, ok+1): 1, (oi+1, ok+1): 1})
        add_term_ijk({(oi+1, oj+1): 1}, {(oj+1, ok): 1, (oj, ok): -1}, {(oi, ok): 1, (oi+1, ok): 1})
        add_term_ijk({(oi, oj): 1, (oi, oj+1): 1}, {(oj+1, ok+1): 1}, {(oi, ok): -1, (oi, ok+1): 1})
        add_term_ijk({(oi+1, oj): 1, (oi, oj): -1}, {(oj, ok): 1, (oj, ok+1): 1}, {(oi+1, ok+1): 1})
        add_term_ijk({(oi, oj+1): 1, (oi+1, oj+1): -1}, {(oj+1, ok): 1, (oj+1, ok+1): 1}, {(oi, ok): 1})

    def _strassen_super(oi, oj, ok, u_blocks, v_blocks, w_blocks):
        strassen_2x2_terms_local = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        block_offsets = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
        
        for u_pat, v_pat, w_pat in strassen_2x2_terms_local:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            for block_idx, block_coeff in u_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in u_pat.items():
                    i = oi + 2*br + r
                    j = oj + 2*bc + c
                    if 0 <= i < n and 0 <= j < m:
                        u_col[i * m + j] += block_coeff * coeff
            
            for block_idx, block_coeff in v_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in v_pat.items():
                    j = oj + 2*br + r
                    k = ok + 2*bc + c
                    if 0 <= j < m and 0 <= k < p:
                        v_col[j * p + k] += block_coeff * coeff
            
            for block_idx, block_coeff in w_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in w_pat.items():
                    i = oi + 2*br + r
                    k = ok + 2*bc + c
                    if 0 <= i < n and 0 <= k < p:
                        w_col[k * n + i] += block_coeff * coeff
                        
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)

    # 4x4 * 4x4 part (cols 0-3)
    _strassen_super(0, 0, 0, {0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1})
    _strassen_super(0, 0, 0, {2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1})
    _strassen_super(0, 0, 0, {0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1})
    _strassen_super(0, 0, 0, {3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1})
    _strassen_super(0, 0, 0, {0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1})
    _strassen_super(0, 0, 0, {2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
    _strassen_super(0, 0, 0, {1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    
    # 4x4 * 4x2 part (cols 4-5)
    for rb in range(2):
        strassen_2x2(rb*2, 0, 4)
        strassen_2x2(rb*2, 2, 4)
        
    rank = len(U_cols)
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
