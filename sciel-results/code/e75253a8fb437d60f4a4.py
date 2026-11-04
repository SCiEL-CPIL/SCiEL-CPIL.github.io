import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Initialization: Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # 2. Precompute constants
    a_diag = np.diag(a)
    b_self = np.diag(b)
    
    # Helper to calculate cost
    def calc_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    current_cost = calc_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # 3. Compute D matrix
    # D[w, l] = sum_k A[l, k] * B[w, p[k]]
    # M = B[:, p]
    # D = A @ M.T
    D = a @ b[:, p].T
    
    # 4. Local Search Loop
    while time.time() < deadline:
        # Calculate Delta matrix
        # d_i = D[p[i], i]
        d_vec = D[p, np.arange(n)]
        
        # M_D part: 2 * (D[p, :].T - d_vec[:, None])
        M_D = 2.0 * (D[p, :].T - d_vec[:, np.newaxis])
        
        # M_A part: 3 * (a_diag[:, None] * b_self[p][None, :] - (a_diag * b_self[p])[:, None])
        b_self_p = b_self[p]
        term_a = a_diag * b_self_p
        M_A = 3.0 * (a_diag[:, np.newaxis] * b_self_p[np.newaxis, :] - term_a[:, np.newaxis])
        
        # M_B part: -(b_self[p][:, None] * a_diag[None, :]) + (b_self[p] * a_diag)[None, :]
        M_B = -(b_self_p[:, np.newaxis] * a_diag[np.newaxis, :]) + term_a[np.newaxis, :]
        
        delta = M_D + M_A + M_B
        
        # Find the most negative delta (best improvement)
        min_idx = np.unravel_index(np.argmin(delta), delta.shape)
        i, j = min_idx
        min_delta = delta[i, j]
        
        if min_delta >= 0:
            break
            
        # Apply swap
        p[i], p[j] = p[j], p[i]
        current_cost += min_delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        # Update D
        # D_new = D_old + (A[:, i] - A[:, j])[:, None] * (B[p[i], :] - B[p[j], :])[None, :]
        # Note: p[i] and p[j] are already swapped, so we need the OLD values for the update formula?
        # Formula derived: D_new[w, l] = D_old[w, l] + (A[l, i] - A[l, j]) * (B[w, v] - B[w, u])
        # where u was at i, v was at j.
        # After swap, p[i]=v, p[j]=u.
        # So B[w, v] is B[w, p[i]] and B[w, u] is B[w, p[j]].
        u, v = p[j], p[i] # u is now at j, v is now at i.
        # Wait, before swap: p[i]=u, p[j]=v.
        # After swap: p[i]=v, p[j]=u.
        # So in the current p, p[i]=v, p[j]=u.
        # The term is (B[w, v] - B[w, u]).
        # So we use B[p[i], :] - B[p[j], :].
        
        diff_a = a[:, i] - a[:, j]
        diff_b = b[p[i], :] - b[p[j], :]
        D += np.outer(diff_a, diff_b)
        
    return best_p.tolist()
