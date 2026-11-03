"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _spectral_init(a, b):
    """Spectral embedding initialization: align leading eigenvectors of A and B."""
    n = len(a)
    try:
        A_sym = a + a.T
        B_sym = b + b.T
        
        k_eig = min(8, n - 1)
        eigvals_A, eigvecs_A = np.linalg.eigh(A_sym)
        eigvals_B, eigvecs_B = np.linalg.eigh(B_sym)
        
        idx_A = np.argsort(np.abs(eigvals_A))[::-1][:k_eig]
        idx_B = np.argsort(np.abs(eigvals_B))[::-1][:k_eig]
        
        V_A = eigvecs_A[:, idx_A]
        V_B = eigvecs_B[:, idx_B]
        
        C_spectral = -V_A @ V_B.T
        row_ind, col_ind = linear_sum_assignment(C_spectral)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p
    except Exception:
        a_sums = a.sum(axis=1)
        b_sums = b.sum(axis=1)
        heavy_order = np.argsort(-a_sums, kind='stable')
        central_order = np.argsort(b_sums, kind='stable')
        p = np.empty(n, dtype=np.int64)
        p[heavy_order] = central_order
        return p


def _fast_swap_delta(a, b, p, u, v):
    """O(n) delta calculation for swapping facilities u and v in permutation p."""
    pu = p[u]
    pv = p[v]

    # d_uv[k] = b[pv, k] - b[pu, k]
    d_uv = b[pv, :] - b[pu, :]
    # d_vu[k] = b[k, pv] - b[k, pu]
    d_vu = b[:, pv] - b[:, pu]

    # Sum over m != u, v of:
    # a[u,m]*(b[pv,p[m]] - b[pu,p[m]])  -> dot(a[u,:], d_uv[p]) but exclude m=u,v
    # a[m,u]*(b[p[m],pv] - b[p[m],pu])  -> dot(a[:,u], d_vu[p]) but exclude m=u,v
    # a[v,m]*(b[pu,p[m]] - b[pv,p[m]])  -> dot(a[v,:], -d_uv[p]) but exclude m=u,v
    # a[m,v]*(b[p[m],pu] - b[p[m],pv])  -> dot(a[:,v], -d_vu[p]) but exclude m=u,v
    
    # We compute the full dot products and subtract the terms for m=u and m=v
    
    s1 = np.dot(a[u, :], d_uv[p])
    s2 = np.dot(a[:, u], d_vu[p])
    s3 = np.dot(a[v, :], d_uv[p])
    s4 = np.dot(a[:, v], d_vu[p])
    
    # Subtract m=u terms
    # For s1 (u,m): m=u -> a[u,u] * d_uv[pu]
    s1 -= a[u, u] * d_uv[pu]
    # For s1 (u,m): m=v -> a[u,v] * d_uv[pv]
    s1 -= a[u, v] * d_uv[pv]
    
    # For s2 (m,u): m=u -> a[u,u] * d_vu[pu]
    s2 -= a[u, u] * d_vu[pu]
    # For s2 (m,u): m=v -> a[v,u] * d_vu[pv]
    s2 -= a[v, u] * d_vu[pv]
    
    # For s3 (v,m): m=u -> a[v,u] * d_uv[pu]
    s3 -= a[v, u] * d_uv[pu]
    # For s3 (v,m): m=v -> a[v,v] * d_uv[pv]
    s3 -= a[v, v] * d_uv[pv]
    
    # For s4 (m,v): m=u -> a[u,v] * d_vu[pu]
    s4 -= a[u, v] * d_vu[pu]
    # For s4 (m,v): m=v -> a[v,v] * d_vu[pv]
    s4 -= a[v, v] * d_vu[pv]
    
    # The delta is s1 + s2 - s3 - s4 + cross_term
    # Cross term: a[u,v]*(b[pv,pu]-b[pu,pv]) + a[v,u]*(b[pu,pv]-b[pv,pu])
    cross = (a[u, v] - a[v, u]) * (b[pv, pu] - b[pu, pv])
    
    return s1 + s2 - s3 - s4 + cross


def _local_search(a, b, p, time_limit=10.0):
    """2-opt local search with fast O(n) delta calculation."""
    n = len(a)
    p = np.array(p, dtype=np.int64)
    current_cost = _qap_cost(a, b, p)
    
    start_time = time.time()
    best_cost = current_cost
    best_p = p.copy()
    
    np.random.seed(42)
    
    while time.time() - start_time < time_limit:
        i, j = np.random.choice(n, 2, replace=False)
        
        delta = _fast_swap_delta(a, b, p, i, j)
        
        if delta < 0:
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # Occasionally accept non-improving swaps to escape local minima?
            # For now, stick to strict descent for stability, but maybe add small prob later
            pass
            
    return best_p.tolist(), int(best_cost)


def _frank_wolfe_step(a, b, p):
    """One Frank-Wolfe step: linearize the objective and solve the assignment problem."""
    n = len(a)
    B_p = b[p]  # B_p[i, k] = B[p[i], k]
    
    # C[i, k] = sum_j a[i,j] * B[p[j], k] + sum_j a[j,i] * B[k, p[j]]
    # C = A @ B_p.T + A.T @ B_p
    C = a @ B_p.T + a.T @ B_p
    
    row_ind, col_ind = linear_sum_assignment(C)
    p_new = np.empty(n, dtype=np.int64)
    p_new[row_ind] = col_ind
    return p_new


def solve(a, b):
    """Solve the QAP instance using spectral init + Frank-Wolfe + Local Search."""
    n = len(a)
    total_start = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Phase 1: Spectral Initialization
    p = _spectral_init(a, b)
    current_cost = _qap_cost(a, b, p)
    
    best_p = p.copy()
    best_cost = current_cost
    
    # Phase 2: Frank-Wolfe Iterations
    # Each FW step is O(n^3) due to Hungarian. n=256 -> ~16M ops.
    # We can afford ~50-100 iterations in ~10-20s.
    fw_start = time.time()
    fw_deadline = fw_start + 20.0
    
    while time.time() < fw_deadline:
        p_new = _frank_wolfe_step(a, b, p)
        new_cost = _qap_cost(a, b, p_new)
        
        if new_cost < current_cost:
            current_cost = new_cost
            p = p_new
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # No improvement. Try a local search to escape, or break if stuck.
            # Let's do a quick local search to see if we can improve from current p
            # before deciding to break.
            p_ls, cost_ls = _local_search(a, b, p, time_limit=2.0)
            if cost_ls < current_cost:
                current_cost = cost_ls
                p = np.array(p_ls)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            # If even local search didn't help, we are likely at a local min of the linearization
            # Try a random perturbation
            np.random.seed(int(time.time() * 1000) % 10000)
            i, j = np.random.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]
            current_cost = _qap_cost(a, b, p)
            # Continue FW loop, but check time
    
    # Phase 3: Final Polish with Local Search
    time_remaining = deadline - (time.time() - total_start)
    if time_remaining > 5:
        p_final, cost_final = _local_search(a, b, best_p, time_limit=min(time_remaining - 2, 60.0))
        if cost_final < best_cost:
            best_p = np.array(p_final)
            best_cost = cost_final
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
