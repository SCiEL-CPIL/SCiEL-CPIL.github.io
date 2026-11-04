import time
import numpy as np
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    deadline = time.time() + 118.0
    
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    def compute_cost(p):
        B_p = b[np.ix_(p, p)]
        return int(np.sum(a * B_p))

    def build_row_cost(p):
        """row_cost[i, l] = sum_k a[i,k] * b[l, p[k]]"""
        b_col_p = b_f[:, p]
        return a_f @ b_col_p

    def swap_delta(i, j, p, row_cost):
        """Compute delta for swapping p[i] and p[j]. O(n) via vectorized dot."""
        u = p[i]
        v = p[j]
        # row_cost terms
        rc = (row_cost[i, v] - row_cost[i, u]) + (row_cost[j, u] - row_cost[j, v])
        # diagonal/cross terms
        d1 = (a_f[i, i] - a_f[i, j]) * (b_f[v, v] - b_f[v, u])
        d2 = (a_f[j, i] - a_f[j, j]) * (b_f[u, v] - b_f[u, u])
        # sum over k != i,j: (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])
        diff_a = a_f[:, i] - a_f[:, j]
        diff_b = b_f[p, v] - b_f[p, u]
        S = np.dot(diff_a, diff_b)
        S -= (a_f[i, i] - a_f[i, j]) * (b_f[u, v] - b_f[u, u])
        S -= (a_f[j, i] - a_f[j, j]) * (b_f[v, u] - b_f[v, v])
        return rc + d1 + d2 + S

    def update_row_cost(p, i, j, row_cost):
        """Incremental update of row_cost after swapping p[i] and p[j]."""
        u = p[i]
        v = p[j]
        da = a_f[:, i] - a_f[:, j]
        db = b_f[:, v] - b_f[:, u]
        row_cost += np.outer(da, db)

    def local_search_first_improvement(p, time_limit):
        """Fast first-improvement 2-opt local search with O(n) delta per pair."""
        p = p.copy()
        cost = compute_cost(p)
        row_cost = build_row_cost(p)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(n):
                if time.time() > time_limit:
                    break
                u = p[i]
                for j in range(i + 1, n):
                    if time.time() > time_limit:
                        break
                    v = p[j]
                    delta = swap_delta(i, j, p, row_cost)
                    if delta < -1e-9:
                        # Accept this swap (first improvement)
                        p[i], p[j] = v, u
                        cost += delta
                        update_row_cost(p, i, j, row_cost)
                        improved = True
                        break  # restart scan from i+1 or full restart
            # After a full pass with no improvement, we're at local optimum
        return p, cost

    def local_search_best_improvement(p, time_limit):
        """Best-improvement 2-opt: find the single best swap per full pass."""
        p = p.copy()
        cost = compute_cost(p)
        row_cost = build_row_cost(p)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            for i in range(n):
                if time.time() > time_limit:
                    break
                u = p[i]
                for j in range(i + 1, n):
                    if time.time() > time_limit:
                        break
                    v = p[j]
                    delta = swap_delta(i, j, p, row_cost)
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            if best_i != -1:
                i, j = best_i, best_j
                p[i], p[j] = p[j], p[i]
                cost += best_delta
                update_row_cost(p, i, j, row_cost)
                improved = True
        return p, cost

    def rank_init():
        flow_sums = a_f.sum(axis=1) + a_f.sum(axis=0)
        dist_sums = b_f.sum(axis=1) + b_f.sum(axis=0)
        fac_order = np.argsort(-flow_sums)
        loc_order = np.argsort(dist_sums)
        p = np.empty(n, dtype=np.int64)
        p[fac_order] = loc_order
        return p

    def spectral_init():
        try:
            a_sym = (a_f + a_f.T) / 2.0
            b_sym = (b_f + b_f.T) / 2.0
            evals_a, evecs_a = np.linalg.eigh(a_sym)
            evals_b, evecs_b = np.linalg.eigh(b_sym)
            idx_a = np.argsort(-np.abs(evals_a))
            idx_b = np.argsort(-np.abs(evals_b))
            V_a = evecs_a[:, idx_a]
            V_b = evecs_b[:, idx_b]
            M = V_a.T @ V_b
            row_ind, col_ind = linear_sum_assignment(-M)
            p = np.empty(n, dtype=np.int64)
            p[row_ind] = col_ind
            return p
        except:
            return None

    def hungarian_refine(p_cur):
        b_col_p = b_f[:, p_cur]
        b_row_p = b_f[p_cur, :]
        C = a_f @ b_col_p.T + a_f.T @ b_row_p
        row_ind, col_ind = linear_sum_assignment(C)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        return p_new

    def simulated_annealing(p, time_limit):
        p = p.copy()
        current_cost = compute_cost(p)
        best_p = p.copy()
        best_cost = current_cost
        row_cost = build_row_cost(p)
        
        T = 500.0
        cooling = 0.995
        min_T = 1.0
        
        while time.time() < time_limit and T > min_T:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            delta = swap_delta(i, j, p, row_cost)
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                update_row_cost(p, i, j, row_cost)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            T *= cooling
        return best_p, best_cost

    # Main Logic
    best_p_overall = None
    best_cost_overall = float('inf')
    
    inits = []
    p_rank = rank_init()
    inits.append(p_rank)
    p_spec = spectral_init()
    if p_spec is not None:
        inits.append(p_spec)
    for _ in range(4):
        inits.append(np.random.permutation(n))
    
    # Phase 1: Quick SA + local search on multiple starts
    # Allocate ~60% of time to this phase
    phase1_end = time.time() + (deadline - time.time()) * 0.55
    
    for idx, p_init in enumerate(inits):
        if time.time() > phase1_end - 2:
            break
        
        p_hung = hungarian_refine(p_init)
        c_hung = compute_cost(p_hung)
        c_init = compute_cost(p_init)
        if c_hung < c_init:
            p_cur = p_hung
        else:
            p_cur = p_init
        
        # Give each restart a slice of phase1 time
        remaining_restarts = len(inits) - idx
        sa_time = min(phase1_end - time.time() - 1, 8.0)
        if sa_time < 1:
            sa_time = 1
        
        p_sa, c_sa = simulated_annealing(p_cur, time.time() + sa_time)
        
        # Quick local search polish
        ls_time = min(phase1_end - time.time() - 0.5, 5.0)
        if ls_time > 0.5:
            p_ls, c_ls = local_search_first_improvement(p_sa, time.time() + ls_time)
            if c_ls < c_sa:
                p_sa, c_sa = p_ls, c_ls
        
        if c_sa < best_cost_overall:
            best_cost_overall = c_sa
            best_p_overall = p_sa
    
    # Phase 2: Thorough local search on the best solution found
    if best_p_overall is not None and time.time() < deadline - 3:
        # First-improvement local search (fast, many passes)
        fi_time = min(deadline - time.time() - 5, 20.0)
        p_fi, c_fi = local_search_first_improvement(best_p_overall, time.time() + fi_time)
        if c_fi < best_cost_overall:
            best_cost_overall = c_fi
            best_p_overall = p_fi
        
        # Best-improvement local search (slower, more precise)
        if time.time() < deadline - 2:
            bi_time = min(deadline - time.time() - 1, 15.0)
            p_bi, c_bi = local_search_best_improvement(best_p_overall, time.time() + bi_time)
            if c_bi < best_cost_overall:
                best_cost_overall = c_bi
                best_p_overall = p_bi
        
        # Final first-improvement pass with remaining time
        if time.time() < deadline - 1:
            p_final, c_final = local_search_first_improvement(best_p_overall, deadline - 1)
            if c_final < best_cost_overall:
                best_cost_overall = c_final
                best_p_overall = p_final
    
    return best_p_overall.tolist()
