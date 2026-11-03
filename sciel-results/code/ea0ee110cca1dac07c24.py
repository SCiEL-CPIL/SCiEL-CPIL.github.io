"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    # Safety margin: 150s limit, aim to finish by 145s
    deadline = time.time() + 145.0

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def local_search(p, current_cost, deadline):
        """
        Fast local search with O(n^2) incremental update of interaction vectors.
        Maintains dot_row[i] = sum_k a[i,k]*b[p[i],p[k]] and dot_col[i] = sum_k a[k,i]*b[p[k],p[i]]
        """
        dot_row = np.zeros(n, dtype=np.float64)
        dot_col = np.zeros(n, dtype=np.float64)
        
        for i in range(n):
            dot_row[i] = np.dot(a[i, :], b[p[i], p])
            dot_col[i] = np.dot(a[:, i], b[p, p[i]])
        
        improved = True
        max_iter = 5000
        iter_count = 0
        time_check_interval = 50
        
        while improved and time.time() < deadline - 3.0 and iter_count < max_iter:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            
            for i in range(n - 1):
                if iter_count % time_check_interval == 0 and time.time() >= deadline - 3.0:
                    return p, current_cost
                
                ai = a[i, :]
                ai_col = a[:, i]
                
                for j in range(i + 1, n):
                    u = p[i]
                    v = p[j]
                    
                    diff_a_rows = ai - a[j, :]
                    diff_b_rows = b[v, p] - b[u, p]
                    s1 = np.dot(diff_a_rows, diff_b_rows)
                    
                    diff_a_cols = ai_col - a[:, j]
                    diff_b_cols = b[p, v] - b[p, u]
                    s2 = np.dot(diff_a_cols, diff_b_cols)
                    
                    ai_i = a[i, i]
                    aj_i = a[j, i]
                    ai_j = a[i, j]
                    aj_j = a[j, j]
                    bu_u = b[u, u]
                    bu_v = b[u, v]
                    bv_u = b[v, u]
                    bv_v = b[v, v]
                    
                    term_ii_s1 = (ai_i - aj_i) * (bv_u - bu_u)
                    term_ii_s2 = (ai_i - aj_i) * (bu_v - bu_u)
                    term_jj_s1 = (ai_j - aj_j) * (bv_v - bu_v)
                    term_jj_s2 = (ai_j - aj_j) * (bu_v - bv_v)
                    interaction_delta = (ai_j - aj_i) * (bv_u - bv_v)
                    
                    d = s1 + s2 - term_ii_s1 - term_ii_s2 - term_jj_s1 - term_jj_s2 + interaction_delta
                    
                    if d < best_delta:
                        best_delta = d
                        best_i = i
                        best_j = j
            
            if best_delta < 0:
                bi = best_i
                bj = best_j
                u = p[bi]
                v = p[bj]
                
                p[bi], p[bj] = v, u
                current_cost += best_delta
                improved = True
                iter_count += 1
                
                mask = np.ones(n, dtype=bool)
                mask[bi] = False
                mask[bj] = False
                ks = np.where(mask)[0]
                
                if len(ks) > 0:
                    b_p_k_v = b[p[ks], v]
                    b_p_k_u = b[p[ks], u]
                    diff_a = a[ks, bi] - a[ks, bj]
                    dot_row[ks] += diff_a * (b_p_k_v - b_p_k_u)
                
                dot_row[bi] = np.dot(a[bi, :], b[v, p])
                dot_row[bj] = np.dot(a[bj, :], b[u, p])
                
                if len(ks) > 0:
                    b_v_pk = b[v, p[ks]]
                    b_u_pk = b[u, p[ks]]
                    diff_a = a[bi, ks] - a[bj, ks]
                    dot_col[ks] += diff_a * (b_v_pk - b_u_pk)
                
                dot_col[bi] = np.dot(a[:, bi], b[p, v])
                dot_col[bj] = np.dot(a[:, bj], b[p, u])
                
            else:
                break
                
        return p, current_cost

    def greedy_init():
        p = np.full(n, -1, dtype=np.int64)
        flow_sums = a.sum(axis=1)
        order = np.argsort(-flow_sums, kind='stable')
        
        assigned = np.zeros(n, dtype=bool)
        assigned_count = 0
        
        for idx in order:
            if assigned_count == 0:
                p[idx] = 0
                assigned[0] = True
                assigned_count += 1
                continue
            
            assigned_indices = np.where(assigned)[0]
            b_loc_p_assigned = b[:, p[assigned_indices]]
            inter_vec = a[idx, :] + a[:, idx]
            inter_vec_assigned = inter_vec[assigned_indices]
            
            unassigned_locs = np.where(~assigned)[0]
            if len(unassigned_locs) == 0:
                break
            
            incs = b_loc_p_assigned[unassigned_locs] @ inter_vec_assigned
            best_idx_in_unassigned = np.argmin(incs)
            best_loc = unassigned_locs[best_idx_in_unassigned]
            
            p[idx] = best_loc
            assigned[best_loc] = True
            assigned_count += 1
            
        return p

    # 1. Initialization
    best_cost = np.inf
    best_p = None
    
    inits = []
    inits.append(greedy_init())
    
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    inits.append(p_rank)
    
    p_rank_rev = np.empty(n, dtype=np.int64)
    p_rank_rev[heavy] = central[::-1]
    inits.append(p_rank_rev)
    
    rng = np.random.RandomState(42)
    inits.append(rng.permutation(n))
    
    for p_init in inits:
        if time.time() > deadline - 15.0:
            break
        
        p_work = p_init.copy()
        cost = compute_cost(p_work)
        
        p_work, cost = local_search(p_work, cost, deadline)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p_work.copy()
    
    if best_p is None:
        best_p = inits[0].copy()
        best_cost = compute_cost(best_p)
    
    # 2. Parallel Tempering with local search
    # Use multiple temperatures to escape local minima
    no_improve_count = 0
    rng = np.random.RandomState(123)
    
    # Temperature schedule
    T_start = max(1.0, best_cost * 0.01)
    T_end = 0.01
    
    # Run ILS with adaptive perturbation
    while time.time() < deadline - 5.0:
        p = best_p.copy()
        
        # Adaptive perturbation: more swaps when stuck
        base_swaps = 3
        extra_swaps = min(no_improve_count, 15)
        num_swaps = base_swaps + extra_swaps
        num_swaps = min(num_swaps, 20)
        
        for _ in range(num_swaps):
            i = rng.randint(0, n)
            j = rng.randint(0, n)
            if i != j:
                p[i], p[j] = p[j], p[i]
        
        current_cost = compute_cost(p)
        
        # Run local search from the perturbed state
        p, current_cost = local_search(p, current_cost, deadline)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > 30:
            no_improve_count = 0

    # Final verification
    if best_p is not None:
        final_cost = compute_cost(best_p)
        if final_cost < best_cost:
            best_cost = final_cost
        return best_p.tolist()
    
    return list(range(n))

# EVOLVE-BLOCK-END
