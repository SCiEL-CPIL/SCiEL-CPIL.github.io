import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _compute_deltas_for_i(a, b, p, i):
    """
    Compute deltas for swapping position i with all j > i.
    Returns an array of deltas for j in range(i+1, n).
    """
    n = len(p)
    u = p[i]
    
    # diff_a[k] = a[i, k] - a[k, i] ? No, a is symmetric.
    # Let's use the formula:
    # delta(i, j) = 2 * sum_{k != i, j} (a[i, k] - a[j, k]) * (b[p[i], p[j]] - b[p[j], p[i]]) ... wait, standard swap delta.
    
    # Let's stick to the previous logic but vectorize over j.
    # Previous logic:
    # diff_a_j = a[i, :] - a[j, :]
    # diff_b_j = b[p[i], :] - b[p[j], :]
    # sum_all_j = dot(diff_a_j, diff_b_j)
    # term_i_j = (a[i, i] - a[j, i]) * (b[p[i], p[i]] - b[p[j], p[i]])
    # term_j_j = (a[i, j] - a[j, j]) * (b[p[i], p[j]] - b[p[j], p[j]])
    # sum_k_neq_j = sum_all_j - term_i_j - term_j_j
    # self_term_j = (a[i, i] + a[j, j] - a[i, j] - a[j, i]) * (b[p[i], p[i]] - b[p[j], p[j]])
    # delta_j = 2 * sum_k_neq_j + self_term_j
    
    # This is hard to fully vectorize over j without advanced indexing because of the p[j] terms.
    # However, we can precompute some parts.
    
    # Let's just optimize the inner loop by avoiding function call overhead and using local variables.
    # Or, better yet, rewrite _tabu_search to be much more efficient or use a different heuristic if TS is too slow.
    # Given the time limit, a simple steepest descent or randomized local search might be safer and faster to implement correctly without complex vectorization bugs.
    
    # Let's try a simpler, robust local search: Randomized Swap Descent.
    pass

def _local_search(a, b, p, cost, time_limit, max_no_improve=500):
    """
    Perform randomized local search with swaps.
    """
    n = len(p)
    p = p.copy()
    current_cost = cost
    best_p = p.copy()
    best_cost = current_cost
    no_improve = 0
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Pick a random pair
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        # Compute delta
        u = p[i]
        v = p[j]
        
        # Vectorized delta calculation for single swap
        # delta = 2 * sum_{k != i, j} (a[i, k] - a[j, k]) * (b[u, p[k]] - b[v, p[k]]) ... 
        # Actually, let's use the explicit formula derived before:
        # delta = 2 * (sum_{k != i, j} (a[i, k] - a[j, k]) * (b[v, p[k]] - b[u, p[k]])) 
        #         + (a[i, i] + a[j, j] - a[i, j] - a[j, i]) * (b[v, v] - b[u, u])
        #         - 2 * (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])  # Wait, the k=i term in sum is (a[i,i]-a[j,i])*(b[v,u]-b[u,u])
        #         - 2 * (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])  # k=j term
        
        # Let's re-verify the "sum_all - terms" approach.
        # sum_all = dot(a[i, :] - a[j, :], b[v, :] - b[u, :])
        # This sums over all k.
        # Term k=i: (a[i, i] - a[j, i]) * (b[v, p[i]] - b[u, p[i]]) = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        # Term k=j: (a[i, j] - a[j, j]) * (b[v, p[j]] - b[u, p[j]]) = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        
        diff_a = a[i, :] - a[j, :]
        diff_b = b[v, :] - b[u, :]
        sum_all = np.dot(diff_a, diff_b)
        
        term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        
        sum_k_neq = sum_all - term_i - term_j
        
        self_term = (a[i, i] + a[j, j] - a[i, j] - a[j, i]) * (b[v, v] - b[u, u])
        
        delta = 2 * sum_k_neq + self_term
        delta = int(delta)
        
        if delta < 0:
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            no_improve = 0
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                # Perturb
                for _ in range(5):
                    i2 = np.random.randint(n)
                    j2 = np.random.randint(n)
                    if i2 != j2:
                        u2 = p[i2]
                        v2 = p[j2]
                        diff_a2 = a[i2, :] - a[j2, :]
                        diff_b2 = b[v2, :] - b[u2, :]
                        sum_all2 = np.dot(diff_a2, diff_b2)
                        term_i2 = (a[i2, i2] - a[j2, i2]) * (b[v2, u2] - b[u2, u2])
                        term_j2 = (a[i2, j2] - a[j2, j2]) * (b[v2, v2] - b[u2, v2])
                        sum_k_neq2 = sum_all2 - term_i2 - term_j2
                        self_term2 = (a[i2, i2] + a[j2, j2] - a[i2, j2] - a[j2, i2]) * (b[v2, v2] - b[u2, u2])
                        delta2 = int(2 * sum_k_neq2 + self_term2)
                        p[i2], p[j2] = p[j2], p[i2]
                        current_cost += delta2
                no_improve = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()

    return best_p, best_cost

def _greedy_initial(a, b):
    """
    Create an initial permutation using a greedy heuristic based on row sums.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow
    heavy = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by ascending centrality
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def solve(a, b):
    n = len(a)
    total_deadline = 115.0  # Safety margin for 120s limit
    start_time = time.time()
    
    # Phase 1: Initialization
    p_greedy = _greedy_initial(a, b)
    cost_greedy = _qap_cost(a, b, p_greedy)
    
    p_random = np.random.permutation(n).astype(np.int64)
    cost_random = _qap_cost(a, b, p_random)
    
    best_p = p_greedy.copy() if cost_greedy < cost_random else p_random.copy()
    best_cost = cost_greedy if cost_greedy < cost_random else cost_random
    
    # Phase 2: Local Search
    # Run local search for the majority of the time
    elapsed = time.time() - start_time
    ls_time = max(1.0, total_deadline - elapsed - 5.0)
    
    p_ls, cost_ls = _local_search(a, b, best_p, best_cost, ls_time)
    
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
        
    # Phase 3: Final Polish (Quick Steepest Descent)
    elapsed = time.time() - start_time
    polish_time = max(0.5, total_deadline - elapsed - 2.0)
    
    if polish_time > 0:
        p_polish = best_p.copy()
        cost_polish = best_cost
        improved = True
        polish_start = time.time()
        
        while improved and (time.time() - polish_start) < polish_time:
            improved = False
            # Check a subset of moves or all? All is O(n^2 * n) = O(n^3). 100^3 = 1e6 ops.
            # With numpy dot, it's fast. Let's do full steepest descent one pass.
            best_delta = 0
            best_i = -1
            best_j = -1
            
            for i in range(n):
                for j in range(i + 1, n):
                    u = p_polish[i]
                    v = p_polish[j]
                    
                    diff_a = a[i, :] - a[j, :]
                    diff_b = b[v, :] - b[u, :]
                    sum_all = np.dot(diff_a, diff_b)
                    
                    term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
                    term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                    
                    sum_k_neq = sum_all - term_i - term_j
                    self_term = (a[i, i] + a[j, j] - a[i, j] - a[j, i]) * (b[v, v] - b[u, u])
                    
                    delta = int(2 * sum_k_neq + self_term)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            
            if best_i != -1:
                p_polish[best_i], p_polish[best_j] = p_polish[best_j], p_polish[best_i]
                cost_polish += best_delta
                improved = True
                if cost_polish < best_cost:
                    best_cost = cost_polish
                    best_p = p_polish.copy()
            else:
                break
        
        if cost_polish < best_cost:
            best_cost = cost_polish
            best_p = p_polish.copy()

    # Ensure we return within time
    if time.time() - start_time > total_deadline:
        # Return best found so far
        pass

    return best_p.tolist()
