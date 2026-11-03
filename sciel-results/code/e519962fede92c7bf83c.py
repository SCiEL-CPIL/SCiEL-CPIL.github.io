# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    def calc_cost_full(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))

    def two_opt_delta(p, i, j):
        """Compute delta cost for swapping facilities i and j in permutation p."""
        li = p[i]
        lj = p[j]
        if li == lj:
            return 0
        d = b[lj, p] - b[li, p]
        d[i] = 0
        d[j] = 0
        a_diff = a[i, :] - a[j, :] + a[:, i] - a[:, j]
        a_diff[i] = 0
        a_diff[j] = 0
        return int(np.dot(a_diff, d))

    def two_opt_polish(p, max_time=None):
        """Greedy 2-opt local search. Returns (p, cost)."""
        cost = calc_cost_full(p)
        if max_time is None:
            max_time = deadline
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(n):
                if time.time() > max_time:
                    break
                for j in range(i + 1, n):
                    if time.time() > max_time:
                        break
                    delta = two_opt_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, cost

    # --- Phase 1: Initial Solution via Rank Matching ---
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy_order = np.argsort(-a_row_sums, kind='stable')
    central_order = np.argsort(b_row_sums, kind='stable')
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    p = p.copy()

    # Initial polish
    p, current_cost = two_opt_polish(p, max_time=start_time + 5.0)
    best_p = p.copy()
    best_cost = current_cost
    
    # Try a few random starts to get a better baseline
    for _ in range(3):
        if time.time() > start_time + 10.0:
            break
        p_rand = np.random.permutation(n).astype(np.int64)
        p_rand, cost_rand = two_opt_polish(p_rand, max_time=start_time + 12.0)
        if cost_rand < best_cost:
            best_cost = cost_rand
            best_p = p_rand.copy()

    # --- Phase 2: Simulated Annealing ---
    # SA parameters
    T_start = abs(best_cost) * 0.01 if best_cost != 0 else 1000.0
    T_end = 1.0
    cooling_rate = 0.995
    max_sa_time = deadline - 10.0
    
    p_sa = best_p.copy()
    cost_sa = best_cost
    T = T_start
    
    # Precompute random indices for speed? No, just use np.random
    while time.time() < max_sa_time:
        # Generate a random swap
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        delta = two_opt_delta(p_sa, i, j)
        
        # Metropolis criterion
        if delta < 0:
            # Accept better solution
            p_sa[i], p_sa[j] = p_sa[j], p_sa[i]
            cost_sa += delta
            if cost_sa < best_cost:
                best_cost = cost_sa
                best_p = p_sa.copy()
        else:
            # Accept worse solution with probability exp(-delta / T)
            if T > 0:
                # Avoid overflow in exp
                if delta / T < 700: 
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        p_sa[i], p_sa[j] = p_sa[j], p_sa[i]
                        cost_sa += delta
        T *= cooling_rate
        
        # Periodically polish to remove local noise
        if int((max_sa_time - time.time()) * 10) % 50 == 0 and time.time() > start_time + 20:
             p_sa, cost_sa = two_opt_polish(p_sa, max_time=min(time.time() + 1.0, max_sa_time))
             if cost_sa < best_cost:
                 best_cost = cost_sa
                 best_p = p_sa.copy()

    # --- Phase 3: Final Polish ---
    p_polish = best_p.copy()
    p_polish, final_cost = two_opt_polish(p_polish, max_time=deadline - 2.0)
    if final_cost < best_cost:
        best_p = p_polish
        best_cost = final_cost

    return best_p.tolist()

# EVOLVE-BLOCK-END
