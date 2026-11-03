"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    # Ensure int64 for exact arithmetic if possible, but float64 is safer for large sums if overflow is a concern.
    # However, the task says "exact integer cost". QAPLIB values are usually small enough for int64.
    # Let's stick to int64 for the cost calculation to ensure exactness, but use float64 for SA probabilities.
    a_int = a.astype(np.int64)
    b_int = b.astype(np.int64)
    
    # 1. Initialization: Rank matching heuristic
    a_sum = a_int.sum(axis=1)
    b_sum = b_int.sum(axis=1)
    heavy_order = np.argsort(-a_sum, kind="stable")
    central_order = np.argsort(b_sum, kind="stable")
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy_order] = central_order
    
    # Helper to calculate cost of a permutation
    def calc_cost(p):
        # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # Vectorized: b_perm = b[p][:, p]
        b_perm = b_int[p][:, p]
        return int(np.sum(a_int * b_perm))
        
    # Helper to calculate delta cost for swapping positions i and j in permutation p
    # p[i] and p[j] are swapped.
    # Let u = p[i], v = p[j].
    # New assignment: i->v, j->u.
    # Old assignment: i->u, j->v.
    # Delta = sum_k (a[i,k] * (b[v, p[k]] - b[u, p[k]]) + a[j,k] * (b[u, p[k]] - b[v, p[k]])) 
    #       + sum_k (a[k,i] * (b[p[k], v] - b[p[k], u]) + a[k,j] * (b[p[k], u] - b[p[k], v]))
    #       - a[i,j]*(b[u,v] + b[v,u]) + a[i,j]*(b[v,u] + b[u,v]) -- wait, diagonal terms i,i and j,j and i,j,j,i
    # Let's derive carefully.
    # C(p) = sum_{x,y} a[x,y] b[p[x], p[y]]
    # Only terms involving indices i or j in x or y change.
    # Terms: (i,i), (i,j), (j,i), (j,j) and (i,k), (j,k), (k,i), (k,j) for k != i,j.
    
    def delta_swap(p, i, j):
        if i == j:
            return 0
        u = p[i]
        v = p[j]
        if u == v: # Should not happen in a permutation
            return 0
            
        # We need sum over k != i, j
        # Let S1[k] = b[v, p[k]] - b[u, p[k]]
        # Let S2[k] = b[p[k], v] - b[p[k], u]
        
        # Create masks to exclude i and j
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        
        # S1 vector: b[v, p[mask]] - b[u, p[mask]]
        # S2 vector: b[p[mask], v] - b[p[mask], u]
        
        # Actually, simpler:
        # Delta = 
        # 1. Terms where x=i or y=i
        # 2. Terms where x=j or y=j
        
        # Let's just compute the change directly using vectorization on the full vectors and subtract the i,j parts.
        
        # Old contribution of i to the cost:
        # sum_k a[i,k] b[u, p[k]] + sum_k a[k,i] b[p[k], u] - a[i,i] b[u,u] (double counted in sums if k=i)
        # This is getting messy with indices. 
        # Let's use the formula:
        # Delta = 
        #   sum_{k != i,j} a[i,k] (b[v, p[k]] - b[u, p[k]]) 
        # + sum_{k != i,j} a[j,k] (b[u, p[k]] - b[v, p[k]])
        # + sum_{k != i,j} a[k,i] (b[p[k], v] - b[p[k], u])
        # + sum_{k != i,j} a[k,j] (b[p[k], u] - b[p[k], v])
        # + (a[i,i] (b[v,v] - b[u,u]))
        # + (a[j,j] (b[u,u] - b[v,v]))
        # + (a[i,j] (b[v,u] - b[u,v]))
        # + (a[j,i] (b[u,v] - b[v,u]))
        
        # Note: b is symmetric in distance matrices usually, but not guaranteed.
        
        # Vectorized calculation for the sums over k != i,j:
        # Let P = p[mask]
        # Let A_i_k = a[i, mask]
        # Let A_j_k = a[j, mask]
        # Let A_k_i = a[mask, i]
        # Let A_k_j = a[mask, j]
        
        # Term 1: sum A_i_k * (b[v, P] - b[u, P])
        # Term 2: sum A_j_k * (b[u, P] - b[v, P])
        # Term 3: sum A_k_i * (b[P, v] - b[P, u])
        # Term 4: sum A_k_j * (b[P, u] - b[P, v])
        
        # Combine:
        # (A_i_k - A_j_k) * (b[v, P] - b[u, P]) + (A_k_i - A_k_j) * (b[P, v] - b[P, u])
        
        diff_b_row = b_int[v, P] - b_int[u, P]
        diff_b_col = b_int[P, v] - b_int[P, u]
        
        coeff_row = a_int[i, mask] - a_int[j, mask]
        coeff_col = a_int[mask, i] - a_int[mask, j]
        
        sum_k = np.dot(coeff_row, diff_b_row) + np.dot(coeff_col, diff_b_col)
        
        # Diagonal and cross terms
        term_ii = a_int[i, i] * (b_int[v, v] - b_int[u, u])
        term_jj = a_int[j, j] * (b_int[u, u] - b_int[v, v])
        term_ij = a_int[i, j] * (b_int[v, u] - b_int[u, v])
        term_ji = a_int[j, i] * (b_int[u, v] - b_int[v, u])
        
        return int(sum_k + term_ii + term_jj + term_ij + term_ji)

    # Parallel Tempering Setup
    num_chains = 10
    # Temperatures: geometric progression
    # T_high should be high enough to accept most moves, T_low close to 0.
    # Initial cost estimate
    init_cost = calc_cost(p_init)
    
    # Heuristic for T_max: average absolute value of delta? 
    # Let's just use a range.
    T_min = 1.0
    T_max = 10000.0
    # Or adaptively: T_max = max(|delta|) / 0.1 ?
    # Let's stick to fixed for now, or calculate based on initial cost magnitude.
    # Cost is ~8e6. Delta can be large.
    # Let's use log spacing.
    temperatures = np.geomspace(T_min, T_max, num_chains)
    
    # Initialize chains
    # Chain 0 is the best one so far (colder)
    # Others are randomized versions of the best
    chains_p = [p_init.copy() for _ in range(num_chains)]
    chains_cost = [init_cost for _ in range(num_chains)]
    
    # Randomize other chains
    for c in range(1, num_chains):
        p_c = p_init.copy()
        random.shuffle(p_c.tolist())
        chains_p[c] = p_c
        chains_cost[c] = calc_cost(p_c)
        
    # Ensure chain 0 is the best
    best_idx = np.argmin(chains_cost)
    if best_idx != 0:
        chains_p[0], chains_p[best_idx] = chains_p[best_idx], chains_p[0]
        chains_cost[0], chains_cost[best_idx] = chains_cost[best_idx], chains_cost[0]
        
    global_best_p = chains_p[0].copy()
    global_best_cost = chains_cost[0]
    
    # Swap acceptance history for temperature adaptation (optional, keep simple first)
    
    def do_move(chain_idx, T):
        p = chains_p[chain_idx]
        cost = chains_cost[chain_idx]
        
        # Random 2-swap
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j:
            return False
            
        delta = delta_swap(p, i, j)
        
        if delta <= 0:
            # Accept
            p[i], p[j] = p[j], p[i]
            cost += delta
            return True
        else:
            # Metropolis criterion
            # P = exp(-delta / T)
            # To avoid underflow, compare log(P) with log(random)
            # -delta / T > log(rand)
            # delta / T < -log(rand)
            # delta < T * (-log(rand))
            # rand is in (0, 1], -log(rand) is in [0, inf)
            # Let's use:
            if random.random() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                cost += delta
                return True
            return False

    # Main Loop
    while time.time() < deadline:
        # 1. Perform moves in all chains
        for c in range(num_chains):
            # Multiple moves per chain per iteration?
            # 1 move is cheap. Let's do a few.
            for _ in range(5):
                if time.time() > deadline:
                    break
                do_move(c, temperatures[c])
                
        # 2. Try swaps between adjacent temperature chains
        # Iterate from cold to hot or hot to cold? 
        # Standard: iterate from 0 to N-2
        for c in range(num_chains - 1):
            if time.time() > deadline:
                break
            T_c = temperatures[c]
            T_next = temperatures[c+1]
            
            cost_c = chains_cost[c]
            cost_next = chains_cost[c+1]
            
            # Delta E = E_next - E_c (if we swap, chain c gets next's state, next gets c's state)
            # Wait, if we swap states, the energy of chain c becomes E_next, and chain next becomes E_c.
            # The change in energy for the system is 0.
            # But we need to check the Metropolis criterion for the swap:
            # P = min(1, exp(-(E_c_new - E_c_old + E_next_new - E_next_old) / T_swap))
            # E_c_new = E_next_old, E_c_old = E_c_old
            # E_next_new = E_c_old, E_next_old = E_next_old
            # Delta = (E_next_old - E_c_old) + (E_c_old - E_next_old) = 0? 
            # No, the criterion is based on the difference in energies at their respective temperatures.
            # Delta E_total = (E_next - E_c)/T_c + (E_c - E_next)/T_next
            # = (E_next - E_c) * (1/T_c - 1/T_next)
            # Let dE = E_next - E_c
            # If dE * (1/T_c - 1/T_next) <= 0, accept.
            # Since T_c < T_next, 1/T_c > 1/T_next, so (1/T_c - 1/T_next) > 0.
            # So if dE <= 0 (next is better), accept.
            # If dE > 0 (next is worse), accept with prob exp(-dE * (1/T_c - 1/T_next))
            
            dE = chains_cost[c+1] - chains_cost[c]
            factor = (1.0 / T_c) - (1.0 / T_next)
            
            if dE <= 0:
                # Swap
                chains_p[c], chains_p[c+1] = chains_p[c+1], chains_p[c]
                chains_cost[c], chains_cost[c+1] = chains_cost[c+1], chains_cost[c]
            else:
                # Calculate acceptance prob
                # Avoid overflow in exp if factor is large? factor is positive.
                # dE * factor could be large.
                # exp(-large) -> 0
                if dE * factor < 700: # exp(-700) is ~1e-304, safe
                    p_acc = np.exp(-dE * factor)
                else:
                    p_acc = 0.0
                    
                if random.random() < p_acc:
                    chains_p[c], chains_p[c+1] = chains_p[c+1], chains_p[c]
                    chains_cost[c], chains_cost[c+1] = chains_cost[c+1], chains_cost[c]
                    
        # 3. Update global best
        # Check if any chain improved global best
        # Optimization: only check if we suspect improvement, or just check all?
        # Checking all is O(N) cost calc if we don't cache? No, cost is cached in chains_cost.
        # So just compare min(chains_cost) with global_best_cost.
        current_min_cost = min(chains_cost)
        if current_min_cost < global_best_cost:
            global_best_cost = current_min_cost
            best_chain_idx = chains_cost.index(current_min_cost)
            global_best_p = chains_p[best_chain_idx].copy()
            
    # Final Polish: Run simple SA on the best solution found to squeeze out more
    # Use a very low temperature
    p_polish = global_best_p.copy()
    cost_polish = global_best_cost
    T_polish = 1.0
    
    # Short SA loop
    polish_deadline = time.time() + 2.0
    while time.time() < polish_deadline:
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j: continue
        delta = delta_swap(p_polish, i, j)
        if delta <= 0 or random.random() < np.exp(-delta / T_polish):
            p_polish[i], p_polish[j] = p_polish[j], p_polish[i]
            cost_polish += delta
            if cost_polish < global_best_cost:
                global_best_cost = cost_polish
                global_best_p = p_polish.copy()

    return global_best_p.tolist()
# EVOLVE-BLOCK-END
