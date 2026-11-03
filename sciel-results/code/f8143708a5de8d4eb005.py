"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    # 150s limit, leave 5s safety margin
    time_limit = 145.0 

    # Ensure float64 for speed in intermediate calculations, 
    # but keep track of integer cost if possible. 
    # For n=150, int64 dot products are fast enough but float64 is safer for 
    # avoiding overflow in intermediate sums if values are large (though QAPLIB is usually safe).
    # Let's stick to float64 for vectorization speed, casting back at the end if needed for exactness.
    # The problem asks for exact integer cost, but the evaluator compares raw value.
    # NumPy float64 has 53 bits of precision. 
    # Max value in QAPLIB tho150: 
    # Flows ~ 10^3, Distances ~ 10^3. Product ~ 10^6. Sum of 150*150=22500 terms -> ~2.2 * 10^10.
    # 2^53 ~ 9 * 10^15. So float64 is exact for this magnitude.
    
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    def compute_cost(p):
        # p is a permutation array of indices 0..n-1
        # cost = sum a[i,j] * b[p[i], p[j]]
        # Vectorized: np.sum(a_f * b_f[p][:, p])
        # b_f[p] creates a matrix where row i is b[p[i]]
        # b_f[p][:, p] creates a matrix where element [i,j] is b[p[i], p[j]]
        return np.sum(a_f * b_f[p][:, p])

    def greedy_construct(seed_rng):
        """
        Construct a permutation by greedily adding facilities one by one.
        At each step, pick the facility that minimizes the incremental cost.
        """
        unassigned = set(range(n))
        p = np.empty(n, dtype=np.int64)
        # We will build p in a random order of positions? No, standard QAP construction 
        # usually fixes the order of facilities or locations.
        # Let's fix the order of facilities as 0..n-1 for simplicity, or randomize?
        # Randomizing the order of facility assignment can lead to diverse solutions.
        
        # Let's use a random order of facilities to assign
        order = list(range(n))
        seed_rng.shuffle(order)
        
        # Keep track of assigned facilities and their locations
        # assigned_facilities: list of facility indices
        # assigned_locations: list of location indices
        assigned_facs = []
        assigned_locs = []
        
        # Incremental cost for adding facility k at location l:
        # Delta = sum_{j in assigned_facs} a[k, j] * b[l, p[j]] + a[j, k] * b[p[j], l]
        # Since a and b are symmetric in QAP, a[k,j] = a[j,k] and b[l, p[j]] = b[p[j], l].
        # So Delta = 2 * sum_{j in assigned_facs} a[k, j] * b[l, p[j]]
        
        # To speed this up, we can maintain a "current cost contribution" vector.
        # Let C[l] be the cost contribution if we placed a new facility at location l.
        # But the contribution depends on WHICH facility is being placed.
        
        # Alternative view:
        # When we place facility k at location l, the cost added is:
        # sum_{j already placed} a[k,j] b[l, loc_j] + sum_{j already placed} a[j,k] b[loc_j, l]
        # = 2 * sum_{j already placed} a[k,j] b[l, loc_j]
        
        # We can precompute for each candidate location l, the vector of distances to already placed locations.
        # But we have to do this for each new facility k.
        
        # Efficient Greedy:
        # 1. Start with empty assignment.
        # 2. For each step t from 0 to n-1:
        #    a. Pick next facility k from remaining (based on order).
        #    b. For each available location l, calculate incremental cost.
        #    c. Pick l with min incremental cost.
        #    d. Update state.
        
        # This is O(n^3) which is 150^3 = 3.3M operations. Very fast.
        
        remaining_facs = set(order)
        remaining_locs = set(range(n))
        
        # To make it faster, we can precompute a[k, j] for all k,j.
        
        for step in range(n):
            k = order[step]
            best_loc = -1
            best_cost = np.inf
            
            # Get indices of already assigned locations
            if step > 0:
                prev_locs = np.array(assigned_locs, dtype=np.int64)
                prev_facs = np.array(assigned_facs, dtype=np.int64)
                
                # We need sum_{j in prev} a[k, j] * b[l, prev_locs[j]]
                # Let A_k = a[k, prev_facs] (vector of length step)
                # For a given l, term is dot(A_k, b[l, prev_locs])
                
                # Precompute B_prev = b[:, prev_locs] (shape n x step)
                # Then for all l, cost[l] = dot(a[k, prev_facs], B_prev[l, :])
                
                # Actually, let's just loop over available locations? 
                # n is 150. Looping 150 times, each doing a dot product of length 150.
                # 150 * 150 * 150 = 3.3M flops. Totally fine.
                
                A_k_prev = a_f[k, prev_facs]
                # B_prev[l] is b[l, prev_locs]
                # We want min over l in remaining_locs of dot(A_k_prev, b_f[l, prev_locs])
                
                # Vectorize over remaining locations
                rem_locs_list = list(remaining_locs)
                rem_locs_arr = np.array(rem_locs_list, dtype=np.int64)
                
                # Matrix M where M[i, j] = b[rem_locs_arr[i], prev_locs[j]]
                M = b_f[rem_locs_arr][:, prev_locs]
                
                # Costs for each remaining location
                costs = M @ A_k_prev
                
                # Find min
                min_idx = np.argmin(costs)
                best_loc = rem_locs_arr[min_idx]
                best_cost = costs[min_idx]
            else:
                # First facility, cost is 0 regardless of location. Pick random.
                rem_locs_list = list(remaining_locs)
                best_loc = rem_locs_list[0]
                best_cost = 0.0

            # Assign k to best_loc
            p[k] = best_loc
            assigned_facs.append(k)
            assigned_locs.append(best_loc)
            remaining_locs.remove(best_loc)

        return p

    def local_search_2_opt(p, cost, rng):
        """
        Standard 2-opt swap local search.
        """
        n = len(p)
        improved = True
        while improved:
            improved = False
            # Random order of pairs
            pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            rng.shuffle(pairs)
            
            for i, j in pairs:
                if time.time() - start_time > time_limit:
                    return p, cost, True # timeout
                
                u = p[i]
                v = p[j]
                
                # Delta for swapping i and j
                # cost = sum a[x,y] b[p[x], p[y]]
                # Swap affects terms involving i or j.
                # Terms:
                # 1. a[i,j] b[u,v] becomes a[i,j] b[v,u] (same if symmetric)
                # 2. a[i,k] b[u, p[k]] for k!=i,j becomes a[i,k] b[v, p[k])
                # 3. a[k,i] b[p[k], u] for k!=i,j becomes a[k,i] b[p[k], v)
                # 4. a[j,k] b[v, p[k]) for k!=i,j becomes a[j,k] b[u, p[k])
                # 5. a[k,j] b[p[k], v) for k!=i,j becomes a[k,j] b[p[k], u)
                
                # Assuming symmetry:
                # Delta = a[i,j](b[v,u]-b[u,v]) + sum_{k!=i,j} [ a[i,k](b[v,p[k]]-b[u,p[k]]) + a[k,i](b[p[k],v]-b[p[k],u]) + ... ]
                # = 2 * a[i,j] * 0 + 2 * sum_{k!=i,j} (a[i,k] - a[j,k]) * (b[v,p[k]] - b[u,p[k]])
                # Wait, let's derive carefully.
                # C_old = a[i,j]b[u,v] + sum_{k!=i,j} (a[i,k]b[u,p[k]] + a[k,i]b[p[k],u] + a[j,k]b[v,p[k]] + a[k,j]b[p[k],v])
                # C_new = a[i,j]b[v,u] + sum_{k!=i,j} (a[i,k]b[v,p[k]] + a[k,i]b[p[k],v] + a[j,k]b[u,p[k]] + a[k,j]b[p[k],u])
                
                # Delta = C_new - C_old
                # = a[i,j](b[v,u]-b[u,v]) + sum_{k!=i,j} [ a[i,k](b[v,p[k]]-b[u,p[k]]) + a[k,i](b[p[k],v]-b[p[k],u]) + a[j,k](b[u,p[k]]-b[v,p[k]]) + a[k,j](b[p[k],u]-b[p[k],v]) ]
                
                # If symmetric b[v,u]=b[u,v], first term 0.
                # If symmetric a, a[i,k]=a[k,i], a[j,k]=a[k,j].
                # Term for k: a[i,k](b[v,p[k]]-b[u,p[k]]) + a[i,k](b[p[k],v]-b[p[k],u]) + a[j,k](b[u,p[k]]-b[v,p[k]]) + a[j,k](b[p[k],u]-b[p[k],v])
                # Note b[p[k],v] = b[v,p[k]] and b[p[k],u] = b[u,p[k]].
                # Term for k: a[i,k](b[v,p[k]]-b[u,p[k]]) + a[i,k](b[v,p[k]]-b[u,p[k]]) - a[j,k](b[v,p[k]]-b[u,p[k]]) - a[j,k](b[v,p[k]]-b[u,p[k]])
                # = 2 * (a[i,k] - a[j,k]) * (b[v,p[k]] - b[u,p[k]])
                
                # So Delta = 2 * sum_{k!=i,j} (a[i,k] - a[j,k]) * (b[v,p[k]] - b[u,p[k]])
                
                # Vectorized calculation:
                # diff_a = a[i] - a[j]
                # diff_b = b[v] - b[u]
                # We need sum over k != i,j of diff_a[k] * diff_b[p[k]]
                # sum_all = dot(diff_a, diff_b[p])
                # subtract terms for k=i and k=j
                # term_i = diff_a[i] * diff_b[p[i]] = (a[i,i]-a[j,i]) * (b[v,u]-b[u,u])
                # term_j = diff_a[j] * diff_b[p[j]] = (a[i,j]-a[j,j]) * (b[v,v]-b[u,v])
                
                diff_a = a_f[i] - a_f[j]
                diff_b = b_f[v] - b_f[u]
                
                # b[p] is the matrix with rows permuted by p
                # diff_b[p] is the matrix with rows permuted by p? No.
                # We want vector d_k = diff_b[p[k]]
                # d = diff_b[p]
                d = diff_b[p]
                
                sum_all = np.dot(diff_a, d)
                
                # Subtract k=i: diff_a[i] * d[i] = diff_a[i] * diff_b[p[i]] = diff_a[i] * diff_b[u]
                # Subtract k=j: diff_a[j] * d[j] = diff_a[j] * diff_b[p[j]] = diff_a[j] * diff_b[v]
                
                term_i = diff_a[i] * diff_b[u]
                term_j = diff_a[j] * diff_b[v]
                
                delta = 2.0 * (sum_all - term_i - term_j)
                
                if delta < -1e-9: # Small epsilon for float errors
                    # Perform swap
                    p[i], p[j] = v, u
                    cost += delta
                    improved = True
                    # Don't break, continue scanning? 
                    # Standard steepest descent or first improvement?
                    # First improvement is faster per pass but might take more passes.
                    # Let's do first improvement to keep moving.
                    break 
        
        return p, cost, False

    def local_search_insertion(p, cost, rng):
        """
        Best Insertion Local Search.
        Pick a facility, remove it, and insert it at the best position.
        This allows for larger moves than 2-opt.
        """
        n = len(p)
        # We will iterate through facilities in random order
        # For each facility, try removing and re-inserting at all other positions
        # This is O(n^3) per full pass. 150^3 = 3.3M. 
        # With 120s, we can do many passes.
        
        # To make it efficient, we need to calculate the delta for removing i and inserting at k.
        
        # Let's stick to 2-opt for now as it's proven and fast, but add a "3-opt" or "block move" later if time permits?
        # Actually, the prompt asks for a MATERIAL change.
        # Greedy Construction is the material change from Rank Matching.
        # 2-opt is standard.
        # Let's add a "Great Deluge" or "Simulated Annealing" element?
        # No, let's stick to Greedy + 2-opt but with MULTIPLE RESTARTS using different random seeds for the greedy phase.
        
        # The current seed program does rank matching + 2-opt + restarts.
        # My change: Greedy Construction + 2-opt.
        # Is that enough? 
        # Greedy construction for QAP is often better than rank matching.
        
        # Let's enhance the local search slightly.
        # Use "Vectorized 2-Opt" with early termination.
        
        return p, cost

    best_p = None
    best_cost = np.inf
    
    # Multiple restarts
    # Each restart: Greedy Construct + Local Search
    # Time budget: 145s
    # Estimate time per restart:
    # Greedy: ~0.01s
    # Local Search: depends on convergence. Say 0.1s - 1s.
    # We can do 100-1000 restarts.
    
    rng = random.Random(42)
    
    while True:
        if time.time() - start_time > time_limit - 5.0: # Leave 5s for final check
            break
        
        # 1. Construct
        p = greedy_construct(rng)
        cost = compute_cost(p)
        
        # 2. Local Search
        p, cost, timed_out = local_search_2_opt(p, cost, rng)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            
        # Optional: Perturb and continue? 
        # If we just restart, we lose the local search progress.
        # But greedy construction is cheap.
        # Let's try a few iterations of local search on the same solution before restarting?
        # The current code does local search until no improvement, then perturbs.
        # My code does local search until no improvement, then RESTARTS (new greedy).
        # This is effectively a multi-start local search.
        
        # To improve diversity, we can vary the greedy strategy.
        # e.g., sometimes pick 2nd best location?
        # Let's keep it simple first.

    # Final verification
    if best_p is None:
        # Fallback
        best_p = np.arange(n)
        best_cost = compute_cost(best_p)

    # Convert to list of ints
    return best_p.tolist()

# EVOLVE-BLOCK-END
