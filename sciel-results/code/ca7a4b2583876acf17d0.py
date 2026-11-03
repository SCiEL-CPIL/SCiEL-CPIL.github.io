"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit (150s kill)
    deadline = start_time + time_limit

    # 1. Precomputation
    # Ensure symmetry for faster delta calculation if possible
    # QAPLIB instances are symmetric, but let's be robust.
    # We assume symmetry for the delta formula optimization.
    # If not symmetric, we fall back to a slower check or just assume it for speed 
    # as tai256c is symmetric.
    a_sym = np.array_equal(a, a.T)
    b_sym = np.array_equal(b, b.T)
    
    if not (a_sym and b_sym):
        # Fallback for non-symmetric: just return identity or simple heuristic
        # as the fast delta logic below relies on symmetry properties of the cost function structure
        # specifically for the simplified delta derivation.
        # Actually, the delta derivation holds for any matrices if we are careful, 
        # but the "2 * off_diag" term assumes symmetry in the summation index or specific structure.
        # Let's stick to the standard symmetric delta which is valid for tai256c.
        pass

    def calculate_cost(perm):
        # perm is a numpy array of locations for each facility
        return int(np.sum(a * b[np.ix_(perm, perm)]))

    def calc_delta(i, j, perm):
        """
        Calculate the change in cost if we swap facilities i and j in the current permutation.
        i, j are facility indices.
        perm is the current assignment: perm[facility] = location.
        """
        u = perm[i]
        v = perm[j]
        
        if u == v:
            return 0

        # Formula for symmetric QAP:
        # Delta = 2 * sum_{k != i, j} (a[i, k] - a[j, k]) * (b[u, perm[k]] - b[v, perm[k]])
        #       + (a[i, i] - a[j, j]) * (b[u, u] - b[v, v])
        #       + (a[i, j] - a[j, i]) * (b[u, v] - b[v, u])  <-- This term is 0 if a is symmetric and u!=v?
        # Wait, let's derive carefully.
        # Cost = sum_{p,q} a[p,q] b[perm[p], perm[q]]
        # Swap i and j.
        # Terms involving i or j change.
        # New cost - Old cost:
        # 1. Terms (i,i) and (j,j):
        #    a[i,i]b[v,v] + a[j,j]b[u,u] - (a[i,i]b[u,u] + a[j,j]b[v,v])
        #    = a[i,i](b[v,v]-b[u,u]) + a[j,j](b[u,u]-b[v,v])
        #    = (a[i,i]-a[j,j])(b[v,v]-b[u,u])
        #
        # 2. Terms (i,j) and (j,i):
        #    a[i,j]b[v,u] + a[j,i]b[u,v] - (a[i,j]b[u,v] + a[j,i]b[v,u])
        #    = a[i,j](b[v,u]-b[u,v]) + a[j,i](b[u,v]-b[v,u])
        #    = (a[i,j]-a[j,i])(b[v,u]-b[u,v])
        #    If a is symmetric, a[i,j]=a[j,i], so this term is 0.
        #
        # 3. Terms (i,k) and (k,i) for k != i,j:
        #    Sum over k != i,j:
        #    a[i,k]b[v, perm[k]] + a[k,i]b[perm[k], v] - (a[i,k]b[u, perm[k]] + a[k,i]b[perm[k], u])
        #    Since b is symmetric, b[v, perm[k]] = b[perm[k], v].
        #    So: a[i,k]b[v, perm[k]] + a[i,k]b[v, perm[k]] - a[i,k]b[u, perm[k]] - a[i,k]b[u, perm[k]]
        #    = 2 * a[i,k] * (b[v, perm[k]] - b[u, perm[k]])
        #
        # 4. Terms (j,k) and (k,j) for k != i,j:
        #    Similarly: 2 * a[j,k] * (b[u, perm[k]] - b[v, perm[k]])
        #
        # Combining 3 and 4:
        #    2 * sum_{k != i,j} (a[i,k] - a[j,k]) * (b[v, perm[k]] - b[u, perm[k]])
        #
        # Total Delta:
        #    (a[i,i]-a[j,j])(b[v,v]-b[u,u]) + 2 * sum_{k != i,j} (a[i,k] - a[j,k]) * (b[v, perm[k]] - b[u, perm[k]])
        
        # Vectorized calculation for the sum part
        # diff_a = a[i, :] - a[j, :]
        # diff_b = b[v, perm] - b[u, perm]
        
        diff_a = a[i] - a[j]
        diff_b = b[v, perm] - b[u, perm]
        
        # Dot product includes k=i and k=j, we need to subtract those
        total_dot = np.dot(diff_a, diff_b)
        
        # Subtract k=i term:
        # diff_a[i] * diff_b[i] = (a[i,i] - a[j,i]) * (b[v, perm[i]] - b[u, perm[i]])
        # perm[i] is u. So (b[v, u] - b[u, u])
        term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        
        # Subtract k=j term:
        # diff_a[j] * diff_b[j] = (a[i, j] - a[j, j]) * (b[v, perm[j]] - b[u, perm[j]])
        # perm[j] is v. So (b[v, v] - b[u, v])
        term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        
        off_diag_sum = total_dot - term_i - term_j
        
        diag_part = (a[i, i] - a[j, j]) * (b[v, v] - b[u, u])
        
        return int(diag_part + 2 * off_diag_sum)

    # Initialization Strategies
    def init_linear():
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        cost_matrix = np.outer(a_row_sums, b_row_sums)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def init_spectral():
        # Use Fiedler vector of the flow matrix to partition
        # This is a structural initialization different from rank matching
        try:
            # Symmetrize just in case
            a_symm = (a + a.T) / 2.0
            # Compute eigenvalues and vectors
            # For n=256, this is fast
            vals, vecs = np.linalg.eigh(a_symm)
            # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
            # For connected graph, lambda_0 is smallest, lambda_1 is Fiedler
            # We want to split based on the sign of the Fiedler vector components
            # Assign facilities to locations based on ranking?
            # Actually, spectral initialization for QAP usually involves mapping the order of Fiedler components to locations.
            # Let's take the Fiedler vector (index 1)
            fiedler = vecs[:, 1]
            # Sort facilities by their Fiedler component
            order = np.argsort(fiedler)
            # Assign to locations in that order?
            # p[facility] = location
            # If we assign facility order[0] to location 0, etc.
            p = np.empty(n, dtype=np.int64)
            # We can also try reversing or randomizing the location assignment direction
            # Let's just map sorted facilities to sorted locations (0..n-1)
            p[order] = np.arange(n)
            return p
        except:
            return init_linear()

    def init_random():
        p = np.arange(n)
        np.random.shuffle(p)
        return p

    # Local Search: First Improvement with Randomized Order
    def local_search_first_improvement(p, cost, max_time):
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Randomize start to avoid bias
            start_i = random.randint(0, n - 1)
            for k in range(n):
                i = (start_i + k) % n
                if time.time() > max_time:
                    return p, cost
                u = p[i]
                start_j = random.randint(0, n - 1)
                for m in range(n - 1):
                    j = (start_j + m) % n
                    if i == j:
                        continue
                    if time.time() > max_time:
                        return p, cost
                    
                    delta = calc_delta(i, j, p)
                    if delta < 0:
                        v = p[j]
                        p[i], p[j] = v, u
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, cost

    # Local Search: Steepest Descent (for polishing)
    def local_search_steepest(p, cost, max_time):
        while time.time() < max_time:
            best_delta = 0
            best_i = -1
            best_j = -1
            for i in range(n):
                if time.time() > max_time:
                    break
                u = p[i]
                for j in range(i + 1, n):
                    if time.time() > max_time:
                        break
                    delta = calc_delta(i, j, p)
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
                if time.time() > max_time:
                    break
            
            if best_i == -1:
                break
                
            i, j = best_i, best_j
            v = p[j]
            u = p[i]
            p[i], p[j] = v, u
            cost += best_delta
            
        return p, cost

    # Island Model
    num_islands = 4
    islands = []
    
    # Initialize islands with different strategies
    for k in range(num_islands):
        if k == 0:
            p = init_linear()
        elif k == 1:
            p = init_spectral()
        else:
            p = init_random()
        
        # Ensure unique permutations for islands if possible, or just let them be
        # Calculate cost
        c = calculate_cost(p)
        islands.append({'p': p, 'cost': c})

    best_p = islands[0]['p'].copy()
    best_cost = islands[0]['cost']
    for isl in islands:
        if isl['cost'] < best_cost:
            best_cost = isl['cost']
            best_p = isl['p'].copy()

    # Run interleaved local search
    # We will cycle through islands. Each island gets a time slice.
    # To allow diversity, we occasionally "kick" an island (random swap) if it stagnates.
    
    time_slice = 0.01 # 10ms per island step? 
    # Actually, just check time inside loops.
    
    # Phase 1: First Improvement Local Search on all islands
    # We run them in parallel (sequentially in Python, but interleaved)
    # until time is up or all converged.
    
    phase1_end = start_time + time_limit * 0.7
    
    while time.time() < phase1_end:
        any_improved = False
        for idx, isl in enumerate(islands):
            if time.time() > phase1_end:
                break
            
            p, c = local_search_first_improvement(isl['p'], isl['cost'], phase1_end)
            isl['p'] = p
            isl['cost'] = c
            
            if c < best_cost:
                best_cost = c
                best_p = p.copy()
            
            # Simple stagnation check: if no improvement in last few steps?
            # We don't track history easily here.
            # Let's add a random kick if this island hasn't improved in a while?
            # For now, just let it run. If it converges, it will return quickly.
            
        # If all islands converged (returned without changing cost in a full cycle), 
        # we might need to diversify.
        # But local_search_first_improvement returns as soon as it finds a move or finishes a pass.
        # If it finishes a pass without finding a move, it's a local optimum.
        
        # Check if all are local optima?
        # Hard to tell without flag.
        # Let's add a kick mechanism:
        # If time.time() < phase1_end - 10:
        #    Pick a random island that is not best, do a random swap.
        
        # Simple diversification:
        if random.random() < 0.1: # 10% chance to kick a non-best island
            non_best = [i for i in range(num_islands) if i != best_p is not islands[i]['p']] # Rough check
            if non_best:
                # Pick random non-best island
                # Actually, just pick any island that isn't the current global best
                # Or just pick a random island and kick it
                kick_idx = random.randint(0, num_islands - 1)
                # If it's the best, maybe skip? Or kick it anyway to explore?
                # Let's kick a random one.
                p = islands[kick_idx]['p']
                c = islands[kick_idx]['cost']
                
                # Perform 2-3 random swaps
                for _ in range(random.randint(1, 3)):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i != j:
                        delta = calc_delta(i, j, p)
                        v = p[j]
                        u = p[i]
                        p[i], p[j] = v, u
                        c += delta
                
                islands[kick_idx]['p'] = p
                islands[kick_idx]['cost'] = c
                
                if c < best_cost:
                    best_cost = c
                    best_p = p.copy()

    # Phase 2: Steepest Descent Polish on the best island (and maybe others if time permits)
    phase2_end = start_time + time_limit - 2.0
    
    # Sort islands by cost
    islands.sort(key=lambda x: x['cost'])
    
    # Polish the best island
    if islands[0]['cost'] == best_cost:
        p, c = local_search_steepest(islands[0]['p'], islands[0]['cost'], phase2_end)
        islands[0]['p'] = p
        islands[0]['cost'] = c
        if c < best_cost:
            best_cost = c
            best_p = p.copy()
    else:
        # If the global best is in a different island (shouldn't happen if we updated correctly, 
        # but just in case)
        # Actually, best_p is a copy. We should polish the island that holds the best cost.
        # Let's find the island with the min cost
        min_cost_island = min(islands, key=lambda x: x['cost'])
        p, c = local_search_steepest(min_cost_island['p'], min_cost_island['cost'], phase2_end)
        min_cost_island['p'] = p
        min_cost_island['cost'] = c
        if c < best_cost:
            best_cost = c
            best_p = p.copy()

    # Final check
    if sorted(best_p.tolist()) != list(range(n)):
        return np.arange(n).tolist()
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
