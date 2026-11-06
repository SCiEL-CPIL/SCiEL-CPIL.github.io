"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    N = 256
    vertices = list(range(2, 257))
    n = len(vertices)
    
    # 1. Precompute conflict graph
    # adj[i] is a bitmask of indices j such that lcm(vertices[i], vertices[j]) <= 256
    # We use integers as bitmasks for fast set operations.
    adj = [0] * n
    for i in range(n):
        vi = vertices[i]
        for j in range(i + 1, n):
            vj = vertices[j]
            # Efficient lcm check: lcm(vi, vj) = vi * vj // gcd(vi, vj)
            # To avoid overflow or large numbers, check vi // gcd * vj <= 256
            g = gcd(vi, vj)
            if (vi // g) * vj <= 256:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # 2. Precompute weights
    weights = [1.0 / v for v in vertices]
    
    # 3. Greedy Initialization
    # Strategy: Sort by weight descending (i.e., value ascending) to pick small numbers first?
    # Or sort by degree? 
    # The insight from the prompt suggests that picking 2 first locks us into a suboptimal cluster.
    # However, 1/2 is the largest weight. Excluding 2 might allow including 3, 4, 5, etc.
    # Let's try multiple greedy strategies with different exclusion sets or ordering.
    
    def greedy_init(exclude_mask=0, sort_key='weight_desc'):
        """
        Greedy construction of an independent set.
        exclude_mask: bitmask of vertices to force exclude.
        sort_key: 'weight_desc' (small values first), 'degree_asc' (least constrained first), etc.
        """
        if sort_key == 'weight_desc':
            order = sorted(range(n), key=lambda i: -weights[i])
        elif sort_key == 'degree_asc':
            # Calculate degrees
            degrees = [bin(adj[i]).count('1') for i in range(n)]
            order = sorted(range(n), key=lambda i: (degrees[i], -weights[i]))
        else:
            order = list(range(n))
            
        selected_mask = 0
        current_sum = 0.0
        available_mask = ((1 << n) - 1) & ~exclude_mask
        
        for i in order:
            if not (available_mask & (1 << i)):
                continue
            # Check if i is still available (not excluded by previous selections)
            # We maintain available_mask as nodes not yet excluded.
            # If we select i, we must remove i and its neighbors from available_mask.
            selected_mask |= (1 << i)
            current_sum += weights[i]
            # Remove i and neighbors from available
            available_mask &= ~(adj[i] | (1 << i))
            
        return selected_mask, current_sum

    best_mask = 0
    best_sum = 0.0
    
    # Try several initial configurations
    # 1. Standard greedy (small values first)
    m, s = greedy_init(exclude_mask=0, sort_key='weight_desc')
    if s > best_sum:
        best_sum = s
        best_mask = m
        
    # 2. Greedy excluding 2
    m, s = greedy_init(exclude_mask=(1 << 0), sort_key='weight_desc') # Vertex 0 is 2
    if s > best_sum:
        best_sum = s
        best_mask = m
        
    # 3. Greedy excluding 2, 3
    m, s = greedy_init(exclude_mask=((1 << 0) | (1 << 1)), sort_key='weight_desc')
    if s > best_sum:
        best_sum = s
        best_mask = m
        
    # 4. Greedy by degree ascending (least constrained first)
    m, s = greedy_init(exclude_mask=0, sort_key='degree_asc')
    if s > best_sum:
        best_sum = s
        best_mask = m

    # 4. Local Search / Hill Climbing
    # We will perform a local search starting from the best initial solution.
    # Moves:
    # a) Add a vertex if it doesn't conflict with any selected vertex.
    # b) Swap a selected vertex with an unselected one if the sum increases.
    # c) Remove a selected vertex if it allows adding multiple others that sum to more? (Complex, skip for now)
    
    def get_selected_indices(mask):
        return [i for i in range(n) if (mask >> i) & 1]
    
    def can_add(mask, i):
        # Check if vertex i can be added to the set defined by mask
        # It must not be in mask, and it must not conflict with any vertex in mask
        if (mask >> i) & 1:
            return False
        # Check conflicts: adj[i] & mask should be 0
        return (adj[i] & mask) == 0

    def try_add(mask, i):
        return mask | (1 << i)
    
    # Optimization: Precompute list of neighbors for faster iteration?
    # adj is already bitmask.
    
    # Iterative improvement
    # To avoid getting stuck, we can use a simple tabu-like mechanism or just restart if stuck.
    # Given time constraints, a simple hill climbing with "add if possible" and "swap if better" is good.
    
    # We will run multiple local search chains from different starting points to escape local optima.
    
    def local_search(start_mask, start_sum, max_iters=10000):
        current_mask = start_mask
        current_sum = start_sum
        improved = True
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline - 5.0:
            improved = False
            iters += 1
            
            # Phase 1: Try to add any vertex that fits
            # We can iterate through all vertices
            for i in range(n):
                if can_add(current_mask, i):
                    # Adding is always beneficial since weights > 0
                    new_mask = current_mask | (1 << i)
                    new_sum = current_sum + weights[i]
                    current_mask = new_mask
                    current_sum = new_sum
                    improved = True
                    # Restart the add phase since adding one might open up others? No, adding only removes options.
                    # So we can break and restart the loop to check for more adds.
                    break
            
            # Phase 2: Try swaps
            if not improved:
                # Get selected indices
                selected = get_selected_indices(current_mask)
                unselected = [i for i in range(n) if not ((current_mask >> i) & 1)]
                
                # To speed up, we only consider swaps where the new vertex has higher weight
                # or where the removal allows other additions.
                # Simple swap: remove i, add j.
                # Condition: j must not conflict with (current_mask without i)
                # And weight[j] > weight[i] is a good heuristic, but not necessary if it unlocks others.
                
                for i in selected:
                    mask_without_i = current_mask & ~(1 << i)
                    # Candidates j: not in mask_without_i, and no conflict with mask_without_i
                    # j must be in unselected (which is same as not in current_mask)
                    # Actually j can be i? No, swap implies distinct.
                    
                    # Iterate over potential j
                    # Optimization: Only check j if weight[j] > weight[i] * 0.5? 
                    # Let's just check all j not in mask_without_i.
                    # To make this faster, we can precompute the "available" mask for each i?
                    # available_mask_for_i = ((1<<n)-1) & ~mask_without_i & ~(neighbors of mask_without_i)
                    
                    # Neighbors of mask_without_i:
                    # We need the union of adj[k] for all k in mask_without_i.
                    # This is expensive to compute from scratch each time.
                    
                    # Alternative: Just iterate j in unselected.
                    for j in unselected:
                        if j == i: continue
                        # Check if j conflicts with mask_without_i
                        if (adj[j] & mask_without_i) == 0:
                            new_sum = current_sum - weights[i] + weights[j]
                            if new_sum > current_sum + 1e-12:
                                current_mask = mask_without_i | (1 << j)
                                current_sum = new_sum
                                improved = True
                                break
                    if improved:
                        break
                        # Restart unselected calculation? 
                        # It's safer to break out and let the while loop restart, 
                        # recalculating selected/unselected.
        
        return current_mask, current_sum

    # Run local search from best initial
    best_mask, best_sum = local_search(best_mask, best_sum)
    
    # Try other initial seeds if time permits
    if time.time() < deadline - 10.0:
        # Try a random restart or different greedy
        # Let's try a greedy that prioritizes "independent" small numbers
        # e.g. exclude 2, 4, 8... powers of 2?
        pass

    # Convert mask back to list of values
    result = [vertices[i] for i in range(n) if (best_mask >> i) & 1]
    
    return result

# EVOLVE-BLOCK-END
