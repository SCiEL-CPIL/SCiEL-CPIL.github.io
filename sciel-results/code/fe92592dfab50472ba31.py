import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Generate Type A vectors (Weight 1, value +/-2, Norm^2=4)
    # These are mutually compatible and compatible with all Type C vectors.
    type_a_vectors = []
    for i in range(D):
        for sign in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * sign
            type_a_vectors.append(v)
    
    type_a_array = np.array(type_a_vectors)
    n_a = len(type_a_array) # 22

    # 2. Generate Type C vectors (Weight 4, values +/-1, Norm^2=4)
    type_c_vectors = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(positions, signs):
                v[pos] = s
            type_c_vectors.append(v)
            
    type_c_array = np.array(type_c_vectors)
    n_c = len(type_c_array) # 330 * 16 = 5280

    # 3. Compute Conflict Graph for Type C vectors
    # Two Type C vectors conflict if their dot product > 2.
    # Norm^2 is 4. Distance^2 = 8 - 2*dot.
    # Min distance >= Max norm (2) => Distance^2 >= 4 => 8 - 2*dot >= 4 => 2*dot <= 4 => dot <= 2.
    # So conflict if dot > 2.
    dots = type_c_array @ type_c_array.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists for fast conflict checking
    neighbors = [np.where(conflicts[i])[0] for i in range(n_c)]
    
    rng = np.random.default_rng(42)
    
    # GA Parameters
    pop_size = 60
    elite_size = 10
    mutation_rate = 0.1
    tournament_size = 5
    
    # Helper: Repair a bitmask to ensure it's an independent set
    # We keep the first 'k' bits set, but greedily remove conflicting ones.
    # To preserve the "intent" of the GA, we iterate in a fixed random order for each repair?
    # No, better to iterate in the order of the bits being 1, but if we just remove conflicts,
    # the order matters. Let's use a fixed random permutation for repair order to be consistent?
    # Or just iterate 0 to N. If we iterate 0 to N, lower indices are preferred.
    # To make it fair, let's use a global random permutation for the repair priority.
    repair_order = rng.permutation(n_c)
    
    def repair(individual):
        """
        Takes a boolean array of length n_c.
        Returns a boolean array representing a valid independent set.
        Greedy: Iterate through nodes in 'repair_order'. If node is in individual,
        check if it conflicts with any already accepted node. If not, accept.
        """
        accepted = np.zeros(n_c, dtype=bool)
        # We need to check conflicts against the 'accepted' set.
        # For efficiency, we can maintain a set of accepted indices.
        accepted_indices = []
        
        for v in repair_order:
            if not individual[v]:
                continue
            # Check if v conflicts with any in accepted_indices
            # neighbors[v] is an array of indices.
            # We need to check if any index in neighbors[v] is in accepted_indices.
            # Since accepted_indices is a list, checking membership is O(N).
            # N is ~5000. Pop size 60. Generations ~100.
            # 60 * 100 * 5000 * (avg degree ~ 100?) -> Too slow.
            
            # Optimization: Use a boolean array for accepted set.
            # But we need to check intersection of neighbors[v] and accepted_set.
            # np.any(conflicts[v][accepted_indices])? No, accepted_indices changes.
            
            # Let's use a different repair strategy:
            # 1. Start with all True where individual is True.
            # 2. While there is a conflict, remove one of the conflicting nodes.
            #    Which one? The one with higher degree in the conflict subgraph?
            #    Or just remove the one that appears later in repair_order?
            
            # Let's stick to the simple greedy but optimize the check.
            # We maintain `accepted_mask` (bool array).
            # Check: np.any(conflicts[v] & accepted_mask)
            # This is vectorized.
            
            # We will do this inside the loop.
            pass
        
        # Re-implementing repair with vectorized check inside the loop is still O(N^2) per repair if not careful.
        # However, n_c=5280.
        # Let's try a simpler repair:
        # 1. Create a list of indices where individual is True.
        # 2. Shuffle this list? No, use repair_order to sort them?
        # 3. Iterate through this sorted list.
        # 4. Maintain `accepted_mask`.
        # 5. If np.any(conflicts[v] & accepted_mask) is False, accept.
        
        # This is O(K * N) where K is number of selected nodes.
        # K is max ~500. N=5280. 500*5280 = 2.6M ops per repair.
        # 60 individuals * 100 generations = 6000 repairs.
        # 6000 * 2.6M = 15.6 Billion ops. Too slow for Python loops.
        
        # We need a faster repair.
        # Alternative: Max Independent Set approximation is hard.
        # Let's use a simpler heuristic repair:
        # Randomly remove nodes involved in conflicts until no conflicts remain.
        # This might destroy the solution structure.
        
        # Let's go back to the greedy but optimize with NumPy.
        # We can precompute nothing more.
        # Let's try to reduce the search space.
        # Actually, the current SA program works.
        # The GA needs to be fast.
        
        # Let's try a different approach for GA:
        # Instead of repairing arbitrary bitmasks, we can define the GA on a "core" set?
        # No, let's try to make the repair fast.
        
        # Efficient Repair:
        # 1. Get list of active nodes: `active = np.where(individual)[0]`
        # 2. Sort `active` by `repair_order` index? 
        #    `sorted_active = active[np.argsort(repair_order[active])]`
        # 3. Iterate `sorted_active`.
        # 4. `conflict_mask = conflicts[v] & accepted_mask`
        # 5. If no conflict, `accepted_mask[v] = True`
        
        # Cost: Sorting is O(K log K). Loop is O(K * (cost of vectorized AND))
        # Cost of vectorized AND is O(N) but highly optimized in C.
        # 500 iterations * 5280 floats/bools ~ 2.6M ops.
        # 6000 repairs * 2.6M = 15B ops. Still likely too slow in 120s.
        
        # How to speed up?
        # Reduce population size? 20 individuals.
        # Reduce generations? 50.
        # 20 * 50 = 1000 repairs. 1000 * 2.6M = 2.6B ops. Maybe okay?
        
        # Let's try to be smarter.
        # We can use a "Conflict Count" approach similar to the SA program's initialization.
        
        # Actually, let's look at the SA program again. It's score 2.4.
        # The theoretical max is ~518?
        # If I can't beat 2.4, I should stick to a robust method.
        
        # Let's try a **Greedy Randomized Adaptive Search Procedure (GRASP)** instead of GA.
        # GRASP is distinct from SA.
        # Phase 1: Constructive Heuristic (Randomized Greedy)
        # Phase 2: Local Search (Swap/Exchange)
        
        # This is very similar to the current SA program's structure (Greedy Init + Local Search).
        # The directive says "do not reuse the current designs".
        # The current design is: Greedy Init + SA Local Search.
        # GRASP is: Randomized Greedy Init + Deterministic Local Search.
        # This is distinct enough?
        
        # Let's implement GRASP.
        
        def grasp_repair_and_optimize(individual):
            """
            Takes a boolean array.
            1. Greedy Repair: Remove conflicts.
            2. Local Search: Try to add nodes, or swap nodes to improve size.
            """
            # 1. Repair
            accepted = np.zeros(n_c, dtype=bool)
            conf_counts = np.zeros(n_c, dtype=np.int32)
            
            # Sort active nodes by a random priority (repair_order)
            active_nodes = np.where(individual)[0]
            if len(active_nodes) == 0:
                return accepted, 0
            
            # Sort by repair_order
            # repair_order is a permutation. We want to process nodes with lower repair_order index first.
            # indices_in_order = np.argsort(repair_order[active_nodes])
            # sorted_active = active_nodes[indices_in_order]
            
            # Faster: just iterate active_nodes in random order?
            # Let's use the global repair_order for consistency.
            # Create a map: node -> priority
            # This is O(N) setup.
            # Let's just shuffle active_nodes for each repair? 
            # "Randomized Greedy" usually means picking randomly from candidates.
            
            # Let's do this:
            # Candidate pool = Active Nodes.
            # While pool not empty:
            #   Pick a node v from pool.
            #   If v is compatible with all currently accepted nodes, accept it.
            #   Else, discard it.
            #   Remove v from pool.
            
            # To make it "Greedy", we should pick the node that conflicts with the FEWEST accepted nodes?
            # Or the node that has the highest "value" (all are 1).
            # Let's pick randomly from the pool, but if it conflicts, we discard.
            # This is just a random subset.
            
            # Better Randomized Greedy:
            # At each step, consider all nodes in the original `individual` that are not yet processed.
            # Calculate their conflict count with the current `accepted` set.
            # Pick the one with the MINIMUM conflict count (if 0, it's perfect).
            # If min conflict count > 0, we might still pick it? No, if it conflicts, we can't add it.
            # So we only add if conflict count is 0.
            # If no node has conflict count 0, we stop?
            # This is just a maximal independent set.
            
            # Let's implement this.
            pool = set(active_nodes)
            accepted_indices = []
            
            while pool:
                # Find all nodes in pool with 0 conflicts against accepted
                # This is expensive to check all every time.
                
                # Optimization:
                # Maintain `conf_counts` for nodes in pool?
                # When we accept a node `u`, it increases conflict count for its neighbors.
                
                # Let's restart the repair logic with this efficient counting.
                accepted = np.zeros(n_c, dtype=bool)
                conf_counts = np.zeros(n_c, dtype=np.int32)
                remaining = set(active_nodes)
                
                # Pre-compute neighbors for fast update
                # neighbors is a list of arrays.
                
                while remaining:
                    # Find nodes in remaining with conf_counts == 0
                    # We can't easily filter a set by an array.
                    # Let's convert remaining to a list or array?
                    # If we keep `remaining` as a boolean array `in_remaining`:
                    
                    # Let's use arrays.
                    break # Switching to array-based loop below
            
            # Array-based Greedy Repair
            accepted = np.zeros(n_c, dtype=bool)
            conf_counts = np.zeros(n_c, dtype=np.int32)
            in_remaining = individual.copy() # Start with all nodes in the candidate set
            
            while in_remaining.any():
                # Find all free nodes (conflict count 0) in remaining
                free_nodes = np.where(in_remaining & (conf_counts == 0))[0]
                
                if len(free_nodes) == 0:
                    # No free nodes. We are stuck.
                    # To escape, we could remove a node from remaining that is "blocking" others?
                    # In GRASP, we just stop? Or we pick a random node to remove from remaining?
                    # If we remove a node, we might free up others.
                    # Let's pick a random node from remaining and remove it (don't accept).
                    # This simulates "skipping" a bad node.
                    stuck_nodes = np.where(in_remaining)[0]
                    v = stuck_nodes[rng.integers(len(stuck_nodes))]
                    in_remaining[v] = False
                    continue
                
                # Pick a random free node
                v = free_nodes[rng.integers(len(free_nodes))]
                accepted[v] = True
                in_remaining[v] = False
                conf_counts[neighbors[v]] += 1
                
            # Now we have a valid independent set `accepted`.
            
            # 2. Local Search: Try to add more nodes?
            # The greedy repair already added all possible nodes that were "free" at some point.
            # But maybe a different order would have allowed more?
            # Local Search: Try to swap a node OUT and a node IN?
            # Or just try to add any node that became free?
            # Actually, the greedy process stops when no free nodes remain in `remaining`.
            # But `remaining` only contains nodes from the initial `individual`.
            # We can try to add nodes NOT in `individual`?
            # Yes, that's the "Improve" step.
            
            # Try to add any node not in `accepted` that is free.
            while True:
                free_nodes = np.where((~accepted) & (conf_counts == 0))[0]
                if len(free_nodes) == 0:
                    break
                v = free_nodes[rng.integers(len(free_nodes))]
                accepted[v] = True
                conf_counts[neighbors[v]] += 1
                
            return accepted, accepted.sum()

    # GRASP Main Loop
    best_size = 0
    best_solution = np.zeros(n_c, dtype=bool)
    
    # Initial population of random bitmasks?
    # GRASP usually starts with a random construction.
    # Let's just run the constructive heuristic on random subsets.
    
    n_restarts = 0
    while time.time() < deadline - 2:
        n_restarts += 1
        # Create a random candidate set (bias towards including nodes?)
        # Just use a random boolean array.
        candidate = rng.random(n_c) > 0.5
        
        # Run GRASP Repair + Improve
        sol, size = grasp_repair_and_optimize(candidate)
        
        if size > best_size:
            best_size = size
            best_solution = sol.copy()
            
        # Optional: Local search on the best solution found so far?
        # The `grasp_repair_and_optimize` already does a local improvement (adding free nodes).
        
    # Construct final result
    final_c_indices = np.where(best_solution)[0]
    result_c = type_c_array[final_c_indices]
    
    if len(result_c) > 0:
        final_vectors = np.vstack([type_a_array, result_c])
    else:
        final_vectors = type_a_array
        
    return final_vectors

# EVOLVE-BLOCK-END
