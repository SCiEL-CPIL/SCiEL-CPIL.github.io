import time
import random
from itertools import combinations, permutations

def construct():
    deadline = time.time() + 140
    n = 16
    points = list(range(n))
    
    # 1. Generate all triples
    all_triples = list(combinations(points, 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # 2. Generate all K(2,2,2) configurations
    # A config is defined by 3 disjoint pairs.
    # We represent a config by the set of 8 triples it contains.
    # We need to map each config to the indices of its 8 triples.
    
    # To avoid duplicates in configs?
    # A specific set of 8 triples defines a unique K(2,2,2) structure?
    # Yes, the 8 triples determine the 6 vertices and the pairing.
    # So we can just generate them.
    
    config_triples = [] # List of lists of triple indices
    
    # Iterate over all 6-subsets
    for six_set in combinations(points, 6):
        # Get all perfect matchings of the 6-set
        # There are 15 matchings.
        # We can generate them by picking pairs.
        # Let vertices be v0..v5.
        # Pairs are (v0, vi), (vj, vk), (vl, vm) where i,j,k,l,m are 1..5.
        # To avoid duplicates, enforce v0 < vi, vj < vk, vl < vm, and vj < vl.
        
        v0, v1, v2, v3, v4, v5 = six_set
        
        # Pairs involving v0: (v0, v1), (v0, v2), (v0, v3), (v0, v4), (v0, v5)
        # For each, pick 2 pairs from the remaining 4 vertices.
        remaining_indices = [1, 2, 3, 4, 5]
        
        for i in range(1, 6):
            pair1 = (v0, six_set[i])
            rem_verts = [six_set[j] for j in remaining_indices if j != i]
            # rem_verts has 4 elements: r0, r1, r2, r3
            # Matchings of 4 elements: 3 ways.
            # (r0,r1),(r2,r3)
            # (r0,r2),(r1,r3)
            # (r0,r3),(r1,r2)
            
            r0, r1, r2, r3 = rem_verts
            
            matchings = [
                ((r0, r1), (r2, r3)),
                ((r0, r2), (r1, r3)),
                ((r0, r3), (r1, r2))
            ]
            
            for m2, m3 in matchings:
                # We have 3 pairs: pair1, m2, m3
                # Ensure they are disjoint (they are by construction)
                # Generate 8 transversal triples
                # pair1 = (a, a'), m2 = (b, b'), m3 = (c, c')
                a, a_prime = pair1
                b, b_prime = m2
                c, c_prime = m3
                
                t_indices = set()
                for u in (a, a_prime):
                    for v in (b, b_prime):
                        for w in (c, c_prime):
                            t = tuple(sorted((u, v, w)))
                            t_indices.add(triple_to_idx[t])
                
                # Add config
                config_triples.append(list(t_indices))
                
    num_configs = len(config_triples)
    
    # 3. Map triple to configs
    # triple_to_configs[i] = list of config indices containing triple i
    triple_to_configs = [[] for _ in range(num_triples)]
    for c_idx, t_indices in enumerate(config_triples):
        for t_idx in t_indices:
            triple_to_configs[t_idx].append(c_idx)
    
    # 4. Greedy Hitting Set with Random Restarts
    best_missing = None
    best_size = float('inf')
    
    max_restarts = 50
    restart_count = 0
    
    while restart_count < max_restarts and time.time() < deadline - 5:
        restart_count += 1
        covered = [False] * num_configs
        missing = []
        
        # Shuffle triples to break ties randomly
        triple_order = list(range(num_triples))
        random.shuffle(triple_order)
        
        # Pre-compute uncovered counts? 
        # We need to update counts as we cover configs.
        # uncovered_count[t] = number of configs in triple_to_configs[t] that are not covered.
        
        uncovered_count = [0] * num_triples
        # Initialize
        for t_idx in range(num_triples):
            # All configs are initially uncovered
            uncovered_count[t_idx] = len(triple_to_configs[t_idx])
        
        while any(not c for c in covered):
            # Find best triple
            best_t = -1
            max_covers = -1
            best_t_idx_in_order = -1
            
            # Iterate through shuffled order to pick first max (or random among max)
            # To make it more random, we can collect all maxes and pick one.
            # But for speed, we can just pick the first one encountered with max value.
            # Since order is shuffled, this introduces randomness.
            
            for pos, t_idx in enumerate(triple_order):
                if uncovered_count[t_idx] > max_covers:
                    max_covers = uncovered_count[t_idx]
                    best_t = t_idx
                    best_t_idx_in_order = pos
            
            if best_t == -1:
                break # Should not happen if valid
            
            # Add to missing
            missing.append(best_t)
            
            # Update covered and counts
            for c_idx in triple_to_configs[best_t]:
                if not covered[c_idx]:
                    covered[c_idx] = True
                    # Decrement counts for all triples in this config
                    for t_idx in config_triples[c_idx]:
                        uncovered_count[t_idx] -= 1
            
            # Remove best_t from consideration? 
            # We can just set its count to 0 or skip it.
            # Since we only add each triple once, we can mark it as "used".
            # But we need to iterate triple_order again.
            # We can just check if t_idx is in missing? That's O(N).
            # Better: maintain a set of used triples or a boolean array.
            
            # Actually, the loop above finds the max among ALL triples.
            # If a triple is already in missing, its uncovered_count might be > 0?
            # No, if it's in missing, it covered some configs.
            # But it might still cover others?
            # Wait, the greedy algorithm picks a triple, covers its configs.
            # We should not pick the same triple again.
            # So we should exclude used triples.
            
            # Let's optimize the search for best_t.
            # We can maintain a list of available triples.
            # Or just check `if t_idx in used_set`.
            
            # For simplicity and speed, let's just use a boolean array `used`.
            # But we are inside the restart loop.
            
            # Let's refactor the inner loop to be efficient.
            
            # Re-doing the inner loop logic properly:
            # We need to find the triple with max uncovered_count among UNUSED triples.
            
            # Let's restart the inner logic with a cleaner state.
            
            # Actually, the code above is inside the `while any(not c...)` loop.
            # I will rewrite the greedy part more cleanly below.
            
            pass # Placeholder, will rewrite

    # Rewriting the greedy part properly inside the restart loop
    
    # We'll define a helper function for one restart
    def run_greedy(order_seed):
        random.seed(order_seed)
        covered = [False] * num_configs
        used = [False] * num_triples
        missing = []
        
        # Initialize counts
        uncovered_count = [0] * num_triples
        for t_idx in range(num_triples):
            uncovered_count[t_idx] = len(triple_to_configs[t_idx])
        
        # Shuffled order for tie-breaking
        triple_order = list(range(num_triples))
        random.shuffle(triple_order)
        
        while True:
            # Check if all covered
            # Optimization: keep track of remaining_configs count
            # Initially num_configs. Decrement when we cover one.
            # If remaining == 0, break.
            
            # Find best triple
            best_t = -1
            max_covers = -1
            
            for t_idx in triple_order:
                if not used[t_idx] and uncovered_count[t_idx] > max_covers:
                    max_covers = uncovered_count[t_idx]
                    best_t = t_idx
            
            if best_t == -1:
                # No triple covers any new config?
                # This means we are stuck or done.
                # If there are uncovered configs, it's a problem.
                # But every config has 8 triples, so if any is uncovered, 
                # its triples must have count > 0?
                # Yes, unless all its triples are used but didn't cover it?
                # No, if a triple is used, it covers ALL its configs.
                # So if a config is uncovered, none of its 8 triples are used.
                # So at least one of its 8 triples has count > 0.
                # So best_t should not be -1 if any config is uncovered.
                break
            
            # Select best_t
            used[best_t] = True
            missing.append(best_t)
            
            # Update
            for c_idx in triple_to_configs[best_t]:
                if not covered[c_idx]:
                    covered[c_idx] = True
                    # Decrement counts for triples in this config
                    for t_idx in config_triples[c_idx]:
                        uncovered_count[t_idx] -= 1
            
            # Check completion
            # We can maintain a counter `remaining_configs`
            # For now, the loop condition `while True` with break is fine.
            # To be safe, add a check:
            # If all covered, break.
            # Checking `all(covered)` is O(N). 
            # We can just maintain a count.
            
            # Let's add a remaining count variable for efficiency.
            
            pass # Placeholder
            
    # Actually, let's just implement it directly in the main loop for clarity and speed.
    
    best_missing = None
    best_size = float('inf')
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        random.seed(restart)
        covered = [False] * num_configs
        used = [False] * num_triples
        missing = []
        remaining_configs = num_configs
        
        uncovered_count = [0] * num_triples
        for t_idx in range(num_triples):
            uncovered_count[t_idx] = len(triple_to_configs[t_idx])
        
        triple_order = list(range(num_triples))
        random.shuffle(triple_order)
        
        while remaining_configs > 0:
            best_t = -1
            max_covers = -1
            
            for t_idx in triple_order:
                if not used[t_idx]:
                    if uncovered_count[t_idx] > max_covers:
                        max_covers = uncovered_count[t_idx]
                        best_t = t_idx
            
            if best_t == -1:
                break # Should not happen
            
            used[best_t] = True
            missing.append(best_t)
            
            for c_idx in triple_to_configs[best_t]:
                if not covered[c_idx]:
                    covered[c_idx] = True
                    remaining_configs -= 1
                    for t_idx in config_triples[c_idx]:
                        uncovered_count[t_idx] -= 1
        
        if len(missing) < best_size:
            best_size = len(missing)
            best_missing = missing[:]
            
        # Optional: Local Search / Polish
        # Try to remove any triple from best_missing if possible?
        # Or swap?
        # Given time, simple greedy with restarts is likely sufficient.
        
    # Construct final result
    if best_missing is None:
        # Fallback: return seed result
        return [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
        
    missing_set = set(best_missing)
    result = []
    for i, t in enumerate(all_triples):
        if i not in missing_set:
            result.append(list(t))
            
    return result

# EVOLVE-BLOCK-END
