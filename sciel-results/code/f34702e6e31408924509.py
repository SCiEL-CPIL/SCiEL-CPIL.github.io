import itertools
import time

def construct():
    n = 16
    points = list(range(n))
    
    # 1. Generate all triples and map them to indices
    all_triples = []
    triple_to_idx = {}
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                t = (i, j, k)
                triple_to_idx[t] = len(all_triples)
                all_triples.append(t)
                
    num_triples = len(all_triples)
    
    # 2. Identify all K(2,2,2) configurations
    # A configuration is defined by 6 distinct points and a partition into 3 pairs.
    # We generate all sets of 6 points, then all perfect matchings.
    configs = [] # List of lists of 8 triple indices
    
    # Precompute matchings for 6 points
    # Points are relative 0..5
    matchings = []
    # Generate all perfect matchings of K6
    # There are 15.
    # We can hardcode or generate.
    def gen_matchings(verts):
        if not verts:
            yield []
        else:
            v0 = verts[0]
            rest = verts[1:]
            for i in range(len(rest)):
                v1 = rest[i]
                new_rest = rest[:i] + rest[i+1:]
                for m in gen_matchings(new_rest):
                    yield [(v0, v1)] + m
                    
    # We need to map the 6 points to actual indices
    # To avoid duplicates, we just iterate over all combinations of 6 points
    # and generate matchings.
    
    # Optimization: 
    # For each set of 6 points, there are 15 matchings.
    # Each matching produces 8 transversals.
    # Total configs = C(16,6) * 15 = 8008 * 15 = 120120.
    
    # We need to map the 8 transversals to triple indices.
    
    # To speed up, we can precompute the 8 transversals for a generic matching
    # and just substitute the points.
    
    # Let's generate the 15 matchings of 6 items (0..5)
    base_matchings = list(gen_matchings(list(range(6))))
    
    # For each combination of 6 points
    for combo in itertools.combinations(range(n), 6):
        # Map 0..5 to combo[0]..combo[5]
        # For each base matching
        for m in base_matchings:
            # m is a list of 3 pairs of indices 0..5
            # Construct the 8 transversals
            # The pairs are (p0, p0'), (p1, p1'), (p2, p2')
            # Transversals are all combinations of choosing one from each pair
            
            # Let's extract the pairs
            # m is [(a,b), (c,d), (e,f)] where a,b,c,d,e,f are 0..5
            # The points are combo[a], combo[b], etc.
            
            # We need the 8 triples.
            # We can iterate 0..7, bit 0 for pair 0, bit 1 for pair 1, bit 2 for pair 2
            
            config_triples = []
            for mask in range(8):
                t_points = []
                for i in range(3):
                    bit = (mask >> i) & 1
                    if bit == 0:
                        t_points.append(combo[m[i][0]])
                    else:
                        t_points.append(combo[m[i][1]])
                t = tuple(sorted(t_points))
                config_triples.append(triple_to_idx[t])
            
            configs.append(config_triples)
    
    # 3. Precompute for each triple, the list of config indices it belongs to
    triple_to_configs = [[] for _ in range(num_triples)]
    for i, cfg in enumerate(configs):
        for t_idx in cfg:
            triple_to_configs[t_idx].append(i)
    
    # 4. Branch and Bound
    # State:
    # - selected_mask: int
    # - available_mask: int
    # - counts: list of ints, count of selected triples in each config
    
    best_size = 0
    best_set = 0
    
    # Heuristic: Order triples? 
    # We will pick the available triple that is in the most "critical" configs.
    # Critical = count is high.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def solve():
        nonlocal best_size, best_set
        
        counts = [0] * len(configs)
        selected_mask = 0
        available_mask = (1 << num_triples) - 1
        
        # We use a stack or recursion
        # Recursion depth max 560.
        
        def backtrack(selected_mask, available_mask, counts, current_size):
            nonlocal best_size, best_set
            
            # Upper bound
            if current_size + bin(available_mask).count('1') <= best_size:
                return
            
            if available_mask == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_set = selected_mask
                return
            
            # Find the "best" triple to branch on
            # We want a triple that is in configs with high counts.
            # Score(t) = sum(1 for c in triple_to_configs[t] if counts[c] >= 6)
            # Or simply pick the first one? 
            # Picking a random one or a fixed order is slow.
            # Let's pick the triple with the highest "pressure".
            
            best_t = -1
            best_score = -1
            
            # Iterate through available bits
            # To make this faster, we can just pick the first available one?
            # No, that leads to deep trees.
            # Let's pick the one that appears in the most configs with count >= 5?
            
            # Optimization: 
            # If we just pick the lowest index, it might be slow.
            # Let's use a simple heuristic: pick the available triple that is in the most configs.
            # Precompute static degree.
            
            # Actually, let's just pick the first available triple for now to ensure correctness,
            # and rely on the pruning from counts.
            # Wait, if we don't prune well, it will TLE.
            
            # Let's try to pick a triple that is in a config with count 6.
            # If we find one, we MUST decide on it (or its partner).
            # Actually, if a config has count 6, the remaining 2 triples in that config:
            # We can pick at most 1 of them.
            # This is a binary constraint.
            
            # Let's find any config with count >= 6.
            # If we find one, we can branch on the remaining triples of that config.
            
            # For simplicity and speed in Python, let's just pick the first available triple.
            # But we need to update counts efficiently.
            
            # To improve, we can skip triples that are "forced out"?
            # No, available_mask already excludes forbidden ones?
            # We need to maintain available_mask such that it only contains "safe" triples.
            # A triple t is unsafe if it is in a config with count 7.
            # So before branching, we should clean available_mask.
            
            # Clean available_mask
            # For each config with count 7, the remaining triple is forbidden.
            # This is O(num_configs) per node, which is 120k. Too slow.
            
            # Alternative: 
            # When we add a triple t, we increment counts for its configs.
            # If a count becomes 7, we remove the other triple from available_mask.
            # If a count becomes 8, we backtrack (invalid).
            
            # So available_mask is always "safe".
            
            # We pick t from available_mask.
            # Let's pick the one with the highest static degree (number of configs) to prune earlier?
            # Or the one that is in the most "tight" configs?
            
            # Let's just pick the first one for now.
            t_idx = (available_mask & -available_mask).bit_length() - 1
            t_bit = 1 << t_idx
            
            # Branch 1: Exclude t
            # We don't need to do anything to counts.
            # Just remove t from available.
            backtrack(selected_mask, available_mask ^ t_bit, counts, current_size)
            
            # Branch 2: Include t
            # Check if t is safe (it should be).
            # Update counts
            affected_configs = []
            valid = True
            for c_idx in triple_to_configs[t_idx]:
                counts[c_idx] += 1
                if counts[c_idx] == 8:
                    valid = False
                    break
                if counts[c_idx] == 7:
                    # Find the other triple in this config
                    # config is a list of 8 indices
                    # We need to find the one that is not selected and not t?
                    # Actually, if count is 7, exactly one triple is missing from the config.
                    # That missing triple is forbidden.
                    cfg = configs[c_idx]
                    # Find the missing one
                    # We can precompute for each config and each position, the other 7?
                    # Or just loop.
                    for other_t in cfg:
                        if other_t != t_idx and not (selected_mask >> other_t) & 1:
                            # other_t is the missing one
                            # Remove it from available_mask
                            available_mask &= ~(1 << other_t)
                            break
                    affected_configs.append(c_idx)
            
            if valid:
                backtrack(selected_mask | t_bit, available_mask, counts, current_size + 1)
            
            # Backtrack counts
            if valid:
                for c_idx in triple_to_configs[t_idx]:
                    counts[c_idx] -= 1
        
        backtrack(selected_mask, available_mask, counts, 0)
        
        return best_set

    start_time = time.time()
    # Run with a timeout? 
    # The problem says 120s.
    # If B&B is too slow, we might TLE.
    # Let's add a simple greedy initial solution to set a good lower bound.
    
    # Greedy initialization
    greedy_set = 0
    greedy_counts = [0] * len(configs)
    available = (1 << num_triples) - 1
    
    # Order triples by degree?
    # Just iterate
    for t_idx in range(num_triples):
        t_bit = 1 << t_idx
        if not (available & t_bit):
            continue
            
        # Check if safe
        safe = True
        for c_idx in triple_to_configs[t_idx]:
            if greedy_counts[c_idx] == 7:
                safe = False
                break
        
        if safe:
            greedy_set |= t_bit
            for c_idx in triple_to_configs[t_idx]:
                greedy_counts[c_idx] += 1
                if greedy_counts[c_idx] == 7:
                    # Find and remove the other one
                    cfg = configs[c_idx]
                    for other_t in cfg:
                        if other_t != t_idx and (greedy_set >> other_t) & 1 == 0:
                            available &= ~(1 << other_t)
                            break
    
    best_size = bin(greedy_set).count('1')
    best_set = greedy_set
    
    # Now run B&B to improve
    # If time is running out, we can stop.
    # But we can't easily interrupt recursion in Python without signals.
    # We'll just run it and hope it's fast enough.
    # To be safe, we can limit the search depth or nodes?
    # No, let's trust the pruning.
    
    # Actually, the B&B above might be too slow in Python.
    # Let's add a node limit.
    
    node_count = 0
    max_nodes = 500000 # ~10-20 seconds?
    
    def backtrack_limited(selected_mask, available_mask, counts, current_size):
        nonlocal best_size, best_set, node_count
        
        if node_count > max_nodes:
            return
            
        node_count += 1
        
        if current_size + bin(available_mask).count('1') <= best_size:
            return
            
        if available_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = selected_mask
            return
            
        t_idx = (available_mask & -available_mask).bit_length() - 1
        t_bit = 1 << t_idx
        
        # Branch 1: Exclude
        backtrack_limited(selected_mask, available_mask ^ t_bit, counts, current_size)
        
        # Branch 2: Include
        affected_configs = []
        valid = True
        for c_idx in triple_to_configs[t_idx]:
            counts[c_idx] += 1
            if counts[c_idx] == 8:
                valid = False
                break
            if counts[c_idx] == 7:
                cfg = configs[c_idx]
                for other_t in cfg:
                    if other_t != t_idx and not (selected_mask >> other_t) & 1:
                        available_mask &= ~(1 << other_t)
                        break
                affected_configs.append(c_idx)
        
        if valid:
            backtrack_limited(selected_mask | t_bit, available_mask, counts, current_size + 1)
        
        if valid:
            for c_idx in triple_to_configs[t_idx]:
                counts[c_idx] -= 1

    counts = [0] * len(configs)
    backtrack_limited(0, (1 << num_triples) - 1, counts, 0)
    
    # Convert best_set to list of triples
    result = []
    for i in range(num_triples):
        if (best_set >> i) & 1:
            result.append(list(all_triples[i]))
            
    return result
