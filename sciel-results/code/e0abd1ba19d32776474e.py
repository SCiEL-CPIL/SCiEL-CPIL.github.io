import random
import itertools

def construct():
    n = 16
    points = list(range(n))
    
    # 1. Generate all triples and map to indices
    all_triples = []
    triple_to_idx = {}
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                t = (i, j, k)
                triple_to_idx[t] = len(all_triples)
                all_triples.append(t)
                
    num_triples = len(all_triples) # 560
    
    # 2. Identify all K(2,2,2) configurations
    # A K(2,2,2) is defined by 6 distinct points partitioned into 3 pairs.
    # The 8 triples are all combinations of one point from each pair.
    # We need to map each triple to the list of configurations it belongs to.
    
    # First, enumerate all sets of 6 points
    configs = []
    config_to_triples = []
    
    # To ensure we don't duplicate configurations, we can iterate through
    # combinations of 6 points and all 15 partitions.
    # However, we just need the mapping triple -> configs.
    
    # It is more efficient to iterate through triples and find their configs?
    # No, iterating through configs is better to build the reverse map.
    
    # There are C(16,6) = 8008 sets of 6 points.
    # For each set, 15 partitions.
    # Total configs = 120,120.
    
    # Let's build config_to_triples and triple_to_configs
    triple_to_configs = [[] for _ in range(num_triples)]
    
    for combo in itertools.combinations(points, 6):
        # Get all partitions of 6 elements into 3 pairs
        # There are 15 such partitions.
        # We can generate them by pairing up.
        # A standard way to generate 1-factorizations of K6.
        
        # Let's just iterate through all permutations of the 6 points and group?
        # Or use a known list of 15 pairings.
        # Pairings of 0,1,2,3,4,5:
        # (0,1)(2,3)(4,5)
        # (0,1)(2,4)(3,5)
        # ...
        
        # Easier: Iterate through all ways to pick 2 for pair 1, 2 for pair 2, 2 for pair 3.
        # To avoid duplicates, enforce ordering of pairs?
        # Actually, the set of 8 triples is the same regardless of which pair is called "pair 1".
        # So we just need the set of 8 triples.
        
        # Let's generate the 8 triples for a given partition.
        # To avoid generating the same set of 8 triples multiple times for the same 6-point set,
        # we can just generate all 15 partitions and add the set of triples.
        # But we need to map *each* triple to the *config ID*.
        # If we use the same config ID for the same set of 8 triples, we are good.
        # Since the set of 8 triples uniquely identifies the partition (and the 6 points),
        # we can just collect the set of triples.
        
        # Optimization: 
        # For a fixed set of 6 points, there are 15 distinct K(2,2,2)s.
        # We can generate them by fixing the first point's partner?
        
        # Let's just use itertools to find all partitions.
        # A partition is a set of 3 disjoint pairs covering the 6 points.
        
        # We can generate them by:
        # p1 = (combo[0], combo[i]) for i in 1..5
        # remaining 4 points, p2 = (rem[0], rem[j]) for j in 1..3
        # p3 = the last 2.
        
        for i in range(1, 6):
            p1 = (combo[0], combo[i])
            rem = [x for x in combo if x not in p1]
            for j in range(1, 4):
                p2 = (rem[0], rem[j])
                p3 = (rem[1], rem[3]) if rem[1] < rem[3] else (rem[3], rem[1])
                # Ensure p3 is sorted and distinct from p1, p2
                # This generates 5 * 3 = 15 partitions.
                
                # Generate the 8 triples
                t_set = []
                for a in p1:
                    for b in p2:
                        for c in p3:
                            t = tuple(sorted((a, b, c)))
                            t_set.append(t)
                
                # Convert to indices
                t_indices = [triple_to_idx[t] for t in t_set]
                
                # Check if this configuration is already seen?
                # The set of 8 triples uniquely identifies the configuration.
                # We can use a set of frozensets of indices to check duplicates?
                # Or just assume the 15 generated are distinct for a given 6-set.
                # Yes, distinct partitions yield distinct sets of edges.
                
                config_id = len(config_to_triples)
                config_to_triples.append(t_indices)
                
                # Update reverse map
                for idx in t_indices:
                    triple_to_configs[idx].append(config_id)
                    
    num_configs = len(config_to_triples)
    
    best_size = 0
    best_set = None
    
    # Greedy with count array
    for restart in range(50): # Increase restarts as it's faster now
        # Random order
        order = list(range(num_triples))
        random.shuffle(order)
        
        selected = [False] * num_triples
        count_conf = [0] * num_configs
        
        for t_idx in order:
            # Check if adding t_idx is safe
            # It is unsafe if any config it belongs to already has 7 selected triples
            safe = True
            for c_idx in triple_to_configs[t_idx]:
                if count_conf[c_idx] == 7:
                    safe = False
                    break
            
            if safe:
                selected[t_idx] = True
                for c_idx in triple_to_configs[t_idx]:
                    count_conf[c_idx] += 1
        
        current_size = sum(selected)
        if current_size > best_size:
            best_size = current_size
            best_set = selected[:]
            
    # Convert best_set to list of lists
    result = [list(all_triples[i]) for i in range(num_triples) if best_set[i]]
    return result
