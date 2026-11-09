import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    n = 16
    
    # 1. Precomputation: Map triples to indices and find all K_{2,2,2} configurations
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # A forbidden configuration is a set of 3 disjoint pairs.
    # The 8 transversals form a K_{2,2,2}.
    # We need to find all such sets of 8 triples.
    # There are C(16,6) ways to choose 6 vertices, and 3 ways to partition them into 3 pairs.
    # Total configurations: 8008 * 3 = 24024.
    
    # For each configuration, we store the bitmask of the 8 triples involved.
    # We also need to know, for each triple, which configurations it belongs to.
    # Instead of storing lists of configs per triple, we can just store the list of 
    # (config_mask, triple_idx) or simply iterate through all configs.
    
    # Better: For each triple t, store a list of masks of the OTHER 7 triples in the configs it belongs to.
    # Actually, simpler: 
    # 1. Generate all 24024 configs.
    # 2. For each config, get the 8 triple indices.
    # 3. For each of the 8 triples, add the mask of the other 7 to a list associated with that triple.
    
    # Let's optimize: 
    # configs is a list of 8-int arrays.
    # We want for each triple i, a list of masks.
    # Let's just store all config masks in an array `config_masks` of shape (num_configs, 8).
    
    config_indices = []
    for vertices in itertools.combinations(range(n), 6):
        v0, v1, v2, v3, v4, v5 = vertices
        # 3 ways to partition 6 items into 3 pairs
        partitions = [
            ((v0, v1), (v2, v3), (v4, v5)),
            ((v0, v2), (v1, v3), (v4, v5)),
            ((v0, v3), (v1, v2), (v4, v5))
        ]
        for p1, p2, p3 in partitions:
            # Generate the 8 transversals
            tris = []
            for a in p1:
                for b in p2:
                    for c in p3:
                        tris.append((a, b, c))
            # Convert to indices
            idxs = [triple_to_idx[t] for t in tris]
            config_indices.append(idxs)
            
    num_configs = len(config_indices)
    # config_indices is a list of lists of 8 ints.
    # Convert to numpy array for fast bitwise ops? 
    # Actually, we need to check if 7 specific bits are set in a 560-bit integer.
    # Python integers are arbitrary precision, so we can use them directly.
    # 560 bits is very small for Python big ints.
    
    # Precompute for each triple i, the list of masks of the other 7 triples in each config it's in.
    # triple_masks[i] will be a list of integers (masks).
    triple_masks = [[] for _ in range(num_triples)]
    
    for cfg in config_indices:
        # cfg is a list of 8 indices
        base_mask = 0
        for idx in cfg:
            base_mask |= (1 << idx)
            
        for i in range(8):
            other_mask = base_mask & ~(1 << cfg[i])
            triple_masks[cfg[i]].append(other_mask)
            
    # Now, for a given system (represented as a set of indices or a bit mask),
    # a triple t is "blocked" if there exists a mask m in triple_masks[t] such that (system_mask & m) == m.
    
    # To speed this up, we can precompute for each triple a list of masks.
    # In the main loop, for each candidate triple, we check its masks against the current system mask.
    
    best_system = None
    best_count = 0
    
    random.seed(42)
    
    # We will try multiple restarts.
    # Each restart:
    # 1. Shuffle triples.
    # 2. Iterate through shuffled triples.
    # 3. For each triple, check if it's blocked.
    # 4. If not blocked, add it.
    
    # Optimization: 
    # Checking if a triple is blocked requires checking all masks in triple_masks[t].
    # Average number of masks per triple: 
    # Each triple is in how many configs?
    # A triple {a,b,c} is in a config if it's a transversal.
    # The other 3 vertices of the config are chosen from the remaining 13.
    # And they must form 3 pairs with a,b,c.
    # For a fixed triple {a,b,c}, the number of configs is:
    # Choose 3 other vertices from 13: C(13,3) = 286.
    # For each set of 3 other vertices, how many ways to pair them with a,b,c?
    # The pairs must be disjoint. 
    # If we fix the triple, the "other" vertices d,e,f must be paired with a,b,c.
    # The pairs are (a, x), (b, y), (c, z) where {x,y,z} = {d,e,f}.
    # There are 3! = 6 ways to assign x,y,z to a,b,c.
    # So 286 * 6 = 1716 configs per triple.
    # This is a lot of masks to check per triple.
    # 1716 checks * 560 triples * restarts... might be slow if done naively in Python.
    
    # Alternative: Maintain a set of "blocked" triples?
    # When we add a triple t, it might block other triples.
    # But the "blocked" status depends on the *entire* current system.
    # A triple t is blocked if it completes a K_{2,2,2}.
    # This happens if there exist 7 other triples in the system that form a K_{2,2,2} with t.
    
    # Let's try to optimize the check.
    # Instead of checking all 1716 masks, can we update a "danger" count?
    # For each config C, let k_C be the number of triples in C that are in the system.
    # A triple t is blocked if there is a config C containing t such that k_C == 8 (if t is in system) 
    # or if we are *considering* adding t, we check if k_C == 7 for the other 7.
    
    # Let's maintain `config_counts`: an array of length num_configs, where config_counts[i] is the number of triples from config i currently in the system.
    # When we add a triple t:
    # 1. For each config C containing t, increment config_counts[C].
    # 2. If config_counts[C] becomes 8, we have a violation! (But we only add if valid, so this shouldn't happen if we check properly).
    # Wait, the check is: Can we add t?
    # We can add t if for ALL configs C containing t, config_counts[C] < 7.
    # Because if config_counts[C] == 7, adding t makes it 8.
    
    # So, for each triple t, we need to know the max config_counts[C] among all C containing t.
    # If max_count < 7, t is safe.
    
    # How to update this efficiently?
    # When we add t, we increment config_counts[C] for all C containing t.
    # This might cause some other triple u (in some C containing t) to become blocked?
    # No, u is blocked if there exists a C' containing u such that config_counts[C'] >= 7 (before adding u).
    # Actually, u is blocked if max_{C' containing u} config_counts[C'] >= 7.
    
    # We can maintain `max_count_for_triple[t]` = max(config_counts[C] for C containing t).
    # When we add t:
    # 1. Check if max_count_for_triple[t] < 7. If yes, t is safe to add.
    # 2. If we add t, for each C containing t, config_counts[C] increases by 1.
    # 3. For each C containing t, for each other triple u in C, max_count_for_triple[u] might increase.
    #    Specifically, if config_counts[C] becomes the new max for u, we update it.
    
    # Number of configs per triple: ~1716.
    # Size of each config: 8.
    # So updating takes 1716 * 8 = ~13728 operations per added triple.
    # 560 triples * 13728 = ~7.7 million operations per restart.
    # This is very fast in Python (probably < 0.1s per restart).
    # We can do many restarts!
    
    # Implementation details:
    # 1. Precompute `triple_configs`: list of lists of config indices for each triple.
    # 2. Precompute `config_triples`: list of lists of triple indices for each config.
    
    triple_configs = [[] for _ in range(num_triples)]
    config_triples = [cfg for cfg in config_indices]
    
    for c_idx, cfg in enumerate(config_indices):
        for t_idx in cfg:
            triple_configs[t_idx].append(c_idx)
            
    # Main Loop
    for trial in range(200): # Try many restarts
        if time.time() > deadline - 2:
            break
            
        random.shuffle(all_triples)
        
        config_counts = [0] * num_configs
        max_count_for_triple = [0] * num_triples
        system_set = set()
        system_list = []
        
        for t in all_triples:
            # t is a tuple (a,b,c)
            t_idx = triple_to_idx[t]
            
            # Check if safe
            if max_count_for_triple[t_idx] >= 7:
                continue
            
            # Add t
            system_set.add(t_idx)
            system_list.append(list(t))
            
            # Update counts
            # For each config containing t
            for c_idx in triple_configs[t_idx]:
                old_count = config_counts[c_idx]
                new_count = old_count + 1
                config_counts[c_idx] = new_count
                
                # Update max_count_for_triple for all other triples in this config
                # If new_count > max_count_for_triple[u], update it
                for u_idx in config_triples[c_idx]:
                    if u_idx == t_idx:
                        continue
                    if new_count > max_count_for_triple[u_idx]:
                        max_count_for_triple[u_idx] = new_count
                        
        if len(system_list) > best_count:
            best_count = len(system_list)
            best_system = system_list[:]
            
    if best_system is None:
        # Fallback
        return [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
        
    return best_system

