import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    vertices = range(n)
    
    # Generate all edges
    all_edges = list(itertools.combinations(vertices, 3))
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges) # 560
    
    # Generate all configurations (3 disjoint pairs)
    # A configuration is defined by 3 disjoint pairs.
    # We need to identify the 8 transversal edges.
    
    # To avoid duplicates, we can just iterate through all sets of 6 vertices
    # and all partitions of these 6 vertices into 3 pairs.
    
    configs = []
    # To speed up, we can precompute partitions of 6 elements into 3 pairs
    # There are 15 such partitions.
    # Let's generate them once.
    verts6 = range(6)
    partitions = []
    # Pair 1: (0,1). Remaining: 2,3,4,5. Pairs: (2,3),(4,5) or (2,4),(3,5) or (2,5),(3,4)
    # Actually, let's just generate them programmatically or hardcode the 15.
    # Hardcoding is safer and faster.
    # The 15 perfect matchings on 6 vertices {0,1,2,3,4,5}:
    matchings = [
        ((0,1),(2,3),(4,5)),
        ((0,1),(2,4),(3,5)),
        ((0,1),(2,5),(3,4)),
        ((0,2),(1,3),(4,5)),
        ((0,2),(1,4),(3,5)),
        ((0,2),(1,5),(3,4)),
        ((0,3),(1,2),(4,5)),
        ((0,3),(1,4),(2,5)),
        ((0,3),(1,5),(2,4)),
        ((0,4),(1,2),(3,5)),
        ((0,4),(1,3),(2,5)),
        ((0,4),(1,5),(2,3)),
        ((0,5),(1,2),(3,4)),
        ((0,5),(1,3),(2,4)),
        ((0,5),(1,4),(2,3)),
    ]
    
    # Map vertex sets to configurations
    # We iterate over all 6-subsets of vertices
    for subset in itertools.combinations(vertices, 6):
        # Map 0..5 to actual vertices
        v = list(subset)
        for m in matchings:
            # m is a tuple of 3 pairs of indices 0..5
            p1 = (v[m[0][0]], v[m[0][1]])
            p2 = (v[m[1][0]], v[m[1][1]])
            p3 = (v[m[2][0]], v[m[2][1]])
            
            # Generate 8 transversals
            edges_in_config = []
            for x1 in p1:
                for x2 in p2:
                    for x3 in p3:
                        edge = tuple(sorted((x1, x2, x3)))
                        edges_in_config.append(edge_to_idx[edge])
            
            # Store config as tuple of edge indices
            configs.append(tuple(edges_in_config))
            
    num_configs = len(configs) # 120120
    
    # Precompute configs_in_edge
    configs_in_edge = [[] for _ in range(num_edges)]
    for c_idx, c in enumerate(configs):
        for e_idx in c:
            configs_in_edge[e_idx].append(c_idx)
    
    best_system = []
    best_count = 0
    
    # Greedy algorithm with random restarts
    # We have ~110 seconds.
    # Each run takes ~0.1s?
    # Let's estimate.
    # 560 edges * 10000 configs = 5.6M ops.
    # Python loop overhead is high. 5.6M iterations might take ~0.5s - 1s.
    # Let's try to optimize the inner loop.
    
    def run_greedy(seed):
        random.seed(seed)
        order = list(range(num_edges))
        random.shuffle(order)
        
        conf_count = [0] * num_configs
        current_edges = set()
        
        for e_idx in order:
            # Check if we can add e_idx
            # We need max(conf_count[c] for c in configs_in_edge[e_idx]) < 7
            # If max is 7, adding makes it 8 -> violation.
            # If max < 7, adding makes it <= 7 -> safe.
            
            # Optimization: If any config has count 7, we can't add.
            # We can break early if we find a 7.
            can_add = True
            for c_idx in configs_in_edge[e_idx]:
                if conf_count[c_idx] == 7:
                    can_add = False
                    break
            
            if can_add:
                current_edges.add(e_idx)
                for c_idx in configs_in_edge[e_idx]:
                    conf_count[c_idx] += 1
        
        return current_edges

    # Run multiple times
    # Let's benchmark first run
    # To be safe, run as many as possible before deadline.
    
    # We can also try a "local search" improvement on the best found so far.
    # But greedy with different orders is a good start.
    
    # Let's run 50-100 times.
    
    # Note: The seed program returned 105.
    # We expect to get > 200.
    
    # To improve further, we can use a weighted greedy or look-ahead.
    # But simple random greedy is robust.
    
    # Let's also consider the "removal" strategy from complete graph.
    # Start with all edges.
    # While violation exists, remove an edge.
    # This might leave fewer edges if we remove too many.
    # Greedy addition usually packs more tightly.
    
    # Let's stick to greedy addition.
    
    # One optimization:
    # If we maintain `conf_count`, we can also track which configs are "saturated" (count 7).
    # But checking the list is fine.
    
    # Let's run the loop.
    
    num_runs = 0
    while time.time() < deadline:
        # Increase seed for diversity
        seed = num_runs
        result = run_greedy(seed)
        if len(result) > best_count:
            best_count = len(result)
            best_system = result
        
        num_runs += 1
        
        # If we have a very high count, maybe we can stop early?
        # Theoretical max?
        # If we can't beat 300, we keep trying.
        
        # Break if we've run for a while and haven't improved?
        # No, just keep going until deadline.
        
        # To avoid infinite loop if time check is slow, add a small sleep? No.
        # The loop body is the bottleneck.
        
        # Let's check if we can optimize the inner check.
        # `configs_in_edge[e_idx]` is a list.
        # `conf_count` is a list.
        # Access is fast.
        
        # If we want to speed up, we can use numpy?
        # `conf_count` as numpy array.
        # `configs_in_edge` as list of arrays.
        # `np.max(conf_count[configs_in_edge[e_idx]]) < 7`
        # This might be faster due to C backend.
        
        # Let's try to implement a numpy version for the check.
        
    # Convert best_system (set of indices) to list of triples
    return [list(all_edges[i]) for i in sorted(best_system)]

# Refining the Numpy approach for speed
import numpy as np

def construct_np():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    vertices = range(n)
    all_edges = list(itertools.combinations(vertices, 3))
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)
    
    matchings = [
        ((0,1),(2,3),(4,5)),
        ((0,1),(2,4),(3,5)),
        ((0,1),(2,5),(3,4)),
        ((0,2),(1,3),(4,5)),
        ((0,2),(1,4),(3,5)),
        ((0,2),(1,5),(3,4)),
        ((0,3),(1,2),(4,5)),
        ((0,3),(1,4),(2,5)),
        ((0,3),(1,5),(2,4)),
        ((0,4),(1,2),(3,5)),
        ((0,4),(1,3),(2,5)),
        ((0,4),(1,5),(2,3)),
        ((0,5),(1,2),(3,4)),
        ((0,5),(1,3),(2,4)),
        ((0,5),(1,4),(2,3)),
    ]
    
    configs = []
    for subset in itertools.combinations(vertices, 6):
        v = list(subset)
        for m in matchings:
            p1 = (v[m[0][0]], v[m[0][1]])
            p2 = (v[m[1][0]], v[m[1][1]])
            p3 = (v[m[2][0]], v[m[2][1]])
            edges_in_config = []
            for x1 in p1:
                for x2 in p2:
                    for x3 in p3:
                        edge = tuple(sorted((x1, x2, x3)))
                        edges_in_config.append(edge_to_idx[edge])
            configs.append(tuple(edges_in_config))
            
    num_configs = len(configs)
    
    # Build configs_in_edge as list of numpy arrays for fast indexing
    configs_in_edge_np = [None] * num_edges
    # First pass to get sizes
    sizes = [0] * num_edges
    for c in configs:
        for e in c:
            sizes[e] += 1
            
    for i in range(num_edges):
        configs_in_edge_np[i] = np.zeros(sizes[i], dtype=np.int32)
        
    # Fill
    # We need to track current index for each edge
    idx_tracker = [0] * num_edges
    for c_idx, c in enumerate(configs):
        for e in c:
            configs_in_edge_np[e][idx_tracker[e]] = c_idx
            idx_tracker[e] += 1
            
    best_system = set()
    best_count = 0
    
    def run_greedy_np(seed):
        random.seed(seed)
        order = list(range(num_edges))
        random.shuffle(order)
        
        conf_count = np.zeros(num_configs, dtype=np.int32)
        current_edges = set()
        
        for e_idx in order:
            # Check max count in configs containing e_idx
            # If max < 7, safe
            c_indices = configs_in_edge_np[e_idx]
            if len(c_indices) > 0:
                max_count = np.max(conf_count[c_indices])
            else:
                max_count = 0
                
            if max_count < 7:
                current_edges.add(e_idx)
                conf_count[c_indices] += 1
                
        return current_edges

    num_runs = 0
    while time.time() < deadline:
        seed = num_runs
        result = run_greedy_np(seed)
        if len(result) > best_count:
            best_count = len(result)
            best_system = result
        num_runs += 1
        
    return [list(all_edges[i]) for i in sorted(best_system)]

def construct():
    return construct_np()
