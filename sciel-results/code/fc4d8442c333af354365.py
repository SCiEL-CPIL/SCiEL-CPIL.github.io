"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product, combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    # Strategy: Find maximum independent set in conflict graph of weight-4 vectors.
    # Conflict if dot product > 2.
    # Candidates: C(11,4) * 16 = 5280 vectors.
    
    # 1. Generate candidates
    candidates = []
    for support in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
    
    cand_arr = np.array(candidates, dtype=np.int64)
    n_cands = len(cand_arr)
    
    # 2. Build Conflict Graph
    # G = cand_arr @ cand_arr.T
    # Conflict if G[i,j] > 2.
    # Dot products are integers in [-4, 4].
    
    # Compute Gram matrix efficiently
    # Using int8 to save memory and potentially speed up
    G = (cand_arr.astype(np.int8) @ cand_arr.T.astype(np.int8))
    
    # Build adjacency lists (neighbors where conflict exists)
    neighbors = [set() for _ in range(n_cands)]
    for i in range(n_cands):
        # Find j where G[i,j] > 2
        # Exclude self (G[i,i]=4)
        conflicts = np.where(G[i] > 2)[0]
        conflicts = conflicts[conflicts != i]
        neighbors[i] = set(conflicts.tolist())
        
    # Convert to numpy arrays for faster iteration if needed, but sets are good for membership.
    # For the greedy algorithm, we need to iterate neighbors. Lists might be faster for iteration.
    neighbor_lists = [list(neighbors[i]) for i in range(n_cands)]
    
    # 3. Greedy Independent Set Algorithm
    # We will use a "Minimum Degree" heuristic with randomization.
    # To improve results, we can try multiple restarts with different random seeds.
    
    best_set = np.array([], dtype=np.int64)
    best_score = 0
    
    rng = np.random.default_rng(42)
    
    # We can also try a "Random Order" greedy as a baseline or alternative.
    # But Min-Degree is usually better for IS.
    
    # Optimization: 
    # Instead of recalculating degrees from scratch or using complex updates,
    # we can maintain a set of active nodes and their current degrees.
    
    max_restarts = 100 # Adjust based on time
    
    for r in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        # Randomize the order of nodes to break ties and explore different local optima
        # We can add random noise to degrees initially, or just randomize tie-breaking.
        # Let's randomize the initial "priority" by adding small random noise to degrees?
        # Or simply, when picking min degree, if there are ties, pick randomly.
        
        active = np.ones(n_cands, dtype=bool)
        # Current degree of each node in the active subgraph
        deg = np.array([len(neighbor_lists[i]) for i in range(n_cands)], dtype=np.int32)
        
        selected_indices = []
        
        # To speed up finding min degree, we can use a simple scan.
        # N=5280. Scan is O(N). Selection is O(N) if we remove neighbors.
        # Total O(N^2) ~ 28M operations per restart.
        # 100 restarts ~ 2.8B operations. Too slow in Python?
        # Let's check timing.
        
        # We can optimize the inner loop.
        # Instead of np.where(active, deg, inf) which creates a new array each time,
        # we can maintain a list of active nodes?
        # Or just scan and check active[i].
        
        while np.any(active):
            if time.time() > deadline - 2:
                break
                
            # Find min degree among active
            min_deg = np.inf
            min_node = -1
            
            # Random tie-breaking: 
            # If multiple nodes have min_deg, pick one randomly.
            # For simplicity in this first pass, let's just pick the first one found.
            # We can improve this later.
            
            for i in range(n_cands):
                if active[i]:
                    if deg[i] < min_deg:
                        min_deg = deg[i]
                        min_node = i
            
            if min_node == -1:
                break
                
            # Select min_node
            selected_indices.append(min_node)
            active[min_node] = False
            
            # Remove neighbors
            # For each neighbor nb of min_node:
            #   If active[nb]:
            #     active[nb] = False
            #     For each neighbor nn of nb:
            #       If active[nn]:
            #         deg[nn] -= 1
            
            # This is correct but potentially slow if degrees are high.
            # Average degree?
            # Let's estimate.
            # Total edges E.
            # Each edge is processed at most twice (once for each endpoint being removed).
            # So total work is O(E).
            
            for nb in neighbor_lists[min_node]:
                if active[nb]:
                    active[nb] = False
                    for nn in neighbor_lists[nb]:
                        if active[nn]:
                            deg[nn] -= 1
        
        if len(selected_indices) > best_score:
            best_score = len(selected_indices)
            best_set = cand_arr[selected_indices].copy()
            
    # Fallback to weight-2 vectors if something went wrong or time ran out before finding anything
    if best_score == 0:
        base_vectors = []
        for i in range(D):
            for j in range(i+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        base_vectors.append(v)
        best_set = np.array(base_vectors, dtype=np.int64)
        
    return best_set

# EVOLVE-BLOCK-END
