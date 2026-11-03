import numpy as np
import random
import time

def construct():
    D = 11
    # Generate all weight-4 vectors
    # There are C(11,4) supports, each with 2^4 sign combinations
    supports = []
    for comb in __import__('itertools').combinations(range(D), 4):
        supports.append(comb)
        
    vectors = []
    for sup in supports:
        for signs in __import__('itertools').product([1, -1], repeat=4):
            v = [0] * D
            for k, s in enumerate(sup):
                v[s] = signs[k]
            vectors.append(v)
            
    V = np.array(vectors, dtype=np.int8)
    N = len(V)
    
    # Compute conflict graph
    # Dot product matrix
    # V is (N, D). V @ V.T is (N, N)
    # We need to identify i != j where dot > 2
    # To save memory and time, we can process in chunks or just do it if N is small enough.
    # N = 5280. N^2 = 27.8M. 
    # NumPy matrix multiplication is highly optimized.
    # Let's try to compute it.
    
    # Optimization: We only care about dot > 2.
    # Dot products are integers.
    
    # For memory efficiency, we can store the neighbor lists.
    # But first, let's get the adjacency structure.
    
    # Since N is ~5000, V @ V.T is ~200MB in float32. Acceptable.
    # But we can be smarter.
    
    # Let's just build the neighbor lists directly.
    # For each pair, check dot.
    # This is O(N^2 * D). 28M * 11 = 300M ops. Might be slow in pure Python, fast in NumPy.
    
    # Let's use NumPy for the dot products.
    # We'll compute the upper triangle to save memory/time?
    # Actually, just compute full matrix and mask.
    
    start_time = time.time()
    
    # Compute dot products
    # To avoid large intermediate arrays if possible, but 5280x5280 is fine.
    dots = V @ V.T
    
    # Conflict if dots[i,j] > 2 and i != j
    # Create adjacency list
    adj = [set() for _ in range(N)]
    
    # Iterate over upper triangle
    # This python loop might be slow. 
    # Alternative: Use np.where to get indices.
    # But we need to build sets.
    
    # Let's try a hybrid: 
    # For each i, find j such that dots[i,j] > 2.
    # This is still O(N^2) in Python if we loop.
    
    # Let's use a vectorized approach to build the adjacency lists?
    # It's hard to build sparse structures directly from dense arrays in a single vectorized step without loops.
    
    # Given 5280 nodes, a nested loop in Python might take:
    # 5280 * 5280 / 2 = 14M iterations.
    # Each iteration: check condition, add to sets.
    # 14M iterations in Python is ~5-10 seconds. Acceptable.
    
    for i in range(N):
        # Find all j > i where dots[i,j] > 2
        # We can use numpy on the row
        row = dots[i, i+1:]
        indices = np.where(row > 2)[0]
        for idx in indices:
            j = idx + i + 1
            adj[i].add(j)
            adj[j].add(i)
            
    # Now we have the conflict graph.
    # Solve MIS using Local Search / Simulated Annealing
    
    # State: in_set (boolean array)
    # conflict_count (int array): number of neighbors in set
    
    def greedy_init():
        in_set = np.zeros(N, dtype=bool)
        conflict_count = np.zeros(N, dtype=np.int32)
        
        # Order vertices by degree?
        # Simple greedy: pick random order, add if no conflict
        order = list(range(N))
        random.shuffle(order)
        
        for v in order:
            if conflict_count[v] == 0:
                in_set[v] = True
                # Update neighbors
                for u in adj[v]:
                    conflict_count[u] += 1
                    # If u was in set? No, we are building it.
                    # Actually, if u was in set, conflict_count[v] would have been > 0.
                    # So we only add if conflict_count[v] == 0.
                    # When we add v, all its neighbors u get their conflict_count increased.
                    # If u is in set, that's a contradiction? 
                    # No, if u is in set, then conflict_count[v] would have been incremented when u was added.
                    # So if conflict_count[v] == 0, none of its neighbors are in set.
                    # So we can safely add v.
        
        return in_set, conflict_count

    def local_search(in_set, conflict_count, max_iters=20000):
        # We want to maximize sum(in_set)
        # Moves:
        # 1. Add v if conflict_count[v] == 0 and not in_set[v]
        # 2. Swap: Remove a set of neighbors N_v of v (where v not in set) and add v.
        #    If len(N_v) == 1, size is same.
        #    If len(N_v) > 1, size decreases.
        #    We want to increase size.
        
        # Better move: 
        # If there is a v not in set with conflict_count[v] == 0, add it.
        # This is a "greedy fill".
        
        # To improve, we can try to "shake" the set.
        # Let's use a simple hill climbing with random restarts or SA.
        
        current_size = np.sum(in_set)
        best_size = current_size
        best_in_set = in_set.copy()
        
        T = 1.0
        cooling = 0.999
        random.seed(42)
        
        for step in range(max_iters):
            # Check for free vertices
            # We need to efficiently find v with conflict_count[v] == 0 and not in_set[v]
            
            # Maintain a list of free vertices?
            # Or just pick random v.
            
            # Let's pick a random v
            v = random.randint(0, N-1)
            
            if in_set[v]:
                continue
            
            # v is not in set
            # Find neighbors in set
            # We have conflict_count[v], which is the count.
            # But we need the actual neighbors to remove them for a swap.
            
            # If conflict_count[v] == 0:
            #   Add v
            if conflict_count[v] == 0:
                in_set[v] = True
                current_size += 1
                for u in adj[v]:
                    conflict_count[u] += 1
                if current_size > best_size:
                    best_size = current_size
                    best_in_set = in_set.copy()
            else:
                # Try swap
                # Get neighbors in set
                # We need to iterate adj[v] and check in_set[u]
                # This is O(deg(v))
                neighbors_in_set = [u for u in adj[v] if in_set[u]]
                
                if not neighbors_in_set:
                    # Should have been caught by conflict_count[v]==0
                    # But conflict_count might be stale? No, we update it.
                    pass
                else:
                    # Move: Remove neighbors_in_set, Add v
                    # Delta = 1 - len(neighbors_in_set)
                    delta = 1 - len(neighbors_in_set)
                    
                    # Accept if delta > 0 or random < exp(delta/T)
                    if delta > 0:
                        # Always accept
                        for u in neighbors_in_set:
                            in_set[u] = False
                            for w in adj[u]:
                                conflict_count[w] -= 1
                        in_set[v] = True
                        current_size += delta
                        for u in adj[v]:
                            conflict_count[u] += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_in_set = in_set.copy()
                    else:
                        # delta <= 0
                        if random.random() < np.exp(delta / T):
                            for u in neighbors_in_set:
                                in_set[u] = False
                                for w in adj[u]:
                                    conflict_count[w] -= 1
                            in_set[v] = True
                            current_size += delta
                            for u in adj[v]:
                                conflict_count[u] += 1
                            if current_size > best_size:
                                best_size = current_size
                                best_in_set = in_set.copy()
            
            T *= cooling
            
            # Time check
            if time.time() - start_time > 100:
                break
                
        return best_in_set

    # Run multiple restarts
    best_size_overall = 0
    best_set_overall = None
    
    # Use remaining time for restarts
    time_limit = 110
    while time.time() - start_time < time_limit:
        in_set, conflict_count = greedy_init()
        # Run local search for a shorter duration
        # Or just one long SA?
        # Let's do a few restarts with moderate SA steps.
        
        # We can just run the SA above, but it has a fixed max_iters.
        # Let's modify local_search to run until time limit or max_iters.
        
        # For simplicity, let's just run greedy_init and then a quick hill climb.
        # Actually, the SA above is good. Let's just call it.
        
        # To make it faster, we can reduce max_iters per restart and do more restarts.
        
        # Let's refine: 
        # The greedy_init is O(N * avg_degree).
        # The local search is O(max_iters * avg_degree).
        
        # Let's just run one long SA from a greedy start.
        # But greedy start might be poor.
        
        # Let's do 10 restarts.
        
        pass # Placeholder

    # Actually, let's implement a more efficient loop.
    
    # Reset time check
    start_time = time.time()
    
    best_size = 0
    best_set = None
    
    for restart in range(20):
        if time.time() - start_time > 110:
            break
            
        in_set, conflict_count = greedy_init()
        
        # Run SA
        T = 1.0
        cooling = 0.999
        current_size = np.sum(in_set)
        
        # Estimate steps based on remaining time?
        # Let's do 10000 steps per restart.
        for step in range(10000):
            if time.time() - start_time > 110:
                break
                
            v = random.randint(0, N-1)
            if in_set[v]:
                continue
            
            if conflict_count[v] == 0:
                in_set[v] = True
                current_size += 1
                for u in adj[v]:
                    conflict_count[u] += 1
            else:
                neighbors_in_set = [u for u in adj[v] if in_set[u]]
                if neighbors_in_set:
                    delta = 1 - len(neighbors_in_set)
                    if delta > 0 or random.random() < np.exp(delta / T):
                        for u in neighbors_in_set:
                            in_set[u] = False
                            for w in adj[u]:
                                conflict_count[w] -= 1
                        in_set[v] = True
                        current_size += delta
                        for u in adj[v]:
                            conflict_count[u] += 1
                        
            T *= cooling
            
        if current_size > best_size:
            best_size = current_size
            best_set = in_set.copy()

    if best_set is None:
        # Fallback
        in_set, _ = greedy_init()
        best_set = in_set

    # Extract vectors
    indices = np.where(best_set)[0]
    result_vectors = V[indices]
    
    return result_vectors
