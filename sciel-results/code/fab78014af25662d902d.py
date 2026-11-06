import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin for safety
    
    N = 256
    # Squares in range [2, 512]
    squares = [i*i for i in range(2, 23)] # 2^2=4 to 22^2=484
    square_set = set(squares)
    
    # 1. Identify forbidden nodes (2v is a square)
    forbidden = set()
    for s in squares:
        if s % 2 == 0:
            v = s // 2
            if 1 <= v <= N:
                forbidden.add(v)
                
    # 2. Build adjacency list for valid nodes
    valid_nodes = [i for i in range(1, N+1) if i not in forbidden]
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    M = len(valid_nodes)
    
    # Adjacency as list of sets for fast intersection
    adj = [set() for _ in range(M)]
    
    for i, u in enumerate(valid_nodes):
        for v in valid_nodes:
            if u >= v: continue
            s = u + v
            if s in square_set:
                j = node_to_idx[v]
                adj[i].add(j)
                adj[j].add(i)
                
    # Helper to convert index to value
    def idx_to_val(i):
        return valid_nodes[i]
        
    def val_to_idx(v):
        return node_to_idx[v]

    # Greedy heuristic: Min-degree in remaining subgraph
    def greedy_mis():
        remaining = set(range(M))
        chosen = []
        while remaining:
            min_deg = float('inf')
            best_node = None
            for u in remaining:
                # len of intersection is faster than sum of generator
                deg = len(adj[u] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_node = u
            chosen.append(best_node)
            # Remove best_node and its neighbors
            to_remove = {best_node}
            to_remove.update(adj[best_node])
            remaining -= to_remove
        return chosen

    # Local Search (1-swap and 0-conflict adds)
    def local_search(initial_set):
        S = set(initial_set)
        improved = True
        while improved:
            improved = False
            # Try to add any node not in S with 0 conflicts
            for u in range(M):
                if u in S: continue
                # Check if u has any neighbors in S
                if not (adj[u] & S):
                    S.add(u)
                    improved = True
                    break # restart loop
            
            if improved: continue
            
            # Try 1-swaps: remove 1 neighbor, add u
            # Only try if u has exactly 1 neighbor in S
            for u in range(M):
                if u in S: continue
                neighbors_in_S = adj[u] & S
                if len(neighbors_in_S) == 1:
                    # The swap is valid: remove the one neighbor, add u
                    S.remove(next(iter(neighbors_in_S)))
                    S.add(u)
                    improved = True
                    break # restart
        return S

    best_set = set(greedy_mis())
    
    # Run multiple restarts with randomization
    num_restarts = 100 # Reduced from 500 to ensure completion
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Randomized Greedy: Pick min degree, break ties randomly
        remaining = set(range(M))
        chosen = []
        while remaining:
            # Calculate degrees for remaining nodes
            degs = {}
            for u in remaining:
                degs[u] = len(adj[u] & remaining)
            
            # Find min degree
            min_deg = min(degs.values())
            # Get all nodes with min degree
            candidates = [u for u in remaining if degs[u] == min_deg]
            best_node = random.choice(candidates)
            
            chosen.append(best_node)
            to_remove = {best_node}
            to_remove.update(adj[best_node])
            remaining -= to_remove
            
        S = set(chosen)
        S = local_search(S)
        
        if len(S) > len(best_set):
            best_set = S

    # Convert back to values
    result = [idx_to_val(i) for i in best_set]
    result.sort()
    return result
