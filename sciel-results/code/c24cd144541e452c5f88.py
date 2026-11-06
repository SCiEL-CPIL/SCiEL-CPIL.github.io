import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe
    
    # 1. Precompute squares
    squares = set()
    i = 2
    while i * i <= 512:
        squares.add(i * i)
        i += 1
    
    # 2. Identify bad nodes (diagonal constraint: 2*v is square)
    bad_nodes = set()
    for v in range(1, 257):
        if 2 * v in squares:
            bad_nodes.add(v)
            
    candidates = [v for v in range(1, 257) if v not in bad_nodes]
    
    # 3. Build adjacency list for conflict graph
    # adj[u] is a set of v such that u+v is a square
    adj = {v: set() for v in candidates}
    for u in candidates:
        for s in squares:
            v = s - u
            if v in adj:
                adj[u].add(v)
                adj[v].add(u)
                
    best_set = set()
    best_size = 0
    
    # Run randomized greedy with local improvement
    while time.time() < deadline:
        # Random shuffle
        shuffled_candidates = candidates[:]
        random.shuffle(shuffled_candidates)
        
        current_set = set()
        
        # Greedy Phase
        for v in shuffled_candidates:
            # Check if v conflicts with any node in current_set
            conflict = False
            for u in current_set:
                if u in adj[v]:
                    conflict = True
                    break
            if not conflict:
                current_set.add(v)
                
        # Local Improvement Phase (Greedy Fill)
        # Try to add any node that wasn't added
        added = True
        while added:
            added = False
            for v in candidates:
                if v in current_set:
                    continue
                conflict = False
                for u in current_set:
                    if u in adj[v]:
                        conflict = True
                        break
                if not conflict:
                    current_set.add(v)
                    added = True
                    
        # Track best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
    # Return the best set found
    # The task asks for a nonempty subset. 
    # If best_set is empty (shouldn't happen), return a valid single element.
    if not best_set:
        return [1] # 1+1=2 not square. Valid.
        
    return list(best_set)
