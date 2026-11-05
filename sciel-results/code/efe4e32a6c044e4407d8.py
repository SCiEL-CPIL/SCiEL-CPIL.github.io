import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 55.0

    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
        
    # Build adjacency lists as sets for fast intersection
    adj = [set() for _ in range(N)]
    valid_vertices = [True] * N
    
    for i in range(N):
        val_i = i + 1
        # Check self-conflict
        if 2 * val_i in squares:
            valid_vertices[i] = False
            continue
            
        for j in range(i + 1, N):
            val_j = j + 1
            if val_i + val_j in squares:
                adj[i].add(j)
                adj[j].add(i)
    
    # List of indices that are valid candidates (no self-conflict)
    candidates = [i for i in range(N) if valid_vertices[i]]
    candidates_set = set(candidates)
    
    best_set = set()
    best_size = 0
    
    def greedy_and_local_search(seed):
        random.seed(seed)
        
        # 1. Greedy Construction
        # Sort by degree + noise
        degrees = {v: len(adj[v]) for v in candidates}
        noise = {v: random.random() for v in candidates}
        sorted_candidates = sorted(candidates, key=lambda v: (degrees[v] + noise[v] * 0.1, v))
        
        available = set(candidates)
        current_set = set()
        
        for v in sorted_candidates:
            if v in available:
                current_set.add(v)
                available.discard(v)
                # Remove neighbors from available
                # adj[v] is a set
                available.difference_update(adj[v])
        
        # 2. Local Search: Add
        # Try to add any vertex not in current_set that doesn't conflict
        improved = True
        while improved:
            improved = False
            for v in candidates:
                if v not in current_set:
                    # Check conflict using set intersection
                    if not adj[v].intersection(current_set):
                        current_set.add(v)
                        improved = True
        
        # 3. Local Search: 1-Swap
        # Try to replace u in current_set with v not in current_set
        for _ in range(20): # More iterations for better local opt
            swapped = False
            set_list = list(current_set)
            random.shuffle(set_list)
            
            for u in set_list:
                if u not in current_set:
                    continue
                
                # Only consider v that are neighbors of u, as non-neighbors 
                # would have been added in the "Add" phase if they didn't conflict with others.
                # Wait, if v is not a neighbor of u, and v is not in current_set, 
                # it must conflict with some w in current_set.
                # Swapping u for v only helps if u was the *only* conflict for v? 
                # No, if v conflicts with w != u, swapping u out doesn't help.
                # So v MUST be a neighbor of u? 
                # Actually, if v is not a neighbor of u, v's conflicts are independent of u.
                # If v couldn't be added before, it has conflicts with current_set.
                # Removing u (which is not a conflict for v) doesn't remove any conflict for v.
                # So v still conflicts.
                # Therefore, we only need to check v in adj[u].
                
                for v in adj[u]:
                    if v in current_set:
                        continue
                    if v not in candidates_set:
                        continue
                    
                    # Check if v conflicts with current_set - {u}
                    # current_set - {u}
                    # We can create a temporary set or just check intersection with current_set 
                    # and ensure the intersection is only u (if u is in adj[v]).
                    # Since v is in adj[u], u is in adj[v].
                    # So adj[v].intersection(current_set) must be exactly {u} for the swap to be valid.
                    
                    conflicts = adj[v].intersection(current_set)
                    if conflicts == {u}:
                        # Perform swap
                        current_set.remove(u)
                        current_set.add(v)
                        swapped = True
                        break
                if swapped:
                    break
            
            if not swapped:
                break
            
            # After a swap, try to add more
            improved_add = True
            while improved_add:
                improved_add = False
                for v in candidates:
                    if v not in current_set:
                        if not adj[v].intersection(current_set):
                            current_set.add(v)
                            improved_add = True
        
        return current_set

    # Run multiple restarts
    num_restarts = 0
    while time.time() - start_time < DEADLINE:
        s = greedy_and_local_search(num_restarts)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
        num_restarts += 1
        
        # Optional: If we found a size > 120, we might be close to optimal.
        # But let's keep searching until deadline.

    # Return list of values (1-based)
    return [i + 1 for i in best_set]
