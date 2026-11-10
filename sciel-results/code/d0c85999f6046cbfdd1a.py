import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Set of all edges as frozensets for easy removal
    all_edges = set()
    for u, v in edges:
        if u < v:
            all_edges.add((u, v))
        else:
            all_edges.add((v, u))
            
    def get_edge(u, v):
        if u < v:
            return (u, v)
        else:
            return (v, u)

    def find_hamiltonian_cycle(available_edges, start=0, neighbor_order=None):
        """
        Finds a Hamiltonian cycle using backtracking.
        available_edges: set of tuples (u, v) with u < v.
        start: vertex to start the path.
        neighbor_order: optional list of vertices to define order, or None for sorted.
        Returns list of vertices in cycle order, or None if not found.
        """
        # Build local adjacency based on available edges
        local_adj = {i: set() for i in range(n)}
        for u, v in available_edges:
            local_adj[u].add(v)
            local_adj[v].add(u)
            
        # If neighbor_order is provided, sort neighbors accordingly
        if neighbor_order:
            order_map = {v: i for i, v in enumerate(neighbor_order)}
            for i in range(n):
                local_adj[i] = sorted(local_adj[i], key=lambda x: order_map.get(x, 100))
        else:
            for i in range(n):
                local_adj[i] = sorted(local_adj[i])

        path = [start]
        visited = {start}
        
        def backtrack(current):
            if len(path) == n:
                # Check if we can close the cycle
                if get_edge(current, start) in available_edges:
                    return True
                return False
            
            for next_v in local_adj[current]:
                if next_v not in visited:
                    path.append(next_v)
                    visited.add(next_v)
                    if backtrack(next_v):
                        return True
                    visited.remove(next_v)
                    path.pop()
            return False
            
        if backtrack(start):
            return path
        return None

    # Strategy: Try to find 3 disjoint cycles.
    # We will try different "seeds" for the first cycle to avoid getting stuck.
    
    # Base neighbor order
    base_order = list(range(n))
    
    # Try a few different permutations of neighbor order for the first cycle
    # This helps find different Hamiltonian cycles.
    attempts = 0
    max_attempts = 50 # Limit retries
    
    # Pre-define a few shuffles
    random.seed(42)
    shuffles = []
    for _ in range(20):
        shuffled = base_order.copy()
        random.shuffle(shuffled)
        shuffles.append(shuffled)
        
    # Add identity and reverse
    shuffles.append(base_order)
    shuffles.append(base_order[::-1])
    
    for order in shuffles:
        if attempts >= max_attempts:
            break
        attempts += 1
        
        # Find C1
        C1 = find_hamiltonian_cycle(all_edges, start=0, neighbor_order=order)
        if not C1:
            continue
            
        # Remove edges of C1
        edges_C1 = set()
        for i in range(len(C1)):
            u = C1[i]
            v = C1[(i+1) % len(C1)]
            edges_C1.add(get_edge(u, v))
            
        remaining_edges = all_edges - edges_C1
        
        # Find C2
        # Try different orders for C2 as well? 
        # Let's just try the same order or a few others.
        C2 = None
        for order2 in [order, order[::-1], base_order]:
            C2 = find_hamiltonian_cycle(remaining_edges, start=0, neighbor_order=order2)
            if C2:
                break
                
        if not C2:
            continue
            
        edges_C2 = set()
        for i in range(len(C2)):
            u = C2[i]
            v = C2[(i+1) % len(C2)]
            edges_C2.add(get_edge(u, v))
            
        remaining_edges_2 = remaining_edges - edges_C2
        
        # Find C3
        C3 = None
        for order3 in [order, order[::-1], base_order]:
            C3 = find_hamiltonian_cycle(remaining_edges_2, start=0, neighbor_order=order3)
            if C3:
                break
                
        if C3:
            # Verify they are disjoint
            # (Implicitly guaranteed by edge removal)
            # Verify they are valid cycles
            # (Implicitly guaranteed by find_hamiltonian_cycle)
            
            # Convert to lists of ints
            return [list(C1), list(C2), list(C3)]
            
    # If we couldn't find 3, return the best we found (e.g., 2 or 1)
    # The task says "Maximize number of cycles".
    # If we only found 1, return 1.
    # We should track the best result.
    
    best_result = []
    
    # Re-run search to find at least 1, then 2, then 3, keeping track of best.
    # Actually, the loop above tries to find 3. If it fails, it doesn't save the partial result.
    # Let's modify the logic to save partial results.
    
    # Reset
    for order in shuffles:
        if attempts >= max_attempts:
            break
        # We already used 'attempts' in the loop above? No, 'attempts' was local to that block if we structure it right.
        # Let's just run a fresh search that saves the best chain found.
        pass
        
    # Simpler: Just run the previous loop, but if it fails to find 3, 
    # it might have found 2 or 1. 
    # But the previous loop `continue`s if C2 or C3 is not found.
    # So it doesn't save C1 if C2 fails.
    
    # Let's do a proper search for max k.
    
    max_found = 0
    best_cycles = []
    
    for order in shuffles:
        C1 = find_hamiltonian_cycle(all_edges, start=0, neighbor_order=order)
        if not C1:
            continue
            
        edges_C1 = set()
        for i in range(len(C1)):
            u = C1[i]
            v = C1[(i+1) % len(C1)]
            edges_C1.add(get_edge(u, v))
            
        remaining_edges = all_edges - edges_C1
        
        C2 = None
        for order2 in [order, order[::-1], base_order]:
            C2 = find_hamiltonian_cycle(remaining_edges, start=0, neighbor_order=order2)
            if C2:
                break
                
        if C2:
            if len(best_cycles) < 2:
                best_cycles = [C1, C2]
            max_found = 2
            
            edges_C2 = set()
            for i in range(len(C2)):
                u = C2[i]
                v = C2[(i+1) % len(C2)]
                edges_C2.add(get_edge(u, v))
                
            remaining_edges_2 = remaining_edges - edges_C2
            
            C3 = None
            for order3 in [order, order[::-1], base_order]:
                C3 = find_hamiltonian_cycle(remaining_edges_2, start=0, neighbor_order=order3)
                if C3:
                    break
                    
            if C3:
                if len(best_cycles) < 3:
                    best_cycles = [C1, C2, C3]
                max_found = 3
                # We can stop early if we found 3
                return [list(C1), list(C2), list(C3)]
        else:
            if len(best_cycles) < 1:
                best_cycles = [C1]
            max_found = 1
            
    return [list(c) for c in best_cycles]
