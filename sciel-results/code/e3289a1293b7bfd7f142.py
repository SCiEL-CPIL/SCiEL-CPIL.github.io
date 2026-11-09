import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_greedy_decomposition(edges, rng, max_cycles=100):
        """
        Greedily find cycles to minimize pieces.
        Returns (list_of_cycles, set_of_remaining_edges)
        """
        if not edges:
            return [], set()
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cycles = []
        remaining_edges = set(edges)
        
        # We want to find cycles that are "good". 
        # Heuristic: Find a cycle, remove it, repeat.
        # To get a good initial solution, we try to find long cycles.
        
        for _ in range(max_cycles):
            if not any(len(neigh) > 0 for neigh in adj.values()):
                break
            
            # Find a cycle using a randomized DFS/Warnsdorff
            # Start from a random vertex with high degree
            vertices = [u for u in adj if adj[u]]
            if not vertices:
                break
                
            # Try a few starts
            best_cycle = None
            for _ in range(20): # Increased attempts per cycle
                start = rng.choice(vertices)
                path = [start]
                visited = {start}
                current = start
                
                while True:
                    neighbors = [n for n in adj[current] if n not in visited]
                    if not neighbors:
                        break
                    
                    # Choose neighbor with fewest options (Warnsdorff)
                    # to close the loop
                    min_deg = float('inf')
                    next_node = None
                    for n in neighbors:
                        deg = len(adj[n] - visited)
                        if deg < min_deg:
                            min_deg = deg
                            next_node = n
                        elif deg == min_deg and rng.random() < 0.5:
                            next_node = n
                            
                    if next_node is None:
                        break
                    
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node
                    
                    if len(path) > 30: # Cap for speed
                        break
                
                if len(path) >= 3 and path[-1] in adj[path[0]]:
                    if best_cycle is None or len(path) > len(best_cycle):
                        best_cycle = path
            
            if best_cycle:
                cycles.append(best_cycle)
                # Remove edges
                for i in range(len(best_cycle)):
                    u, v = best_cycle[i], best_cycle[(i+1) % len(best_cycle)]
                    e = (min(u,v), max(u,v))
                    remaining_edges.discard(e)
                    adj[u].discard(v)
                    adj[v].discard(u)
            else:
                break
                
        return cycles, remaining_edges

    def decompose(edges):
        if not edges:
            return []
            
        best_pieces = None
        best_score = float('inf')
        
        # Multiple restarts to find a better decomposition
        # Each restart uses a different random seed for the greedy process
        num_restarts = 100
        
        for r in range(num_restarts):
            if time.time() - start_time > DEADLINE:
                break
                
            rng = random.Random(1000 + r)
            cycles, remaining = find_greedy_decomposition(edges, rng)
            
            score = len(cycles) + len(remaining)
            
            if score < best_score:
                best_score = score
                best_pieces = []
                for c in cycles:
                    best_pieces.append(list(c))
                for e in remaining:
                    best_pieces.append([e[0], e[1]])
                
                # Early exit if we found a perfect decomposition? 
                # A perfect decomposition would have 0 remaining edges.
                # The minimum number of cycles is not trivially known, but 
                # if we get 0 remaining edges, it's a strong candidate.
                if not remaining:
                    # We can't do better than 0 remaining edges in terms of "waste",
                    # but we might reduce the number of cycles by merging?
                    # Merging cycles is hard. 
                    # However, 0 remaining edges is a good sign.
                    # Let's not break immediately as we might find a solution with fewer cycles.
                    pass

        if best_pieces is None:
             # Fallback: all edges
             return [[u, v] for u, v in edges]
             
        return best_pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        results.append(decompose(edges))
    return results
