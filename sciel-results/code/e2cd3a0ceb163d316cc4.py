import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin for 60s limit
    
    n = 12
    best_edges = None
    best_score = float('inf')
    
    # Helper to check constraints and count cycle lengths
    def evaluate(edges):
        # Check simple graph, no loops
        if len(edges) != len(set(tuple(sorted(e)) for e in edges)):
            return float('inf'), None
        for u, v in edges:
            if u == v or u < 0 or u >= n or v < 0 or v >= n:
                return float('inf'), None
        
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check min degree
        for i in range(n):
            if len(adj[i]) < 3:
                return float('inf'), None
                
        # Check triangle-free (girth >= 4)
        # A triangle is a cycle of length 3
        # For each edge (u,v), check if they have common neighbors
        # Or simpler: for each u, for each pair in adj[u], if they are connected, triangle
        for u in range(n):
            neighbors = adj[u]
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    if neighbors[j] in adj[neighbors[i]]:
                        return float('inf'), None
                        
        # Find all simple cycles and their lengths
        # Use a DFS approach to find cycles
        # Since n=12, we can use a bitset or just list
        # We want distinct lengths
        lengths = set()
        
        # To avoid finding the same cycle multiple times, we can enforce that the smallest vertex in the cycle is the start
        # And the second smallest is the first step?
        # Standard cycle enumeration is expensive, but n=12 is small.
        
        # We only need to know if there is a cycle of length L.
        # Max cycle length is 12.
        # We can check for cycles of length 4, 5, 6, ...
        
        # Optimized: Just find all cycles and add lengths to set.
        # Use DFS from each node, but only visit nodes > start to avoid duplicates?
        # Actually, we just need the SET of lengths.
        
        # We can use a simple DFS that tracks path.
        # To prevent exponential blowup, we limit depth to n.
        
        def find_cycles(start):
            nonlocal lengths
            stack = [(start, [start], {start})]
            while stack:
                node, path, visited = stack.pop()
                for nei in adj[node]:
                    if nei == start and len(path) >= 4:
                        lengths.add(len(path))
                    elif nei not in visited and nei > start: 
                        # Only allow nodes > start to avoid counting same cycle multiple times from different starts
                        # This works if we consider undirected cycles.
                        # A cycle is uniquely identified by its set of vertices and edges.
                        # The condition `nei > start` ensures we only form cycles where `start` is the minimum vertex.
                        stack.append((nei, path + [nei], visited | {nei}))
                        
        for i in range(n):
            if time.time() > deadline:
                return float('inf'), None
            find_cycles(i)
            
        if not lengths:
            # No cycles found? But min degree 3 implies cycles.
            # Unless graph is a tree? No, min degree 3 on 12 vertices must have cycles.
            return float('inf'), None
            
        return len(lengths), edges

    # Initial random valid graph generation
    # Start with a known valid graph (e.g., two K33) and perturb
    # Or just random edge addition until valid?
    
    # Let's use the seed graph as initial
    seed_edges = set()
    for i in range(6):
        for u, v in [(i, (i+1)%6), (i+6, (i+1)%6+6), (i, i+6)]:
            seed_edges.add(tuple(sorted((u, v))))
            
    current_edges = list(seed_edges)
    score, _ = evaluate(current_edges)
    if score < best_score:
        best_score = score
        best_edges = current_edges
        
    # Local Search
    random.seed(42)
    max_iter = 5000
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        if not current_edges:
            break
            
        # Pick a random edge to remove
        u, v = random.choice(current_edges)
        new_edges = [e for e in current_edges if e != (u, v) and e != (v, u)]
        
        # Pick a random new edge to add
        # Choose two non-adjacent vertices
        candidates = []
        for a in range(n):
            for b in range(a+1, n):
                if (a, b) not in set(map(tuple, new_edges)) and (b, a) not in set(map(tuple, new_edges)):
                    # Check if adding this creates a triangle immediately?
                    # We can just evaluate
                    candidates.append((a, b))
                    
        if not candidates:
            continue
            
        new_u, new_v = random.choice(candidates)
        new_edges.append((new_u, new_v))
        
        # Evaluate
        s, _ = evaluate(new_edges)
        if s < score:
            score = s
            current_edges = new_edges
            if score < best_score:
                best_score = score
                best_edges = current_edges
                if best_score == 1:
                    # Can't do better than 1
                    break
        else:
            # Accept with some probability (Simulated Annealing style)
            # Or just reject to keep valid?
            # Since we need to maintain validity, and evaluate returns inf for invalid,
            # we only accept if valid and better.
            # To escape local minima, we might need to accept worse valid states.
            # But "worse" means more distinct lengths.
            # Let's just do hill climbing for now.
            pass
            
    # Return best found
    if best_edges is None:
        # Fallback to seed
        return [list(e) for e in sorted(seed_edges)]
        
    return [list(e) for e in sorted(best_edges)]
