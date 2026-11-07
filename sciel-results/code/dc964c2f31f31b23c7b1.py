import random
import time
import sys
from collections import deque

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    n = 12
    min_deg = 3
    
    best_edges = None
    best_value = float('inf')
    
    # Helper to get distinct cycle lengths
    def get_cycle_lengths(edges):
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        lengths = set()
        
        # Find all simple cycles
        # For n=12, we can use DFS from each node
        visited_global = set()
        
        for start in range(n):
            pass
        # Better: Use a standard algorithm to enumerate all simple cycles (Johnson's algorithm or similar)
        # For n=12, a simple DFS with pruning is sufficient.
        
        cycles = set()
        
        def dfs(u, start, path, visited):
            if len(path) > 1 and u == start:
                if len(path) >= 3: # Triangle check is separate, but cycles must be >= 4
                    # Normalize cycle to avoid duplicates
                    # Cycle is path[0:-1]
                    c_edges = frozenset((min(path[i], path[(i+1)%len(path)]), max(path[i], path[(i+1)%len(path)])) for i in range(len(path)))
                    cycles.add(c_edges)
                    lengths.add(len(path))
                return
            
            for v in adj[u]:
                if v == start and len(path) >= 4: # Girth 4 constraint check during search?
                    # If we find a cycle, record it
                    c_edges = frozenset((min(path[i], path[(i+1)%len(path)]), max(path[i], path[(i+1)%len(path)])) for i in range(len(path)))
                    cycles.add(c_edges)
                    lengths.add(len(path))
                elif v not in visited and v > start: # Optimization: only visit vertices > start to avoid duplicate cycles
                    # This optimization only works if we want unique cycles, but we need ALL cycles to count lengths.
                    # Actually, to count distinct lengths, we just need to know if a cycle of length L exists.
                    # We can stop early if we find all possible lengths? No.
                    pass
        
        # The above DFS is tricky to implement correctly for all cycles.
        # Given n=12, we can just do a full search.
        
        # Let's use a simpler approach:
        # Since n is small, we can iterate over all subsets? No.
        # We can use networkx if available? No, standard library only.
        
        # Let's use a basic DFS from each node, allowing revisits only to start.
        # To avoid counting the same cycle multiple times, we can restrict the start node to be the smallest node in the cycle.
        
        for start in range(n):
            # DFS
            stack = [(start, [start], {start})]
            while stack:
                u, path, visited = stack.pop()
                for v in adj[u]:
                    if v == start and len(path) >= 4:
                        # Found a cycle
                        lengths.add(len(path))
                    elif v not in visited and v > start:
                        stack.append((v, path + [v], visited | {v}))
                        
        return lengths

    # Random search
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Generate a random graph
        # Try to generate 3-regular graphs
        # Or just random edges and check
        
        # Method: Start with a random 3-regular graph (if possible)
        # Or start with a base graph and mutate
        
        # Let's try generating random bipartite graphs (to ensure triangle-free)
        # Split vertices into two sets of 6
        # Random edges between sets
        # Ensure min degree 3
        
        left = list(range(6))
        right = list(range(6, 12))
        
        edges = []
        # Ensure min degree 3
        # Each left node needs 3 neighbors in right
        # Each right node needs 3 neighbors in left
        # This is a 3-regular bipartite graph on 12 vertices.
        
        # Generate a random 3-regular bipartite graph
        # This is equivalent to a random 3-regular graph on 6+6 vertices in the bipartite sense.
        # We can use the configuration model or just random matching.
        
        # Simple method:
        # Create a list of "stubs" for left and right
        # Shuffle and match
        
        left_stubs = [i for i in range(6) for _ in range(3)]
        right_stubs = [i for i in range(6, 12) for _ in range(3)]
        
        random.shuffle(right_stubs)
        
        current_edges = set()
        valid = True
        for u in left_stubs:
            v = right_stubs.pop()
            if u == v: # No loops in bipartite
                valid = False
                break
            e = (min(u, v), max(u, v))
            if e in current_edges:
                # Multi-edge, invalid for simple graph
                valid = False
                break
            current_edges.add(e)
            
        if not valid:
            continue
            
        edges = [list(e) for e in current_edges]
        
        # Check min degree
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if min(deg) < min_deg:
            continue
            
        # Check triangle-free (bipartite ensures this)
        
        # Calculate distinct cycle lengths
        lengths = get_cycle_lengths(edges)
        
        if not lengths:
            # No cycles? Disconnected or tree.
            # Task says "must contain a cycle".
            continue
            
        val = len(lengths)
        if val < best_value:
            best_value = val
            best_edges = edges
            
            if best_value == 1:
                # Optimal found
                break
                
    if best_edges is None:
        # Fallback to seed-like structure
        # Hexagonal prism
        es = set()
        for i in range(6):
            es.add((i, (i+1)%6))
            es.add((i+6, (i+1)%6+6))
            es.add((i, i+6))
        best_edges = [list(e) for e in sorted(es)]
        
    return best_edges

# EVOLVE-BLOCK-END
