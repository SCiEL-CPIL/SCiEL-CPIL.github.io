import time
import random
import itertools
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit
    
    best_edges = None
    best_score = float('inf')
    
    n = 12
    # Try to find a graph with few distinct cycle lengths
    # We know K3,3 U K3,3 has {4, 6} -> count 2.
    # Seed has {4, 6, 12} -> count 3.
    # So K3,3 U K3,3 is already an improvement.
    
    # Construct K3,3 U K3,3
    edges = []
    for i in range(3):
        for j in range(3):
            edges.append([i, j+3])
    for i in range(3):
        for j in range(3):
            edges.append([i+6, j+9])
            
    # Verify K3,3 U K3,3
    # Degrees: 3.
    # Cycles: 4, 6.
    # Count: 2.
    
    # Can we do better? Count 1?
    # Search for a graph with count 1.
    
    def get_cycle_lengths(edges):
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        lengths = set()
        # Find all simple cycles up to length 12
        # This is expensive, but n=12 is small.
        # We can use DFS from each vertex, keeping track of path.
        # To avoid duplicates, only start from min vertex in cycle?
        # Or just collect all lengths.
        
        visited_cycles = set()
        
        def dfs(start, current, path, visited):
            if len(path) > 12:
                return
            if current == start and len(path) >= 4:
                # Check if this cycle is new
                # Normalize cycle: rotate to start with min vertex
                min_idx = min(range(len(path)), key=lambda i: path[i])
                norm = tuple(path[min_idx:] + path[:min_idx])
                if norm not in visited_cycles:
                    visited_cycles.add(norm)
                    lengths.add(len(path)-1)
                return
            for neighbor in adj[current]:
                if neighbor == start and len(path) >= 4:
                    dfs(start, start, path + [start], visited)
                elif neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(start, neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)
        
        for i in range(n):
            dfs(i, i, [i], {i})
            
        return lengths

    current_lengths = get_cycle_lengths(edges)
    current_count = len(current_lengths)
    
    if current_count < best_score:
        best_score = current_count
        best_edges = edges
        
    # Try to find a graph with count 1
    # Random search
    attempts = 0
    while time.time() < deadline and attempts < 500:
        attempts += 1
        # Generate random bipartite graph
        # Split vertices into A (0-5) and B (6-11)
        # Ensure min degree 3
        # Try to avoid 6-cycles? Hard to enforce directly.
        
        # Just random edges
        edges = []
        for i in range(6):
            for j in range(6):
                if random.random() < 0.5:
                    edges.append([i, j+6])
        
        # Check min degree
        deg = [0]*n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            
        if any(d < 3 for d in deg):
            continue
            
        # Check triangle-free (bipartite is always triangle-free)
        
        lengths = get_cycle_lengths(edges)
        count = len(lengths)
        
        if count < best_score:
            best_score = count
            best_edges = edges
            if count == 1:
                break

    return [list(e) for e in best_edges]
