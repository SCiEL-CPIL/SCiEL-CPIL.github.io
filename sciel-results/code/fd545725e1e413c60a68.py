"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 12, 'minimum_degree': 3, 'girth': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = P['n']
    min_deg = P['minimum_degree']
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # We'll use a greedy approach to build a graph with:
    # 1. Minimum degree >= 3
    # 2. Girth >= 4 (no triangles)
    # 3. Minimize distinct cycle lengths
    
    # Strategy: Try to construct graphs that have few distinct cycle lengths.
    # The theoretical minimum is 2 (e.g., only 4-cycles and 6-cycles).
    
    # First, try a known good structure: a bipartite graph with specific properties.
    # For n=12, we can try to create a graph where all cycles have length 4 or 6.
    
    # Approach: Use a greedy algorithm with backtracking to minimize distinct cycle lengths.
    
    # Helper function to check if adding an edge creates a triangle
    def creates_triangle(adj, u, v):
        # Check if u and v share a common neighbor
        for w in adj[u]:
            if w in adj[v]:
                return True
        return False
    
    # Helper function to find all simple cycles in the graph
    def find_all_cycles(adj, n):
        cycles = []
        
        # Use DFS to find all simple cycles
        # This is exponential in worst case, but for n=12 it should be manageable
        # We use a more efficient approach: find cycles by looking at paths
        
        # For small graphs, we can use a simple approach
        # Find all cycles of length 4, 5, 6, 7, 8
        
        def dfs(start, current, visited, path):
            if len(path) >= 4 and current == start:
                cycle = tuple(sorted(path))
                if cycle not in seen_cycles:
                    seen_cycles.add(cycle)
                    cycles.append(path[:])
                return
            
            for neighbor in adj[current]:
                if neighbor == start and len(path) >= 4:
                    cycle = tuple(sorted(path + [start]))
                    if cycle not in seen_cycles:
                        seen_cycles.add(cycle)
                        cycles.append(path + [start])
                elif neighbor not in visited and neighbor > start:
                    # Only explore neighbors with higher index to avoid duplicates
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(start, neighbor, visited, path)
                    path.pop()
                    visited.remove(neighbor)
        
        seen_cycles = set()
        for start in range(n):
            visited = {start}
            dfs(start, start, visited, [start])
        
        return cycles
    
    # Helper function to get distinct cycle lengths
    def get_distinct_cycle_lengths(adj, n):
        cycles = find_all_cycles(adj, n)
        lengths = set()
        for cycle in cycles:
            lengths.add(len(cycle))
        return lengths
    
    # Helper function to check if graph is valid
    def is_valid(adj, n):
        # Check minimum degree
        for i in range(n):
            if len(adj[i]) < min_deg:
                return False
        # Check girth >= 4 (no triangles)
        for i in range(n):
            for j in adj[i]:
                for k in adj[j]:
                    if k == i:
                        return False
                    if k > j and k in adj[i]:
                        return False
        # Check that there is at least one cycle
        cycles = find_all_cycles(adj, n)
        if not cycles:
            return False
        return True
    
    # Greedy construction: start with empty graph, add edges greedily
    # to satisfy minimum degree while avoiding triangles
    
    best_score = float('inf')
    best_edges = None
    
    # Try different strategies
    
    # Strategy 1: Complete bipartite K_{6,6}
    # This has degree 6 for all vertices, girth 4, and cycles of length 4 and 6
    adj = [[] for _ in range(n)]
    for i in range(6):
        for j in range(6, 12):
            adj[i].append(j)
            adj[j].append(i)
    
    if is_valid(adj, n):
        cycle_lengths = get_distinct_cycle_lengths(adj, n)
        score = len(cycle_lengths)
        if score < best_score:
            best_score = score
            best_edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        best_edges.append([i, j])
    
    # Strategy 2: Try to find a graph with only one cycle length (unlikely but worth checking)
    # This is very constrained, so we'll try a few specific structures
    
    # Strategy 3: Greedy with backtracking for better solutions
    # Use a time-limited search
    
    import random
    random.seed(42)
    
    # Try random restarts with greedy construction
    attempts = 0
    while time.time() < deadline and attempts < 50:
        attempts += 1
        
        # Start with a random set of edges
        adj = [[] for _ in range(n)]
        edge_set = set()
        
        # Add edges to satisfy minimum degree
        for _ in range(20):  # Max iterations
            # Find vertices with degree < min_deg
            low_deg_vertices = [i for i in range(n) if len(adj[i]) < min_deg]
            if not low_deg_vertices:
                break
            
            # Try to add an edge from a low-degree vertex to another vertex
            progress = False
            random.shuffle(low_deg_vertices)
            for u in low_deg_vertices:
                candidates = [v for v in range(n) if v != u and v not in adj[u] and not creates_triangle(adj, u, v)]
                random.shuffle(candidates)
                for v in candidates:
                    if len(adj[v]) < min_deg or random.random() < 0.3:
                        # Add edge
                        adj[u].append(v)
                        adj[v].append(u)
                        edge_set.add((min(u,v), max(u,v)))
                        progress = True
                        break
                if progress:
                    break
            
            if not progress:
                break
        
        # Check if we achieved minimum degree
        if all(len(adj[i]) >= min_deg for i in range(n)):
            if is_valid(adj, n):
                cycle_lengths = get_distinct_cycle_lengths(adj, n)
                score = len(cycle_lengths)
                if score < best_score:
                    best_score = score
                    best_edges = [list(e) for e in sorted(edge_set)]
        
        if time.time() > deadline - 10:
            break
    
    # Return the best result found
    if best_edges is None:
        # Fallback: return the K_{6,6} structure
        edges = []
        for i in range(6):
            for j in range(6, 12):
                edges.append([i, j])
        return edges
    else:
        return best_edges

# EVOLVE-BLOCK-END
