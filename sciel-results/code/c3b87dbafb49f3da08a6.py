"""Program: minimize distinct simple cycle lengths on 12 vertices with min degree 3, girth >= 4."""

import time
import sys
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph on 12 vertices (0..11) that minimizes the number of distinct
    simple cycle lengths.

    The baseline is two disjoint K_{3,3} components, which have cycle lengths {4, 6} (value 2).
    
    We perform a bounded search for any 12-vertex graph (connected or not) that has only
    1 distinct cycle length. If found, we return it. Otherwise, we return the
    K_{3,3} ∪ K_{3,3} construction (value 2).
    """
    
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin: return well before 60s kill
    
    # -----------------------------------------------------------------
    # Helper functions
    # -----------------------------------------------------------------
    
    def make_k33_union():
        edges = []
        # Component 1: K_{3,3} on {0,1,2} x {3,4,5}
        for u in [0, 1, 2]:
            for v in [3, 4, 5]:
                edges.append([u, v])
        # Component 2: K_{3,3} on {6,7,8} x {9,10,11}
        for u in [6, 7, 8]:
            for v in [9, 10, 11]:
                edges.append([u, v])
        return edges

    def validate_and_extract_edges(edge_list, n=12):
        """
        Validate edges and compute distinct cycle lengths.
        Returns (valid, edge_list, cycle_lengths_set).
        """
        # Check basic structure
        if not edge_list:
            return False, None, None
        
        # Remove duplicates and check for loops
        seen = set()
        unique_edges = []
        for e in edge_list:
            if len(e) != 2:
                return False, None, None
            u, v = e
            if u == v:
                return False, None, None
            if not (0 <= u < n and 0 <= v < n):
                return False, None, None
            key = (min(u, v), max(u, v))
            if key in seen:
                return False, None, None
            seen.add(key)
            unique_edges.append(e)
        
        edges = unique_edges
        
        # Check degrees
        deg = defaultdict(int)
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        
        # Ensure all 12 vertices have degree >= 3
        for i in range(n):
            if deg[i] < 3:
                return False, None, None
        
        # Check girth >= 4 (triangle-free)
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    for w in range(v + 1, n):
                        if adj[u][w] and adj[v][w]:
                            return False, None, None
        
        # Must contain a cycle
        # Count edges: if E < V, no cycle (for forest). 
        # A graph with min degree 3 on 12 vertices has at least 18 edges.
        # A forest on 12 vertices has at most 11 edges. So a cycle is guaranteed.
        
        # Count distinct cycle lengths
        def get_cycle_lengths(n, edges):
            adj = defaultdict(list)
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            cycle_lengths = set()
            
            # We will find all simple cycles.
            # To avoid duplicates, we only count cycles where the start vertex is the smallest in the cycle.
            
            for start in range(n):
                # DFS
                stack = [(start, [start])]
                while stack:
                    node, path = stack.pop()
                    for neighbor in adj[node]:
                        if neighbor == start and len(path) > 2:
                            # Found a cycle
                            cycle_lengths.add(len(path))
                        elif neighbor > start and neighbor not in path:
                            # Only visit neighbors > start to avoid redundant cycles
                            # and ensure start is the minimum
                            stack.append((neighbor, path + [neighbor]))
            
            return cycle_lengths

        try:
            cycle_lengths = get_cycle_lengths(n, edges)
            if not cycle_lengths:
                return False, None, None # No cycle found
        except Exception:
            return False, None, None

        return True, edges, cycle_lengths

    # -----------------------------------------------------------------
    # Step 1: Verify baseline
    # -----------------------------------------------------------------
    baseline_edges = make_k33_union()
    valid, baseline_list, baseline_cycles = validate_and_extract_edges(baseline_edges)
    
    if not valid:
        # Should never happen, but return something valid if it does
        return make_k33_union()
    
    best_value = len(baseline_cycles)
    best_edges = baseline_list
    
    # -----------------------------------------------------------------
    # Step 2: Search for value-1 graphs
    # -----------------------------------------------------------------
    # We search for graphs with only one distinct cycle length.
    # This is a hard search, so we use a heuristic approach.
    
    # We can try to construct graphs that are "cycle-uniform" in some sense.
    # One idea: regular bipartite graphs.
    # Another idea: graphs with high symmetry.
    
    # Let's try a small randomized search with a strict time budget.
    # We will generate random graphs with min degree 3 and check their cycle lengths.
    
    import random
    
    # Seed for reproducibility
    random.seed(42)
    
    search_start = time.time()
    search_deadline = search_start + 30.0  # Leave 20s for validation and return
    
    n = 12
    min_degree = 3
    
    # Generate random graphs
    # We'll use a simple approach: start with a random graph and add edges until min degree is satisfied.
    
    attempts = 0
    max_attempts = 10000  # Cap on attempts
    
    while time.time() < search_deadline and attempts < max_attempts:
        attempts += 1
        
        # Generate a random graph
        edges = []
        adj = defaultdict(set)
        
        # Start with a random bipartite graph (to ensure triangle-free)
        # Split vertices into two sets
        left = list(range(0, n, 2))
        right = list(range(1, n, 2))
        
        # Add random edges between left and right
        for u in left:
            for v in right:
                if random.random() < 0.5:
                    edges.append([u, v])
                    adj[u].add(v)
                    adj[v].add(u)
        
        # Ensure min degree
        # Check degrees
        deg = {i: len(adj[i]) for i in range(n)}
        
        # Add edges to satisfy min degree
        for i in range(n):
            while deg[i] < min_degree:
                # Find a vertex to connect to
                candidates = [j for j in range(n) if j != i and j not in adj[i] and deg[j] < 5]
                if not candidates:
                    break
                j = random.choice(candidates)
                edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
                deg[i] += 1
                deg[j] += 1
        
        # Check if we have a valid graph
        valid, edge_list, cycle_lengths = validate_and_extract_edges(edges)
        
        if valid and len(cycle_lengths) < best_value:
            best_value = len(cycle_lengths)
            best_edges = edge_list
            
            if best_value == 1:
                break  # Can't do better than 1
    
    # -----------------------------------------------------------------
    # Step 3: Return the best graph found
    # -----------------------------------------------------------------
    
    # Ensure type safety
    result = [[int(u), int(v)] for u, v in best_edges]
    
    # Deadline guard
    if time.time() > deadline:
        return result
    
    return result

# EVOLVE-BLOCK-END
