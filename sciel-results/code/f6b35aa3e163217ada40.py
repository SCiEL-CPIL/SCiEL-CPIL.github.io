import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    n = 12
    best_edges = None
    best_distinct = float('inf')
    
    # Try to find a bipartite 3-regular graph with fewest distinct cycle lengths
    # Part A: 0..5, Part B: 6..11
    
    def count_distinct_cycle_lengths(adj):
        """Count distinct lengths of all simple cycles in the graph."""
        n = len(adj)
        cycle_lengths = set()
        
        def find_cycles(start, current, visited, path, length):
            nonlocal cycle_lengths
            for nxt in adj[current]:
                if nxt == start and length >= 3:
                    # Found a cycle - only count if start is the smallest vertex
                    # to avoid duplicates
                    if start == min(path):
                        cycle_lengths.add(length + 1)
                elif nxt not in visited and nxt > start:
                    # Only visit vertices > start to avoid duplicate cycles
                    visited.add(nxt)
                    path.append(nxt)
                    find_cycles(start, nxt, visited, path, length + 1)
                    path.pop()
                    visited.remove(nxt)
        
        for start in range(n):
            visited = {start}
            path = [start]
            find_cycles(start, start, visited, path, 0)
        
        return len(cycle_lengths) if cycle_lengths else 0
    
    def is_triangle_free(adj):
        """Check if graph is triangle-free."""
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                for w in adj[u]:
                    if w == v:
                        continue
                    if w in adj[v]:
                        return False
        return True
    
    def min_degree_ok(adj):
        return all(len(adj[i]) >= 3 for i in range(n))
    
    # Generate all 3-regular bipartite graphs on 6+6 vertices
    # Each vertex in A connects to 3 vertices in B
    # This is equivalent to 3-regular bipartite graphs
    
    A = list(range(6))
    B = list(range(6, 12))
    
    # Represent as: for each a in A, a set of 3 neighbors in B
    # We'll backtrack to generate all such graphs
    
    def backtrack(a_idx, adj, deg_b):
        nonlocal best_edges, best_distinct, deadline
        
        if time.time() > deadline:
            return
        
        if a_idx == 6:
            # Check if all B vertices have degree 3
            if all(deg_b[b] == 3 for b in B):
                # Build adjacency list
                adj_list = [set() for _ in range(n)]
                for a in A:
                    for b in adj[a]:
                        adj_list[a].add(b)
                        adj_list[b].add(a)
                
                if not is_triangle_free(adj_list):
                    return
                
                if min_degree_ok(adj_list):
                    # Count distinct cycle lengths
                    distinct = count_distinct_cycle_lengths(adj_list)
                    if distinct < best_distinct:
                        best_distinct = distinct
                        # Build edge list
                        edges = []
                        for a in A:
                            for b in adj[a]:
                                edges.append([a, b])
                        best_edges = edges
                        if best_distinct == 1:
                            return  # Can't do better
            
            return
        
        # Try all combinations of 3 neighbors for A[a_idx]
        a = A[a_idx]
        for combo in itertools.combinations(B, 3):
            # Check if this assignment is valid (deg_b constraints)
            valid = True
            for b in combo:
                if deg_b[b] >= 3:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Also check: can remaining A vertices still satisfy their degree?
            # For each b in B, remaining capacity = 3 - deg_b[b]
            # Remaining A vertices = 6 - a_idx - 1 (after this one)
            # Each needs 3 neighbors, total slots = (6 - a_idx - 1) * 3
            # Total remaining capacity in B = sum(3 - deg_b[b] - (1 if b in combo else 0))
            remaining_a = 5 - a_idx
            remaining_slots = remaining_a * 3
            remaining_cap = sum(3 - deg_b[b] - (1 if b in combo else 0) for b in B)
            if remaining_slots > remaining_cap:
                continue
            
            # Assign
            adj[a] = list(combo)
            for b in combo:
                deg_b[b] += 1
            
            backtrack(a_idx + 1, adj, deg_b)
            
            # Undo
            for b in combo:
                deg_b[b] -= 1
            adj[a] = []
    
    adj = {a: [] for a in A}
    deg_b = {b: 0 for b in B}
    backtrack(0, adj, deg_b)
    
    # If we found a good solution, return it
    if best_edges is not None:
        return best_edges
    
    # Fallback: return the seed graph (prism)
    es = set()
    for i in range(6):
        for u, v in [(i, (i+1) % 6), (i+6, (i+1) % 6 + 6), (i, i+6)]:
            es.add(tuple(sorted((u, v))))
    return [list(e) for e in sorted(es)]

# EVOLVE-BLOCK-END
