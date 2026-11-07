"""Program: minimize distinct simple cycle lengths on 12 vertices with min degree 3, girth >= 4."""

import time
import sys
import itertools
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph on 12 vertices (0..11) that minimizes the number of distinct
    simple cycle lengths.

    The baseline is two disjoint K_{3,3} components, which have cycle lengths {4, 6} (value 2).
    We search for any 12-vertex graph (connected or not) that has only 1 distinct
    cycle length. If found, we return it. Otherwise, we return the
    K_{3,3} ∪ K_{3,3} construction (value 2).
    """
    
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin: return well before 90s kill

    # -----------------------------------------------------------------
    # Step 1: Define the baseline graph (K_{3,3} ∪ K_{3,3})
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

    def get_cycle_lengths(n, edges):
        """Find all distinct simple cycle lengths in the graph."""
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        cycle_lengths = set()
        
        for start in range(n):
            # DFS to find cycles where start is the minimum vertex in the cycle
            stack = [(start, [start])]
            while stack:
                node, path = stack.pop()
                for neighbor in adj[node]:
                    if neighbor == start and len(path) > 2:
                        cycle_lengths.add(len(path))
                    elif neighbor > start and neighbor not in path:
                        stack.append((neighbor, path + [neighbor]))
        
        return cycle_lengths

    def check_triangle_free(n, edges):
        """Check if graph is triangle-free."""
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    for w in range(v + 1, n):
                        if adj[u][w] and adj[v][w]:
                            return False
        return True

    def validate_graph(n, edges):
        """
        Validate graph: no loops, no duplicates, min degree 3, girth >= 4, has cycle.
        Returns (valid, cycle_lengths_set).
        """
        # Check for loops and duplicates
        seen = set()
        for u, v in edges:
            if u == v:
                return False, None
            key = (min(u, v), max(u, v))
            if key in seen:
                return False, None
            seen.add(key)
        
        # Check degrees
        deg = defaultdict(int)
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        
        for i in range(n):
            if deg[i] < 3:
                return False, None
        
        # Check triangle-free
        if not check_triangle_free(n, edges):
            return False, None
        
        # Find cycle lengths
        try:
            cycle_lengths = get_cycle_lengths(n, edges)
            if not cycle_lengths:
                return False, None
        except Exception:
            return False, None
        
        return True, cycle_lengths

    # -----------------------------------------------------------------
    # Step 2: Verify baseline
    # -----------------------------------------------------------------
    baseline_edges = make_k33_union()
    valid, baseline_cycles = validate_graph(12, baseline_edges)
    
    if not valid:
        # Fallback: hardcoded
        baseline_cycles = {4, 6}

    best_value = len(baseline_cycles)
    best_edges = baseline_edges

    # -----------------------------------------------------------------
    # Step 3: Search for value 1
    # -----------------------------------------------------------------
    n = 12
    
    # Targeted search for 3-regular graphs with girth 5
    # Value 1 requires all cycles to be the same length.
    # For 3-regular graphs on 12 vertices, girth 5 is the max possible (Moore bound for girth 6 is 14).
    # If a graph has girth 5 and no 6-cycles, it might have value 1 (only 5-cycles).
    
    # Helper to check if a graph has girth 5 (no 3 or 4 cycles)
    def has_girth_5(n, edges):
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check for triangles (3-cycles)
        for u in range(n):
            for i in range(len(adj[u])):
                for j in range(i+1, len(adj[u])):
                    if adj[u][i] in adj[adj[u][j]]:
                        return False
        
        # Check for 4-cycles
        # A 4-cycle exists if there are two vertices with two common neighbors
        # Or more simply: BFS from each node up to depth 2
        for start in range(n):
            visited = {start: 0}
            queue = [(start, 0)]
            while queue:
                node, depth = queue.pop(0)
                if depth >= 2:
                    continue
                for neighbor in adj[node]:
                    if neighbor == start and depth == 2:
                        return False # Found 4-cycle
                    if neighbor not in visited:
                        visited[neighbor] = depth + 1
                        queue.append((neighbor, depth + 1))
                    elif visited[neighbor] == depth + 1 and neighbor != start:
                        # This check is tricky, simpler to just count common neighbors
                        pass
        
        # More efficient 4-cycle check:
        # For each pair of nodes (u, v), count common neighbors.
        # If any pair has >= 2 common neighbors, there is a 4-cycle?
        # No, if u and v are not adjacent, and have 2 common neighbors, it's a 4-cycle.
        # If u and v are adjacent, and have 1 common neighbor, it's a triangle.
        # We already ruled out triangles.
        # So we just need to check if any non-adjacent pair has >= 2 common neighbors.
        adj_matrix = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj_matrix[u][v] = True
            adj_matrix[v][u] = True
            
        for u in range(n):
            for v in range(u+1, n):
                if not adj_matrix[u][v]:
                    common = 0
                    for w in range(n):
                        if adj_matrix[u][w] and adj_matrix[v][w]:
                            common += 1
                            if common >= 2:
                                return False
        return True

    # Search for 3-regular graphs with girth 5
    random.seed(42)
    tries = 0
    while time.time() < deadline - 5 and tries < 10000:
        tries += 1
        items = [i for i in range(n) for _ in range(3)]
        random.shuffle(items)
        
        edge_set = set()
        valid_graph = True
        for i in range(0, len(items), 2):
            u, v = items[i], items[i+1]
            if u == v:
                valid_graph = False
                break
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid_graph = False
                break
            edge_set.add(key)
        
        if not valid_graph or len(edge_set) != 18:
            continue
        
        edges = [list(e) for e in edge_set]
        
        # Quick filter: check girth 5
        if not has_girth_5(n, edges):
            continue
            
        # Full validation
        valid, cycle_lengths = validate_graph(n, edges)
        if valid:
            if len(cycle_lengths) == 1:
                best_value = 1
                best_edges = edges
                break
            elif len(cycle_lengths) < best_value:
                best_value = len(cycle_lengths)
                best_edges = edges
    
    # -----------------------------------------------------------------
    # Step 4: Return best found
    # -----------------------------------------------------------------
    result_edges = best_edges
    
    # Final type safety check
    final_result = []
    for e in result_edges:
        u, v = e
        if isinstance(u, bool) or isinstance(v, bool):
            raise ValueError("Bool detected")
        final_result.append([int(u), int(v)])
        
    # Check for duplicates/loops in final result
    seen = set()
    for e in final_result:
        u, v = e
        if u == v:
            raise ValueError("Loop detected")
        key = (min(u, v), max(u, v))
        if key in seen:
            raise ValueError("Duplicate edge detected")
        seen.add(key)
        
    return final_result

# EVOLVE-BLOCK-END
