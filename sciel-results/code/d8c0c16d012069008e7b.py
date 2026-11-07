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
    
    def has_only_4_cycles(n, adj):
        """Check if all simple cycles have length exactly 4."""
        # First check: if there's a 6-cycle, return False
        # A 6-cycle exists if there are two vertices at distance 3 from each other
        # through different intermediate paths of length 3
        
        # More direct: enumerate all 6-cycles
        # A 6-cycle is a sequence v0-v1-v2-v3-v4-v5-v0 where all consecutive are edges
        # and v0 < v1 < ... < v5 (to avoid duplicates, use canonical form)
        
        for v0 in range(n):
            for v1 in adj[v0]:
                if v1 <= v0:
                    continue
                for v2 in adj[v1]:
                    if v2 == v0 or v2 <= v0:
                        continue
                    for v3 in adj[v2]:
                        if v3 == v1 or v3 == v0 or v3 <= v0:
                            continue
                        for v4 in adj[v3]:
                            if v4 == v2 or v4 == v1 or v4 == v0 or v4 <= v0:
                                continue
                            for v5 in adj[v4]:
                                if v5 == v3 or v5 == v2 or v5 == v1 or v5 == v0 or v5 <= v0:
                                    continue
                                if v5 in adj[v0]:
                                    # Found a 6-cycle
                                    return False
        return True

    def count_distinct_cycle_lengths_fast(n, adj):
        """Count distinct cycle lengths using BFS from each vertex."""
        cycle_lengths = set()
        
        for start in range(n):
            # BFS to find shortest path back to start through each neighbor
            # This finds cycles of various lengths
            visited = [[False] * n for _ in range(n)]  # visited[step][vertex]
            for step in range(2, n + 1):  # cycle length = step + 1
                if step + 1 > n:
                    break
            
            # Better: DFS with depth limit
            stack = [(start, [start], 0)]
            while stack:
                node, path, depth = stack.pop()
                if depth >= n:
                    continue
                for neighbor in adj[node]:
                    if neighbor == start and depth >= 3:
                        cycle_lengths.add(depth + 1)
                    elif neighbor > start and neighbor not in path and depth < n - 1:
                        stack.append((neighbor, path + [neighbor], depth + 1))
        
        return cycle_lengths

    # Strategy A: Search for bipartite graphs with all 4-cycles only
    # A graph where every cycle is a 4-cycle is called a "4-cycle graph" or has girth exactly 4
    # and no cycles of length >= 6.
    
    # For bipartite graphs with parts of size a and b (a+b=12), min degree 3:
    # - K_{3,3} ∪ K_{3,3}: has 4 and 6 cycles
    # - K_{4,4} minus edges: need to eliminate 6-cycles
    # - K_{5,5} minus edges: more complex
    
    # Key insight: A 6-cycle in a bipartite graph exists iff there are two vertices 
    # in the same part with at least 3 common neighbors (forming a 6-cycle through them)
    # OR more precisely, a 6-cycle v0-v1-v2-v3-v4-v5-v0 with v0,v2,v4 in one part and v1,v3,v5 in other.
    
    # For all cycles to be 4-cycles, we need: no 6-cycles, no 8-cycles, etc.
    # No 6-cycle means: for any two vertices u,w in the same part, they share at most 2 common neighbors.
    # But that's not sufficient - we also need no longer cycles.
    
    # Let's try: bipartite graph with parts A (size 6) and B (size 6), min degree 3
    # such that no 6-cycles exist.
    
    # Try all bipartite graphs with 6+6 split, min degree 3, no 6-cycles
    part_a = list(range(6))
    part_b = list(range(6, 12))
    
    # Number of edges needed: at least 6*3 = 18 edges (min degree 3)
    # Total possible edges: 36
    
    # Search over edge subsets of size 18-24 (to keep it tractable)
    # This is still too large for exhaustive search, so use random search
    
    random.seed(42)
    found_value_1 = False
    
    # Try random bipartite graphs with 6+6 split
    for trial in range(20000):
        if time.time() > deadline - 15:
            break
        
        # Random bipartite graph with min degree 3
        adj_matrix = [[False] * 12 for _ in range(12)]
        valid = True
        
        # Start with empty, add random edges ensuring min degree 3
        edge_set = set()
        
        # First ensure each vertex has degree >= 3
        # Randomly assign 3 edges to each vertex in part A
        for u in part_a:
            neighbors = random.sample(part_b, 3)
            for v in neighbors:
                edge_set.add((u, v))
        
        # Now part B vertices may have low degree, add more edges
        for v in part_b:
            current_deg = sum(1 for u in part_a if (u, v) in edge_set)
            if current_deg < 3:
                missing = [u for u in part_a if (u, v) not in edge_set]
                if len(missing) < 3 - current_deg:
                    valid = False
                    break
                for u in random.sample(missing, 3 - current_deg):
                    edge_set.add((u, v))
        
        if not valid:
            continue
        
        # Verify min degree 3
        deg = [0] * 12
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
        if any(d < 3 for d in deg):
            continue
        
        # Build adjacency
        adj = defaultdict(list)
        for u, v in edge_set:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check for 6-cycles using the fast method
        if has_only_4_cycles(12, adj):
            # Verify no triangles (should be automatic for bipartite)
            # Count distinct cycle lengths
            cycle_lengths = count_distinct_cycle_lengths_fast(12, adj)
            if len(cycle_lengths) == 1:
                best_value = 1
                best_edges = [list(e) for e in edge_set]
                found_value_1 = True
                break
    
    if not found_value_1:
        # Try non-bipartite random graphs (3-regular)
        for _ in range(10000):
            if time.time() > deadline - 10:
                break
            
            # Generate random 3-regular graph using configuration model
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
            valid, cycle_lengths = validate_graph(n, edges)
            if valid and len(cycle_lengths) == 1:
                best_value = 1
                best_edges = edges
                break
    
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
