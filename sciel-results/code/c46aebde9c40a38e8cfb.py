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
    
    # Helper to check if a graph has girth >= 5 (no 3 or 4 cycles)
    def has_girth_at_least_5(n, edges):
        adj_matrix = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj_matrix[u][v] = True
            adj_matrix[v][u] = True
            
        # Check for triangles
        for u in range(n):
            for v in range(u+1, n):
                if adj_matrix[u][v]:
                    for w in range(v+1, n):
                        if adj_matrix[u][w] and adj_matrix[v][w]:
                            return False
        
        # Check for 4-cycles: non-adjacent pairs with >= 2 common neighbors
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

    # Backtracking search for 3-regular graphs with girth >= 5
    # We build edges one at a time, checking constraints incrementally
    def backtrack_search_3reg_girth5(n, deadline):
        """
        Backtracking search for 3-regular triangle-free, no-4-cycle graphs.
        Returns list of edges or None if not found within deadline.
        """
        adj = [set() for _ in range(n)]
        edges = []
        
        def can_add_edge(u, v):
            """Check if adding edge (u,v) maintains girth >= 5 and degree constraints."""
            if u == v:
                return False
            if v in adj[u]:
                return False
            if len(adj[u]) >= 3 or len(adj[v]) >= 3:
                return False
            
            # Check triangle: u-v-w-u where w is common neighbor of u and v
            for w in adj[u]:
                if w in adj[v]:
                    return False
            
            # Check 4-cycle: u-a-b-v-u where a in adj[u], b in adj[v], a-b edge
            for a in adj[u]:
                for b in adj[v]:
                    if a != v and b != u and b in adj[a]:
                        return False
            
            return True
        
        def is_solvable():
            """Quick feasibility check."""
            total_remaining = sum(3 - len(adj[i]) for i in range(n))
            if total_remaining % 2 != 0:
                return False
            return True
        
        # Use recursive backtracking with symmetry breaking
        def backtrack():
            if time.time() > deadline:
                return None
            
            if not is_solvable():
                return None
            
            # Check if all vertices have degree 3
            if all(len(adj[i]) == 3 for i in range(n)):
                return list(edges)
            
            # Find the vertex with minimum remaining degree > 0
            min_remaining = 4
            target_vertex = -1
            for i in range(n):
                rem = 3 - len(adj[i])
                if 0 < rem < min_remaining:
                    min_remaining = rem
                    target_vertex = i
            
            if target_vertex == -1:
                return None
            
            # Try adding edges from target_vertex to other vertices
            # Order neighbors by how constrained they are
            candidates = []
            for j in range(n):
                if j != target_vertex and j not in adj[target_vertex] and len(adj[j]) < 3:
                    candidates.append(j)
            
            # Sort candidates: prefer those with higher degree (more constrained)
            candidates.sort(key=lambda x: -len(adj[x]))
            
            for j in candidates:
                if time.time() > deadline:
                    return None
                
                if can_add_edge(target_vertex, j):
                    # Add edge
                    adj[target_vertex].add(j)
                    adj[j].add(target_vertex)
                    edges.append((target_vertex, j))
                    
                    result = backtrack()
                    if result is not None:
                        return result
                    
                    # Undo
                    adj[target_vertex].remove(j)
                    adj[j].remove(target_vertex)
                    edges.pop()
            
            return None
        
        return backtrack()
    
    # First, try backtracking search for 3-regular girth-5 graph
    found_edges = backtrack_search_3reg_girth5(n, deadline - 10)
    
    if found_edges is not None:
        # Validate and check cycle lengths
        valid, cycle_lengths = validate_graph(n, [[u, v] for u, v in found_edges])
        if valid:
            if len(cycle_lengths) == 1:
                best_value = 1
                best_edges = [[u, v] for u, v in found_edges]
            elif len(cycle_lengths) < best_value:
                best_value = len(cycle_lengths)
                best_edges = [[u, v] for u, v in found_edges]
    
    # If not found, try randomized search with incremental constraint checking
    if best_value > 1:
        random.seed(42)
        tries = 0
        while time.time() < deadline - 5 and tries < 5000:
            tries += 1
            # Random 3-regular graph construction via configuration model
            # but with rejection for triangles and 4-cycles
            items = [i for i in range(n) for _ in range(3)]
            random.shuffle(items)
            
            adj = [set() for _ in range(n)]
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
                # Check triangle
                tri = False
                for w in adj[u]:
                    if w in adj[v]:
                        tri = True
                        break
                if tri:
                    valid_graph = False
                    break
                # Check 4-cycle: u-a-b-v
                four = False
                for a in adj[u]:
                    for b in adj[v]:
                        if a != v and b != u and b in adj[a]:
                            four = True
                            break
                    if four:
                        break
                if four:
                    valid_graph = False
                    break
                
                edge_set.add(key)
                adj[u].add(v)
                adj[v].add(u)
            
            if not valid_graph or len(edge_set) != 18:
                continue
            
            edges = [list(e) for e in edge_set]
            
            # Quick filter: check girth >= 5
            if not has_girth_at_least_5(n, edges):
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
    # Step 3b: Try specific known 3-regular girth-5 candidates
    # -----------------------------------------------------------------
    if best_value > 1 and time.time() < deadline - 10:
        # Try the complete bipartite graph K_{3,3} + K_{3,3} is already baseline
        # Try other structured graphs
        
        # Attempt: Take a 3-regular graph on 12 vertices with girth 5
        # One approach: start from known small cages
        # The (3,5)-cage is the Petersen graph (10 vertices)
        # For 12 vertices, we can try to find 3-regular girth-5 graphs
        
        # Let's try a more systematic search with better heuristics
        random.seed(123)
        for attempt in range(2000):
            if time.time() > deadline - 5:
                break
            
            # Generate random 3-regular graph using perfect matching approach
            # Split into 3 perfect matchings
            vertices = list(range(n))
            matchings = []
            
            success = True
            for m in range(3):
                # Find a perfect matching avoiding previous matchings
                used = set()
                for prev in matchings:
                    for u, v in prev:
                        used.add((u, v))
                        used.add((v, u))
                
                # Greedy matching
                remaining = list(range(n))
                matching = []
                random.shuffle(remaining)
                
                i = 0
                while i < len(remaining):
                    u = remaining[i]
                    # Find a partner for u
                    found = False
                    for j in range(i+1, len(remaining)):
                        v = remaining[j]
                        if (u, v) not in used and (v, u) not in used:
                            matching.append((u, v))
                            used.add((u, v))
                            used.add((v, u))
                            remaining[j] = remaining[-1]
                            remaining.pop()
                            found = True
                            break
                    if not found:
                        success = False
                        break
                    # Don't increment i, we removed an element
                if not success:
                    break
            
            if not success or len(matchings) != 3:
                continue
            
            # Combine matchings into edges
            all_edges = []
            edge_set = set()
            for matching in matchings:
                for u, v in matching:
                    key = (min(u, v), max(u, v))
                    if key in edge_set:
                        success = False
                        break
                    edge_set.add(key)
                    all_edges.append(key)
            
            if not success or len(all_edges) != 18:
                continue
            
            edges = [list(e) for e in all_edges]
            
            # Check girth >= 5
            if not has_girth_at_least_5(n, edges):
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
