import time
import sys
import random

_START = None

_DEADLINE = None

def _check_validity(edges):
    """Check all constraints: 12 vertices, min degree >= 3, no triangles, has a cycle."""
    n = 12
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        adj[u].add(v)
        adj[v].add(u)
    # Check min degree
    for i in range(n):
        if len(adj[i]) < 3:
            return False
    # Check triangle-free
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w != u and w in adj[u]:
                    return False
    # Check has a cycle (simple graph with min degree 3 on 12 vertices always has a cycle,
    # but let's verify: a tree on 12 vertices has 11 edges and avg degree < 2, so min degree 3 guarantees cycle)
    return True

def _get_distinct_cycle_lengths(edges):
    """
    Computes the set of distinct lengths of all simple cycles in the graph.
    """
    n = 12
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    cycle_lengths = set()
    
    def dfs(start, current, visited):
        if len(current) > 12:
            return
        
        for neighbor in adj[current[-1]]:
            if neighbor == start and len(current) >= 4:
                length = len(current)
                cycle_lengths.add(length)
            elif neighbor > start and neighbor not in visited:
                visited.add(neighbor)
                dfs(start, current + [neighbor], visited)
                visited.remove(neighbor)

    for v in range(n):
        dfs(v, [v], {v})

    return cycle_lengths

def construct():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 85.0  # Safety margin before 90s kill

    def make_k33(offset):
        edges = []
        for a in range(3):
            for b in range(3):
                u = offset + a
                v = offset + 3 + b
                edges.append([int(u), int(v)])
        return edges

    # Fallback: Two disjoint K_{3,3} components -> cycle lengths {4, 6}, count = 2
    raw_edges = make_k33(0) + make_k33(6)
    seen = set()
    fallback = []
    for u, v in raw_edges:
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            fallback.append([key[0], key[1]])
    fallback.sort(key=lambda x: (x[0], x[1]))

    # Try to find a graph with only 1 distinct cycle length
    # Strategy: random search for triangle-free min-degree-3 graphs with single cycle length
    # A connected 4-regular bipartite graph on 12 vertices might work
    # Or: try to find a graph where all cycles have the same length
    
    best_result = fallback
    best_score = 2  # Two K3,3 gives 2 distinct cycle lengths

    # Try specific constructions first
    # 1. 3-regular bipartite graph that is a "cage" - e.g., incidence of Fano plane won't work (not bipartite)
    # 2. Try: two copies of a 6-vertex 3-regular bipartite graph with only 4-cycles
    #    A 3-regular bipartite graph on 6 vertices is K_{3,3}, which has 4-cycles and 6-cycles.
    #    Can we modify?
    
    # Let's try: a 3-regular graph on 12 vertices that is bipartite and has girth 6+
    # The (3,6)-cage is the Heawood graph with 14 vertices - too many.
    # For 12 vertices: try random 3-regular bipartite graphs
    
    rng = random.Random(42)
    found_single = False
    
    # Try many random 3-regular bipartite graphs
    # Bipartition: A = {0..5}, B = {6..11}
    # Each vertex in A needs degree 3, each in B needs degree 3
    # This is a 3-regular bipartite graph on 12 vertices
    
    for trial in range(50000):
        if time.time() > _DEADLINE - 5:
            break
        
        # Generate random 3-regular bipartite graph
        # Use configuration model
        a_side = list(range(6))
        b_side = list(range(6, 12))
        
        # Create stubs
        a_stubs = []
        for v in a_side:
            a_stubs.extend([v] * 3)
        b_stubs = []
        for v in b_side:
            b_stubs.extend([v] * 3)
        
        rng.shuffle(a_stubs)
        rng.shuffle(b_stubs)
        
        edges = []
        valid = True
        seen_pairs = set()
        for u, v in zip(a_stubs, b_stubs):
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in seen_pairs:
                valid = False
                break
            seen_pairs.add(key)
            edges.append([u, v])
        
        if not valid or len(edges) != 18:
            continue
        
        # Check triangle-free (guaranteed if bipartite, but verify)
        adj = [set() for _ in range(12)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        is_triangle_free = True
        for u in range(12):
            for v in adj[u]:
                for w in adj[v]:
                    if w != u and w in adj[u]:
                        is_triangle_free = False
                        break
                if not is_triangle_free:
                    break
            if not is_triangle_free:
                break
        
        if not is_triangle_free:
            continue
        
        # Check min degree (should be 3 by construction)
        min_deg = min(len(adj[i]) for i in range(12))
        if min_deg < 3:
            continue
        
        # Find all simple cycle lengths
        cycle_lengths = set()
        
        def dfs_cyc(start, current, visited):
            if len(current) > 13:
                return
            for neighbor in adj[current[-1]]:
                if neighbor == start and len(current) >= 4:
                    cycle_lengths.add(len(current))
                elif neighbor > start and neighbor not in visited:
                    visited.add(neighbor)
                    dfs_cyc(start, current + [neighbor], visited)
                    visited.remove(neighbor)
        
        for v in range(12):
            dfs_cyc(v, [v], {v})
        
        if len(cycle_lengths) == 1:
            found_single = True
            best_result = sorted([[min(u,v), max(u,v)] for u, v in edges], key=lambda x: (x[0], x[1]))
            best_score = 1
            break
        elif len(cycle_lengths) < best_score:
            best_score = len(cycle_lengths)
            best_result = sorted([[min(u,v), max(u,v)] for u, v in edges], key=lambda x: (x[0], x[1]))
    
    # Also try: a single connected component with specific structure
    # For example, the dodecahedral graph is 3-regular on 20 vertices (too many)
    # The Petersen graph is 3-regular on 10 vertices, girth 5, cycle lengths {5, 6}
    # What about adding 2 vertices to Petersen?
    
    # Try: 4-regular bipartite graph
    # 4-regular on 12 vertices: 24 edges total
    # Bipartite: A={0..5}, B={6..11}, each vertex degree 4
    for trial in range(20000):
        if time.time() > _DEADLINE - 5:
            break
        
        a_side = list(range(6))
        b_side = list(range(6, 12))
        
        a_stubs = []
        for v in a_side:
            a_stubs.extend([v] * 4)
        b_stubs = []
        for v in b_side:
            b_stubs.extend([v] * 4)
        
        rng.shuffle(a_stubs)
        rng.shuffle(b_stubs)
        
        edges = []
        valid = True
        seen_pairs = set()
        for u, v in zip(a_stubs, b_stubs):
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in seen_pairs:
                valid = False
                break
            seen_pairs.add(key)
            edges.append([u, v])
        
        if not valid or len(edges) != 24:
            continue
        
        adj = [set() for _ in range(12)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        is_triangle_free = True
        for u in range(12):
            for v in adj[u]:
                for w in adj[v]:
                    if w != u and w in adj[u]:
                        is_triangle_free = False
                        break
                if not is_triangle_free:
                    break
            if not is_triangle_free:
                break
        
        if not is_triangle_free:
            continue
        
        min_deg = min(len(adj[i]) for i in range(12))
        if min_deg < 3:
            continue
        
        cycle_lengths = set()
        
        def dfs_cyc2(start, current, visited):
            if len(current) > 13:
                return
            for neighbor in adj[current[-1]]:
                if neighbor == start and len(current) >= 4:
                    cycle_lengths.add(len(current))
                elif neighbor > start and neighbor not in visited:
                    visited.add(neighbor)
                    dfs_cyc2(start, current + [neighbor], visited)
                    visited.remove(neighbor)
        
        for v in range(12):
            dfs_cyc2(v, [v], {v})
        
        if len(cycle_lengths) == 1:
            found_single = True
            best_result = sorted([[min(u,v), max(u,v)] for u, v in edges], key=lambda x: (x[0], x[1]))
            best_score = 1
            break
        elif len(cycle_lengths) < best_score:
            best_score = len(cycle_lengths)
            best_result = sorted([[min(u,v), max(u,v)] for u, v in edges], key=lambda x: (x[0], x[1]))
    
    if best_score == 2:
        return fallback
    return best_result
