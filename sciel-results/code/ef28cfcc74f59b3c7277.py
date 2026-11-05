"""Paley(29) baseline + fast local search to try to reach n=30."""
import time
import random
import numpy as np

# Deadline guard: return best found with safety margin before 150s wall-clock.
START_TIME = time.time()
DEADLINE = 140.0  # 10s safety margin

def _paley29():
    """Paley graph on 29 vertices: edge (i,j) is color 0 if (j-i) mod 29 is a quadratic residue."""
    n = 29
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
    # Build adjacency matrix for color 0 (the Paley graph)
    adj0 = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n in squares:
                adj0[i, j] = 1
                adj0[j, i] = 1
    # Color 1 is the complement
    adj1 = 1 - adj0
    np.fill_diagonal(adj1, 0)
    np.fill_diagonal(adj0, 0)
    return adj0, adj1

def _has_mono_k222(adj):
    """Check if graph 'adj' contains K_{2,2,2} as a subgraph.
    K_{2,2,2} is the complete tripartite graph with parts of size 2.
    Equivalently, it's the octahedron graph: 6 vertices, each vertex connected to 4 others (all except its part-mate).
    
    A subgraph isomorphic to K_{2,2,2} exists iff there are 6 vertices that can be partitioned
    into 3 pairs {a,b}, {c,d}, {e,f} such that every vertex is connected to every vertex not in its pair.
    Equivalently: pick 3 disjoint pairs; all 12 cross-edges must be present.
    
    Faster check: K_{2,2,2} has 6 vertices each of degree 4 within the subgraph.
    We can check: for each set of 3 disjoint edges (a matching of size 3), check if the 12 cross edges exist.
    But that's O(n^6) worst case. For n=30, that's too slow.
    
    Alternative: K_{2,2,2} is the complement of 3 disjoint edges (K2 x 3).
    So adj contains K_{2,2,2} iff there exist 6 vertices where the induced subgraph has all edges except 3 specific non-edges (the matching).
    
    Let's use: for each pair of disjoint edges (a,b) and (c,d), check if there exist (e,f) disjoint from {a,b,c,d}
    such that all cross edges exist. This is O(n^6) still.
    
    Better approach for small n: enumerate all 6-subsets. C(30,6) = 593775. For each, check if it's K_{2,2,2}.
    That's feasible.
    """
    n = len(adj)
    from itertools import combinations
    for combo in combinations(range(n), 6):
        a, b, c, d, e, f = combo
        # Check all 15 pairs: K_{2,2,2} has 12 edges (all except 3 matching edges)
        # The 3 non-edges form a perfect matching on the 6 vertices.
        # There are 15 possible perfect matchings of 6 vertices.
        # K_{2,2,2} corresponds to exactly one structure: 3 pairs, all cross-edges present.
        # So we need: there exists a partition of {a,b,c,d,e,f} into 3 pairs
        # such that all 12 cross-edges are present and the 3 within-pair edges are absent.
        
        # The 15 perfect matchings of 6 labeled vertices:
        # We can enumerate them. But simpler: just check if the induced subgraph on these 6 vertices
        # is exactly K_{2,2,2}. K_{2,2,2} has 12 edges, each vertex degree 4.
        
        # Count edges in induced subgraph
        edges = 0
        for i_idx in range(6):
            for j_idx in range(i_idx + 1, 6):
                if adj[combo[i_idx], combo[j_idx]]:
                    edges += 1
        if edges != 12:
            continue
        # Check each vertex has degree 4 within the 6
        for i_idx in range(6):
            deg = 0
            for j_idx in range(6):
                if i_idx != j_idx and adj[combo[i_idx], combo[j_idx]]:
                    deg += 1
            if deg != 4:
                break
        else:
            return True
    return False

def _is_valid(adj0, adj1):
    """Check neither color class contains K_{2,2,2}."""
    return not _has_mono_k222(adj0) and not _has_mono_k222(adj1)

def _flip_edge(adj0, adj1, i, j):
    """Flip the color of edge (i,j)."""
    adj0[i, j] = 1 - adj0[i, j]
    adj0[j, i] = adj0[i, j]
    adj1[i, j] = 1 - adj1[i, j]
    adj1[j, i] = adj1[i, j]

def _try_add_vertex(adj0, adj1):
    """Try to add a new vertex connected to all existing vertices.
    Returns (new_adj0, new_adj1) or None if no valid extension found quickly."""
    n = len(adj0)
    new_n = n + 1
    # We need to assign colors to edges from new vertex to all existing vertices.
    # 2^n possibilities. For n=29, that's 536 million - too many.
    # Instead, use backtracking with pruning.
    
    # Actually, let's try a smarter approach: for the new vertex, the edges to existing vertices
    # must not create K_{2,2,2} in either color. This is complex.
    # Let's just try random assignments for the new vertex's edges.
    
    best_adj0 = None
    best_adj1 = None
    
    # Try random colorings for the new vertex's edges
    for attempt in range(2000):
        if time.time() > DEADLINE:
            break
        new_colors = np.random.randint(0, 2, size=n)
        # Build new adjacency matrices
        new_adj0 = np.zeros((new_n, new_n), dtype=np.uint8)
        new_adj1 = np.zeros((new_n, new_n), dtype=np.uint8)
        # Copy existing
        new_adj0[:n, :n] = adj0
        new_adj1[:n, :n] = adj1
        # Add new vertex edges
        for i in range(n):
            if new_colors[i] == 0:
                new_adj0[i, new_n - 1] = 1
                new_adj0[new_n - 1, i] = 1
            else:
                new_adj1[i, new_n - 1] = 1
                new_adj1[new_n - 1, i] = 1
        # Check validity
        if _is_valid(new_adj0, new_adj1):
            return new_adj0, new_adj1
    return None

def construct():
    global START_TIME, DEADLINE
    START_TIME = time.time()
    DEADLINE = START_TIME + 140.0
    
    # Start with Paley(29)
    adj0, adj1 = _paley29()
    
    # Verify it's valid
    if not _is_valid(adj0, adj1):
        # Fallback: use Paley(17)
        n17 = 17
        squares17 = set()
        for x in range(1, n17):
            squares17.add((x * x) % n17)
        adj0 = np.zeros((n17, n17), dtype=np.uint8)
        adj1 = np.zeros((n17, n17), dtype=np.uint8)
        for i in range(n17):
            for j in range(i + 1, n17):
                if (j - i) % n17 in squares17:
                    adj0[i, j] = 1
                    adj0[j, i] = 1
                else:
                    adj1[i, j] = 1
                    adj1[j, i] = 1
        n = n17
    else:
        n = len(adj0)
    
    best_n = n
    best_adj0 = adj0.copy()
    best_adj1 = adj1.copy()
    
    # Try to extend to n+1 using random search
    if n < 30:
        result = _try_add_vertex(adj0, adj1)
        if result is not None:
            adj0, adj1 = result
            n = n + 1
            best_n = n
            best_adj0 = adj0.copy()
            best_adj1 = adj1.copy()
    
    # Try local search: flip edges to find a valid n=30 coloring
    if best_n < 30 and time.time() < DEADLINE:
        # Start from best_n coloring and try to add vertex with local search
        # Or: try to find a valid 30-vertex coloring directly using simulated annealing
        # This is very hard. Let's try a few random restarts of adding a vertex.
        
        for restart in range(50):
            if time.time() > DEADLINE - 5:
                break
            # Perturb the base coloring
            adj0_p = best_adj0.copy()
            adj1_p = best_adj1.copy()
            # Flip a few random edges
            for _ in range(5):
                i, j = random.sample(range(best_n), 2)
                _flip_edge(adj0_p, adj1_p, i, j)
            # Try to add a vertex
            if _is_valid(adj0_p, adj1_p):
                result = _try_add_vertex(adj0_p, adj1_p)
                if result is not None:
                    adj0, adj1 = result
                    n = n + 1
                    best_n = n
                    best_adj0 = adj0.copy()
                    best_adj1 = adj1.copy()
                    break
    
    # Convert to list format
    colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            colors.append(int(best_adj0[i, j]))
    
    return {'n': best_n, 'colors': colors}
