"""
Dense triangle-free graph on 16 vertices with chromatic number >= 4.
Strategy:
  1. Start from the Grötzsch graph (11 vertices, 20 edges) + 5 isolated vertices.
  2. Greedy addition of vertices 11..15, connecting each to a large independent set.
  3. Intensive local search (edge additions/removals) to maximize edges while
     maintaining triangle-free and non-3-colorable properties.
"""

import time
import itertools
import random

DEADLINE_MARGIN = 5.0  # seconds before the 150s kill


def _is_triangle_free(adj, n):
    """Check if graph with adjacency list is triangle-free."""
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        return False
    return True


def _is_3_colorable(adj, n):
    """Backtracking 3-colorability check with forward checking."""
    color = [-1] * n
    # Order vertices by degree descending for better pruning
    order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        # Determine available colors
        used = set()
        for nb in adj[v]:
            if color[nb] != -1:
                used.add(color[nb])
        for c in range(3):
            if c not in used:
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
        return False
    
    return backtrack(0)


def _is_valid(adj, n):
    """Check triangle-free and not 3-colorable."""
    if not _is_triangle_free(adj, n):
        return False
    if _is_3_colorable(adj, n):
        return False
    return True


def _count_edges(adj, n):
    return sum(len(adj[i]) for i in range(n)) // 2


def _add_edge(adj, u, v, n):
    """Try adding edge (u,v). Returns True if valid."""
    if u == v or v in adj[u]:
        return False
    # Check triangle-free: no common neighbor
    common = set(adj[u]) & set(adj[v])
    if common:
        return False
    # Add edge
    adj[u].append(v)
    adj[v].append(u)
    # Check 3-colorability
    if _is_3_colorable(adj, n):
        # Remove edge
        adj[u].remove(v)
        adj[v].remove(u)
        return False
    return True


def _remove_edge(adj, u, v):
    """Remove edge (u,v)."""
    if v in adj[u]:
        adj[u].remove(v)
        adj[v].remove(u)


def _find_independent_sets(adj, n):
    """Find all maximal independent sets (small n, so brute force is ok)."""
    # For n=16, we can use a simple approach
    # Actually, let's just find independent sets of size >= 4
    # A vertex set S is independent if no two vertices in S are adjacent
    independent_sets = []
    for r in range(4, n + 1):
        for S in itertools.combinations(range(n), r):
            S_set = set(S)
            is_independent = True
            for u in S:
                for v in S:
                    if u != v and v in adj[u]:
                        is_independent = False
                        break
                if not is_independent:
                    break
            if is_independent:
                independent_sets.append(S)
    return independent_sets


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    n = 16
    random.seed(42)
    
    # Build Grötzsch graph on vertices 0..10
    adj = [set() for _ in range(n)]
    # Pentagon: 0-1-2-3-4-0
    pentagon = [(0,1), (1,2), (2,3), (3,4), (4,0)]
    for u, v in pentagon:
        adj[u].add(v)
        adj[v].add(u)
    # Mycielski: vertices 5..9 connect to non-neighbors of corresponding pentagon vertex
    # Vertex 5+i connects to pentagon vertices not adjacent to i
    for i in range(5):
        pent_adj = set(adj[i])  # neighbors of i in pentagon
        for j in range(5):
            if j != i and j not in pent_adj:
                adj[5+i].add(j)
                adj[j].add(5+i)
    # Vertex 10 connects to all of 5..9
    for i in range(5):
        adj[10].add(5+i)
        adj[5+i].add(10)
    
    # Convert to lists for consistency
    adj = [list(s) for s in adj]
    
    # Verify initial graph
    assert _is_triangle_free(adj, n), "Grötzsch graph should be triangle-free"
    assert not _is_3_colorable(adj, n), "Grötzsch graph should not be 3-colorable"
    
    initial_edges = _count_edges(adj, n)
    
    # Phase 1: Greedy addition of vertices 11..15
    for new_v in range(11, 16):
        if time.time() > deadline:
            break
        # Find the largest independent set in the current graph (vertices 0..new_v-1)
        # that we can connect new_v to while keeping triangle-free
        # new_v can connect to any independent set S (no two vertices in S are adjacent)
        # because new_v has no edges yet, so adding edges from new_v to S won't create triangles
        # as long as S is independent.
        
        # Find all independent sets in the subgraph on vertices 0..new_v-1
        best_set = []
        for r in range(5, 0, -1):  # Try larger sets first
            found = False
            for S in itertools.combinations(range(new_v), r):
                S_set = set(S)
                is_independent = True
                for u in S:
                    for v in S:
                        if u != v and v in adj[u]:
                            is_independent = False
                            break
                    if not is_independent:
                        break
                if is_independent:
                    best_set = list(S)
                    found = True
                    break
            if found:
                break
        
        # Connect new_v to best_set
        for v in best_set:
            adj[new_v].append(v)
            adj[v].append(new_v)
    
    # Convert back to lists (already lists)
    current_edges = _count_edges(adj, n)
    
    # Phase 2: Local search - try adding edges
    # Generate all possible non-edges
    all_pairs = list(itertools.combinations(range(n), 2))
    
    best_adj = [list(s) for s in adj]
    best_edges = current_edges
    
    # Iterative local search
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        improved = False
        
        # Try adding each non-edge
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            if v in adj[u]:
                continue
            # Check triangle-free
            common = set(adj[u]) & set(adj[v])
            if common:
                continue
            # Try adding
            adj[u].append(v)
            adj[v].append(u)
            if not _is_3_colorable(adj, n):
                # Valid! Keep the edge
                improved = True
                current_edges += 1
            else:
                # Remove
                adj[u].remove(v)
                adj[v].remove(u)
        
        # Try removing edges and re-adding different ones (simulated annealing-like)
        if not improved and iteration % 10 == 0:
            # Try removing a random edge and then re-optimizing
            existing_edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
            if existing_edges:
                u, v = random.choice(existing_edges)
                _remove_edge(adj, u, v)
                current_edges -= 1
                # Now try to add edges
                for u2, v2 in all_pairs:
                    if time.time() > deadline:
                        break
                    if v2 in adj[u2]:
                        continue
                    common = set(adj[u2]) & set(adj[v2])
                    if common:
                        continue
                    adj[u2].append(v2)
                    adj[v2].append(u2)
                    if not _is_3_colorable(adj, n):
                        current_edges += 1
                    else:
                        adj[u2].remove(v2)
                        adj[v2].remove(u2)
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [list(s) for s in adj]
        
        if not improved:
            # No improvement, break after a few stagnant iterations
            if iteration > 50:
                break
    
    # Use best found
    adj = best_adj
    current_edges = best_edges
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    
    # Final verification
    assert len(edges) == current_edges
    assert _is_triangle_free(adj, n), "Result should be triangle-free"
    assert not _is_3_colorable(adj, n), "Result should not be 3-colorable"
    
    return edges
