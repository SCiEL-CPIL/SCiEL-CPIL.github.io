"""Rewritten program: branch-and-bound maximum independent set on the bad-triple hypergraph."""
# EVOLVE-BLOCK-START
import sys
from itertools import combinations
from math import lcm

def _build_conflict_sets():
    """
    For each vertex v in 1..128, compute the set of other vertices u such that
    there exists a third vertex w (w != u, w != v) with lcm(u,v) == lcm(u,w) == lcm(v,w).
    Actually, we need to be careful: a triple (a,b,c) is bad if lcm(a,b)==lcm(a,c)==lcm(b,c).
    We want to find the maximum subset with NO such triple.
    
    This is a 3-uniform hypergraph independent set problem.
    However, solving maximum independent set in a 3-uniform hypergraph is NP-hard.
    We can use branch and bound.
    
    Alternative approach: Note that if lcm(a,b) == lcm(a,c) == lcm(b,c) = L, then a,b,c all divide L,
    and for any pair, their lcm is L. This means that for any prime power p^k || L, at most one of a,b,c
    can have the full p^k exponent... actually no. Let's think differently.
    
    Key insight: If lcm(a,b) = lcm(a,c) = lcm(b,c) = L, then a|L, b|L, c|L, and for each prime p,
    max(v_p(a), v_p(b)) = v_p(L), max(v_p(a), v_p(c)) = v_p(L), max(v_p(b), v_p(c)) = v_p(L).
    This implies that for each prime p, at least two of a,b,c have v_p equal to v_p(L).
    
    Let's just build the list of bad triples and then use a branching algorithm.
    Given the size (128 vertices), the number of bad triples might be manageable.
    """
    n = 128
    # Precompute LCM for all pairs
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    
    return bad_triples

def _solve_max_independent_set():
    """
    Use branch and bound to find the maximum independent set in the 3-uniform hypergraph
    where vertices are 1..128 and hyperedges are bad triples.
    """
    n = 128
    bad_triples = _build_conflict_sets()
    
    # For each vertex, store the set of hyperedges it belongs to
    vertex_edges = [set() for _ in range(n + 1)]
    for idx, (a, b, c) in enumerate(bad_triples):
        vertex_edges[a].add(idx)
        vertex_edges[b].add(idx)
        vertex_edges[c].add(idx)
    
    # For each hyperedge, store its vertices
    edge_vertices = [(a, b, c) for a, b, c in bad_triples]
    
    # Branch and bound
    # State: current set of included vertices (as a bitmask or set), current set of excluded vertices
    # We want to maximize |included| such that no hyperedge is fully contained in included.
    
    # Use a recursive approach with pruning.
    # To make it efficient, we can use the following strategy:
    # - Maintain a set of "available" vertices (not yet decided)
    # - Maintain a set of "forbidden" hyperedges (hyperedges that have at least one vertex already excluded)
    # - Actually, a hyperedge is "active" if all its vertices are still available or included.
    #   If any vertex is excluded, the hyperedge is no longer a constraint.
    
    # Let's use a different formulation:
    # We process vertices one by one. For each vertex, we decide include or exclude.
    # A hyperedge becomes "satisfied" (no longer a constraint) if any of its vertices is excluded.
    
    # State: 
    #   - index of next vertex to process
    #   - set of active hyperedges (hyperedges where all vertices are still unprocessed or included)
    #   - count of included vertices so far
    
    # For efficiency, represent active hyperedges as a set of indices.
    
    # Precompute for each vertex, the list of hyperedges it participates in
    vertex_hyperedges = [set() for _ in range(n + 1)]
    for idx, (a, b, c) in enumerate(bad_triples):
        vertex_hyperedges[a].add(idx)
        vertex_hyperedges[b].add(idx)
        vertex_hyperedges[c].add(idx)
    
    best_size = 0
    best_set = None
    
    # Use iterative deepening or just branch and bound
    # Since n=128, and the hypergraph is sparse, this should work.
    
    # To speed up, we can use a greedy upper bound:
    # The maximum independent set size is at most n - (minimum vertex cover of hypergraph)
    # But computing min vertex cover is also hard.
    
    # Simpler upper bound: if there are k active hyperedges, and each has 3 vertices,
    # we need to exclude at least ceil(k/3) vertices... not quite.
    
    # Let's just do a straightforward branch and bound with good pruning.
    
    def branch(idx, active_edges, included_count, included_set):
        nonlocal best_size, best_set
        
        # Upper bound: included_count + (n - idx) 
        # This is a weak bound. We can improve it.
        # If active_edges is empty, we can include all remaining vertices.
        if not active_edges:
            total = included_count + (n - idx)
            if total > best_size:
                best_size = total
                best_set = included_set | set(range(idx + 1, n + 1))
            return
        
        # If idx > n, we're done
        if idx > n:
            if included_count > best_size:
                best_size = included_count
                best_set = included_set.copy()
            return
        
        # Pruning: even if we include all remaining vertices, can we beat best?
        if included_count + (n - idx) <= best_size:
            return
        
        # Option 1: Exclude vertex idx
        # When we exclude vertex idx, all hyperedges containing idx are no longer active
        new_active_exclude = active_edges - vertex_hyperedges[idx]
        branch(idx + 1, new_active_exclude, included_count, included_set)
        
        # Option 2: Include vertex idx
        # Only valid if no active hyperedge is fully satisfied by including idx
        # i.e., no active hyperedge has all its other two vertices already included
        # Actually, if we include idx, then any active hyperedge containing idx will
        # become "bad" if the other two vertices are also included.
        # So we need to check: for each active hyperedge containing idx, are the other
        # two vertices already in included_set? If so, we cannot include idx.
        
        can_include = True
        for eidx in active_edges & vertex_hyperedges[idx]:
            a, b, c = edge_vertices[eidx]
            # Check if the other two vertices are in included_set
            others = [v for v in (a, b, c) if v != idx]
            if all(v in included_set for v in others):
                can_include = False
                break
        
        if can_include:
            new_included_set = included_set | {idx}
            # When we include idx, the active hyperedges containing idx are still active
            # (they haven't been "satisfied" yet, but they're now at risk)
            # Actually, the active hyperedges remain the same, but we've added a constraint.
            # Wait, no. Active hyperedges are those where all vertices are not yet excluded.
            # Including a vertex doesn't make a hyperedge inactive; it makes it more dangerous.
            # The hyperedge is still active. It becomes "bad" only when all 3 vertices are included.
            # So new_active_include = active_edges (same as before)
            branch(idx + 1, active_edges, included_count + 1, new_included_set)
    
    # Initial call
    all_edges = set(range(len(bad_triples)))
    branch(1, all_edges, 0, set())
    
    if best_set is None:
        # Fallback: return all vertices (shouldn't happen)
        return list(range(1, n + 1))
    
    return sorted(best_set)

def construct():
    return _solve_max_independent_set()
# EVOLVE-BLOCK-END
