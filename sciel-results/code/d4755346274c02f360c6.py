"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _bron_kerbosch_maximal_cliques(adj, n):
    """Find all maximal cliques in a graph using Bron-Kerbosch with pivot."""
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            cliques.append(frozenset(R))
            return
        # Pivot selection
        if P:
            # Choose pivot with max degree in P
            u = max(P, key=lambda v: sum(1 for w in P if adj[v][w]))
            candidates = [v for v in P if not adj[v][u]]
        else:
            candidates = []
        
        for v in candidates:
            R.add(v)
            P_new = P & adj[v]
            X_new = X & adj[v]
            bk(R, P_new, X_new)
            R.remove(v)
            P.remove(v)
            X.add(v)
    
    bk(frozenset(), set(range(n)), frozenset())
    return cliques

def _find_maximal_sets(n, edges):
    """Find all maximal cliques and maximal independent sets."""
    # Build adjacency matrix
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    # Convert to set-based for Bron-Kerbosch
    adj_sets = [set() for _ in range(n)]
    for u, v in edges:
        adj_sets[u].add(v)
        adj_sets[v].add(u)
    
    # Find maximal cliques
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) > 0:
                cliques.append(frozenset(R))
            return
        if P:
            u = max(P, key=lambda v: len(P & adj_sets[v]))
            candidates = [v for v in P if v not in adj_sets[u]]
        else:
            candidates = []
        
        for v in candidates:
            R = R | {v}
            P_new = P & adj_sets[v]
            X_new = X & adj_sets[v]
            bk(R, P_new, X_new)
            R = R - {v}
            P = P - {v}
            X = X | {v}
    
    bk(frozenset(), set(range(n)), frozenset())
    
    # Find maximal independent sets (cliques in complement)
    comp_adj_sets = [set() for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u != v and not adj[u][v]:
                comp_adj_sets[u].add(v)
    
    indep_sets = []
    
    def bk_comp(R, P, X):
        if not P and not X:
            if len(R) > 0:
                indep_sets.append(frozenset(R))
            return
        if P:
            u = max(P, key=lambda v: len(P & comp_adj_sets[v]))
            candidates = [v for v in P if v not in comp_adj_sets[u]]
        else:
            candidates = []
        
        for v in candidates:
            R = R | {v}
            P_new = P & comp_adj_sets[v]
            X_new = X & comp_adj_sets[v]
            bk_comp(R, P_new, X_new)
            R = R - {v}
            P = P - {v}
            X = X | {v}
    
    bk_comp(frozenset(), set(range(n)), frozenset())
    
    # Combine and deduplicate
    all_sets = list(set(cliques) | set(indep_sets))
    
    # Remove subsets: keep only maximal sets
    maximal = []
    all_sets_sorted = sorted(all_sets, key=len, reverse=True)
    for s in all_sets_sorted:
        is_maximal = True
        for t in maximal:
            if s <= t:
                is_maximal = False
                break
        if is_maximal:
            maximal.append(s)
    
    return maximal

def _solve_set_cover(universe, sets, time_limit=55.0):
    """Solve minimum set cover using branch and bound."""
    start_time = time.time()
    n = len(universe)
    universe_set = set(universe)
    
    # Precompute set representations
    set_list = [set(s) for s in sets]
    
    # Filter: only keep sets that are subsets of universe
    set_list = [s for s in set_list if s <= universe_set]
    
    # Remove redundant sets (subsets of other sets)
    set_list_sorted = sorted(set_list, key=len, reverse=True)
    minimal_sets = []
    for s in set_list_sorted:
        is_subset = False
        for t in minimal_sets:
            if s <= t:
                is_subset = True
                break
        if not is_subset:
            minimal_sets.append(s)
    set_list = minimal_sets
    
    if not set_list:
        return None
    
    best_cover = [None]
    best_size = [n]
    
    # Sort sets by size descending for better pruning
    set_list.sort(key=lambda s: len(s), reverse=True)
    
    def dfs(remaining, chosen, depth):
        if time.time() - start_time > time_limit:
            return
        if not remaining:
            if depth < best_size[0]:
                best_size[0] = depth
                best_cover[0] = list(chosen)
            return
        
        # Lower bound: at least ceil(|remaining| / max_set_size)
        max_size = max(len(s) for s in set_list if s & remaining) if any(s & remaining for s in set_list) else 0
        if max_size == 0:
            return
        lb = (len(remaining) + max_size - 1) // max_size
        if depth + lb >= best_size[0]:
            return
        
        # Choose a vertex with fewest covering sets
        vert_counts = {}
        for s in set_list:
            if s & remaining:
                for v in s & remaining:
                    vert_counts[v] = vert_counts.get(v, 0) + 1
        
        if not vert_counts:
            return
        
        # Pick vertex with minimum count
        target_v = min(vert_counts, key=lambda v: vert_counts[v])
        
        # Only consider sets containing target_v
        candidate_sets = [s for s in set_list if target_v in s and s & remaining]
        if not candidate_sets:
            return
        
        # Try each candidate set
        for s in candidate_sets:
            new_remaining = remaining - s
            chosen.append(s)
            dfs(new_remaining, chosen, depth + 1)
            chosen.pop()
            if time.time() - start_time > time_limit:
                return
    
    dfs(universe_set, [], 0)
    
    return best_cover[0]

def construct():
    n = P['n']
    edges = P['edges']
    
    # Quick deadline guard
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Find all maximal cliques and independent sets
    maximal_sets = _find_maximal_sets(n, edges)
    
    # Convert to list of lists
    sets_for_cover = [list(s) for s in maximal_sets]
    
    # Solve set cover
    cover = _solve_set_cover(list(range(n)), sets_for_cover, time_limit=deadline - time.time() - 1.0)
    
    if cover is None:
        # Fallback: greedy set cover
        remaining = set(range(n))
        chosen = []
        while remaining:
            best_set = max(sets_for_cover, key=lambda s: len(set(s) & remaining))
            chosen.append(best_set)
            remaining -= set(best_set)
        cover = chosen
    
    # Convert to proper format: list of lists of ints
    result = []
    for part in cover:
        result.append([int(v) for v in part])
    
    return result
# EVOLVE-BLOCK-END
