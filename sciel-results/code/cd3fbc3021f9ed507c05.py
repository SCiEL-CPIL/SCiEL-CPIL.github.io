"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_graph():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_clique(vertices, adj):
    verts = list(vertices)
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            if verts[j] not in adj[verts[i]]:
                return False
    return True

def _is_independent(vertices, adj):
    verts = list(vertices)
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            if verts[j] in adj[verts[i]]:
                return False
    return True

def _find_maximal_cliques(adj):
    n = len(adj)
    cliques = []
    
    def expand(r, p, x):
        if not p and not x:
            cliques.append(set(r))
            return
        while p or x:
            v = p.pop() if p else x.pop()
            r.add(v)
            expand(r, {u for u in p if u in adj[v]}, {u for u in x if u in adj[v]})
            r.discard(v)
            x.add(v)
    
    expand(set(), set(range(n)), set())
    return cliques

def _find_maximal_independent_sets(adj):
    n = len(adj)
    # Independent sets in G are cliques in complement
    comp_adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u != v and v not in adj[u]:
                comp_adj[u].add(v)
    return _find_maximal_cliques(comp_adj)

def _greedy_cover(universe, sets, adj):
    """Greedy set cover heuristic to get initial upper bound."""
    uncovered = set(universe)
    cover = []
    remaining_sets = list(sets)
    
    while uncovered and remaining_sets:
        best_set = None
        best_gain = -1
        for s in remaining_sets:
            gain = len(s & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_set = s
        if best_gain == 0:
            break
        cover.append(best_set)
        uncovered -= best_set
        remaining_sets.remove(best_set)
    
    if uncovered:
        # Add singletons for uncovered vertices
        for v in uncovered:
            cover.append(frozenset([v]))
    
    return cover

def _lower_bound(uncovered, sets, adj):
    """
    Lower bound: for each uncovered vertex v, count how many sets in 'sets' contain v.
    The minimum over v of this count is a lower bound on additional sets needed.
    Actually, a better lower bound: partition uncovered into disjoint groups where each group
    must be covered by a distinct set. Use a simple packing argument.
    
    Simpler: max over vertices of (1 / max coverage) ... not quite.
    
    Let's use: the number of vertices in uncovered divided by the maximum size of any set intersecting uncovered.
    This is a valid lower bound.
    """
    if not uncovered:
        return 0
    max_inter = 0
    for s in sets:
        inter = len(s & uncovered)
        if inter > max_inter:
            max_inter = inter
    if max_inter == 0:
        return len(uncovered)
    return (len(uncovered) + max_inter - 1) // max_inter

def _branch_and_bound(universe, sets, adj, deadline):
    """
    Branch and bound for minimum set cover.
    universe: set of vertices
    sets: list of frozensets (candidate cliques and independent sets)
    Returns: list of frozensets forming a cover, or None if timeout.
    """
    best_cover = _greedy_cover(universe, sets, adj)
    best_size = len(best_cover)
    
    # Sort sets by size descending for better pruning
    sets_sorted = sorted(sets, key=lambda s: -len(s))
    
    # Remove sets that are subsets of other sets (dominated)
    filtered_sets = []
    for s in sets_sorted:
        dominated = False
        for t in filtered_sets:
            if s <= t:
                dominated = True
                break
        if not dominated:
            filtered_sets.append(s)
    
    sets = filtered_sets
    
    # Convert to list of lists for efficiency
    sets_list = [list(s) for s in sets]
    
    uncovered = set(universe)
    current_cover = []
    
    def dfs(uncovered, current_cover, sets_remaining):
        nonlocal best_cover, best_size
        
        if time.time() > deadline:
            return True  # timeout signal
        
        if not uncovered:
            if len(current_cover) < best_size:
                best_size = len(current_cover)
                best_cover = list(current_cover)
            return False
        
        # Lower bound
        lb = _lower_bound(uncovered, sets_list, adj)
        if len(current_cover) + lb >= best_size:
            return False
        
        # Choose vertex with fewest covering sets
        v_counts = {}
        for s in sets_list:
            for v in s:
                if v in uncovered:
                    v_counts[v] = v_counts.get(v, 0) + 1
        
        if not v_counts:
            return False
        
        v = min(v_counts, key=lambda x: v_counts[x])
        
        # Branch: pick each set containing v
        candidates = [s for s in sets_list if v in s]
        
        for s in candidates:
            new_uncovered = uncovered - set(s)
            current_cover.append(s)
            
            timeout = dfs(new_uncovered, current_cover, sets_list)
            current_cover.pop()
            
            if timeout:
                return True
        
        return False
    
    timeout = dfs(uncovered, [], sets_list)
    
    if best_cover:
        return [frozenset(s) for s in best_cover]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s to be safe within 90s limit
    
    adj = _build_graph()
    n = P['n']
    universe = frozenset(range(n))
    
    # Find all maximal cliques and maximal independent sets
    max_cliques = _find_maximal_cliques(adj)
    max_independent_sets = _find_maximal_independent_sets(adj)
    
    # Also include all singletons
    all_sets = set()
    for c in max_cliques:
        all_sets.add(frozenset(c))
    for i in max_independent_sets:
        all_sets.add(frozenset(i))
    for v in range(n):
        all_sets.add(frozenset([v]))
    
    # Convert to list
    sets_list = list(all_sets)
    
    # Use branch and bound
    result = _branch_and_bound(universe, sets_list, adj, deadline)
    
    if result is None:
        # Fallback: greedy
        result = _greedy_cover(universe, sets_list, adj)
    
    # Convert to list of lists
    parts = [list(s) for s in result]
    
    # Verify validity
    es = {tuple(e) for e in P['edges']}
    for part in parts:
        if not part:
            continue
        verts = part
        hits = [(u, w) in es or (w, u) in es for i, u in enumerate(verts) for w in verts[i+1:]]
        if not (all(hits) or not any(hits)):
            # Invalid partition, fall back to trivial
            parts = [[v] for v in range(n)]
            break
    
    # Ensure each vertex appears exactly once
    all_vertices = [v for part in parts for v in part]
    if sorted(all_vertices) != list(range(n)):
        parts = [[v] for v in range(n)]
    
    return parts

# EVOLVE-BLOCK-END
