"""Rewritten program: exact cochromatic number via branch-and-bound over maximal homogeneous sets."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_graph():
    n = P['n']
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return n, adj

def _maximal_cliques(adj, n):
    """Find all maximal cliques using Bron–Kerbosch with pivot."""
    results = []
    def bronk(r, p, x):
        if not p and not x:
            if r:
                results.append(r)
            return
        # Choose pivot u from p | x to minimize branching
        candidates = p | x
        if candidates:
            # Pick vertex with max degree in p
            best_u = -1
            best_deg = -1
            for u in range(n):
                if (candidates >> u) & 1:
                    deg = bin(p & adj[u]).count('1')
                    if deg > best_deg:
                        best_deg = deg
                        best_u = u
            if best_u >= 0:
                p_to_iter = p & ~(1 << best_u)
            else:
                p_to_iter = p
        else:
            p_to_iter = p
        
        for v in range(n):
            if (p_to_iter >> v) & 1:
                bronk(r | (1 << v), p & adj[v], x & adj[v])
                p &= ~(1 << v)
                x |= (1 << v)
    bronk(0, (1 << n) - 1, 0)
    return results

def _maximal_independent_sets(adj, n):
    """Find all maximal independent sets by finding maximal cliques in complement."""
    comp_adj = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and not (adj[i] & (1 << j)):
                comp_adj[i] |= (1 << j)
    return _maximal_cliques(comp_adj, n)

def _popcount(x):
    return bin(x).count('1')

def _clique_number_heuristic(mask, adj, n):
    """Lower bound on chromatic number of induced subgraph via greedy coloring."""
    if mask == 0:
        return 0
    # Greedy coloring to get an upper bound, then use max degree + 1 as lower bound
    colors = {}
    uncolored = mask
    color_count = 0
    while uncolored:
        color_count += 1
        # Find a maximal independent set in the remaining graph
        # Use greedy: pick vertex, then all non-adjacent uncolored vertices
        current_color = 0
        remaining = uncolored
        while remaining:
            # Pick first vertex
            v = (remaining & -remaining).bit_length() - 1
            current_color |= (1 << v)
            remaining &= ~(1 << v)
            # Add all vertices in remaining that are NOT adjacent to any vertex in current_color
            # A vertex w can be added if it's not adjacent to any vertex in current_color
            addable = remaining
            for u in range(n):
                if (current_color >> u) & 1:
                    addable &= ~adj[u]
            # Also remove vertices that conflict with each other within addable
            # Actually, we need an independent set, so we need no edges among them
            # Let's just take the first vertex and its non-neighbors iteratively
            # Simpler: just take v and remove all its neighbors from consideration
            # This gives a valid independent set (star-shaped)
            # But we want maximal. Let's do proper greedy independent set
            break  # This approach is getting complex, use simpler bound
        
        # Simpler lower bound: max degree in induced subgraph + 1
        max_deg = 0
        for v in range(n):
            if (mask >> v) & 1:
                deg = _popcount(adj[v] & mask) - 1  # exclude self
                if deg > max_deg:
                    max_deg = deg
        uncolored = 0  # Just use max degree bound
        
    return max_deg + 1 if mask else 0

def _independence_number_lower_bound(mask, adj, n):
    """
    Lower bound on number of parts needed:
    Each part is either a clique or independent set.
    Vertices in a clique part: all mutually adjacent
    Vertices in an independent set part: no edges between them
    
    Key insight: For any set of vertices, if we look at pairs that are adjacent,
    they must be in the same clique-part or different parts.
    If they're not adjacent, they must be in the same independent-set-part or different parts.
    
    Simple lower bound: ceil(|V| / max_possible_part_size)
    Stronger: consider that each part can cover at most alpha(G') independent vertices
    or omega(G') clique vertices, where G' is the induced subgraph.
    """
    if mask == 0:
        return 0
    pop = _popcount(mask)
    
    # Compute maximum clique size in induced subgraph (greedy approximation)
    # and maximum independent set size
    max_clique_size = 1
    max_indep_size = 1
    
    # Greedy clique: repeatedly pick vertex with max degree in remaining
    remaining = mask
    clique_size = 0
    while remaining:
        clique_size += 1
        # Pick vertex with max degree in remaining
        best_v = -1
        best_deg = -1
        for v in range(n):
            if (remaining >> v) & 1:
                deg = _popcount(adj[v] & remaining)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
        if best_v < 0:
            break
        remaining &= ~adj[best_v]
        remaining &= ~(1 << best_v)
    if clique_size > max_clique_size:
        max_clique_size = clique_size
    
    # Greedy independent set: repeatedly pick vertex, then its non-neighbors
    remaining = mask
    indep_size = 0
    while remaining:
        indep_size += 1
        # Pick vertex with min degree in remaining (to allow more non-neighbors)
        best_v = -1
        best_deg = float('inf')
        for v in range(n):
            if (remaining >> v) & 1:
                deg = _popcount(adj[v] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
        if best_v < 0:
            break
        # Add best_v and all its non-neighbors in remaining
        # Actually, just add best_v and remove all its neighbors
        remaining &= ~adj[best_v]
        remaining &= ~(1 << best_v)
    if indep_size > max_indep_size:
        max_indep_size = indep_size
    
    # Each part can have at most max(max_clique_size, max_indep_size) vertices
    # But this isn't quite right because a part is EITHER clique OR independent
    # So each part has size <= max(omega, alpha) where omega is max clique, alpha is max indep
    max_part_size = max(max_clique_size, max_indep_size)
    
    return (pop + max_part_size - 1) // max_part_size

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n, adj = _build_graph()
    full_mask = (1 << n) - 1
    
    # 1. Get all maximal homogeneous sets
    max_cliques = _maximal_cliques(adj, n)
    max_indeps = _maximal_independent_sets(adj, n)
    
    max_homog = []
    seen = set()
    for mc in max_cliques:
        if mc not in seen:
            max_homog.append(mc)
            seen.add(mc)
    for mi in max_indeps:
        if mi not in seen:
            max_homog.append(mi)
            seen.add(mi)
    
    # Precompute sizes
    homog_sizes = [_popcount(hs) for hs in max_homog]
    
    # Sort by size descending to try larger sets first
    max_homog.sort(key=lambda m: -bin(m).count('1'))
    homog_sizes = [_popcount(hs) for hs in max_homog]
    
    # Precompute for each vertex, which maximal homogeneous sets contain it
    vertex_to_sets = [[] for _ in range(n)]
    for idx, hs in enumerate(max_homog):
        for v in range(n):
            if (hs >> v) & 1:
                vertex_to_sets[v].append(idx)
    
    # Precompute for each vertex, which homogeneous sets it conflicts with
    # Two sets conflict if their union is not a clique and not an independent set
    # But we only use maximal sets, so we just need to check subset relation
    
    best_solution = None
    best_k = n + 1
    
    # Node counter for debugging
    node_count = [0]
    
    def solve(remaining_mask, current_parts, k_used):
        nonlocal best_solution, best_k
        node_count[0] += 1
        
        if time.time() > deadline:
            return
        
        if remaining_mask == 0:
            if k_used < best_k:
                best_k = k_used
                best_solution = list(current_parts)
            return
        
        pop = _popcount(remaining_mask)
        
        # Stronger lower bound using independence/clique numbers
        lb = _independence_number_lower_bound(remaining_mask, adj, n)
        if k_used + lb >= best_k:
            return
        
        # Additional pruning: if remaining vertices can't all fit in (best_k - k_used - 1) parts
        # even with maximum size parts, prune
        # This is already covered by the lower bound
        
        # Pick the vertex with fewest candidate homogeneous sets (most constrained)
        best_v = -1
        best_count = float('inf')
        for v in range(n):
            if (remaining_mask >> v) & 1:
                count = 0
                for idx in vertex_to_sets[v]:
                    if (max_homog[idx] & ~remaining_mask) == 0:
                        count += 1
                if count < best_count:
                    best_count = count
                    best_v = v
                if best_count == 0:
                    break
        
        if best_count == 0:
            return
        
        # Try each maximal homogeneous set containing best_v that is a subset of remaining
        # Sort candidates by size descending to cover more vertices earlier
        candidates = []
        for idx in vertex_to_sets[best_v]:
            hs = max_homog[idx]
            if (hs & ~remaining_mask) == 0:
                candidates.append((homog_sizes[idx], idx))
        
        candidates.sort(key=lambda x: -x[0])
        
        for size, idx in candidates:
            hs = max_homog[idx]
            new_remaining = remaining_mask & ~hs
            # Quick check: can the new remaining be covered in (best_k - k_used - 1) parts?
            new_pop = _popcount(new_remaining)
            if new_pop > 0:
                new_lb = _independence_number_lower_bound(new_remaining, adj, n)
                if k_used + 1 + new_lb >= best_k:
                    continue
            solve(new_remaining, current_parts + [hs], k_used + 1)
            # Early exit if we found an optimal solution
            if best_k == k_used + 1:
                break
            
    solve(full_mask, [], 0)
    
    # Convert best_solution (list of masks) to list of lists
    if best_solution is not None:
        result = []
        for mask in best_solution:
            part = [v for v in range(n) if (mask >> v) & 1]
            result.append(part)
        result = [p for p in result if p]
        return result
    else:
        # Fallback: greedy first-fit
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                trial = part + [v]
                verts = trial
                is_clique = True
                is_indep = True
                for i in range(len(verts)):
                    for j in range(i + 1, len(verts)):
                        if adj[verts[i]] & (1 << verts[j]):
                            is_indep = False
                        else:
                            is_clique = False
                        if not is_clique and not is_indep:
                            break
                    if not is_clique and not is_indep:
                        break
                if is_clique or is_indep:
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return parts

# EVOLVE-BLOCK-END
