"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def _paley_graph(q):
    """Return (n, edges) for the Paley graph on q vertices (q prime, q ≡ 1 mod 4)."""
    residues = {(x * x) % q for x in range(1, q)}
    residues.discard(0)
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    return q, edges


def _build_adjacency(edges, n):
    """Build adjacency list from edge list."""
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _has_k5(adj, n):
    """Check if graph has a K5 using adjacency lists. Returns True if found."""
    # Find all K4s and check if any vertex connects to all 4
    for i in range(n):
        neighbors_i = list(adj[i])
        for j in neighbors_i:
            if j <= i:
                continue
            common_ij = adj[i] & adj[j]
            for k in common_ij:
                if k <= j:
                    continue
                common_ijk = common_ij & adj[k]
                for l in common_ijk:
                    if l <= k:
                        continue
                    # Found K4: (i, j, k, l)
                    common_ijkl = common_ijk & adj[l]
                    if common_ijkl:
                        return True
    return False


def _has_ind5(adj, n):
    """Check if graph has an independent set of size 5. Returns True if found."""
    # Build complement adjacency
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                comp_adj[i].add(j)
                comp_adj[j].add(i)
    
    # Find K5 in complement
    return _has_k5(comp_adj, n)


def _is_valid(edges, n):
    """Check if graph is (5,5)-free."""
    if n < 5:
        return True
    adj = _build_adjacency(edges, n)
    if _has_k5(adj, n):
        return False
    if _has_ind5(adj, n):
        return False
    return True


def _count_k4s_in_subset(adj, subset, n):
    """Count K4s in the subgraph induced by subset. Early exit if found."""
    subset_set = set(subset)
    for i in subset:
        neighbors_i = adj[i] & subset_set
        for j in neighbors_i:
            if j <= i:
                continue
            common_ij = neighbors_i & adj[j]
            for k in common_ij:
                if k <= j:
                    continue
                common_ijk = common_ij & adj[k]
                for l in common_ijk:
                    if l > k:
                        return True
    return False


def _count_ind4s_in_subset(adj, subset, n):
    """Check if there's an independent 4-set in subset. Early exit if found."""
    subset_set = set(subset)
    # Non-edges in subset
    for i in subset:
        non_neighbors_i = subset_set - adj[i] - {i}
        for j in non_neighbors_i:
            if j <= i:
                continue
            common_non_ij = non_neighbors_i & (subset_set - adj[j])
            for k in common_non_ij:
                if k <= j:
                    continue
                common_non_ijk = common_non_ij & (subset_set - adj[k])
                for l in common_non_ijk:
                    if l > k:
                        return True
    return False


def _greedy_extend(base_edges, base_n, target_n, seed=42, deadline=None):
    """
    Try to extend a (5,5)-free graph on base_n vertices to target_n vertices
    by greedily adding vertices with carefully chosen edges.
    Returns (n, edges) or None if fails.
    """
    rng = random.Random(seed)
    edges = list(base_edges)
    n = base_n
    
    # Build adjacency from base
    adj = _build_adjacency(edges, n)
    
    for new_v in range(base_n, target_n):
        if deadline and time.time() > deadline:
            return None
        
        # Try multiple random edge assignments with different strategies
        found = False
        for attempt in range(300):
            # Strategy: use structured patterns based on Paley-like properties
            # or random with bias
            if attempt < 100:
                # Random with 0.5 bias
                new_adj = set()
                for v in range(n):
                    if rng.random() < 0.5:
                        new_adj.add(v)
            elif attempt < 200:
                # Try to mimic Paley structure: connect based on quadratic residues
                # This is a heuristic - not guaranteed to work but might find valid configs
                new_adj = set()
                for v in range(n):
                    # Use a pseudo-random but structured pattern
                    if ((v * (new_v + 1)) % n) in {x*x % n for x in range(1, n//2+1)}:
                        new_adj.add(v)
                    elif rng.random() < 0.3:
                        new_adj.add(v)
            else:
                # More aggressive: try to balance density
                new_adj = set()
                density = 0.5 + rng.gauss(0, 0.1)
                density = max(0.3, min(0.7, density))
                for v in range(n):
                    if rng.random() < density:
                        new_adj.add(v)
            
            # Quick feasibility check: count potential violations
            # K4 check among new_adj
            if len(new_adj) >= 4:
                if _count_k4s_in_subset(adj, list(new_adj), n):
                    continue
            
            # Independent 4-set check among non-new_adj
            non_adj = set(range(n)) - new_adj
            if len(non_adj) >= 4:
                if _count_ind4s_in_subset(adj, list(non_adj), n):
                    continue
            
            # Valid! Add the vertex
            for v in new_adj:
                edges.append((min(v, new_v), max(v, new_v)))
                adj[v].add(new_v)
            adj.append(set(new_adj))
            n += 1
            found = True
            break
        
        if not found:
            return None
    
    return n, edges


def construct():
    """
    Construct a (5,5)-free graph. Start from Paley(37) and try to extend it.
    Returns (n, edges).
    """
    deadline = time.time() + 270  # 300s limit, keep 30s margin
    
    # Start with Paley(37)
    base_n, base_edges = _paley_graph(37)
    
    # Verify base is valid
    adj_base = _build_adjacency(base_edges, base_n)
    if _has_k5(adj_base, base_n) or _has_ind5(adj_base, base_n):
        # Shouldn't happen, but fallback
        return base_n, base_edges
    
    best_n = base_n
    best_edges = base_edges
    best_adj = adj_base
    
    # Try to extend with different seeds and strategies
    # First, try a few quick greedy extensions
    for seed in range(200):
        if time.time() > deadline - 60:
            break
        result = _greedy_extend(base_edges, base_n, base_n + 1, seed=seed, deadline=deadline - 60)
        if result is not None:
            n, edges = result
            adj_new = _build_adjacency(edges, n)
            if not _has_k5(adj_new, n) and not _has_ind5(adj_new, n):
                if n > best_n:
                    best_n = n
                    best_edges = edges
                    best_adj = adj_new
    
    # Try to extend further from the best found so far
    for target_offset in range(1, 8):
        if time.time() > deadline - 30:
            break
        target_n = best_n + target_offset
        for seed in range(100):
            if time.time() > deadline - 30:
                break
            result = _greedy_extend(best_edges, best_n, target_n, seed=seed, deadline=deadline - 30)
            if result is not None:
                n, edges = result
                adj_new = _build_adjacency(edges, n)
                if not _has_k5(adj_new, n) and not _has_ind5(adj_new, n):
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                        best_adj = adj_new
                    break
    
    return best_n, best_edges

# EVOLVE-BLOCK-END
