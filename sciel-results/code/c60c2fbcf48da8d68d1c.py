"""Clique partitioning via most-constrained-first edge selection and greedy clique growth."""
import time
from random import Random


def _random_graph(n, p, seed):
    """Generate random graph using the same RNG as the verifier."""
    rng = Random(seed)
    adj = [0] * n  # bitmask adjacency
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    return adj


def _expand_clique(rem, start, n, strategy="constrained"):
    """
    Expand a clique starting from `start` in the graph defined by `rem`.
    strategy: 'constrained' - pick vertex with fewest common neighbors (most constrained),
              'max' - pick vertex with most common neighbors
    """
    clique = [start]
    cand = rem[start]
    
    while cand:
        best = None
        best_score = -1 if strategy == "max" else float('inf')
        temp = cand
        while temp:
            low = temp & (-temp)
            u = low.bit_length() - 1
            common = (rem[u] & cand).bit_count()
            if strategy == "constrained":
                # Most constrained: fewest common neighbors
                if common < best_score:
                    best_score = common
                    best = u
                if common == 0:
                    break
            else:
                # Max: most common neighbors
                if common > best_score:
                    best_score = common
                    best = u
            temp ^= low
        if best is None:
            break
        clique.append(best)
        cand &= rem[best]
    
    return clique if len(clique) >= 2 else None


def _solve_instance(adj, n, deadline):
    """Find clique partition for a single graph using most-constrained-first strategy."""
    rem = [adj[i] for i in range(n)]
    remaining_edges = sum(rem[i].bit_count() for i in range(n)) // 2
    cliques = []
    
    if remaining_edges == 0:
        return cliques
    
    while remaining_edges > 0:
        if time.time() > deadline:
            break
        
        # Find the most constrained edge: edge (u,v) where u and v have fewest common neighbors
        # This edge is hardest to extend into a large clique
        best_edge = None
        best_common = float('inf')
        
        # Iterate through all vertices with remaining edges
        for u in range(n):
            if rem[u] == 0:
                continue
            temp = rem[u]
            while temp:
                low = temp & (-temp)
                v = low.bit_length() - 1
                if v > u:  # Only consider each edge once
                    common = (rem[u] & rem[v]).bit_count()
                    if common < best_common:
                        best_common = common
                        best_edge = (u, v)
                    if common == 0:
                        break
                temp ^= low
        
        if best_edge is None:
            break
        
        u, v = best_edge
        
        if best_common == 0:
            # No common neighbors, just add this edge
            cliques.append([u, v])
            rem[u] &= ~(1 << v)
            rem[v] &= ~(1 << u)
            remaining_edges -= 1
            continue
        
        # Build a clique starting from the edge (u, v)
        # Start with u, then try to include v and others
        clique = [u]
        cand = rem[u] & rem[v]  # Common neighbors of u and v
        
        # Add v to clique
        clique.append(v)
        cand &= rem[v]  # Now cand is common neighbors of both u and v
        
        # Continue expanding with most-constrained strategy
        while cand:
            best = None
            best_common = float('inf')
            temp = cand
            while temp:
                low = temp & (-temp)
                w = low.bit_length() - 1
                common = (rem[w] & cand).bit_count()
                if common < best_common:
                    best_common = common
                    best = w
                if common == 0:
                    break
                temp ^= low
            if best is None:
                break
            clique.append(best)
            cand &= rem[best]
        
        if len(clique) < 2:
            # Fallback: just add the edge
            cliques.append([u, v])
            rem[u] &= ~(1 << v)
            rem[v] &= ~(1 << u)
            remaining_edges -= 1
            continue
        
        # Add the clique and remove all its edges
        m = len(clique)
        cliques.append(clique)
        for i in range(m):
            for j in range(i + 1, m):
                a, b = clique[i], clique[j]
                rem[a] &= ~(1 << b)
                rem[b] &= ~(1 << a)
        remaining_edges -= m * (m - 1) // 2
    
    # Handle any remaining edges (if deadline hit)
    if remaining_edges > 0:
        for i in range(n):
            if time.time() > deadline:
                break
            temp = rem[i]
            while temp:
                if time.time() > deadline:
                    break
                low = temp & (-temp)
                j = low.bit_length() - 1
                cliques.append([i, j])
                rem[i] &= ~(1 << j)
                rem[j] &= ~(1 << i)
                temp &= temp - 1
                rem[i] &= ~(1 << j)
    
    # Local improvement: try to merge cliques that form a larger clique together
    # This is a simple pass - for each pair of cliques, check if their union is a clique
    improved = True
    max_iterations = 10
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        if time.time() > deadline:
            break
        
        for i in range(len(cliques)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(cliques)):
                if time.time() > deadline:
                    break
                
                # Check if union of cliques[i] and cliques[j] is a clique
                # Build the combined set
                combined = set(cliques[i]) | set(cliques[j])
                if len(combined) < 2:
                    continue
                
                # Check all pairs in combined
                is_clique = True
                for a in combined:
                    for b in combined:
                        if a == b:
                            continue
                        if not (adj[a] >> b) & 1:
                            is_clique = False
                            break
                    if not is_clique:
                        break
                
                if is_clique:
                    # Merge: replace cliques[i] and cliques[j] with the union
                    cliques[i] = list(combined)
                    cliques.pop(j)
                    improved = True
                    break
            if improved:
                break
    
    return cliques


def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            results.append([])
            continue
        
        adj = _random_graph(n, p, seed)
        cliques = _solve_instance(adj, n, deadline)
        results.append(cliques)
    
    return results

