"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = time.time() + 145.0  # safety margin before 150s kill

def _has_k4(n, adj):
    """Check if graph with adjacency list `adj` (list of sets) has a K4."""
    # Check all 4-tuples via edges
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                continue
            # common neighbors of i and j
            common = adj[i] & adj[j]
            for k in common:
                if k <= j:
                    continue
                # check if k and some l > k form K4 with i,j
                common2 = adj[i] & adj[j] & adj[k]
                for l in common2:
                    if l <= k:
                        continue
                    # i,j,k,l all mutually adjacent
                    return True
    return False

def _has_indep6(n, adj):
    """Check if graph has an independent set of size 6."""
    # Use branch and bound
    # complement graph: non-edges become edges
    # We need a clique of size 6 in the complement
    # Build complement adjacency
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                comp_adj[i].add(j)
                comp_adj[j].add(i)
    
    # Find clique of size 6 in complement via backtracking
    def _find_clique(candidates, target, size):
        if size == 0:
            return True
        if not candidates:
            return False
        if len(candidates) < size:
            return False
        # Try each vertex in candidates
        for v in list(candidates):
            if size == 1:
                return True
            new_cands = candidates & comp_adj[v]
            if len(new_cands) >= size - 1:
                if _find_clique(new_cands, target, size - 1):
                    return True
            candidates.discard(v)
        return False
    
    return _find_clique(set(range(n)), n, 6)

def _is_valid(n, adj):
    return not _has_k4(n, adj) and not _has_indep6(n, adj)

def _paley17():
    """Return Paley graph on 17 vertices as adjacency list."""
    squares = {x * x % 17 for x in range(1, 17)}
    n = 17
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % 17 in squares:
                adj[i].add(j)
                adj[j].add(i)
    return adj

def _try_add_vertex(n, adj, new_vertex, edges_to_add):
    """Try adding a new vertex with given edges to existing vertices.
    Returns (success, new_adj) or (False, None)."""
    new_n = n + 1
    new_adj = [set(s) for s in adj]
    new_adj.append(set(edges_to_add))
    for v in edges_to_add:
        new_adj[v].add(new_vertex)
    
    # Check K4: any K4 must include the new vertex or be in old graph
    # Old graph is K4-free, so check K4s involving new vertex
    # K4 with new vertex: need triangle among neighbors of new vertex
    neighbors = set(edges_to_add)
    for i in neighbors:
        for j in neighbors:
            if i >= j:
                continue
            if j in new_adj[i]:
                # i, j are adjacent; check common neighbors in `neighbors`
                common = new_adj[i] & new_adj[j] & neighbors
                for k in common:
                    if k > j:
                        return False, None
    # Check independent set of 6: any independent set of 6 either avoids new vertex (old graph OK)
    # or includes new vertex, meaning 5 independent vertices among non-neighbors of new vertex
    non_neighbors = set(range(n)) - neighbors
    if len(non_neighbors) >= 5:
        # Check if non_neighbors contains independent set of size 5 in original graph
        # i.e., clique of size 5 in complement restricted to non_neighbors
        comp_restricted = [set() for _ in range(n)]
        for i in non_neighbors:
            for j in non_neighbors:
                if i < j and j not in adj[i]:
                    comp_restricted[i].add(j)
                    comp_restricted[j].add(i)
        
        def _find_clique5(candidates, size):
            if size == 0:
                return True
            if not candidates:
                return False
            if len(candidates) < size:
                return False
            for v in list(candidates):
                if size == 1:
                    return True
                new_cands = candidates & comp_restricted[v]
                if len(new_cands) >= size - 1:
                    if _find_clique5(new_cands, size - 1):
                        return True
                candidates.discard(v)
            return False
        
        if _find_clique5(set(non_neighbors), 5):
            return False, None
    
    return True, new_adj

def _grow_from_paley(adj):
    """Try to grow the Paley graph by adding vertices one at a time."""
    n = len(adj)
    best_n = n
    best_adj = [set(s) for s in adj]
    
    while time.time() < _DEADLINE - 5:
        # Try to add a vertex
        added = False
        # Try different edge patterns for the new vertex
        # The new vertex's neighborhood must be a set S such that:
        # 1. S contains no triangle (else K4)
        # 2. V\S contains no independent set of size 5 (else indep 6 with new vertex)
        # Try random subsets
        
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > _DEADLINE - 5:
                break
            # Random subset of existing vertices
            # Bias toward medium-sized neighborhoods
            size = random.randint(max(1, n // 4), min(n - 1, 3 * n // 4))
            candidates = random.sample(range(n), size)
            edges_to_add = set(candidates)
            
            success, new_adj = _try_add_vertex(n, adj, n, edges_to_add)
            if success:
                adj = new_adj
                n += 1
                best_n = n
                best_adj = [set(s) for s in adj]
                added = True
                break
        
        if not added:
            # Try smaller or larger neighborhoods
            for size in range(1, n):
                if time.time() > _DEADLINE - 5:
                    break
                for _ in range(20):
                    candidates = random.sample(range(n), size)
                    edges_to_add = set(candidates)
                    success, new_adj = _try_add_vertex(n, adj, n, edges_to_add)
                    if success:
                        adj = new_adj
                        n += 1
                        best_n = n
                        best_adj = [set(s) for s in adj]
                        added = True
                        break
                if added:
                    break
        
        if not added:
            break
    
    return best_n, best_adj

def _to_dict(n, adj):
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
    return {'n': n, 'edges': edges}

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 145.0
    
    random.seed(42)
    
    # Start from Paley 17
    adj = _paley17()
    
    # Verify initial graph
    assert _is_valid(17, adj), "Paley 17 should be valid"
    
    best_result = _to_dict(17, adj)
    
    # Try to grow
    n, adj = _grow_from_paley(adj)
    
    if n > 17:
        result = _to_dict(n, adj)
        # Final verification
        if _is_valid(n, adj):
            best_result = result
    
    return best_result
# EVOLVE-BLOCK-END
