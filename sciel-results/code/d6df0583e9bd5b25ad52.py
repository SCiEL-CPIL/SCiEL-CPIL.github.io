"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def _deadline_guard(start, limit=115.0):
    return (time.time() - start) < limit

def _generate_instance(n, p, seed):
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _clique_partition(edges, n, time_limit=110.0):
    """
    Partition edges into cliques (each clique has >= 2 vertices, covers every edge exactly once).
    We use a greedy approach: repeatedly find a maximal clique among remaining edges,
    remove all edges of that clique, and continue.
    
    To minimize number of cliques, we want each clique to be as large as possible.
    Strategy: For each remaining edge (u,v), try to extend {u,v} into the largest clique
    possible using only remaining edges. Pick the edge that gives the largest such clique.
    """
    start = time.time()
    # Build adjacency from edges
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # remaining edges as a set of frozensets for O(1) lookup
    remaining = set()
    for u, v in edges:
        remaining.add((u, v) if u < v else (v, u))
    
    cliques = []
    
    while remaining:
        if not _deadline_guard(start, time_limit):
            break
        
        # Try to find a large clique from remaining edges
        # Strategy: pick an edge, then greedily extend it
        best_clique = None
        best_size = 1
        
        # Sample edges to find a good starting edge (for speed)
        edge_list = list(remaining)
        # Try a few candidate edges
        candidates = edge_list[:min(50, len(edge_list))]
        
        for (u, v) in candidates:
            # Try to build a clique starting from {u, v}
            clique = {u, v}
            # Candidates must be adjacent to both u and v in remaining graph
            candidates_v = adj[u] & adj[v]
            # Also must have edges to u and v still remaining
            candidates_v = {x for x in candidates_v if (min(u,x), max(u,x)) in remaining and (min(v,x), max(v,x)) in remaining}
            
            # Greedily add vertices
            changed = True
            while changed:
                changed = False
                for x in list(candidates_v):
                    if x in clique:
                        continue
                    # Check if x is adjacent to all in clique via remaining edges
                    ok = True
                    for y in clique:
                        if (min(x,y), max(x,y)) not in remaining:
                            ok = False
                            break
                    if ok:
                        clique.add(x)
                        candidates_v.discard(x)
                        changed = True
            
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique
        
        if best_clique is None or len(best_clique) < 2:
            # Fallback: just take any single edge
            e = next(iter(remaining))
            best_clique = {e[0], e[1]}
        
        # Remove all edges within best_clique from remaining
        clique_list = list(best_clique)
        for i in range(len(clique_list)):
            for j in range(i + 1, len(clique_list)):
                a, b = clique_list[i], clique_list[j]
                key = (a, b) if a < b else (b, a)
                remaining.discard(key)
        
        cliques.append(clique_list)
    
    return cliques

def construct():
    start = time.time()
    
    def instance(n, p, seed):
        edges = _generate_instance(n, p, seed)
        cliques = _clique_partition(edges, n, time_limit=50.0)
        return cliques
    
    result1 = instance(32, 0.6, 601)
    result2 = instance(64, 0.55, 602)
    
    return [result1, result2]
# EVOLVE-BLOCK-END
