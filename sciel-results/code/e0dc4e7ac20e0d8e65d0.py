"""Improved program: degree-sequence targeting + fast incremental validity + local search."""
# EVOLVE-BLOCK-START
import time
import random

def _check_validity(adj, deg, n, edges):
    """Check no 3-edge path s-u-v-t (s!=t) has deg(s)==deg(t)."""
    for (u, v) in edges:
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u:
                    continue
                if s == t:
                    continue
                if deg[s] == deg[t]:
                    return False
    return True

def _try_add_edge(adj, deg, n, edges, edge_set, u, v):
    """Try to add edge (u,v). Returns True if added, False if invalid."""
    # Quick check: adding edge changes deg[u] and deg[v]
    # We need to check all 3-paths involving the new edge AND paths affected by degree changes
    # Simpler: just do full check on the new edge list
    # But that's O(E * deg^2). For n=15, E<=105, deg<=14, so O(105*196) ~ 20k ops per check.
    # With 105 edges to try, that's ~2M ops. Fast enough.
    
    new_edges = edges + [[u, v]]
    new_adj = [set() for _ in range(n)]
    new_deg = [0] * n
    for (a, b) in new_edges:
        new_adj[a].add(b)
        new_adj[b].add(a)
        new_deg[a] += 1
        new_deg[b] += 1
    
    if _check_validity(new_adj, new_deg, n, new_edges):
        # Commit
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        edges.append([u, v])
        edge_set.add((u, v))
        return True
    return False

def _try_remove_edge(adj, deg, n, edges, edge_set, u, v):
    """Remove edge (u,v). Returns the edge that was removed."""
    adj[u].discard(v)
    adj[v].discard(u)
    deg[u] -= 1
    deg[v] -= 1
    edges.remove([u, v])
    edge_set.discard((u, v))
    return [u, v]

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # Safety margin before 90s kill
    n = 15
    
    # Strategy: 
    # 1. Start with a known valid dense graph
    # 2. Use local search to try to add more edges
    # 3. If stuck, try removing one edge to free up degree conflicts, then add different edges
    
    # Phase 1: Greedy construction with smart ordering
    # Key insight: vertices with equal degrees cannot be at distance 2 or 3 in a way that forms a 3-path
    # Actually the constraint is: for any edge (u,v), and any s in N(u)\{v}, t in N(v)\{u}, s!=t, we need deg(s) != deg(t)
    
    # Let's try a different approach: build a graph where most vertices have distinct degrees
    # with at most a few pairs of equal-degree vertices that are "far apart"
    
    # Start empty
    adj = [set() for _ in range(n)]
    deg = [0] * n
    edges = []
    edge_set = set()
    
    # Candidate edges sorted by a heuristic: prioritize edges that connect vertices
    # which will end up with very different degrees
    # Use a random-ish but deterministic order with multiple restarts
    
    best_edges = []
    best_count = 0
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
            
        # Shuffle candidate edges with a seed for reproducibility
        rng = random.Random(restart * 7919)
        candidates = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(candidates)
        
        # Bias: put edges involving high-index vertices first (they can have higher degrees)
        candidates.sort(key=lambda e: (-(e[0] + e[1]), rng.random()))
        
        adj = [set() for _ in range(n)]
        deg = [0] * n
        edges = []
        edge_set = set()
        
        for (u, v) in candidates:
            if time.time() > deadline - 2:
                break
            if (u, v) in edge_set:
                continue
            if _try_add_edge(adj, deg, n, edges, edge_set, u, v):
                pass
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = [list(e) for e in edges]
            
        # Local search: try to add remaining edges
        for (u, v) in candidates:
            if time.time() > deadline - 2:
                break
            if (u, v) in edge_set:
                continue
            if _try_add_edge(adj, deg, n, edges, edge_set, u, v):
                pass
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = [list(e) for e in edges]
            
        # Try 1-swap: remove one edge, try to add two others
        if time.time() > deadline - 2:
            break
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(len(edges)):
                if time.time() > deadline - 2:
                    break
                e = edges[i]
                u, v = e[0], e[1]
                # Remove this edge
                _try_remove_edge(adj, deg, n, edges, edge_set, u, v)
                
                # Try to add any missing edge
                for (a, b) in candidates:
                    if time.time() > deadline - 2:
                        break
                    if (a, b) in edge_set:
                        continue
                    if _try_add_edge(adj, deg, n, edges, edge_set, a, b):
                        # Try to add one more
                        for (c, d) in candidates:
                            if time.time() > deadline - 2:
                                break
                            if (c, d) in edge_set:
                                continue
                            if _try_add_edge(adj, deg, n, edges, edge_set, c, d):
                                if len(edges) > best_count:
                                    best_count = len(edges)
                                    best_edges = [list(e) for e in edges]
                                improved = True
                                # Keep the new state, break inner loops
                                break
                        break
                    # If we couldn't add, restore the removed edge
                    _try_add_edge(adj, deg, n, edges, edge_set, u, v)
                    break
                
                # Check if we broke out cleanly
                if (u, v) not in edge_set:
                    # The edge was removed and not restored; check if we added enough
                    if len(edges) <= len(best_edges):
                        # Restore
                        _try_add_edge(adj, deg, n, edges, edge_set, u, v)
    
    # Final validation
    if best_edges:
        # Verify
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for (u, v) in best_edges:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        if not _check_validity(adj, deg, n, best_edges):
            # Fallback to a simple valid graph
            best_edges = _fallback_valid(n)
    
    # Ensure proper types
    result = [[int(u), int(v)] for (u, v) in best_edges]
    return result

def _fallback_valid(n):
    """Return a simple valid graph as fallback."""
    # Star graph: vertex 0 connected to all others. deg(0)=14, deg(others)=1
    # Check: any 3-path s-u-v-t. In a star, the only 3-paths are s-0-t where s,t are leaves.
    # But that's a 2-edge path (s-0-t). A 3-edge path would be s-0-x-y where x is a leaf... 
    # Wait, in a star, from 0 you go to a leaf, but leaves only connect to 0. So no 3-edge paths exist.
    # So star is valid.
    edges = []
    for v in range(1, n):
        edges.append([0, v])
    return edges
# EVOLVE-BLOCK-END
