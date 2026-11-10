"""K3,3-free graph construction on 20 vertices: randomized init + greedy repair + local search."""
import time
import random
import itertools

N = 20
DEADLINE = 140  # seconds; leave 10s safety margin before 150s kill

def _has_k33(adj, a, b, c):
    """Check if vertices a,b,c have >=3 common neighbors (forms K3,3)."""
    # Common neighbors of a,b,c
    cn = adj[a] & adj[b] & adj[c]
    return len(cn) >= 3

def _count_k33_violations(adj):
    """Count number of triples with >=3 common neighbors."""
    count = 0
    for a, b, c in itertools.combinations(range(N), 3):
        cn = adj[a] & adj[b] & adj[c]
        if len(cn) >= 3:
            count += 1
    return count

def _is_k33_free(adj):
    """Check if the graph is K3,3-free."""
    for a, b, c in itertools.combinations(range(N), 3):
        cn = adj[a] & adj[b] & adj[c]
        if len(cn) >= 3:
            return False
    return True

def _greedy_add_edges(adj, rng, time_limit):
    """Try to add edges greedily while maintaining K3,3-free property."""
    improved = True
    while improved and (time.time() - time_limit) < (DEADLINE - 2):
        improved = False
        # Try all non-edges in random order
        candidates = []
        for u in range(N):
            for v in range(u+1, N):
                if not (adj[u] >> v) & 1:
                    candidates.append((u, v))
        rng.shuffle(candidates)
        for u, v in candidates:
            if (time.time() - time_limit) > (DEADLINE - 2):
                break
            # Check if adding edge (u,v) creates a K3,3
            # A K3,3 with edge (u,v) means: u and v are in one part of size 3,
            # and there are 3 common neighbors in the other part.
            # More precisely: after adding (u,v), check all triples.
            # But we can be smarter: the new edge (u,v) can only create violations
            # involving u or v in the triple of the "3 vertices" side, or u or v in the "3 neighbors" side.
            
            # Actually, let's just check: adding (u,v) might create a K3,3.
            # The K3,3 has two parts of 3. The edge (u,v) is within one part.
            # So {u,v} must be subset of one of the two 3-sets.
            # Case 1: u,v in the "vertices" part (the part whose common neighbors we count).
            #   Then we need a third vertex w such that |N(u) ∩ N(v) ∩ N(w)| >= 3 after adding (u,v).
            #   Before adding, |N(u) ∩ N(v) ∩ N(w)| might be 2, and adding (u,v) doesn't change N(w) but changes N(u),N(v).
            #   Wait, adding edge (u,v) means v becomes a neighbor of u and u becomes a neighbor of v.
            #   So N(u) gains v, N(v) gains u.
            #   For triple (u, v, w): common neighbors = (N(u)∪{v}) ∩ (N(v)∪{u}) ∩ N(w)
            #   = (N(u) ∩ N(v) ∩ N(w)) ∪ (v ∩ N(v) ∩ N(w)) ∪ (u ∩ N(u) ∩ N(w)) ∪ ... 
            #   This is getting complex. Let me just do the full check for affected triples.
            
            # Simpler approach: temporarily add, check only triples involving u or v, then revert.
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            ok = True
            # Check all triples that include u or v
            for a, b, c in itertools.combinations(range(N), 3):
                if u not in (a,b,c) and v not in (a,b,c):
                    continue
                cn = adj[a] & adj[b] & adj[c]
                if len(cn) >= 3:
                    ok = False
                    break
            
            if not ok:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                improved = True
                break  # restart with new state
    
    return adj

def _local_search(adj, rng, time_limit):
    """Local search: try swapping edges to increase density while staying K3,3-free."""
    best_adj = [x for x in adj]
    best_edges = sum(len(bin(x)) for x in best_adj) // 2
    
    while (time.time() - time_limit) < (DEADLINE - 3):
        # Pick a random non-edge to add
        non_edges = []
        for u in range(N):
            for v in range(u+1, N):
                if not (adj[u] >> v) & 1:
                    non_edges.append((u, v))
        if not non_edges:
            break
        u, v = rng.choice(non_edges)
        
        # Try to add (u,v); if it creates violations, try to fix by removing an edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        violations = []
        for a, b, c in itertools.combinations(range(N), 3):
            if u not in (a,b,c) and v not in (a,b,c):
                continue
            cn = adj[a] & adj[b] & adj[c]
            if len(cn) >= 3:
                violations.append((a,b,c,cn))
        
        if not violations:
            # Successful addition
            cur_edges = sum(len(bin(x)) for x in adj) // 2
            if cur_edges > best_edges:
                best_edges = cur_edges
                best_adj = [x for x in adj]
            continue
        
        # Try to fix by removing one edge from the violation
        fixed = False
        for (a, b, c, cn) in violations:
            # Try removing an edge incident to one of the common neighbors
            for w in cn:
                # Try removing edge (w, a), (w, b), or (w, c)
                for x in (a, b, c):
                    if (adj[w] >> x) & 1:
                        # Remove edge (w,x)
                        adj[w] &= ~(1 << x)
                        adj[x] &= ~(1 << w)
                        # Check if now K3,3-free (only check affected triples)
                        ok = True
                        for a2, b2, c2 in itertools.combinations(range(N), 3):
                            if u not in (a2,b2,c2) and v not in (a2,b2,c2) and w not in (a2,b2,c2) and x not in (a2,b2,c2):
                                continue
                            cn2 = adj[a2] & adj[b2] & adj[c2]
                            if len(cn2) >= 3:
                                ok = False
                                break
                        if ok:
                            fixed = True
                            break
                        else:
                            # Revert
                            adj[w] |= (1 << x)
                            adj[x] |= (1 << w)
                    if fixed:
                        break
                if fixed:
                    break
            if fixed:
                break
        
        if not fixed:
            # Revert the addition
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    
    return best_adj

def _random_init(p, rng):
    """Generate a random graph with edge probability p."""
    adj = [0] * N
    for u in range(N):
        for v in range(u+1, N):
            if rng.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    return adj

def _repair(adj, rng, time_limit):
    """Remove edges to make graph K3,3-free."""
    changed = True
    while changed and (time.time() - time_limit) < (DEADLINE - 2):
        changed = False
        for a, b, c in itertools.combinations(range(N), 3):
            cn = adj[a] & adj[b] & adj[c]
            if len(cn) >= 3:
                # Remove a random edge from the common neighborhood
                w = rng.choice(list(cn))
                # Remove edge (w, a)
                adj[w] &= ~(1 << a)
                adj[a] &= ~(1 << w)
                changed = True
                break
    return adj

def construct():
    start_time = time.time()
    
    best_adj = None
    best_edges = 0
    
    # Try multiple random initializations with different densities
    rng = random.Random(42)
    
    # Densities to try: from sparse to moderately dense
    # K2,18 has 36 edges. We want to beat that.
    # Complete graph has 190 edges. K3,3-free max is somewhere in between.
    densities = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    
    # Also try starting from K2,18 and adding
    # K2,18: vertices 0,1 connected to 2..19
    k218_adj = [0] * N
    for a in range(2):
        for b in range(2, 20):
            k218_adj[a] |= (1 << b)
            k218_adj[b] |= (1 << a)
    
    # Greedy add to K2,18
    adj_k218 = [x for x in k218_adj]
    adj_k218 = _greedy_add_edges(adj_k218, rng, start_time)
    e = sum(len(bin(x)) for x in adj_k218) // 2
    if e > best_edges:
        best_edges = e
        best_adj = [x for x in adj_k218]
    
    # Local search on K2,18 result
    if (time.time() - start_time) < (DEADLINE - 10):
        adj_ls = _local_search([x for x in best_adj], rng, start_time)
        e = sum(len(bin(x)) for x in adj_ls) // 2
        if e > best_edges:
            best_edges = e
            best_adj = [x for x in adj_ls]
    
    # Try random initializations
    for p in densities:
        if (time.time() - start_time) > (DEADLINE - 15):
            break
        
        # Multiple restarts per density
        for restart in range(3):
            if (time.time() - start_time) > (DEADLINE - 10):
                break
            
            adj = _random_init(p, rng)
            
            # Repair to make K3,3-free
            adj = _repair(adj, rng, start_time)
            
            # Greedy add edges
            adj = _greedy_add_edges(adj, rng, start_time)
            
            e = sum(len(bin(x)) for x in adj) // 2
            if e > best_edges:
                best_edges = e
                best_adj = [x for x in adj]
            
            # Local search
            if (time.time() - start_time) < (DEADLINE - 5):
                adj_ls = _local_search([x for x in adj], rng, start_time)
                e = sum(len(bin(x)) for x in adj_ls) // 2
                if e > best_edges:
                    best_edges = e
                    best_adj = [x for x in adj_ls]
    
    # Final verification and edge extraction
    if best_adj is None:
        # Fallback to K2,18
        best_adj = k218_adj
    
    # Verify K3,3-free
    if not _is_k33_free(best_adj):
        # Should not happen, but fallback
        best_adj = k218_adj
    
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
