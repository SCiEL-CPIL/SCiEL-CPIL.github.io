import time
import random
import numpy as np

N = 20
DEADLINE_FRACTION = 0.92

def construct():
    start_time = time.time()
    deadline = start_time + 150 * DEADLINE_FRACTION

    def _time_left():
        return deadline - time.time()

    def _edge_count(adj):
        return int(np.sum(adj) // 2)

    def _find_violations(adj):
        """Find all edges (u,v) whose removal is needed to break K3,3.
        Returns list of (u, v) edges that are part of some K3,3."""
        violations = set()
        # A K3,3 exists if there are vertices a,b,c on one side and d,e,f on other
        # such that all 9 edges exist.
        # Equivalently: for any triple {x,y,z} and any triple {p,q,r} disjoint from it,
        # if all 9 cross-edges exist, we have K3,3.
        # Faster: for each pair (x,y), let common = N(x) & N(y). If |common| >= 3,
        # then pick any 3 from common, say {a,b,c}. Then {x,y} are the other side.
        # But we need 3 on each side. So: for triple {x,y,z}, if |N(x)&N(y)&N(z)| >= 3,
        # violation.
        
        # For each triple of vertices, check if they have >= 3 common neighbors
        for x in range(N):
            for y in range(x+1, N):
                common_xy = adj[x] & adj[y]
                # common_xy has bits for common neighbors of x and y
                # We need to check if among common neighbors, there are 3 that also share
                # Actually, we need: exists z such that N(x)&N(y)&N(z) has >= 3 bits
                # i.e., for z in common_xy, check popcount(N(x)&N(y)&N(z)) >= 3
                # But that's expensive. Alternative:
                # K3,3 with partition {x,y,z} | {a,b,c} iff
                # a,b,c all in N(x)&N(y)&N(z)
                # So: for each triple (x,y,z), compute C = N(x)&N(y)&N(z), if |C|>=3 -> violation
                for z in range(y+1, N):
                    common_xyz = common_xy & adj[z]
                    if np.bit_count(common_xyz) >= 3:
                        # Violation: edges between {x,y,z} and any 3 of common_xyz form K3,3
                        # We need to remove some edge. Mark all cross edges as potential removals
                        for a in range(N):
                            if (common_xyz >> a) & 1:
                                for b in [x, y, z]:
                                    violations.add((min(b,a), max(b,a)))
        return list(violations)

    def _repair(adj):
        """Remove edges to eliminate all K3,3 violations."""
        changed = True
        while changed:
            if _time_left() < 1:
                break
            violations = _find_violations(adj)
            if not violations:
                break
            changed = False
            # Remove one edge from a random violation
            random.shuffle(violations)
            for u, v in violations:
                if (adj[u] >> v) & 1:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    changed = True
                    break
        return adj

    def _greedy_add(adj):
        """Try to add missing edges that don't create K3,3."""
        improved = True
        while improved:
            if _time_left() < 1:
                break
            improved = False
            missing = [(u, v) for u in range(N) for v in range(u+1, N) 
                      if not ((adj[u] >> v) & 1)]
            random.shuffle(missing)
            for u, v in missing:
                if _time_left() < 0.5:
                    break
                # Tentatively add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                # Check if this creates a K3,3
                # Quick check: does adding (u,v) create a triple with >= 3 common neighbors?
                # For any pair in N(u), check if they share >= 2 neighbors in N(v)
                # Simpler: just do full check via _find_violations but that's slow
                # Use a targeted check
                valid = True
                # Check triples containing u: for any x,y in N(u)\{v}, if N(x)&N(y)&N(v) has >= 2
                nu = adj[u]
                for x in range(N):
                    if (nu >> x) & 1 and x != v:
                        for y in range(x+1, N):
                            if (nu >> y) & 1:
                                common = adj[x] & adj[y] & adj[v]
                                if np.bit_count(common) >= 2:
                                    valid = False
                                    break
                        if not valid:
                            break
                if valid:
                    # Check triples containing v
                    nv = adj[v]
                    for x in range(N):
                        if (nv >> x) & 1 and x != u:
                            for y in range(x+1, N):
                                if (nv >> y) & 1:
                                    common = adj[x] & adj[y] & adj[u]
                                    if np.bit_count(common) >= 2:
                                        valid = False
                                        break
                            if not valid:
                                break
                if not valid:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    improved = True
        return adj

    def _local_search(adj):
        """Swap edges to improve density while maintaining K3,3-free."""
        best_adj = adj.copy()
        best_count = _edge_count(adj)
        no_improve = 0
        
        while _time_left() > 2:
            if no_improve > 50:
                break
            # Pick a random existing edge to remove
            existing = [(u, v) for u in range(N) for v in range(u+1, N) 
                       if (adj[u] >> v) & 1]
            if not existing:
                break
            u1, v1 = random.choice(existing)
            adj[u1] ^= (1 << v1)
            adj[v1] ^= (1 << u1)
            
            # Try to add a random missing edge
            missing = [(u, v) for u in range(N) for v in range(u+1, N) 
                      if not ((adj[u] >> v) & 1)]
            if not missing:
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                break
            
            u2, v2 = random.choice(missing)
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            
            # Check validity
            valid = True
            nu = adj[u2]
            for x in range(N):
                if (nu >> x) & 1 and x != v2:
                    for y in range(x+1, N):
                        if (nu >> y) & 1:
                            common = adj[x] & adj[y] & adj[v2]
                            if np.bit_count(common) >= 2:
                                valid = False
                                break
                    if not valid:
                        break
            if valid:
                nv = adj[v2]
                for x in range(N):
                    if (nv >> x) & 1 and x != u2:
                        for y in range(x+1, N):
                            if (nv >> y) & 1:
                                common = adj[x] & adj[y] & adj[u2]
                                if np.bit_count(common) >= 2:
                                    valid = False
                                    break
                        if not valid:
                            break
            
            if valid:
                count = _edge_count(adj)
                if count > best_count:
                    best_count = count
                    best_adj = adj.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                adj[u2] ^= (1 << v2)
                adj[v2] ^= (1 << u2)
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                no_improve += 1
        
        return best_adj

    def _dense_random_init(p=0.7):
        """Create a dense random graph and repair it."""
        adj = [0] * N
        for u in range(N):
            for v in range(u+1, N):
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        adj = _repair(adj)
        adj = _greedy_add(adj)
        return adj

    best_adj = None
    best_count = 0

    # Multiple restarts with different densities
    for p in [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3]:
        if _time_left() < 3:
            break
        adj = _dense_random_init(p)
        count = _edge_count(adj)
        if count > best_count:
            best_count = count
            best_adj = adj.copy()
    
    # Local search on best
    if best_adj is not None and _time_left() > 5:
        adj_ls = _local_search(best_adj)
        count = _edge_count(adj_ls)
        if count > best_count:
            best_count = count
            best_adj = adj_ls.copy()

    # More restarts with kick-and-repair
    while _time_left() > 3:
        if best_adj is None:
            adj = _dense_random_init(0.5)
        else:
            # Kick: remove some random edges
            adj = best_adj.copy()
            existing = [(u, v) for u in range(N) for v in range(u+1, N) 
                       if (adj[u] >> v) & 1]
            random.shuffle(existing)
            kick_size = random.randint(3, 8)
            for i in range(min(kick_size, len(existing))):
                u, v = existing[i]
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            adj = _repair(adj)
            adj = _greedy_add(adj)
        
        count = _edge_count(adj)
        if count > best_count:
            best_count = count
            best_adj = adj.copy()
            
            if _time_left() > 5:
                adj_ls = _local_search(best_adj)
                count = _edge_count(adj_ls)
                if count > best_count:
                    best_count = count
                    best_adj = adj_ls.copy()

    if best_adj is None:
        # Fallback: empty graph
        return []

    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
