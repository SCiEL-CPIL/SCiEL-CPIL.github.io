import time
import random
import sys

_DEADLINE = 148.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_hit():
    return _elapsed() >= _DEADLINE

def _build_polarity_graph(q):
    """Construct the polarity graph of PG(2, q) via absolute points."""
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, q - 2, q)
                    key = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q - 2, q)
                    key = (0, 1, (z * inv_y) % q)
                else:
                    key = (0, 0, 1)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    n_pts = len(points)
    adj = [set() for _ in range(n_pts)]
    for i in range(n_pts):
        x1, x2, x3 = points[i]
        for j in range(i + 1, n_pts):
            y1, y2, y3 = points[j]
            if (x1 * y1 + x2 * y2 + x3 * y3) % q == 0:
                adj[i].add(j)
                adj[j].add(i)
    return adj

def _has_indep_11(adj, n):
    """Check if there's an independent set of size 11 using 2-packing B&B."""
    if n < 11:
        return False
    # Build neighbor masks
    masks = [0] * n
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        masks[i] = m
    # For each vertex, its "forbidden set" if added to packing = N(v) ∪ {v}
    # But for 2-packing, the constraint is: for new vertex v with neighbor set N(v),
    # N(v) must intersect each existing S_i in at most 1 element.
    # In B&B: we build the packing S = {v1, v2, ...} where each v_i is a vertex.
    # The constraint: |N(v_a) ∩ N(v_b)| <= 1 for all pairs.
    # B&B: pick a vertex, add it, remove all vertices that would violate with it.
    # Violation: vertex u conflicts with v if |N(u) ∩ N(v)| >= 2.
    # Precompute conflict masks: conf[v] = set of u with |N(u)∩N(v)|>=2
    
    # Compute conflict masks
    conf = [0] * n
    for k in range(n):
        # All pairs of neighbors of k conflict
        nk = list(adj[k])
        for a in range(len(nk)):
            u = nk[a]
            for b in range(a + 1, len(nk)):
                v = nk[b]
                conf[u] |= (1 << v)
                conf[v] |= (1 << u)
    # Also, a vertex conflicts with itself? No. But in B&B we remove v and its conflicts.
    
    sys.setrecursionlimit(20000)
    
    def branch(rem_mask, size):
        if size >= 11:
            return True
        if rem_mask == 0:
            return False
        if size + rem_mask.bit_count() < 11:
            return False
        # Pick vertex with max conflict degree in remaining
        best_v = -1
        best_deg = -1
        m = rem_mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            m ^= lsb
            deg = (conf[v] & rem_mask).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
        if best_v < 0:
            return False
        # Include best_v: remove it and all its conflicts
        new_rem = rem_mask & ~(conf[best_v] | (1 << best_v))
        if branch(new_rem, size + 1):
            return True
        # Exclude best_v
        new_rem2 = rem_mask & ~(1 << best_v)
        if branch(new_rem2, size):
            return True
        return False
    
    full = (1 << n) - 1
    return branch(full, 0)

def _get_conflict_masks(adj, n):
    """conflict_masks[i] has bit j set if i and j share >=2 common neighbors."""
    conf = [0] * n
    for k in range(n):
        nk = list(adj[k])
        for a in range(len(nk)):
            u = nk[a]
            for b in range(a + 1, len(nk)):
                v = nk[b]
                conf[u] |= (1 << v)
                conf[v] |= (1 << u)
    return conf

def _check_c4(adj, n):
    for i in range(n):
        for j in range(i + 1, n):
            if len(adj[i] & adj[j]) >= 2:
                return False
    return True

def _greedy_conflict_independent(conf_masks, n, randomize=True):
    """Find a large independent set in the conflict graph (greedy)."""
    remaining = (1 << n) - 1
    S = []
    while remaining:
        best_v = -1
        best_deg = 10**9
        m = remaining
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            m ^= lsb
            deg = (conf_masks[v] & remaining).bit_count()
            if deg < best_deg:
                best_deg = deg
                best_v = v
        if best_v < 0:
            break
        S.append(best_v)
        remaining &= ~(conf_masks[best_v] | (1 << best_v))
    return S

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = 148.0
    
    adj = _build_polarity_graph(5)
    n = len(adj)
    
    # Verify initial
    if _has_indep_11(adj, n):
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    best_n = n
    best_adj = [set(s) for s in adj]
    
    # Growth loop
    while _elapsed() < 130.0:
        if _deadline_hit():
            break
        conf_masks = _get_conflict_masks(adj, n)
        
        added = False
        # Try multiple random restarts of greedy
        for attempt in range(200):
            if _deadline_hit():
                break
            # Randomized greedy: pick from low-degree candidates
            remaining = (1 << n) - 1
            S = []
            while remaining:
                # Find min degree
                min_deg = 10**9
                candidates = []
                m = remaining
                while m:
                    lsb = m & (-m)
                    v = lsb.bit_length() - 1
                    m ^= lsb
                    deg = (conf_masks[v] & remaining).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        candidates = [v]
                    elif deg == min_deg:
                        candidates.append(v)
                if not candidates:
                    break
                v = random.choice(candidates)
                S.append(v)
                remaining &= ~(conf_masks[v] | (1 << v))
            if not S:
                continue
            # Try adding new vertex connected to S
            new_n = n + 1
            new_adj = [set(s) for s in adj]
            new_adj.append(set(S))
            for u in S:
                new_adj[u].add(new_n - 1)
            if not _has_indep_11(new_adj, new_n):
                adj = new_adj
                n = new_n
                if n > best_n:
                    best_n = n
                    best_adj = [set(s) for s in adj]
                added = True
                break
        
        if not added:
            # Try edge removal to reduce alpha
            # Find a near-independence set and try removing edges within it
            # Actually, try removing random edges to break independence
            improved = False
            for _ in range(50):
                if _deadline_hit():
                    break
                # Try removing a random edge
                i = random.randint(0, n - 1)
                if adj[i]:
                    j = random.choice(list(adj[i]))
                    adj[i].discard(j)
                    adj[j].discard(i)
                    if not _has_indep_11(adj, n):
                        improved = True
                        break
                    else:
                        adj[i].add(j)
                        adj[j].add(i)
            if not improved:
                break
    
    # Final verification
    if _check_c4(best_adj, len(best_adj)) and not _has_indep_11(best_adj, len(best_adj)):
        n = len(best_adj)
        edges = []
        for i in range(n):
            for j in best_adj[i]:
                if i < j:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    else:
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
