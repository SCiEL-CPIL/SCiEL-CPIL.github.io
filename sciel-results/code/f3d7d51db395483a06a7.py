"""Search for a small 5-regular graph with girth >= g.

Unlike the seed (which builds a full 1116-vertex cage), we only need a k-regular
graph whose shortest cycle has length >= g; it need not be a cage.  We therefore
generate random k-regular graphs and verify their girth exactly with
non-backtracking walks, keeping the smallest verified one.  A deadline guard
returns the best valid result before the wall-clock limit.
"""
import random
import time
import numpy as np
from array import array

# ---------------- deterministic RNG ----------------
def _xorshift64(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 7
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

def _randrange(rng, n):
    return _xorshift64(rng) % n

# ---------------- k-regular graph generation ----------------
def _pairing_regular(n, k, rng):
    """Return adjacency (list of lists) of a random simple k-regular graph on n
    vertices via a random perfect matching of n*k stubs.  Returns None on
    failure (self-loop or parallel edge)."""
    m = n * k
    stubs = array('I', (i for i in range(n) for _ in range(k)))
    # random perfect matching
    perm = list(range(m))
    for i in range(m - 1, 0, -1):
        j = _randrange(rng, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    adj = [[] for _ in range(n)]
    seen = [None] * n
    for a in range(0, m, 2):
        u = stubs[perm[a]]
        v = stubs[perm[a + 1]]
        if u == v:
            return None
        if seen[u] is not None:
            if v in seen[u]:
                return None
            seen[u].append(v)
        else:
            seen[u] = [v]
        adj[u].append(v)
        if seen[v] is not None:
            seen[v].append(u)
        else:
            seen[v] = [u]
        adj[v].append(u)
    return adj

def _swap_regular(adj, n, k, rng):
    """One random 2-switch preserving k-regularity; returns True if it produced
    a simple graph (no self-loops / parallel edges)."""
    m = n * k
    stubs = array('I', (i for i in range(n) for _ in range(k)))
    a = _randrange(rng, m)
    b = _randrange(rng, m - 1)
    if b >= a:
        b += 1
    c = _randrange(rng, m - 2)
    if c >= a:
        c += 1
    if c >= b:
        c += 1
    d = _randrange(rng, m - 3)
    for x in (a, b, c):
        if d >= x:
            d += 1
    u1, v1 = stubs[a], stubs[b]
    u2, v2 = stubs[c], stubs[d]
    # remove edges (u1,v1) and (u2,v2); require they are distinct
    if u1 == v1 or u2 == v2:
        return False
    if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
        return False
    adj[u1].remove(v1)
    adj[v1].remove(u1)
    adj[u2].remove(v2)
    adj[v2].remove(u2)
    # add edges (u1,u2) and (v1,v2); forbid self-loops / parallel edges
    if u1 == u2 or v1 == v2:
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        return False
    if u2 in adj[u1] or v2 in adj[v1]:
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        return False
    adj[u1].append(u2)
    adj[u2].append(u1)
    adj[v1].append(v2)
    adj[v2].append(v1)
    return True

# ---------------- exact girth via non-backtracking walks ----------------
def _girth(adj, g):
    """Return the length of the shortest cycle, or a value >= g if none shorter
    exists. Uses BFS on non-backtracking walks (directed edges)."""
    n = len(adj)
    if n == 0:
        return g
    
    # State: (u, v) means we are at v, came from u.
    # dist[(u, v)] = length of shortest non-backtracking walk ending in state (u, v).
    # Initialize: all directed edges (u, v) have dist 1.
    # We can start a walk from any vertex v to any neighbor w.
    # So states (v, w) have dist 1.
    
    from collections import deque
    
    # Map (u, v) to distance.
    # Since n is small, we can use a dict or a 2D array.
    # 2D array of size n x n, initialized to INF.
    INF = 1 << 28
    dist = [[INF] * n for _ in range(n)]
    queue = deque()
    
    # Initialize all directed edges
    for v in range(n):
        for w in adj[v]:
            dist[v][w] = 1
            queue.append((v, w))
            
    girth = INF
    
    while queue:
        u, v = queue.popleft()
        d = dist[u][v]
        
        # Check for cycle: if u is a neighbor of v, and we have a path from u to v
        # of length d, then the edge (v, u) closes a cycle of length d+1.
        # Wait. The state (u, v) means we came from u to v.
        # The walk is ... -> u -> v.
        # If u is adjacent to v, we can go v -> u.
        # The cycle would be ... -> u -> v -> u.
        # Length of walk so far is d. Adding edge (v, u) makes d+1.
        # But we must ensure the walk is non-backtracking.
        # The step before v was u. The next step is u.
        # So we are backtracking!
        # Non-backtracking means we cannot go v -> u if we just came from u.
        # So a cycle of length 2 is not allowed (and not in simple graphs).
        # A cycle of length L: v0 -> v1 -> ... -> v_{L-1} -> v0.
        # States: (v0, v1) dist 1, (v1, v2) dist 2, ..., (v_{L-1}, v0) dist L.
        # When we reach state (v_{L-1}, v0), we have a walk of length L ending at v0,
        # coming from v_{L-1}.
        # Since v_{L-1} is adjacent to v0, we have a cycle of length L.
        # So if dist[u][v] < INF, and u is a neighbor of v (which it is by definition of state),
        # does that imply a cycle?
        # No. The state (u, v) just means there is a path u->v.
        # If we started at u, and ended at v, and u is adjacent to v,
        # then we have a cycle u->...->v->u.
        # The length of the path is d. The cycle length is d+1.
        # BUT, we must ensure that the start vertex of the walk is u.
        # My initialization sets dist[v][w] = 1 for ALL edges (v, w).
        # This corresponds to walks of length 1 starting at v and ending at w.
        # So for state (u, v) with dist d, the walk started at some vertex s,
        # went s -> ... -> u -> v.
        # If s == u, then we have a cycle u -> ... -> v -> u of length d+1.
        # But we don't track s.
        
        # Alternative:
        # A cycle of length L exists if there is a non-backtracking walk of length L-1
        # from v to u, where (u, v) is an edge.
        # My dist[u][v] is the shortest NB walk ending at (u, v).
        # This walk has length dist[u][v].
        # It ends at v, came from u.
        # If we can "close" the cycle by going v -> u, we get a cycle of length dist[u][v] + 1.
        # However, the move v -> u is forbidden if the previous vertex was u.
        # The state is (u, v), so the previous vertex IS u.
        # So we CANNOT close the cycle immediately.
        
        # We need to find a state (x, v) such that x is adjacent to v, and the walk
        # did NOT start at x? No.
        
        # Correct check:
        # A cycle of length L exists iff there is a non-backtracking walk of length L-1
        # from v to w, where (w, v) is an edge, AND the walk does not use the edge (w, v)
        # as the first step? No, non-backtracking handles that.
        
        # Let's use the property:
        # Girth is the smallest L such that there exists a state (u, v) with dist[u][v] = L-1
        # and u is a neighbor of v?
        # No.
        
        # Let's just BFS and check if we reach a state (u, v) where u is the START of the walk?
        # We don't track start.
        
        # Standard method:
        # For each edge (u, v), find the shortest NB walk from v to u that does not start with (v, u).
        # This is equivalent to finding the shortest cycle containing (u, v).
        
        # Since n is small, we can do this:
        # For each vertex s, do a BFS of NB walks starting at s.
        # dist_s[v][p] = shortest NB walk from s to v, coming from p.
        # If we reach state (u, s) with dist d, then we have a cycle of length d+1?
        # Walk: s -> ... -> u -> s.
        # The state is (u, s). The walk ended at s, came from u.
        # The length is d.
        # The cycle is s -> ... -> u -> s.
        # The edge (u, s) closes it.
        # Is it non-backtracking?
        # The step before s was u. The next step is s.
        # We are returning to s.
        # The walk was s -> v1 -> ... -> u -> s.
        # The last step is u -> s.
        # The step before was v_{L-2} -> u.
        # Since it's NB, v_{L-2} != s (unless L=2, which is not possible).
        # So the cycle is valid.
        
        # So: For each start s, BFS.
        # If we reach state (u, s), we found a cycle of length dist[u][s] + 1?
        # No, dist[u][s] is the length of the walk s -> ... -> u -> s.
        # So the cycle length is dist[u][s].
        
        # We can stop BFS when dist >= g.
        
        for s in range(n):
            # dist_s[v][p]
            dist_s = [[INF] * n for _ in range(n)]
            q = deque()
            for w in adj[s]:
                dist_s[s][w] = 1
                q.append((s, w))
                
            while q:
                u, v = q.popleft()
                d = dist_s[u][v]
                
                # If v == s, we have a cycle of length d.
                if v == s:
                    if d < girth:
                        girth = d
                    # Since BFS, first time we hit s is the shortest cycle from s.
                    # We can break? No, other starts might have smaller cycles.
                    # But for this s, we found the shortest.
                    # We can stop exploring from this s?
                    # Yes, because any other path to s would be longer.
                    break
                
                if d + 1 >= girth:
                    continue
                    
                for w in adj[v]:
                    if w != u:
                        if dist_s[v][w] > d + 1:
                            dist_s[v][w] = d + 1
                            q.append((v, w))
                            
            if girth < g:
                return girth
                
    return girth if girth < INF else g

# ---------------- main entry point ----------------
def construct(k, g):
    deadline = time.time() + 110.0   # safety margin under the 150 s limit
    rng = 0x9E3779B97F4A7C15
    best = None
    best_n = None

    def consider(n, adj):
        nonlocal best, best_n
        if _girth(adj, g) >= g:
            best = adj
            best_n = n
            return True
        return False

    # (1) try a few fixed sizes that may already satisfy the girth
    if k == 5 and g == 9:
        # For small n, random 5-regular graphs will have small girth.
        # Skip to larger n where girth >= 9 is more plausible.
        for n in range(20, 80, 2):  # even n only (n*k must be even, k=5 odd, so n even)
            if time.time() > deadline:
                break
            for _ in range(200):
                if time.time() > deadline:
                    break
                adj = _pairing_regular(n, k, rng)
                if adj is None:
                    continue
                if consider(n, adj):
                    return n, [(i, j) for i in range(n) for j in adj[i] if i < j]

    # (2) growing n with 2-switch local search
    n = 2 * k
    while n <= 400 and time.time() < deadline:
        adj = None
        for _ in range(500):
            if time.time() > deadline:
                break
            cand = _pairing_regular(n, k, rng)
            if cand is not None:
                adj = cand
                break
        if adj is not None:
            consider(n, adj)
            for _ in range(5000):
                if time.time() > deadline:
                    break
                if _swap_regular(adj, n, k, rng):
                    if consider(n, adj):
                        return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
        n += 2  # step by 2 to keep n*k even when k is odd

    if best is not None:
        return best_n, [(i, j) for i in range(best_n) for j in best[i] if i < j]

    # (3) guaranteed fallback: the Lazebnik-Ustimenko-Woldar construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
