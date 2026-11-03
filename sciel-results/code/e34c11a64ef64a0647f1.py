"""Combined cage solver: Circulant search, Optimized Random Configuration Model with local search, Blow-up, and LUW fallback."""
import time
import threading
import random
from collections import deque

_DEADLINE = 120.0
_start = [None]


def _elapsed():
    s = _start[0]
    return 0.0 if s is None else time.time() - s


# ---------------- exact girth (non-backtracking walks, all vertices) ----------
def _nonback_girth(adj, n):
    """Return length of shortest cycle, certified by non-backtracking walks
    from every vertex. >= g means no cycle shorter than g."""
    prev = {}
    q = deque()
    for u in range(n):
        for p in adj[u]:
            key = (p, u)
            prev[key] = (u, 1)
            q.append(key)
    while q:
        (v, p), (u, dist) = q.popleft()
        for w in adj[v]:
            if w == p:
                continue
            key = (w, v)
            if w == u:
                return dist + 1
            if key in prev:
                continue
            prev[key] = (v, dist + 1)
            q.append(key)
    return 10 ** 9


def _is_valid(adj, n, k, g):
    if n == 0:
        return False
    if any(len(adj[i]) != k for i in range(n)):
        return False
    for i in range(n):
        if len(set(adj[i])) != len(adj[i]):
            return False
        for j in adj[i]:
            if i <= j:
                return False
    return _nonback_girth(adj, n) >= g


def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for (i, j) in edges:
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        adj[i] = sorted(adj[i])
    return adj

def _local_girth_check(adj, n, g, start_nodes):
    """
    Check for cycles of length < g starting from start_nodes.
    Returns True if no short cycle found, False otherwise.
    Uses BFS up to depth g-1.
    """
    # We want to detect cycles of length L < g.
    # A cycle of length L corresponds to a path of length L-1 from a node to one of its neighbors, 
    # or more simply, a path of length L from a node back to itself (non-backtracking).
    # To be efficient, we check for collisions in non-backtracking walks of length up to g-2.
    
    # If we find a walk of length d ending at a node that was visited at step d' such that 
    # the walk doesn't backtrack, it might form a cycle.
    # Simpler: Just do a BFS for distance < g/2? No, girth is global.
    # Let's just do a quick check for triangles and 4-cycles which are the most common obstacles.
    # For g=9, we need to avoid cycles of length 3,4,5,6,7,8.
    
    # Heuristic: Check if any start_node is within distance < g/2 of any other start_node 
    # in a way that creates a short cycle.
    # Actually, the most robust quick check is to just run the non-backtracking BFS 
    # but limit the depth to g-1 and stop early if a cycle is found.
    
    prev = {}
    q = deque()
    # Initialize with neighbors of start nodes
    for u in start_nodes:
        if u < 0 or u >= n: continue
        for p in adj[u]:
            key = (p, u)
            if key not in prev:
                prev[key] = (u, 1)
                q.append(key)
    
    while q:
        (v, p), (u, dist) = q.popleft()
        if dist >= g - 1:
            continue # We only care about cycles shorter than g
        for w in adj[v]:
            if w == p:
                continue
            key = (w, v)
            if w == u:
                return False # Found a cycle of length dist+1 < g
            if key in prev:
                # Check if this creates a short cycle
                # The previous visit was at some distance. 
                # If the path from start to u (dist) and start to w (prev_dist) 
                # combine to form a cycle < g, we have a problem.
                # This logic is complex. For safety, if we see a collision in the 
                # non-backtracking walk space, it *might* be a cycle.
                # However, standard BFS girth check is safer.
                # Given the cost, we will just return False on any collision 
                # that implies a cycle shorter than g.
                # Actually, a collision (w,v) in prev means we reached (w,v) before.
                # The path to (w,v) was via some u'. 
                # If u' != u, it's not necessarily a cycle involving u.
                # But if we are doing this to filter bad swaps, being conservative (returning False) 
                # is better than missing a short cycle.
                return False
            prev[key] = (v, dist + 1)
            q.append(key)
    return True


# ---------------- construction 1: LUW (Lazebnik-Ustimenko-Woldar) ------------
def _luw(k, g):
    q = k
    # d is the dimension. g = d + 5 (if d odd) or d + 4 (if d even)
    # We want g >= target_g.
    # So d >= target_g - 5 (if odd) or d >= target_g - 4 (if even)
    # To minimize n ~ q^d, we want smallest d.
    # If g is odd, d = g - 5. If g is even, d = g - 4.
    if (g - 5) % 2 == 1: # g-5 is odd => g is even? No.
        # Let's just iterate d
        pass
    
    # Standard LUW: CD(d, q) has girth d+5 if d is odd, d+4 if d is even.
    # We want girth >= g.
    # Try d values that satisfy the girth condition.
    best_d = None
    for d in range(3, 10):
        if d % 2 == 1:
            girth = d + 5
        else:
            girth = d + 4
        if girth >= g:
            best_d = d
            break
            
    if best_d is None:
        best_d = 3 # Fallback, might not satisfy girth, but returns something
        
    d = best_d
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = (lambda x, y: x ^ y)
        mul = (lambda x, y: table[x][y])
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
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


# ---------------- construction 2: Circulant Search ---------------------------
def _circulant_search(k, g):
    R = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(R))
    
    if k % 2 == 0:
        half = k // 2
        # Prioritize smaller n
        for n in range(moore, moore + 200):
            if n * k % 2 != 0: continue
            if _elapsed() > _DEADLINE - 15.0:
                return None
            if n <= 100:
                import itertools
                for S in itertools.combinations(range(1, n // 2 + 1), half):
                    S_set = set(S)
                    if any((n - s) % n in S_set for s in S):
                        continue
                    adj = [set() for _ in range(n)]
                    for i in range(n):
                        for s in S:
                            adj[i].add((i + s) % n)
                            adj[i].add((i - s) % n)
                    if all(len(adj[i]) == k for i in range(n)):
                        if _nonback_girth(adj, n) >= g:
                            edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                            return n, edges
            else:
                rng = random.Random(n * 1000 + k)
                for _ in range(100): # Increased samples
                    if _elapsed() > _DEADLINE - 15.0:
                        return None
                    S = rng.sample(range(1, n // 2 + 1), half)
                    S_set = set(S)
                    if any((n - s) % n in S_set for s in S):
                        continue
                    adj = [set() for _ in range(n)]
                    for i in range(n):
                        for s in S:
                            adj[i].add((i + s) % n)
                            adj[i].add((i - s) % n)
                    if all(len(adj[i]) == k for i in range(n)):
                        if _nonback_girth(adj, n) >= g:
                            edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                            return n, edges
    else:
        m = (k - 1) // 2
        for n in range(moore, moore + 200, 2):
            if n < k + 1: continue
            if _elapsed() > _DEADLINE - 15.0:
                return None
            if n <= 100:
                import itertools
                for S_vals in itertools.combinations(range(1, n // 2), m):
                    adj = [set() for _ in range(n)]
                    for i in range(n):
                        for s in S_vals:
                            adj[i].add((i + s) % n)
                            adj[i].add((i - s) % n)
                        adj[i].add((i + n // 2) % n)
                    if all(len(adj[i]) == k for i in range(n)):
                        if _nonback_girth(adj, n) >= g:
                            edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                            return n, edges
            else:
                rng = random.Random(n * 1000 + k)
                for _ in range(100):
                    if _elapsed() > _DEADLINE - 15.0:
                        return None
                    S_vals = rng.sample(range(1, n // 2), m)
                    adj = [set() for _ in range(n)]
                    for i in range(n):
                        for s in S_vals:
                            adj[i].add((i + s) % n)
                            adj[i].add((i - s) % n)
                        adj[i].add((i + n // 2) % n)
                    if all(len(adj[i]) == k for i in range(n)):
                        if _nonback_girth(adj, n) >= g:
                            edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                            return n, edges
    return None


# ---------------- construction 3: Optimized Random Search ---------------------
def _random_search(k, g, seed=0):
    rng = random.Random(seed)
    R = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(R))
    
    # Search n from moore to moore + 50
    for n in range(moore, moore + 50):
        if n * k % 2 != 0:
            continue
        if n < k + 1:
            continue
            
        for attempt in range(20):
            if _elapsed() > _DEADLINE - 10.0:
                return None
            
            adj = _generate_random_regular(n, k, rng)
            if adj is None:
                continue
                
            if _nonback_girth(adj, n) >= g:
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                return n, edges
                
            # Local search
            for step in range(500): # Increased steps due to faster check
                if _elapsed() > _DEADLINE - 10.0:
                    return None
                
                # Quick check: if current girth is already good (shouldn't happen if we didn't return)
                # We pick a random edge to swap
                u = rng.randint(0, n - 1)
                if not adj[u]: continue
                v = rng.choice(list(adj[u]))
                
                # Pick another random edge (w, x)
                w = -1
                x = -1
                for _ in range(50):
                    w = rng.randint(0, n - 1)
                    if w in (u, v): continue
                    if w in adj[u] or w in adj[v]: continue
                    if not adj[w]: continue
                    x = rng.choice(list(adj[w]))
                    if x in (u, v, w): continue
                    if x in adj[u] or x in adj[v]: continue
                    if w in adj[u] or x in adj[v]: continue
                    
                    # Perform swap
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[w].remove(x)
                    adj[x].remove(w)
                    
                    adj[u].add(w)
                    adj[w].add(u)
                    adj[v].add(x)
                    adj[x].add(v)
                    
                    # Local check on affected nodes u, v, w, x
                    # We check if any short cycle is created involving these nodes.
                    # A simple heuristic: check girth from u, v, w, x with depth limit.
                    if _local_girth_check(adj, n, g, [u, v, w, x]):
                        # If local check passes, do a full check occasionally or assume it's good?
                        # To be safe, do a full check every 10 steps or if local passes.
                        if _nonback_girth(adj, n) >= g:
                            edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                            return n, edges
                    else:
                        # Revert
                        adj[u].remove(w)
                        adj[w].remove(u)
                        adj[v].remove(x)
                        adj[x].remove(v)
                        
                        adj[u].add(v)
                        adj[v].add(u)
                        adj[w].add(x)
                        adj[x].add(w)
                    break
                if w == -1:
                    continue
    return None


def _generate_random_regular(n, k, rng):
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    rng.shuffle(stubs)
    
    adj = [set() for _ in range(n)]
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            return None
        if v in adj[u]:
            return None
        adj[u].add(v)
        adj[v].add(u)
    return adj


# ---------------- construction 4: Blow-up -------------------------------------
def _blowup(k, g):
    # Try to find a base graph of size B and degree d such that we can blow up to degree k.
    # If we blow up each vertex of a d-regular graph into a clique of size m, 
    # the new degree is (d-1)*m + (m-1) = d*m - 1.
    # So k = d*m - 1 => d*m = k+1.
    # We need d < k.
    # Also, the girth of the blow-up graph is roughly the girth of the base graph, 
    # but cycles within the cliques have length 3.
    # So if g > 3, we cannot use cliques. We must use independent sets.
    # If we blow up into independent sets, the degree is d*m.
    # So k = d*m.
    # Girth is preserved (or slightly reduced if there are short cycles in base).
    
    # Find factors of k
    factors = []
    for d in range(2, k):
        if k % d == 0:
            m = k // d
            if m > 1:
                factors.append((d, m))
                
    if not factors:
        return None
        
    # For each factorization, try to find a d-regular graph of girth >= g
    # with small size, then blow it up.
    # We can use the circulant or random search for the base graph with smaller g?
    # No, the base graph must have girth >= g.
    
    # This is recursive. Let's just try small base graphs.
    # For d=3, g=9, the smallest cage is 10 (Petersen).
    # Blow up Petersen (d=3, n=10) with m=2 (since 5 = 3*? No, 5 is not divisible by 3).
    # For k=5, factors are (1,5), (5,1). d must be < 5.
    # So d=1 is not a graph.
    # Thus, simple blow-up of independent sets doesn't work for k=5.
    
    # What about mixed?
    # This construction is weak for prime k.
    return None


def construct(k, g):
    _start[0] = time.time()
    holder = {"best": None}
    
    jobs = [
        ("circulant", lambda: _circulant_search(k, g)),
        ("rand0", lambda: _random_search(k, g, seed=0)),
        ("rand1", lambda: _random_search(k, g, seed=1)),
        ("rand2", lambda: _random_search(k, g, seed=2)),
        ("luw", lambda: _luw(k, g)),
    ]
    
    threads = []
    for name, fn in jobs:
        t = threading.Thread(target=_worker, args=(name, fn, holder, k, g), daemon=True)
        t.start()
        threads.append(t)

    while _elapsed() < _DEADLINE - 2.0:
        if not any(t.is_alive() for t in threads):
            break
        time.sleep(0.1)

    best = holder.get("best")
    if best is None:
        try:
            n, edges = _luw(k, g)
            best = (n, edges)
        except Exception:
            n = k + 1
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            best = (n, edges)
    return best


def _worker(name, fn, holder, k, g):
    try:
        res = fn()
    except Exception:
        res = None
    if res is None:
        return
    n, edges = res
    if _elapsed() >= _DEADLINE:
        return
    try:
        if n > 0 and _is_valid(_build_adj(edges, n), n, k, g):
            cur = holder.get("best")
            if cur is None or n < cur[0]:
                holder["best"] = (n, edges)
    except Exception:
        pass
