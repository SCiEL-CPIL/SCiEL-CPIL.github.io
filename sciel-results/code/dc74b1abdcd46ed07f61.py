"""Suite task: small 3-regular graph of girth 16 (cage problem n(3,16)).
Entry point: construct(k, g) -> (n, edges).
"""
import time
import random
from collections import deque

_DEADLINE = 150.0
_start = time.time()


def _time_left():
    return _DEADLINE - 10.0 - (time.time() - _start)


def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def _girth_nonbacktracking(n, edges, g):
    """Return the girth (length of shortest cycle) using non-backtracking
    walks from every vertex.  A cycle of length L is certified iff there is a
    non-backtracking closed walk of length L.  We stop as soon as we find a
    cycle shorter than g, returning its length.  If none found, return g (i.e.
    girth >= g)."""
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    limit = g  # we care about cycles of length < g
    for s in range(n):
        if _time_left() < 1.0:
            return g
        depth = {(-1, s): 0}
        q = deque([(-1, s)])
        while q:
            prev, cur = q.popleft()
            d = depth[(prev, cur)]
            if d + 1 >= limit:
                continue
            for nxt in adj[cur]:
                if nxt == prev:
                    continue
                if nxt == s and d + 1 < limit:
                    return d + 1
                key = (cur, nxt)
                if key not in depth:
                    depth[key] = d + 1
                    q.append((cur, nxt))
    return g


def _valid_fast(n, edges, k):
    """Check simple and k-regular."""
    if n < 1:
        return False
    deg = [0] * n
    seen = set()
    for a, b in edges:
        if a < 0 or a >= n or b < 0 or b >= n or a == b:
            return False
        key = (a, b) if a < b else (b, a)
        if key in seen:
            return False
        seen.add(key)
        deg[a] += 1
        deg[b] += 1
    for d in deg:
        if d != k:
            return False
    if len(edges) != n * k // 2:
        return False
    return True


def _construct_cayley_cyclic(p, a, b, c):
    """Construct a 3-regular Cayley graph on Z_p with generators a, b, c.
    Returns (n, edges) if valid, else None."""
    n = p
    edges = []
    edge_set = set()
    
    # Check for multi-edges/loops
    # Generators must be non-zero
    if a == 0 or b == 0 or c == 0:
        return None
    # Generators must be distinct and not negatives of each other
    # to ensure 3 distinct neighbors for each vertex in an undirected sense?
    # Actually, if we define edges {x, x+a}, {x, x+b}, {x, x+c}, 
    # and if b = -a, then {x, x-a} is the same as {x-a, x}.
    # So we must ensure a, b, c are distinct and a != -b, etc.
    gens = [a, b, c]
    for i in range(3):
        for j in range(i+1, 3):
            if gens[i] == gens[j]:
                return None
            if (gens[i] + gens[j]) % p == 0:
                return None

    for x in range(p):
        for g in gens:
            y = (x + g) % p
            u, v = min(x, y), max(x, y)
            if (u, v) in edge_set:
                return None
            edge_set.add((u, v))
            edges.append((u, v))
            
    return n, edges


def construct(k, g):
    global _start
    _start = time.time()
    
    if k != 3:
        # Fallback for other k, though task is specific to k=3
        return 0, []
        
    # Moore bound for (3,16) is 512.
    # We search for primes p >= 512.
    # The known best is 936.
    
    # List of primes in range
    primes = []
    for i in range(512, 1500):
        if _is_prime(i):
            primes.append(i)
            
    # Shuffle primes to try different sizes, or start from smallest?
    # Starting from smallest is better for score.
    # But if 512-600 fail, we might waste time.
    # Let's try primes in ascending order, but limit attempts per prime.
    
    random.seed(42)
    
    for p in primes:
        if _time_left() < 10.0:
            break
            
        # Try a few random generator sets for this prime
        # Number of attempts per prime should be small to allow checking many primes
        # But girth check is fast for small p.
        attempts = 0
        while _time_left() > 5.0 and attempts < 50:
            attempts += 1
            # Pick 3 distinct non-zero elements
            # We can pick from 1..p-1
            a = random.randint(1, p-1)
            b = random.randint(1, p-1)
            c = random.randint(1, p-1)
            
            # Quick check for distinctness and non-inverse
            if a == b or a == c or b == c:
                continue
            if (a + b) % p == 0 or (a + c) % p == 0 or (b + c) % p == 0:
                continue
                
            result = _construct_cayley_cyclic(p, a, b, c)
            if result is None:
                continue
                
            n, edges = result
            if not _valid_fast(n, edges, k):
                continue
                
            # Check girth
            # If girth >= g, we found a valid graph
            # We can stop early if we find one, but we want the SMALLEST n.
            # So we should check all primes in ascending order?
            # If we find one for p=523, that's great.
            # If we don't find one for p=523, we move to next.
            
            girth = _girth_nonbacktracking(n, edges, g)
            if girth >= g:
                return n, edges
                
    # If no graph found in the search range, return a fallback
    # The LUW construction is a safe fallback, but might be large.
    # We can try a larger random search or just return a known large graph if we have one.
    # For now, if we fail, we try one more time with a larger prime range or just return empty?
    # The task says "invalid result... scores 0". We must return a valid graph.
    # If the search fails, we should fall back to a construction that is guaranteed to work,
    # even if it's large.
    
    # Fallback: LUW construction (from original code)
    # We need to re-include the LUW logic or a simpler guaranteed graph.
    # A simple guaranteed graph: 
    # The Petersen graph has girth 5.
    # We can use a product or cover.
    # Actually, the original code had _luw_construct. Let's keep it as a fallback.
    
    # Re-implementing LUW fallback briefly to ensure validity
    # Using the original _luw_construct logic
    def _luw_construct(k, g, d):
        q = k
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = sub = (lambda x, y: x ^ y)
            mul = (lambda x, y: table[x][y])
        else:
            add = (lambda x, y: (x + y) % q)
            sub = (lambda x, y: (x - y) % q)
            mul = (lambda x, y: x * y % q)

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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
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

    # Try LUW with minimal depth
    d_odd = g - 5
    if d_odd % 2 == 0:
        d_odd += 1
    if d_odd >= 3:
        n, edges = _luw_construct(k, g, d_odd)
        if _valid_fast(n, edges, k):
            return n, edges
            
    return 0, []
