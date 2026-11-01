"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
import math as _math
from collections import deque

def _verify_girth_fast(adj, n, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise. Optimized with early termination."""
    for start in range(n):
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _luw_construct(q, g):
    """LUW construction for (q+1)-regular graphs."""
    if q == 3:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        add, sub, mul = (lambda x,y:(x+y)%q), (lambda x,y:(x-y)%q), (lambda x,y:x*y%q)
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        start = (0, (None,) + (0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges
    elif q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add, sub, mul = (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        start = (0, (None,) + (0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges
    else:
        p = q
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        add, sub, mul = (lambda x,y:(x+y)%p), (lambda x,y:(x-y)%p), (lambda x,y:x*y%p)
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        def neighbours(vertex):
            side, x = vertex
            for first in range(p):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        start = (0, (None,) + (0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

def _build_lift_petersen(m, voltages):
    """Build a voltage graph lift of the Petersen graph with given voltages.
    Returns (n, adj, edges) or None if invalid."""
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    n = 10 * m
    adj = [set() for _ in range(n)]
    edges = []
    
    for u, v in petersen_edges:
        vol = voltages[(u, v)]
        for i in range(m):
            a = u * m + i
            b = v * m + (i + vol) % m
            if b not in adj[a]:
                adj[a].add(b)
                adj[b].add(a)
                edges.append((a, b))
    
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    
    return n, adj, edges

def _build_lift_k33(m, voltages):
    """Build a voltage graph lift of K3,3 with given voltages.
    K3,3 has 6 vertices, 9 edges. Lift has 6m vertices.
    Returns (n, adj, edges) or None if invalid."""
    k33_edges = [
        (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5)
    ]
    
    n = 6 * m
    adj = [set() for _ in range(n)]
    edges = []
    
    for u, v in k33_edges:
        vol = voltages[(u, v)]
        for i in range(m):
            a = u * m + i
            b = v * m + (i + vol) % m
            if b not in adj[a]:
                adj[a].add(b)
                adj[b].add(a)
                edges.append((a, b))
    
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    
    return n, adj, edges

def _is_prime(m):
    """Check if m is prime."""
    if m < 2:
        return False
    if m < 4:
        return True
    if m % 2 == 0 or m % 3 == 0:
        return False
    i = 5
    while i * i <= m:
        if m % i == 0 or m % (i + 2) == 0:
            return False
        i += 6
    return True

def _all_primitive_roots(m, limit=50):
    """Find up to `limit` primitive roots modulo m, or return empty list if m not prime."""
    if not _is_prime(m):
        return []
    phi = m - 1
    factors = []
    temp = phi
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            factors.append(d)
            while temp % d == 0:
                temp //= d
        d += 1
    if temp > 1:
        factors.append(temp)
    
    roots = []
    for g in range(2, m):
        if len(roots) >= limit:
            break
        is_primitive = True
        for p in factors:
            if pow(g, phi // p, m) == 1:
                is_primitive = False
                break
        if is_primitive:
            roots.append(g)
    return roots

def _thue_morse(n):
    """Return the Thue-Morse sequence of length n (0-indexed)."""
    tm = [0] * n
    for i in range(1, n):
        tm[i] = tm[i >> 1] ^ (i & 1)
    return tm

def _thue_morse_shifts(m):
    """Generate Thue-Morse based voltage shifts for a lift of size m.
    Uses the Thue-Morse sequence to create shifts that have low-order
    cancellation properties, structurally minimizing short closed walks."""
    tm = _thue_morse(m)
    # Map Thue-Morse bits to shifts: use both the bit and its complement
    # to create a set of distinct shifts with good distribution
    shifts = []
    for i in range(m):
        # Use the Thue-Morse bit to create a shift pattern
        # Alternate between i and m-1-i based on the bit
        if tm[i] == 0:
            shifts.append(i)
        else:
            shifts.append(m - 1 - i)
    return shifts

def _thue_morse_shifts_v2(m):
    """Alternative Thue-Morse shift: use the sequence directly as a permutation.
    Creates a bijection that has the self-similar structure of Thue-Morse."""
    tm = _thue_morse(m)
    # Create a permutation based on Thue-Morse ordering
    indices = list(range(m))
    # Sort by Thue-Morse value (stable sort preserves original order for ties)
    # Actually, let's use a different approach: map each index to its
    # Thue-Morse-complement position
    perm = [0] * m
    for i in range(m):
        # The Thue-Morse complement of i
        perm[i] = m - 1 - i if tm[i] == 1 else i
    return perm

def _cubic_shifts(m):
    """Generate cubic polynomial shifts: shift[i] = i^3 mod m.
    Cubic polynomials have different collision properties than linear/powers."""
    return [(i * i * i) % m for i in range(m)]

def _fibonacci_shifts(m):
    """Generate Fibonacci-based shifts for good distribution."""
    if m <= 1:
        return list(range(m))
    fib = [0, 1]
    for i in range(2, m + 1):
        fib.append((fib[i-1] + fib[i-2]) % m)
    # Use the Fibonacci sequence as shifts (may not be a permutation, but good distribution)
    return [fib[i] for i in range(m)]

def _build_lift_generic(base_edges, m, voltages):
    """Generic voltage lift builder. base_edges: list of (u,v) pairs.
    Returns (n, adj, edges) or None."""
    n = len(set([u for u,v in base_edges] + [v for u,v in base_edges])) * m
    adj = [set() for _ in range(n)]
    edges = []
    
    for u, v in base_edges:
        vol = voltages.get((u, v), 0)
        for i in range(m):
            a = u * m + i
            b = v * m + (i + vol) % m
            if b not in adj[a]:
                adj[a].add(b)
                adj[b].add(a)
                edges.append((a, b))
    
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    
    return n, adj, edges

def _try_lift_with_shifts(base_edges, base_n, m, g, shifts):
    """Try a voltage lift with given shifts. Returns (n, edges) or None."""
    # Assign shifts to edges cyclically
    voltages = {}
    for i, (u, v) in enumerate(base_edges):
        vol = shifts[i % len(shifts)]
        voltages[(u, v)] = vol
        voltages[(v, u)] = (-vol) % m
    
    result = _build_lift_generic(base_edges, m, voltages)
    if result is None:
        return None
    n, adj, edges = result
    if _verify_girth_fast(adj, n, g):
        return n, edges
    return None

def _random_3reg(n, g, seed):
    """Generate a random 3-regular graph and check girth."""
    rng = _random.Random(seed)
    stubs = []
    for v in range(n):
        for _ in range(3):
            stubs.append(v)
    rng.shuffle(stubs)
    adj = [set() for _ in range(n)]
    edges = []
    valid = True
    for i in range(0, n * 3, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v or v in adj[u]:
            valid = False
            break
        adj[u].add(v)
        adj[v].add(u)
        edges.append((u, v))
    if not valid:
        return None
    visited = set([0])
    stack = [0]
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if w not in visited:
                visited.add(w)
                stack.append(w)
    if len(visited) != n:
        return None
    if _verify_girth_fast(adj, n, g):
        return n, edges
    return None

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 115
    
    if k != 3:
        q = k
        n, edges = _luw_construct(q, g)
        return n, edges
    
    # Moore bound for k=3, girth g
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2
    
    best_result = None
    best_n = float('inf')
    
    # K3,3 base edges (6 vertices, 9 edges)
    k33_edges = [
        (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5)
    ]
    
    # Petersen base edges (10 vertices, 15 edges)
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # Strategy 1: Systematic algebraic voltage lifts with multiple shift families
    # For each prime m, try multiple primitive roots and shift patterns
    
    m_min_k33 = max(1, (moore + 5) // 6)
    m_min_petersen = max(1, (moore + 9) // 10)
    
    # Search K3,3 lifts with various algebraic shift patterns
    for m in range(m_min_k33, m_min_k33 + 400):
        if _time.time() > deadline - 15:
            break
        n = 6 * m
        if n < moore:
            continue
        if n >= best_n:
            break
        
        found = False
        
        # Try Thue-Morse shifts
        tm_shifts = _thue_morse_shifts(m)
        result = _try_lift_with_shifts(k33_edges, 6, m, g, tm_shifts)
        if result is not None:
            n_lift, edges = result
            if n_lift < best_n:
                best_n = n_lift
                best_result = (n_lift, edges)
            found = True
            break
        
        # Try Thue-Morse v2
        tm2_shifts = _thue_morse_shifts_v2(m)
        result = _try_lift_with_shifts(k33_edges, 6, m, g, tm2_shifts)
        if result is not None:
            n_lift, edges = result
            if n_lift < best_n:
                best_n = n_lift
                best_result = (n_lift, edges)
            found = True
            break
        
        # Try cubic shifts
        cub_shifts = _cubic_shifts(m)
        result = _try_lift_with_shifts(k33_edges, 6, m, g, cub_shifts)
        if result is not None:
            n_lift, edges = result
            if n_lift < best_n:
                best_n = n_lift
                best_result = (n_lift, edges)
            found = True
            break
        
        # Try multiple primitive roots with power shifts
        if _is_prime(m):
            roots = _all_primitive_roots(m, limit=10)
            for root in roots:
                if _time.time() > deadline - 15:
                    break
                # Power shifts: shift[i] = root^i mod m
                power_shifts = [pow(root, i, m) for i in range(m)]
                result = _try_lift_with_shifts(k33_edges, 6, m, g, power_shifts)
                if result is not None:
                    n_lift, edges = result
                    if n_lift < best_n:
                        best_n = n_lift
                        best_result = (n_lift, edges)
                    found = True
                    break
        
        if found:
            break
        
        # Try Fibonacci shifts
        fib_shifts = _fibonacci_shifts(m)
        result = _try_lift_with_shifts(k33_edges, 6, m, g, fib_shifts)
        if result is not None:
            n_lift, edges = result
            if n_lift < best_n:
                best_n = n_lift
                best_result = (n_lift, edges)
            break
    
    # Search Petersen lifts with algebraic shifts
    for m in range(m_min_petersen, m_min_petersen + 300):
        if _time.time() > deadline - 15:
            break
        n = 10 * m
        if n < moore:
            continue
        if n >= best_n:
            break
        
        found = False
        
        # Try Thue-Morse shifts
        tm_shifts = _thue_morse_shifts(m)
        result = _try_lift_with_shifts(petersen_edges, 10, m, g, tm_shifts)
        if result is not None:
            n_lift, edges = result
            if n_lift < best_n:
                best_n = n_lift
                best_result = (n_lift, edges)
            found = True
            break
        
        # Try multiple primitive roots
        if _is_prime(m):
            roots = _all_primitive_roots(m, limit=8)
            for root in roots:
                if _time.time() > deadline - 15:
                    break
                power_shifts = [pow(root, i, m) for i in range(m)]
                result = _try_lift_with_shifts(petersen_edges, 10, m, g, power_shifts)
                if result is not None:
                    n_lift, edges = result
                    if n_lift < best_n:
                        best_n = n_lift
                        best_result = (n_lift, edges)
                    found = True
                    break
        
        if found:
            break
    
    # Strategy 2: LUW construction as baseline
    if _time.time() < deadline - 20:
        n_luw, edges_luw = _luw_construct(3, g)
        if n_luw < best_n:
            best_n = n_luw
            best_result = (n_luw, edges_luw)
    
    # Strategy 3: Random 3-regular graphs
    if _time.time() < deadline - 15 and best_result is None:
        n_try = moore
        if n_try % 2 != 0:
            n_try += 1
        while n_try < 5000 and _time.time() < deadline - 10:
            for attempt in range(50):
                if _time.time() > deadline - 10:
                    break
                result = _random_3reg(n_try, g, seed=attempt * 104729 + n_try)
                if result is not None:
                    n_r, edges_r = result
                    if n_r < best_n:
                        best_n = n_r
                        best_result = (n_r, edges_r)
                    break
            n_try += 32
    
    # Fallback: LUW construction
    if best_result is None:
        n_luw, edges_luw = _luw_construct(3, g)
        best_result = (n_luw, edges_luw)
    
    return best_result
# EVOLVE-BLOCK-END
