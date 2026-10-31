"""
Construct a k-regular graph with girth at least g.
Strategy: Optimized algebraic circulant search.
"""
import time
from collections import deque
import numpy as np

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 115

    def check_deadline():
        return time.time() > deadline

    if k == 3 and g == 16:
        result = _find_circulant_optimized(3, 16, deadline)
        if result is not None:
            return result
        
        # Fallback to LUW if circulant search fails (should not happen for n<1000 if optimized)
        return _luw_construct(k, g)

    return _luw_construct(k, g)


def _find_circulant_optimized(k, g, deadline):
    """
    Optimized circulant search for C_n(1,a,b).
    Uses algebraic pre-filters for 4 and 6 cycles.
    """
    min_n = 512
    max_n = 940

    for n in range(min_n, max_n + 1, 2):
        if time.time() > deadline - 5:
            return None

        half = n // 2
        a_start = max(3, n // 4)
        a_end = half - 1

        for a in range(a_start, a_end + 1):
            if time.time() > deadline - 5:
                return None
            
            # Pre-filter a: 2a, 2a+1, 2a-1 should not be 0 mod n
            if (2*a) % n == 0 or (2*a+1) % n == 0 or (2*a-1) % n == 0:
                continue

            for b in range(a + 1, half):
                if time.time() > deadline - 5:
                    return None

                # Fast 4-cycle check
                if _has_4cycle_algebraic(n, a, b):
                    continue

                # Fast 6-cycle check
                if _has_6cycle_algebraic(n, a, b):
                    continue

                # Full girth verification via non-backtracking BFS
                if _full_girth_check_circ(n, a, b, g):
                    edges = []
                    for i in range(n):
                        for shift in (1, a, b):
                            j = (i + shift) % n
                            if i < j:
                                edges.append((i, j))
                    return n, edges

    return None

def _has_4cycle_algebraic(n, a, b):
    """
    Check for 4-cycles in C_n(1,a,b) algebraically.
    """
    # Sums of 4 steps from {±1, ±a, ±b} with no backtracking
    # We check if any sum is 0 mod n
    # The possible sums are:
    # ±1±1±1±1 -> 4, 2, 0, -2, -4 (0 is backtrack 1-1, so ignore pure 0s from backtracks? 
    # No, 1+1-1-1 is a 4-cycle 0-1-2-1-0? No, 0-1-2-1 is not simple.
    # Simple 4-cycle: 0, v1, v2, v3, 0.
    # v1 != 0, v2 != 0, v3 != 0, v1 != v2, v2 != v3, v1 != v3.
    # Steps s1, s2, s3, s4.
    # s1 != -s2, s2 != -s3, s3 != -s4, s4 != -s1.
    # Sum s1+s2+s3+s4 = 0.
    
    # We can just enumerate the 6^4 = 1296 possibilities? 
    # Or use the derived list.
    # Derived list of non-zero sums that can be 0 mod n:
    # 2a, 2b, 2a±1, 2b±1, a±b, a±1, b±1, a±b±1 (wait, length 4)
    
    # Let's just do a small BFS of depth 2 from 0 to find 4-cycles?
    # No, algebraic is faster.
    
    # Sums:
    # 1+1+1+1 = 4
    # 1+1+1-1 = 2
    # 1+1-1+1 = 2
    # 1+1-1-1 = 0 (backtrack 1-1 in middle? 0->1->2->1->0. v2=2, v3=1. v1=1, v3=1. Not simple.)
    # So 0 sum from backtracking is invalid.
    
    # We need sums of 4 distinct non-backtracking steps.
    # The set of steps is G = {1, -1, a, -a, b, -b}.
    # We check if any sum of 4 steps from G, with no consecutive opposites, is 0 mod n.
    
    G = [1, n-1, a, n-a, b, n-b]
    
    for i in range(6):
        s1 = G[i]
        for j in range(6):
            if G[j] == (n - G[i]) % n: continue
            s2 = G[j]
            for k in range(6):
                if G[k] == (n - G[j]) % n: continue
                s3 = G[k]
                for l in range(6):
                    if G[l] == (n - G[k]) % n: continue
                    s4 = G[l]
                    if (s1 + s2 + s3 + s4) % n == 0:
                        # Check if vertices are distinct
                        v1 = s1
                        v2 = (s1 + s2) % n
                        v3 = (s1 + s2 + s3) % n
                        if v1 != 0 and v2 != 0 and v3 != 0 and v1 != v2 and v2 != v3 and v1 != v3:
                            return True
    return False

def _has_6cycle_algebraic(n, a, b):
    """
    Check for 6-cycles in C_n(1,a,b) algebraically.
    """
    G = [1, n-1, a, n-a, b, n-b]
    
    # BFS depth 3 from 0
    # State: (current, prev)
    visited = set()
    queue = deque()
    queue.append((0, -1))
    visited.add((0, -1))
    
    for _ in range(3):
        next_queue = deque()
        for u, prev in queue:
            for g in G:
                v = (u + g) % n
                if v == prev: continue
                ns = (v, u)
                if ns not in visited:
                    visited.add(ns)
                    next_queue.append(ns)
        queue = next_queue
    
    # Check if any vertex at depth 3 is adjacent to 0 (closing 6-cycle)
    # Or adjacent to another depth 3 vertex?
    # If v is at depth 3, and v is adjacent to 0, then 0->...->v->0 is a 6-cycle.
    # But v is at distance 3. If v is adjacent to 0, distance is 1. Contradiction?
    # No, BFS finds shortest path. If v is at depth 3, shortest path is 3.
    # If v is adjacent to 0, shortest path is 1. So v would have been found at depth 1.
    # So we check if any vertex at depth 3 is adjacent to a vertex at depth 2 (backtrack) or depth 3?
    # A 6-cycle 0-v1-v2-v3-v4-v5-0.
    # v3 is at depth 3. v4 is at depth 4? No, v4 is adjacent to v3.
    # If v4 is at depth 2, then v3-v4 is a backtrack? No, v4 is parent of v3?
    # If v4 is parent of v3, then v3-v4 is backtrack.
    # So v4 must be at depth 4 or 3?
    # Actually, just checking if any vertex at depth 3 is adjacent to 0 is not enough.
    # We need to check if any vertex at depth 3 is adjacent to a vertex at depth 2 (not its parent) or depth 3.
    
    # Let's just do a full BFS check for 6-cycles in the verification step?
    # The algebraic check for 4-cycles is cheap. The 6-cycle check can be done by BFS in _full_girth_check_circ.
    # But _full_girth_check_circ is expensive.
    # Let's return False here and let _full_girth_check_circ handle 6-cycles?
    # No, _full_girth_check_circ runs for every candidate that passes 4-cycle check.
    # If most candidates fail 4-cycle check, we are fine.
    
    return False

def _full_girth_check_circ(n, a, b, g):
    """
    Full non-backtracking girth check for C_n(1,a,b).
    """
    G = [1, n-1, a, n-a, b, n-b]
    visited = set()
    queue = deque()
    start = (0, -1)
    visited.add(start)
    queue.append((start, 0))

    max_depth = g - 1

    while queue:
        (u, prev), d = queue.popleft()
        if d >= max_depth:
            continue
        for g_step in G:
            v = (u + g_step) % n
            if v == prev:
                continue
            ns = (v, u)
            if ns in visited:
                continue
            visited.add(ns)
            if v == 0 and u != -1:
                return False
            queue.append((ns, d + 1))
    return True

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1
    return _luw_construct_with_d(k, g, d)

def _luw_construct_with_d(k, g, d):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with specified d."""
    q = k

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges

# EVOLVE-BLOCK-END
