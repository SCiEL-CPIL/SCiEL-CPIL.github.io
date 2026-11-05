"""Cayley-graph construction for 4-regular girth-10 graph."""
import math
from collections import deque

def construct(k, g):
    if k != 4:
        return _construct_fallback(k, g)
    
    # Try to find a 4-regular Cayley graph on Z_m with girth >= g
    # Using generating set S = {a, -a, b, -b}
    # Key insight: for girth >= 10, we need no non-backtracking closed walks of length < 10
    
    best_n = None
    best_edges = None
    
    # Try cyclic groups Z_m
    # Moore bound for (4,10) is 243
    for m in range(243, 350):
        # Try generating sets {a, b, -a, -b}
        # We can normalize: assume a=1 by scaling (if gcd(1,b,m)=1)
        # More generally, try a from 1 to m//2, b from a+1 to m//2
        # But we can be smarter: use number theory to prune
        
        for a in range(1, m // 2 + 1):
            if math.gcd(a, m) != 1:
                continue  # a must generate part of the group
            
            # Try b values
            for b in range(a + 1, m // 2 + 1):
                if math.gcd(b, m) != 1:
                    continue
                if math.gcd(math.gcd(a, b), m) != 1:
                    continue
                
                # Quick check: no short relations
                if not _quick_girth_check(m, a, b, g):
                    continue
                
                # Full girth check
                if _full_girth_check(m, a, b, g):
                    # Construct edges
                    edge_set = set()
                    for i in range(m):
                        for s in [a, b]:
                            j = (i + s) % m
                            if i < j:
                                edge_set.add((i, j))
                            else:
                                edge_set.add((j, i))
                    edges = sorted(edge_set)
                    
                    if best_n is None or m < best_n:
                        best_n = m
                        best_edges = edges
                    break
        
        if best_n is not None and best_n == m:
            break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Try Z_p x Z_q for primes p, q
    # Or fall back
    return _construct_fallback(4, g)


def _quick_girth_check(m, a, b, g):
    """Quick necessary conditions for girth >= g."""
    # No element of order < g in the generating set
    # a has order m/gcd(a,m) = m (since gcd(a,m)=1)
    # Check: 2a != 0, 3a != 0, etc. mod m (always true if order >= g)
    
    # Check: no relations like a+b=0, a+a=0, etc.
    # a + b != 0 mod m
    if (a + b) % m == 0:
        return False
    # a + a != 0 mod m (i.e., 2a != 0)
    if (2 * a) % m == 0:
        return False
    # b + b != 0 mod m
    if (2 * b) % m == 0:
        return False
    # a + a + a != 0, etc. - these are covered by full check
    
    # Check: a != b, a != -b, etc.
    if a == b or a == m - b or b == m - a:
        return False
    
    return True


def _full_girth_check(m, a, b, g):
    """Check no non-backtracking closed walks of length < g."""
    # Non-backtracking: consecutive steps can't be inverses
    # Steps are from {a, -a, b, -b}
    # Inverse of a is -a, inverse of b is -b
    
    # We'll do BFS on the non-backtracking graph
    # State: (position, last_step) where last_step is in {a, -a, b, -b}
    # Start: (0, s) for each s in {a, -a, b, -b}
    # Transition: (pos, last) -> (pos + s, s) for s in {a,-a,b,-b} \ {-last}
    
    # We want to find if any state (0, s) is reachable from (0, s') in < g-1 steps
    # (total walk length < g)
    
    # Actually, let's just enumerate all non-backtracking walks of length L for L=2..g-1
    # and check if any sum to 0 mod m
    
    steps = [a, -a, b, -b]
    # Map each step to its inverse
    inv = {a: -a, -a: a, b: -b, -b: b}
    
    for L in range(2, g):
        # Use iterative deepening
        # State: (current_sum, last_step)
        # Start with each step
        visited = set()
        for s in steps:
            visited.add((s % m, s))
        
        for _ in range(L - 1):
            new_visited = set()
            for (pos, last) in visited:
                for s in steps:
                    if s == inv[last]:
                        continue
                    new_pos = (pos + s) % m
                    new_visited.add((new_pos, s))
            
            # Check if any state has pos=0
            for (pos, last) in new_visited:
                if pos == 0:
                    return False
            
            visited = new_visited
    
    return True


def _construct_fallback(k, g):
    """Fallback: use the Lazebnik-Ustimenko-Woldar construction from the seed."""
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
