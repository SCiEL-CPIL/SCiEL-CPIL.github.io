"""Cayley-graph construction for 4-regular girth-10 graph with deadline guard."""
import math
import time

def construct(k, g):
    deadline = time.time() + 115  # 115s safety margin within 150s limit
    
    if k != 4:
        return _construct_fallback(k, g)
    
    # Strategy: search for a 4-regular Cayley graph on Z_m with generating set
    # S = {a, -a, b, -b} that has girth >= g.
    # Optimizations:
    # 1. Precompute coprime candidates per m
    # 2. Quick algebraic rejection before expensive girth check
    # 3. Try a=1 first, then small a values
    # 4. Early termination when valid graph found
    
    best_n = None
    best_edges = None
    
    # Moore bound for (4,10) is 243
    for m in range(243, 400):
        if time.time() > deadline - 3:
            break
        
        # Precompute all values coprime to m in range [1, m//2]
        coprimes = [x for x in range(1, m // 2 + 1) if math.gcd(x, m) == 1]
        
        found_for_m = False
        # Try a=1 first, then other a values in order
        for a in coprimes:
            if time.time() > deadline - 3:
                break
            if found_for_m:
                break
            
            # Quick check: 2a != 0 mod m (ensures a != -a)
            if (2 * a) % m == 0:
                continue
            
            # Try b > a, coprime to m
            for b in coprimes:
                if b <= a:
                    continue
                if time.time() > deadline - 2:
                    break
                
                # Quick algebraic checks for short cycles
                # 2-cycles: already excluded by a != b, a != -b
                if a == m - b or b == m - a:
                    continue
                
                # 3-cycles: a+b, a-b, etc. must not be 0 or +/-a or +/-b
                s1 = (a + b) % m
                s2 = (a - b) % m
                s3 = (-a + b) % m
                s4 = (-a - b) % m
                # Check these aren't 0, +/-a, +/-b
                bad = {0, a, m - a, b, m - b}
                if s1 in bad or s2 in bad or s3 in bad or s4 in bad:
                    continue
                
                # 4-cycles: check (a+b)-a-b = 0 always, but also other combos
                # a+a-b-b=0, a+b-b-a=0 trivially; need non-trivial 4-cycles
                # e.g., a, a, -b, -b: 2a-2b=0 -> a=b (excluded)
                # a, -a, b, -b: 0 (trivial backtrack, excluded by non-backtracking)
                # a, b, -a, -b: b-a=0 -> a=b (excluded)
                # So 4-cycles are mostly excluded by the above; do full check
                
                # Full girth check
                if _check_girth_fast(m, a, b, g):
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
                    found_for_m = True
                    break
        
        if found_for_m:
            break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: use the Lazebnik-Ustimenko-Woldar construction
    return _construct_fallback(4, g)


def _check_girth_fast(m, a, b, g):
    """Check that the Cayley graph of Z_m with S={a,-a,b,-b} has girth >= g.
    
    Uses BFS on the non-backtracking walk graph.
    State: (position, last_step_index) where last_step_index in {0,1,2,3}
    corresponding to steps [a, -a, b, -b].
    
    We check for closed non-backtracking walks of length L for L = 2, ..., g-1.
    If any such walk exists, girth < g.
    """
    steps = [a, -a, b, -b]
    inv = [1, 0, 3, 2]
    
    # Precompute step additions modulo m
    step_mod = [s % m for s in steps]
    
    # Layer 1: all single steps from 0
    current = set()
    for i in range(4):
        pos = step_mod[i]
        if pos == 0:
            return False
        current.add((pos, i))
    
    # Layers 2 through g-1
    for L in range(2, g):
        next_layer = set()
        for (pos, last) in current:
            for s_idx in range(4):
                if s_idx == inv[last]:
                    continue
                new_pos = (pos + step_mod[s_idx]) % m
                if new_pos == 0:
                    return False
                next_layer.add((new_pos, s_idx))
        
        current = next_layer
        # Early exit: if layer is empty, no more walks possible
        if not current:
            break
    
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
