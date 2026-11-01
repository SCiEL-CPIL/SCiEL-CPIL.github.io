import time
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    def check_deadline():
        return time.time() > deadline

    # For the specific case k=3, g=16:
    # 1. First, try to find a small circulant graph C_n(1, a, b).
    # 2. If that fails, fall back to the LUW construction (which is guaranteed to work).
    
    if k == 3 and g == 16:
        # Search for circulant graphs
        try:
            n_circ, edges_circ = _find_circulant_cage(3, 16, deadline)
            if n_circ is not None:
                return n_circ, edges_circ
        except Exception:
            pass

    # Fallback: Lazebnik-Ustimenko-Woldar graph
    return _luw_construct(k, g)


def _find_circulant_cage(k, g, deadline):
    """
    Search for a circulant graph C_n(1, a, b) with girth at least g.
    Assumes k=3.
    """
    # Moore bound for k=3, g=16 (even girth):
    # n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i = 2 * (2^8 - 1) = 510
    # n must be even. So start from 512.
    min_n = 512
    
    # Search space - focus on promising ranges
    # The known cage is 936. We try to find something smaller or equal.
    # If we don't find anything small, we stop and let LUW handle it.
    # LUW(3, 16) has d=11, q=3, n = 2 * 3^11 = 354294. That's huge.
    # Wait, the seed program's LUW implementation might be different or I need to check the size.
    # Actually, for k=3, g=16, the LUW graph CD(11, 3) has 2 * 3^11 vertices.
    # Is there a smaller LUW? No, d must be odd and >= g-5=11.
    # So LUW gives ~354k vertices.
    # The best known is 936.
    # So finding a circulant is crucial for a good score.
    
    max_n = 936 # We don't need to go higher than the known best if we have a fallback, 
                # but maybe there's a circulant between 512 and 936?
                # Let's search up to 936. If we find one, it's better than or equal to 936.
                # If we find one < 936, it's a new record (unlikely but possible).
                
    for n in range(min_n, max_n + 1, 2): 
        if time.time() > deadline - 5:
            return None, None
        
        # Generate pairs (a, b)
        # To save time, we can be smart about (a, b).
        # Symmetries: (a, b) ~ (b, a) ~ (n-a, b) ~ etc.
        # We can restrict a < b and a < n/2, b < n/2.
        
        # Heuristic: try small a, b first? Or random?
        # Let's try a structured search with pruning.
        
        for a in range(2, n // 2):
            if time.time() > deadline - 5:
                return None, None
                
            # Pruning for a: if a is too small, it might create short cycles with 1?
            # 1 and a: cycle length is n/gcd(n, a). If n/gcd(n, a) < g, fail.
            # But we also have b.
            
            for b in range(a + 1, n // 2):
                # Quick filters for 4-cycles
                if _quick_4cycle_check(n, a, b):
                    continue
                
                # Quick filter for 6-cycles
                if _quick_6cycle_check(n, a, b):
                    continue
                
                # Full girth check
                if _has_short_cycle_circulant(n, a, b, g):
                    continue
                
                # If we pass the girth check, we have a candidate.
                # Construct edges
                edges = []
                for i in range(n):
                    for shift in [1, a, b]:
                        j = (i + shift) % n
                        if i < j:
                            edges.append((i, j))
                        
                return n, edges
    
    return None, None


def _quick_4cycle_check(n, a, b):
    """Quick check for 4-cycles in C_n(1, a, b)."""
    # 4-cycle: 0 - x - y - z - 0 where each step is ±1, ±a, ±b
    # A 4-cycle exists if s1 + s2 = 0 (mod n) for some distinct generators s1, s2?
    # No, 4-cycle is v0-v1-v2-v3-v0.
    # v1 = v0 + s1
    # v2 = v1 + s2
    # v3 = v2 + s3
    # v0 = v3 + s4
    # Sum s1+s2+s3+s4 = 0.
    # Also v1 != v3, v0 != v2.
    
    # Simpler check:
    # If (s1 + s2) is a generator, we might have a 2-step path that overlaps?
    # Actually, the standard quick check for circulant graphs with 3 generators:
    # Check if any sum of two distinct generators (with signs) equals 0 or another generator.
    # If s1 + s2 = s3, then 0-s1-(s1+s2)-s2-0 is not a cycle, it's a walk.
    # 0 --s1--> x --s2--> y. If y is connected to 0 by a generator s3, then 0-x-y-0 is a 3-cycle.
    # So if s1+s2 = s3 (or -s3), we have a triangle.
    # For 4-cycles: 0 --s1--> x --s2--> y --s3--> z --s4--> 0.
    # If s1+s2+s3 = s4 (or -s4), and vertices are distinct.
    
    # Let's use the property that C_n(1, a, b) is bipartite iff n is even and all generators are odd?
    # If n is even and 1, a, b are all odd, then the graph is bipartite, so no odd cycles.
    # 4-cycles are even.
    
    # Let's just check if any pair of generators sums to 0 or another generator.
    gens = [1, -1, a, -a, b, -b]
    for i in range(len(gens)):
        for j in range(i+1, len(gens)):
            s = (gens[i] + gens[j]) % n
            if s == 0:
                # 2-cycle? No, simple graph. But 0->u->0 is not allowed in simple graph unless loop.
                # But 0--s1-->u--s2-->0 means s1+s2=0. This is just the edge u-0 traversed twice.
                # It doesn't form a cycle of length 2 in a simple graph.
                # However, if s1 != s2, it's just the same edge.
                # If s1 == s2, it's a loop (not allowed).
                pass
            else:
                # If s is a generator, we have a 3-cycle (triangle) if the vertices are distinct.
                # 0, s1, s1+s2.
                # If s1+s2 = s3, then 0-s1-s3-0 is a triangle.
                if s in gens or (n-s) in gens:
                    # Check for triangle
                    # Vertices: 0, gens[i], (gens[i]+gens[j])%n
                    v1 = gens[i]
                    v2 = (v1 + gens[j]) % n
                    if v1 != 0 and v2 != 0 and v1 != v2:
                        return True
    return False


def _quick_6cycle_check(n, a, b):
    """Quick check for 6-cycles in C_n(1, a, b)."""
    # This is harder to do analytically quickly.
    # We'll rely on the BFS check which is fast enough for n~500-900.
    # The BFS check is O(n * 3) per n.
    # Number of (a,b) pairs is ~ n^2/4.
    # Total ~ n^3. For n=900, 900^3 = 729 million. Too slow.
    # We need better pruning or fewer checks.
    
    # Alternative: Random search for (a,b)?
    # Or smarter enumeration.
    
    # Let's skip this quick check and rely on the BFS, but optimize the BFS.
    # Or, limit the search space.
    
    # Actually, for n=512 to 936, the number of pairs is small.
    # n=512: ~64k pairs. BFS is ~1500 ops. 64k * 1500 = 96M ops. ~1s in Python? Maybe 5s.
    # n=936: ~110k pairs. 110k * 2800 = 300M ops. ~20s.
    # So we can afford to check all pairs for n up to ~700.
    # For larger n, we might need to be selective.
    
    return False


def _has_short_cycle_circulant(n, a, b, g):
    """
    Check if the circulant graph C_n(1, a, b) has a cycle shorter than g.
    Uses BFS from 0.
    Returns True if a short cycle is found, False otherwise.
    """
    visited = [-1] * n
    parent = [-1] * n
    visited[0] = 0
    queue = deque([(0, 0)]) # (node, depth)
    
    max_depth = g // 2
    
    while queue:
        u, depth = queue.popleft()
        
        for shift in [1, a, b]:
            v1 = (u + shift) % n
            v2 = (u - shift) % n
            
            for v in [v1, v2]:
                if visited[v] == -1:
                    visited[v] = depth + 1
                    parent[v] = u
                    if depth + 1 <= max_depth:
                        queue.append((v, depth + 1))
                else:
                    if parent[v] == u or parent[u] == v:
                        continue
                    
                    cycle_len = depth + 1 + visited[v]
                    if cycle_len < g:
                        return True
                        
    return False


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1  # make it odd
    
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
