import time
import itertools

def _moore_bound(k, g):
    if g % 2 == 0:
        r = g // 2
        return 2 * (k * (k - 1) ** (r - 1))
    else:
        r = (g - 1) // 2
        return 1 + k * (k - 1) ** r

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction: q-regular bipartite graph
    with girth >= d+5 for odd d (or d+4 for even d), q prime or 4."""
    q = k
    # Determine d from girth g
    # Girth is at least d + 5 if d is odd, d + 4 if d is even
    # We want girth >= g
    if (g - 5) % 2 == 1:
        d = g - 5  # d odd, girth >= d+5 = g
    else:
        d = g - 4  # d even, girth >= d+4 = g
    
    # For q=4, use GF(4) with XOR-based operations
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q
    
    # Rule for computing neighbours
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    pos = 0
    while pos < len(order):
        vertex = order[pos]
        pos += 1
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _check_cyclic_girth(n, d1, d2, g):
    """Check if Cayley graph on Z_n with S={d1, -d1, d2, -d2, n/2} has girth >= g."""
    if n % 2 != 0:
        return False
    half_n = n // 2
    if d1 == 0 or d2 == 0 or d1 == half_n or d2 == half_n:
        return False
    if d1 == d2 or d1 == n - d2:
        return False
    
    generators = [d1, n-d1, d2, n-d2, half_n]
    # Map generator value to index for backtracking check
    # Backtracking means s_next == -s_prev
    # In Z_n, -s is n-s.
    
    # We need to check if there is a cycle of length < g.
    # A cycle is a sequence s_1...s_L such that sum s_i = 0,
    # no prefix sum is 0, and s_{i+1} != -s_i.
    
    # BFS on the "state space" of (current_sum, last_gen_index, length)
    # But we can just do DFS/BFS up to depth g-1.
    # Since g is small (9), depth is 8.
    # Branching factor is at most 4 (can't go back).
    # 4^8 = 65536 paths.
    # We can prune if sum becomes 0.
    
    # To be faster, we can use the linear constraint approach if it's too slow, 
    # but for n~1000 and few d1,d2 candidates, this might be fine.
    # However, we are scanning many d1, d2.
    # Let's use the precomputed relations approach for speed.
    
    # For this specific small g, direct BFS is simpler to implement correctly.
    # We'll optimize by early termination.
    
    # State: (current_sum, last_gen_idx)
    # We start at sum=0, last_gen_idx=-1.
    # We want to reach sum=0 at length L < g.
    # But we must not reach sum=0 before L.
    
    # Actually, we just need to check if 0 is reachable from 0 in < g steps
    # without visiting 0 in between and without backtracking.
    
    # This is equivalent to checking if the graph has a cycle of length < g.
    # In a Cayley graph, this is a relation.
    
    # Let's just generate all non-backtracking walks of length 3 to g-1
    # and check if any sums to 0.
    
    # Precompute all valid sequences of generator indices of length L
    # such that no adjacent indices are "inverse".
    # Inverse pairs: (0,1), (1,0), (2,3), (3,2). (4,4) is self-inverse, so (4,4) is backtracking?
    # Yes, if s_i = n/2, then -s_i = n/2. So s_{i+1} cannot be 4 if s_i is 4.
    
    inv_pairs = {(0,1), (1,0), (2,3), (3,2), (4,4)}
    
    for L in range(3, g):
        # Generate all sequences of length L
        # To avoid generating millions, we can do a DFS
        def dfs(current_sum, last_idx, length):
            if length == L:
                return current_sum == 0
            for next_idx in range(5):
                if (last_idx, next_idx) in inv_pairs:
                    continue
                new_sum = (current_sum + generators[next_idx]) % n
                if new_sum == 0 and length + 1 < L:
                    # Prefix sum is 0, not a simple cycle
                    continue
                if new_sum == 0 and length + 1 == L:
                    return True
                if dfs(new_sum, next_idx, length + 1):
                    return True
            return False
        
        for start_idx in range(5):
            if dfs(generators[start_idx], start_idx, 1):
                return False
                
    return True

def construct(k, g):
    """Construct a k-regular graph with girth >= g.
    Returns (n, edges) where edges is a list of (i, j) pairs."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Try Cyclic Interval Construction
    # k must be 5 for this specific symmetric set {d1, -d1, d2, -d2, n/2}
    if k == 5:
        mb = _moore_bound(k, g)
        # n must be even
        if mb % 2 != 0:
            mb += 1
            
        # Scan n
        # We expect n to be close to MB.
        # Limit search to reasonable range
        max_n = 1100
        
        for n in range(mb, max_n + 1, 2):
            if time.time() > deadline:
                break
                
            half_n = n // 2
            # d1, d2 in 1..half_n-1
            # We can restrict d1 < d2
            for d1 in range(1, half_n):
                if time.time() > deadline:
                    break
                for d2 in range(d1 + 1, half_n):
                    if _check_cyclic_girth(n, d1, d2, g):
                        # Found valid graph
                        edges = []
                        # Generate edges
                        # Vertices 0..n-1
                        # Neighbours of i: i+d1, i-d1, i+d2, i-d2, i+n/2
                        # To avoid duplicates, only add edge if i < j
                        for i in range(n):
                            nbs = [
                                (i + d1) % n,
                                (i - d1) % n,
                                (i + d2) % n,
                                (i - d2) % n,
                                (i + half_n) % n
                            ]
                            for j in nbs:
                                if i < j:
                                    edges.append((i, j))
                        return n, edges
                        
    # Fallback to LUW
    n, edges = _luw_construct(k, g)
    return n, edges
