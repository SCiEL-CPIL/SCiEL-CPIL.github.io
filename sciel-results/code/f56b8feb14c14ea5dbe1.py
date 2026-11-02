import time
import sys
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    # Moore bound: n >= k + k*(k-1)^(ceil((g-1)/2)/2 - 1) * (something)
    # For k-regular graph of girth g, Moore bound M(k,g):
    # If g is even: M = k * ((k-1)^(g/2 - 1) - 1) / (k - 2)
    # If g is odd: M = 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i = 1 + k * ((k-1)^((g-1)/2) - 1)/(k-2)
    
    def moore_bound(k, g):
        if g % 2 == 0:
            h = g // 2 - 1
            return k * ((k-1)**h - 1) // (k - 2) + 1
        else:
            h = (g - 1) // 2
            return 1 + k * ((k-1)**h - 1) // (k - 2)
    
    mb = moore_bound(k, g)
    
    # LUW construction
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def gf_add(x, y): return x ^ y
        def gf_sub(x, y): return x ^ y
        def gf_mul(x, y): return table[x][y]
    else:
        def gf_add(x, y): return (x + y) % q
        def gf_sub(x, y): return (x - y) % q
        def gf_mul(x, y): return x * y % q

    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
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
                    y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                else:
                    y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    queue = [0]
    head = 0
    
    while head < len(order):
        if time.time() > deadline:
            break
        v_idx = queue[head]
        head += 1
        v = order[v_idx]
        for other in neighbours(v):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(len(order) - 1)
            if v[0] == 0:
                edges.append((v_idx, index[other]))
                
    luw_n = len(order)
    luw_edges = edges
    
    # Verify girth with non-backtracking walks
    def check_girth(n, adj, g):
        """Check that no cycle shorter than g exists using non-backtracking walks."""
        for s in range(n):
            q2 = deque()
            dist = {}
            for v in adj[s]:
                q2.append((v, s))
                dist[(v, s)] = 1
            
            while q2:
                cur, prev = q2.popleft()
                d_cur = dist[(cur, prev)]
                if d_cur >= g - 1:
                    continue
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    if nxt == s:
                        if d_cur + 1 < g:
                            return False
                    else:
                        if (nxt, cur) not in dist:
                            dist[(nxt, cur)] = d_cur + 1
                            q2.append((nxt, cur))
        return True
    
    # Verify the LUW graph
    girth_ok = check_girth(luw_n, [[i for i in range(luw_n) if i != v] for v in range(luw_n)], g)
    # Actually we need the real adjacency list
    adj = [[] for _ in range(luw_n)]
    for u, v in luw_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    girth_ok = check_girth(luw_n, adj, g)
    
    # Precompute non-backtracking walk coefficients for circulant graph check
    # For a 2-step circulant graph C_n(a, b), we need to check that no non-backtracking
    # walk of length < g returns to the start.
    # A walk is determined by the sequence of steps (each +a, -a, +b, -b).
    # Non-backtracking means consecutive steps can't be +a followed by -a (or vice versa),
    # and +b followed by -b (or vice versa).
    # The total displacement is sum of steps. We need this to be 0 mod n for a cycle.
    
    valid_coeffs = set()
    if g > 3:
        states = set()
        # Length 1
        states.add((1, 1, 0))   # step +a, coeff_a=1, coeff_b=0
        states.add((-1, -1, 0)) # step -a
        states.add((2, 0, 1))   # step +b
        states.add((-2, 0, -1)) # step -b
        
        for L in range(2, g):
            new_states = set()
            for last_step, c1, c2 in states:
                for step in [1, -1, 2, -2]:
                    if step == -last_step:
                        continue
                    nc1 = c1 + (1 if step == 1 else -1 if step == -1 else 0)
                    nc2 = c2 + (1 if step == 2 else 0) + (-1 if step == -2 else 0)
                    new_states.add((step, nc1, nc2))
            states = new_states
            if L >= 3:
                for _, c1, c2 in states:
                    valid_coeffs.add((c1, c2))
    
    # Search for circulant graph C_n(a, b)
    # n must be at least moore_bound
    # We search n in [moore_bound, min(luw_n, moore_bound + 300)]
    # For each n, we try random offsets a, b
    
    best_n = luw_n
    best_edges = luw_edges
    
    if girth_ok and luw_n > mb:
        n_max_search = min(luw_n - 1, mb + 300)
        
        for n in range(mb, n_max_search + 1):
            if time.time() > deadline - 10:
                break
            
            # Try random offsets
            # Number of trials per n - scale with remaining time
            remaining = deadline - 10 - time.time()
            if remaining < 1:
                break
            trials = max(10, int(remaining * 20))
            found = False
            
            # Try some deterministic candidates first
            candidates = set()
            # Common small offsets
            for a in range(1, min(n // 2, 20) + 1):
                for b in range(1, min(n // 2, 20) + 1):
                    if a != b:
                        candidates.add((a, b))
            
            # Add random candidates
            for _ in range(trials):
                a = random.randint(1, n // 2 - 1)
                b = random.randint(1, n // 2 - 1)
                if a != b:
                    candidates.add((a, b))
            
            for a, b in candidates:
                if time.time() > deadline - 5:
                    break
                
                # Check girth using precomputed coefficients
                valid = True
                for c1, c2 in valid_coeffs:
                    if (c1 * a + c2 * b) % n == 0:
                        valid = False
                        break
                
                if valid:
                    # Construct edges
                    edges_set = set()
                    for i in range(n):
                        j1 = (i + a) % n
                        if i < j1:
                            edges_set.add((i, j1))
                        else:
                            edges_set.add((j1, i))
                        
                        j2 = (i + b) % n
                        if i < j2:
                            edges_set.add((i, j2))
                        else:
                            edges_set.add((j2, i))
                    
                    # Verify with full girth check
                    adj2 = [[] for _ in range(n)]
                    for u, v in edges_set:
                        adj2[u].append(v)
                        adj2[v].append(u)
                    
                    # Check regular
                    if any(len(adj2[v]) != k for v in range(n)):
                        continue
                    
                    if check_girth(n, adj2, g):
                        best_n = n
                        best_edges = list(edges_set)
                        found = True
                        break
        
        if found:
            return best_n, best_edges
    
    # Fallback to LUW
    return luw_n, luw_edges
# EVOLVE-BLOCK-END
