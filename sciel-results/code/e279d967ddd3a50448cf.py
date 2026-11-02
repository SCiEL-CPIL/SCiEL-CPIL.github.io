import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill

    # Precompute valid non-backtracking walk coefficients for girth check
    # For a 4-regular graph with generators {a, b, -a, -b} on Z_n,
    # a non-backtracking walk of length L has coefficient (c1, c2) where
    # c1 = net count of a-steps, c2 = net count of b-steps
    # We need c1*a + c2*b != 0 mod n for all valid (c1, c2) with 3 <= L <= g-1
    
    valid_coeffs = set()
    if g > 3:
        # State: (last_step_type, c1, c2) where last_step_type in {1,-1,2,-2}
        # 1 = a, -1 = -a, 2 = b, -2 = -b
        states = set()
        # Length 1 walks
        states.add((1, 1, 0))
        states.add((-1, -1, 0))
        states.add((2, 0, 1))
        states.add((-2, 0, -1))
        
        for L in range(2, g):
            new_states = set()
            for last_step, c1, c2 in states:
                for step in [1, -1, 2, -2]:
                    if step == -last_step:  # non-backtracking constraint
                        continue
                    nc1 = c1 + (1 if step == 1 else -1 if step == -1 else 0)
                    nc2 = c2 + (1 if step == 2 else -1 if step == -2 else 0)
                    new_states.add((step, nc1, nc2))
            states = new_states
            if L >= 3:
                for _, c1, c2 in states:
                    valid_coeffs.add((c1, c2))
    
    # Moore bound for k-regular graph of girth g
    if g % 2 == 0:
        R = g // 2
        moore_bound = k + k * sum((k - 1) ** i for i in range(R - 1))
    else:
        R = (g - 1) // 2
        moore_bound = k + k * sum((k - 1) ** i for i in range(R))
    
    # For k=4, g=10: R=5, moore_bound = 4 + 4*(1 + 3 + 9 + 27) = 4 + 4*40 = 164? 
    # Wait, let me recalculate. Moore bound for even girth 2R:
    # n >= k + k*(k-1)^(R-1) + k*(k-1)^R ... no
    # Standard Moore bound: for girth g = 2R (even), n >= k + k*sum_{i=0}^{R-2}(k-1)^i * (k-1) ... 
    # Actually: n >= 1 + k*sum_{i=0}^{R-1}(k-1)^i for odd, 
    # n >= k*sum_{i=0}^{R-1}(k-1)^i for even? Let me use the standard formula.
    # For k-regular, girth g:
    # If g = 2R: n >= k * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(R-1)) = k * ((k-1)^R - 1)/(k-2)
    # If g = 2R+1: n >= 1 + k * ((k-1)^R - 1)/(k-2)
    
    if k > 2:
        if g % 2 == 0:
            R = g // 2
            moore_bound = k * ((k - 1) ** R - 1) // (k - 2)
        else:
            R = (g - 1) // 2
            moore_bound = 1 + k * ((k - 1) ** R - 1) // (k - 2)
    else:
        moore_bound = g  # degenerate case
    
    # Strategy 1: Search for circulant graph C(n; a, b)
    # Try n from moore_bound to moore_bound + 200
    best_result = None
    
    n_max_search = min(moore_bound + 200, 600)  # cap to avoid too long
    
    for n in range(moore_bound, n_max_search + 1):
        if time.time() > deadline - 20:
            break
        
        # Try random pairs (a, b)
        # a and b should be in [1, n//2] and a != b, a+b != 0 mod n, etc.
        num_trials = 200
        found = False
        
        for trial in range(num_trials):
            if time.time() > deadline - 15:
                break
            
            a = random.randint(1, n // 2)
            b = random.randint(1, n // 2)
            
            # Basic checks: a != b, a != n-b, a+b != 0 mod n
            if a == b or a == n - b or (a + b) % n == 0:
                continue
            # Also need 2a != 0, 2b != 0 (no 2-cycles)
            if 2 * a % n == 0 or 2 * b % n == 0:
                continue
            
            # Check girth using precomputed coefficients
            valid = True
            for c1, c2 in valid_coeffs:
                if (c1 * a + c2 * b) % n == 0:
                    valid = False
                    break
            
            if valid:
                # Construct edges for C(n; a, b)
                edges = set()
                for i in range(n):
                    j1 = (i + a) % n
                    if i < j1:
                        edges.add((i, j1))
                    else:
                        edges.add((j1, i))
                    
                    j2 = (i + b) % n
                    if i < j2:
                        edges.add((i, j2))
                    else:
                        edges.add((j2, i))
                
                # Verify degree is exactly k=4
                # In C(n; a, b), each vertex has degree 4 if a, b, -a, -b are distinct
                # and no self-loops
                if len(edges) == n * k // 2:
                    best_result = (n, list(edges))
                    found = True
                    break
        
        if found:
            return best_result
    
    # Strategy 2: Try bipartite Cayley graph on Z_160 (from current program)
    if time.time() < deadline - 10:
        def build_bipartite_cayley(group_order, generators):
            n = group_order
            S = [g % n for g in generators]
            edges = []
            for h in range(n):
                for s in S:
                    h2 = (h + s) % n
                    v1 = h
                    v2 = h2 + n
                    edges.append((v1, v2))
            return 2 * n, edges
        
        def check_girth(n, edges, target_girth):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            half_girth = target_girth // 2
            
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = deque([start])
                
                while queue:
                    u = queue.popleft()
                    if dist[u] >= half_girth:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < target_girth:
                                return False
            return True
        
        random.seed(42)
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            S = [random.randint(1, 159) for _ in range(4)]
            g_val = 160
            for s in S:
                g_val = math.gcd(g_val, s)
            if g_val != 1:
                continue
            
            n, edges = build_bipartite_cayley(160, S)
            if check_girth(n, edges, 10):
                return n, edges
    
    # Strategy 3: Fallback to LUW construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        gf_add = lambda x, y: x ^ y
        gf_sub = lambda x, y: x ^ y
        gf_mul = lambda x, y: table[x][y]
    else:
        gf_add = lambda x, y: (x + y) % q
        gf_sub = lambda x, y: (x - y) % q
        gf_mul = lambda x, y: x * y % q
    
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
    
    return len(order), edges
