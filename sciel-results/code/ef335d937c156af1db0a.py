import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Tries Cayley graphs on SL(2, q) for small q first to minimize n.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin

    if k != 3:
        # Fallback for other k values
        pass

    # Helper to build SL(2, q) Cayley graph
    def try_sl2_cayley(q, poly=None):
        """
        Try to find a 3-regular Cayley graph on SL(2, q) with girth >= g.
        poly: irreducible polynomial for GF(2^m) if q is power of 2, else None for prime q.
        """
        # 1. Field Arithmetic Setup
        if poly is None:
            # Prime field Z_q
            field_add = lambda a, b: (a + b) % q
            field_sub = lambda a, b: (a - b) % q
            field_mul = lambda a, b: (a * b) % q
            field_inv = lambda a: pow(a, q - 2, q) if a != 0 else 0
            field_div = lambda a, b: field_mul(a, field_inv(b)) if b != 0 else 0
            # Element representation: (a, b, c, d)
            # det = ad - bc = 1
        else:
            # Extension field GF(2^m)
            m = len(bin(poly)) - 3 # Approx, better to pass m
            # Actually, let's just use the table method from before for GF(16)
            # But for generality, let's stick to the specific implementation for q=16 if needed.
            # For now, we will only call this for prime q or specific GF(16) case.
            # To keep it simple, I'll implement the GF(16) logic separately if q is power of 2.
            # But the directive suggests structural change. Let's focus on prime q first.
            # If q is power of 2, we use the previous logic.
            # Let's assume q is prime for this new attempt, and keep the old GF(16) logic as a separate branch.
            return None # Fallback to other methods

        # 2. Generate SL(2, q) elements
        elements = []
        elem_to_idx = {}
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    # ad - bc = 1 => ad = 1 + bc
                    # If a == 0, then -bc = 1 => bc = -1
                    # If a != 0, d = (1 + bc) / a
                    if a == 0:
                        # bc = -1 mod q
                        if b == 0:
                            continue
                        # c = -1 / b
                        c_val = field_div(-1, b)
                        if 0 <= c_val < q:
                            d = 0 # Since a=0, ad=0, so -bc=1 => bc=-1. d is free? No, det=1.
                            # Wait, if a=0, det = -bc = 1 => bc = -1.
                            # d can be anything? No, the matrix is [[0, b], [c, d]].
                            # det = 0*d - b*c = -bc. So -bc = 1.
                            # d is not constrained by det?
                            # Yes, d is free?
                            # No, SL(2,q) requires det=1.
                            # If a=0, -bc=1. d is arbitrary?
                            # Let's check. [[0, b], [c, d]]. det = -bc.
                            # So d is indeed free in the determinant constraint.
                            # However, we need to generate unique elements.
                            # Let's iterate d as well.
                            for d in range(q):
                                e = (a, b, c_val, d)
                                if e not in elem_to_idx:
                                    elem_to_idx[e] = len(elements)
                                    elements.append(e)
                    else:
                        # d = (1 + bc) / a
                        bc = field_mul(b, c)
                        rhs = field_add(1, bc)
                        d = field_div(rhs, a)
                        e = (a, b, c, d)
                        if e not in elem_to_idx:
                            elem_to_idx[e] = len(elements)
                            elements.append(e)
        
        n = len(elements)
        
        # 3. Find Involutions
        # M^2 = I => Tr(M) = 0 => a + d = 0 => d = -a
        # And M != I
        involutions = []
        for e in elements:
            a, b, c, d = e
            if field_add(a, d) == 0 and not (a == 1 and b == 0 and c == 0 and d == 1):
                involutions.append(e)
        
        if len(involutions) < 3:
            return None

        # 4. Search for generators
        random.seed(42) # Fixed seed for reproducibility in trials
        
        def multiply(m1, m2):
            a1, b1, c1, d1 = m1
            a2, b2, c2, d2 = m2
            aa = field_add(field_mul(a1, a2), field_mul(b1, c2))
            ab = field_add(field_mul(a1, b2), field_mul(b1, d2))
            ba = field_add(field_mul(c1, a2), field_mul(d1, c2))
            bb = field_add(field_mul(c1, b2), field_mul(d1, d2))
            return (aa, ab, ba, bb)

        def check_girth(adj, n_nodes, target_girth):
            max_depth = (target_girth - 1) // 2
            for s in range(n_nodes):
                dist = [-1] * n_nodes
                parent = [-1] * n_nodes
                dist[s] = 0
                queue = deque([s])
                while queue:
                    u = queue.popleft()
                    if dist[u] >= max_depth:
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

        # Try random sets of 3 involutions
        # Limit trials to fit in time
        max_trials = 50
        for _ in range(max_trials):
            if time.time() > deadline - 30:
                break
            S = random.sample(involutions, 3)
            
            # Build adjacency
            adj = [[] for _ in range(n)]
            for i, e in enumerate(elements):
                for s in S:
                    f = multiply(e, s)
                    j = elem_to_idx.get(f)
                    if j is not None:
                        adj[i].append(j)
            
            # Check connectivity
            visited = [False] * n
            queue = deque([0])
            visited[0] = True
            count = 1
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        count += 1
                        queue.append(v)
            
            if count != n:
                continue
                
            # Check girth
            if check_girth(adj, n, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges

        return None

    # Try small prime fields first
    # q=11 -> n=1320
    # q=13 -> n=2184
    # q=17 -> n=4896 (too big, skip if 13 works)
    
    result = try_sl2_cayley(11)
    if result:
        return result
        
    if time.time() < deadline - 30:
        result = try_sl2_cayley(13)
        if result:
            return result

    # Fallback to GF(16) logic from original if primes fail
    # Re-implementing GF(16) logic concisely
    if time.time() < deadline - 20:
        q = 16
        poly = 0b10011
        mul_table = [[0] * q for _ in range(q)]
        for i in range(q):
            for j in range(q):
                res = 0
                a = i
                b = j
                while b:
                    if b & 1:
                        res ^= a
                    a <<= 1
                    if a >= q:
                        a ^= poly
                    b >>= 1
                mul_table[i][j] = res
        
        gf_add = lambda a, b: a ^ b
        gf_sub = lambda a, b: a ^ b
        gf_mul = lambda a, b: mul_table[a][b]
        gf_inv = lambda a: 0 if a == 0 else next(i for i in range(1, q) if gf_mul(a, i) == 1)
        gf_div = lambda a, b: gf_mul(a, gf_inv(b)) if b != 0 else 0

        elements = []
        elem_to_idx = {}
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    if a == 0:
                        if gf_mul(b, c) == 1:
                            for d in range(q):
                                e = (a, b, c, d)
                                if e not in elem_to_idx:
                                    elem_to_idx[e] = len(elements)
                                    elements.append(e)
                    else:
                        bc = gf_mul(b, c)
                        rhs = gf_add(1, bc)
                        d = gf_div(rhs, a)
                        e = (a, b, c, d)
                        if e not in elem_to_idx:
                            elem_to_idx[e] = len(elements)
                            elements.append(e)
        
        n = len(elements)
        involutions = []
        for e in elements:
            a, b, c, d = e
            if gf_add(a, d) == 0 and not (a == 1 and b == 0 and c == 0 and d == 1):
                involutions.append(e)
        
        def multiply(m1, m2):
            a1, b1, c1, d1 = m1
            a2, b2, c2, d2 = m2
            aa = gf_add(gf_mul(a1, a2), gf_mul(b1, c2))
            ab = gf_add(gf_mul(a1, b2), gf_mul(b1, d2))
            ba = gf_add(gf_mul(c1, a2), gf_mul(d1, c2))
            bb = gf_add(gf_mul(c1, b2), gf_mul(d1, d2))
            return (aa, ab, ba, bb)

        random.seed(42)
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            S = random.sample(involutions, 3)
            adj = [[] for _ in range(n)]
            for i, e in enumerate(elements):
                for s in S:
                    f = multiply(e, s)
                    j = elem_to_idx.get(f)
                    if j is not None:
                        adj[i].append(j)
            
            visited = [False] * n
            queue = deque([0])
            visited[0] = True
            count = 1
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        count += 1
                        queue.append(v)
            
            if count != n:
                continue
                
            if check_girth(adj, n, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges

    # Final Fallback: LUW
    def build_luw(d, q_val):
        if q_val == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add_fn = lambda x, y: x ^ y
            sub_fn = lambda x, y: x ^ y
            mul_fn = lambda x, y: table[x][y]
        else:
            add_fn = lambda x, y: (x + y) % q_val
            sub_fn = lambda x, y: (x - y) % q_val
            mul_fn = lambda x, y: x * y % q_val

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q_val):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                    else:
                        y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
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

    # Try LUW with small d
    q_val = 3
    for d in range(5, 15):
        if time.time() > deadline:
            break
        n_luw, edges_luw = build_luw(d, q_val)
        if n_luw < 2:
            continue
        deg = [0] * n_luw
        for u, v in edges_luw:
            deg[u] += 1
            deg[v] += 1
        if any(dd != k for dd in deg):
            continue
        # Quick girth check
        if check_girth_luw(n_luw, edges_luw, g):
            return n_luw, edges_luw

    # If all else fails, return the last LUW result
    return n_luw, edges_luw

def check_girth_luw(n, edges, girth_target):
    max_depth = (girth_target - 1) // 2
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = deque([s])
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < girth_target:
                        return False
    return True
# EVOLVE-BLOCK-END
