import random
import sys
from collections import deque

def construct(k, g):
    if k != 3:
        raise ValueError("Only k=3 supported")

    def get_girth(adj, n, limit):
        min_girth = float('inf')
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                        if min_girth < limit:
                            return min_girth
        return min_girth

    def try_group(elements, elem_to_idx, mat_mul, n, g):
        # Find involutions
        involutions = []
        for e in elements:
            # Check M^2 = I
            m2 = mat_mul(e, e)
            # Identity check depends on field, but (1,0,0,1) is standard
            if m2 == (1, 0, 0, 1) and e != (1, 0, 0, 1):
                involutions.append(e)
                
        random.seed(42)
        max_attempts = 5000
        for _ in range(max_attempts):
            S = random.sample(involutions, 3)
            adj = [[] for _ in range(n)]
            for i, e in enumerate(elements):
                for s in S:
                    f = mat_mul(e, s)
                    j = elem_to_idx[f]
                    adj[i].append(j)
            
            # Connectivity
            visited = [False] * n
            q_d = deque([0])
            visited[0] = True
            count = 1
            while q_d:
                u = q_d.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        count += 1
                        q_d.append(v)
            if count != n:
                continue
                
            girth = get_girth(adj, n, g)
            if girth >= g:
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
        return None

    # 1. Try SL(2, p) for primes
    for p in [11, 13, 17]:
        elements = []
        for a in range(p):
            for b in range(p):
                for c in range(p):
                    if a == 0:
                        if (b * c) % p == (p - 1) % p:
                            for d in range(p):
                                elements.append((a, b, c, d))
                    else:
                        bc = (b * c) % p
                        rhs = (1 + bc) % p
                        a_inv = pow(a, p-2, p)
                        d = (rhs * a_inv) % p
                        elements.append((a, b, c, d))
        
        n = len(elements)
        elem_to_idx = {e: i for i, e in enumerate(elements)}
        
        def mat_mul(m1, m2, p=p):
            a1, b1, c1, d1 = m1
            a2, b2, c2, d2 = m2
            aa = (a1*a2 + b1*c2) % p
            ab = (a1*b2 + b1*d2) % p
            ba = (c1*a2 + d1*c2) % p
            bb = (c1*b2 + d1*d2) % p
            return (aa, ab, ba, bb)
            
        res = try_group(elements, elem_to_idx, mat_mul, n, g)
        if res:
            return res

    # 2. Try SL(2, 2^m)
    # Reuse GF logic from original or implement simple one
    # For m=4 (q=16), N=4080. Already known to work but maybe better generators?
    # The original code used random search. We can try more attempts or different seeds.
    # But let's try m=3 (q=8) first.
    
    # Implement GF(2^m)
    def build_sl2_gf2m(m):
        q = 1 << m
        # Poly for GF(16) was 0b10011
        # For GF(8), x^3 + x + 1 = 0b1011
        polys = {3: 0b1011, 4: 0b10011, 5: 0b100101}
        poly = polys.get(m, 0b10011) # Default to 16
        
        # Build tables
        add_t = [[0]*q for _ in range(q)]
        mul_t = [[0]*q for _ in range(q)]
        for i in range(q):
            for j in range(q):
                add_t[i][j] = i ^ j
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
                mul_t[i][j] = res
        
        inv_t = [0]*q
        inv_t[1] = 1
        for i in range(2, q):
            for j in range(2, q):
                if mul_t[i][j] == 1:
                    inv_t[i] = j
                    inv_t[j] = i
                    break
        
        elements = []
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    if a == 0:
                        if mul_t[b][c] == 1:
                            for d in range(q):
                                elements.append((a, b, c, d))
                    else:
                        bc = mul_t[b][c]
                        rhs = add_t[1][bc]
                        d = mul_t[rhs][inv_t[a]]
                        elements.append((a, b, c, d))
                        
        n = len(elements)
        elem_to_idx = {e: i for i, e in enumerate(elements)}
        
        def mat_mul(m1, m2):
            a1, b1, c1, d1 = m1
            a2, b2, c2, d2 = m2
            aa = add_t[mul_t[a1][a2]][mul_t[b1][c2]]
            ab = add_t[mul_t[a1][b2]][mul_t[b1][d2]]
            ba = add_t[mul_t[c1][a2]][mul_t[d1][c2]]
            bb = add_t[mul_t[c1][b2]][mul_t[d1][d2]]
            return (aa, ab, ba, bb)
            
        return elements, elem_to_idx, mat_mul, n

    for m in [3, 4, 5]:
        elements, elem_to_idx, mat_mul, n = build_sl2_gf2m(m)
        res = try_group(elements, elem_to_idx, mat_mul, n, g)
        if res:
            return res

    raise RuntimeError("Failed to find valid graph")
