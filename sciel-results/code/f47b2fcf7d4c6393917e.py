"""Construct a 3-regular graph with girth >= g on as few vertices as possible."""
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 135  # 15s safety margin before 150s kill

    def check_girth_fast(n, adj, k, g):
        """
        Check if graph has girth >= g using BFS from every vertex.
        For girth >= g (even), the BFS tree from any vertex up to depth r = g//2 - 1
        must be a perfect tree with no early collisions.
        """
        r = g // 2 - 1
        expected_counts = [1] + [k * (k - 1) ** i for i in range(r)]
        
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            counts = [0] * (r + 1)
            counts[0] = 1
            
            while queue:
                u = queue.popleft()
                if dist[u] >= r:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] <= r:
                            counts[dist[v]] += 1
                            queue.append(v)
            
            if counts != expected_counts:
                return False
        return True

    def random_regular_graph(n, k, seed=None):
        """Generate a random k-regular graph on n vertices using configuration model."""
        import random
        if seed is not None:
            random.seed(seed)
        if n * k % 2 != 0:
            return None
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        adj = [[] for _ in range(n)]
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None
            if v in adj[u]:
                return None
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def margulis_graph(p):
        """
        Construct the Margulis 3-regular graph on SL(2, Z_p).
        Size is p(p^2-1).
        """
        # Generate all matrices in SL(2, Z_p)
        mats = []
        for a in range(p):
            for b in range(p):
                for c in range(p):
                    if a == 0:
                        # bc = -1 mod p
                        if (b * c) % p == (p - 1) % p:
                            for d in range(p):
                                mats.append((0, b, c, d))
                    else:
                        inv_a = pow(a, p - 2, p)
                        for b2 in range(p):
                            for c2 in range(p):
                                d = (1 + b2 * c2) * inv_a % p
                                mats.append((a, b2, c2, d))
        
        mat_to_idx = {m: i for i, m in enumerate(mats)}
        
        def mul(m1, m2):
            a1, b1, c1, d1 = m1
            a2, b2, c2, d2 = m2
            a = (a1 * a2 + b1 * c2) % p
            b = (a1 * b2 + b1 * d2) % p
            c = (c1 * a2 + d1 * c2) % p
            d = (c1 * b2 + d1 * d2) % p
            return (a, b, c, d)
        
        gen_A = (1, 1, 0, 1)
        gen_B = (1, 0, 1, 1)
        gen_C = (0, 1, p - 1, 0)
        
        n = len(mats)
        adj = [[] for _ in range(n)]
        
        for i, m in enumerate(mats):
            m1 = mul(m, gen_A)
            m2 = mul(m, gen_B)
            m3 = mul(m, gen_C)
            
            j1 = mat_to_idx[m1]
            j2 = mat_to_idx[m2]
            j3 = mat_to_idx[m3]
            
            adj[i].append(j1)
            adj[i].append(j2)
            adj[i].append(j3)
        
        return adj

    def luw_construction(k, d):
        """
        LUW (Lubotzky-Phillips-Sarnak) type construction for 3-regular graphs.
        Returns (n, edges, adj).
        """
        q = k
        # d determines the depth; graph has girth >= 2d+2
        # Size is at most 2 * q * q^d = 2 * 3 * 3^d
        
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
                    y[c] = (x[c] + y[a] * x[b]) % q if side == 0 else (x[c] - x[a] * y[b]) % q
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
        
        n = len(order)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        return n, edges, adj

    # Strategy:
    # 1. Try Margulis for small primes (fast, deterministic, known good girth)
    # 2. Try LUW with small d values (d=7 gives girth >= 16, ~13k vertices)
    # 3. Try random search for smaller n
    # 4. Fallback to LUW with original d

    # 1. Margulis for p=7 (n=336), p=11 (n=1320)
    # For p=7, girth of Margulis graph is at least 2*floor(log_2(7))+2 = 2*2+2 = 6? No.
    # Actually, the girth of the Margulis graph on SL(2,p) with these generators
    # is related to the word length. For p=7, n=336, the girth might be too small.
    # Let's just check it.
    
    for p in [7, 11]:
        if time.time() > deadline - 30:
            break
        try:
            adj = margulis_graph(p)
            n = len(adj)
            if check_girth_fast(n, adj, k, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
        except:
            pass

    # 2. LUW with d=7 (girth >= 16 guaranteed, n ~ 2*3*3^7 = 13122)
    # This is a HUGE improvement over d=11 (n ~ 1M)
    for d in [7, 8]:
        if time.time() > deadline - 20:
            break
        try:
            n, edges, adj = luw_construction(k, d)
            if check_girth_fast(n, adj, k, g):
                return n, edges
        except:
            pass

    # 3. Random search for n in [512, 1500]
    import random
    
    best_n = None
    best_edges = None
    
    # Try a few random graphs
    for n in range(512, 900, 16):
        if time.time() > deadline - 10:
            break
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            adj = random_regular_graph(n, k)
            if adj is None:
                continue
            if check_girth_fast(n, adj, k, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges

    # 4. Fallback: LUW with original d
    d_orig = g - 5 if (g - 5) % 2 else g - 4
    n, edges, adj = luw_construction(k, d_orig)
    return n, edges

# EVOLVE-BLOCK-END
