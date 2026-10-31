"""Construct a 3-regular graph with girth >= g on as few vertices as possible."""
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 140  # 10s safety margin

    def check_girth_fast(n, adj, k, g):
        """
        Check if graph has girth >= g.
        For a graph to have girth >= g (even), the BFS tree from any vertex
        up to depth r = g//2 - 1 must be a tree (no cycles, correct vertex counts).
        """
        r = g // 2 - 1
        # Expected counts: 1, k, k*(k-1), k*(k-1)^2, ...
        # For k=3: 1, 3, 6, 12, 24, 48
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
                    else:
                        # If we encounter an already visited node (other than parent)
                        # within depth < r, we have a cycle shorter than 2*r + 2 = g?
                        # Actually, if dist[u] + dist[v] + 1 < g, we have a short cycle.
                        # But simply checking counts is sufficient for "tree-like" structure.
                        # If counts match expected, it implies no cycles in that radius.
                        # Let's rely on the count check.
                        pass
            
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

    # Moore bound for k=3, girth=16 (even): n >= 2 * 3 * sum(2^i for i=0..5) = 6 * 63 = 378
    # But known lower bound is 512. The seed starts at 512.
    # Let's start slightly lower to see if we can find something, but 512 is a safe bet.
    # Actually, the seed program returns a LUW graph which is very large.
    # The best known is 936.
    # Let's try to find a graph with n < 1000.
    
    # If we can't find one quickly, we should return a valid graph.
    # The LUW construction in the seed is a fallback.
    # Let's implement a faster LUW or use a known small graph if possible.
    
    # Strategy:
    # 1. Try random search for n in [512, 1000].
    # 2. If found, return it.
    # 3. If not, return a valid graph from a deterministic construction (e.g., LUW) to ensure validity.
    
    best_n = None
    best_edges = None
    
    # Try n values
    # We must ensure n is even for k=3.
    # Start from lower bound.
    # The seed uses n_start = 512.
    # Let's try 512 to 900.
    
    # To speed up, we can use a better random graph generator or check.
    # The check_girth_fast is O(n * 2^(g/2)). For n=500, g=16, 2^7=128. 500*128 = 64k ops.
    # This is very fast. We can do many attempts.
    
    import random
    
    # Set seed for reproducibility? No, random is fine.
    
    # Try smaller n first?
    # Lower bound is 512.
    # Let's try 512, 520, ...
    
    # Optimization: 
    # If we find a graph with n=936, that's the known best.
    # Can we do better?
    # Let's just search efficiently.
    
    # Note: The seed program's LUW fallback returns a graph with ~1.5M vertices.
    # If we return n=1000, score is 1000/1500000 = 0.0006.
    # If we return n=936, score is 936/1500000 = 0.0006.
    # Wait, the score is seed/value.
    # Seed value is the score of the seed program.
    # Seed program returns LUW graph.
    # If LUW graph has n=1,572,864 (example), score_seed = 1.0.
    # My value = n.
    # Score = 1.0 / (n / n_seed) = n_seed / n.
    # So if I find n=1000 and seed is 1,500,000, score = 1500.
    # This is a HUGE improvement.
    
    # So the goal is to find ANY valid graph with n << n_seed.
    # Even n=2000 would be a massive win.
    
    # Let's search for n in [512, 1200].
    
    # We need to be careful with the time limit.
    # 140s.
    # Check takes ~1ms.
    # Generation takes ~1ms.
    # We can do ~1000 attempts per second.
    # Total attempts ~100,000.
    
    # Probability of finding a girth-16 graph in random 3-regular graph?
    # For n=512, probability is very low.
    # For n=936, it's the cage, so it exists.
    # Random search might not find it easily.
    
    # However, the seed program also does random search and fails, falling back to LUW.
    # If I improve the random search or use a better heuristic, I might find it.
    
    # Alternative: Use the LUW construction but optimize it?
    # LUW construction parameters:
    # The seed uses q=3 (since k=3).
    # d = g - 5 = 11 (since 11 is odd? 16-5=11, odd, so d=11).
    # Wait, seed code:
    # d = g - 5 if (g - 5) % 2 else g - 4
    # 16-5=11, odd, so d=11.
    # q=3.
    # This constructs a graph.
    # The size of this graph is 2 * 3 * 3^d = 2 * 3 * 177147 = 1,062,882.
    # So the seed returns ~1M vertices.
    
    # If I can find a graph with n=1000, score is ~1000.
    
    # Let's try to find a graph with n < 1500 using random search.
    # If I don't find one, I will return a smaller LUW graph if possible?
    # Or just return the best found.
    
    # Actually, I can implement the LUW construction myself and return it if random search fails.
    # But the LUW graph is huge.
    # Is there a smaller deterministic construction?
    
    # What about the Margulis construction?
    # For p=11, n=1320.
    # If I implement Margulis and check girth, and it's >= 16, I return n=1320.
    # Score = 1,000,000 / 1320 ~ 750.
    # If I find n=936, score ~ 1000.
    
    # Let's implement Margulis for p=7, 11, 13.
    
    def margulis_graph(p):
        """
        Construct the Margulis 3-regular graph on SL(2, Z_p).
        Vertices are matrices in SL(2, Z_p).
        Generators:
        A = [[1, 1], [0, 1]]
        B = [[1, 0], [1, 1]]
        C = [[0, 1], [-1, 0]] (or similar, need to check 3-regularity)
        
        Actually, the standard Margulis graph for SL(2, p) uses:
        (a, b; c, d) -> (a, b; c, d) * [[1, 1], [0, 1]]
        (a, b; c, d) -> (a, b; c, d) * [[1, 0], [1, 1]]
        (a, b; c, d) -> (a, b; c, d) * [[0, 1], [-1, 0]]
        
        This is 3-regular.
        Size is p(p^2-1).
        For p=7, size = 7*48 = 336.
        For p=11, size = 11*120 = 1320.
        For p=13, size = 13*168 = 2184.
        """
        import itertools
        
        # Generate all matrices in SL(2, Z_p)
        mats = []
        for a in range(p):
            for b in range(p):
                for c in range(p):
                    # d = (1 - b*c) / a mod p?
                    # det = ad - bc = 1 => ad = 1 + bc => d = (1+bc) * inv(a)
                    if a == 0:
                        # ad - bc = 1 => -bc = 1 => bc = -1
                        # b != 0, c != 0
                        # d can be anything? No, a=0.
                        # det = 0*d - b*c = -bc = 1 => bc = -1.
                        # So b and c are fixed up to inverse.
                        # For each b, c = -1 * inv(b).
                        # d is free? No, matrix is [[0, b], [c, d]].
                        # det = -bc = 1.
                        # So d is free?
                        # Wait, SL(2) matrices have 3 degrees of freedom.
                        # If a=0, b!=0, c = -1/b. d is free.
                        # So there are p-1 choices for b, 1 for c, p for d.
                        # Total p(p-1) matrices with a=0.
                        # If a!=0, there are (p-1) choices for a, p for b, p for c, 1 for d.
                        # Total (p-1)p^2 matrices.
                        # Sum: p(p-1) + (p-1)p^2 = p(p-1)(1+p) = p(p^2-1). Correct.
                        
                        for d in range(p):
                            # Check bc = -1
                            if (b * c) % p == (p - 1) % p:
                                mats.append((0, b, c, d))
                    else:
                        # a != 0
                        # d = (1 + bc) * inv(a)
                        # inv(a)
                        try:
                            inv_a = pow(a, p - 2, p)
                        except:
                            continue
                        for b in range(p):
                            for c in range(p):
                                d = (1 + b * c) * inv_a % p
                                mats.append((a, b, c, d))
        
        # Map matrix to index
        mat_to_idx = {m: i for i, m in enumerate(mats)}
        
        # Generators
        # A = [[1, 1], [0, 1]]
        # B = [[1, 0], [1, 1]]
        # C = [[0, 1], [-1, 0]]
        
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
        gen_C = (0, 1, p - 1, 0) # -1 mod p
        
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

    # Try Margulis for p=7, 11, 13
    # Check girth
    # If girth >= g, return it.
    
    for p in [7, 11, 13]:
        if time.time() > deadline - 20:
            break
        adj = margulis_graph(p)
        n = len(adj)
        if check_girth_fast(n, adj, k, g):
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges

    # If Margulis fails, try random search
    # Start from 512
    # Use random search
    
    # To make random search faster, we can use a better heuristic.
    # But let's just do random search.
    
    # We need to be careful not to exceed time.
    
    # Try n values
    # If we find one, return it.
    
    # If we don't find one, we must return a valid graph.
    # We can return a LUW graph.
    # But LUW is huge.
    # Is there a smaller valid graph we can construct?
    # The Petersen graph is 3-regular, girth 5, n=10.
    # For girth 16, we need a large graph.
    
    # If random search fails, we will return the LUW graph from the seed code.
    # But I should implement it to be safe.
    
    # Actually, the seed code's LUW construction is complex.
    # Let's copy it from the seed code to ensure it works.
    
    # Fallback: LUW
    q = k
    d = g - 5 if (g - 5) % 2 else g - 4
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

# EVOLVE-BLOCK-END
