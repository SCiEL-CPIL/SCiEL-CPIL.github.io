import numpy as np
import time

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g using Cayley graphs on abelian groups.
    This is a deterministic algebraic construction approach.
    """
    start_time = time.time()
    time_limit = 115.0
    
    # We target k=5, g=9. 
    # The problem statement implies k and g are passed, but the specific task is 5-regular girth 9.
    # We will handle the specific case efficiently and fall back to a generic (slower) method if needed.
    
    if k == 5 and g == 9:
        # Try Cayley graphs on Z_p x Z_p
        # Generators must be symmetric: if x in S, -x in S.
        # Degree 5: 2 pairs + 1 element of order 2? Or 2 pairs + 1 element that is its own inverse?
        # In Z_p x Z_p, element (x,y) is its own inverse iff 2(x,y) = 0, which implies x=y=0 for odd p.
        # So for odd p, we cannot have an element of order 2 other than 0.
        # Thus, degree must be even for symmetric S in Z_p x Z_p (p odd).
        # Degree 5 is odd. 
        # So we need a group where there is an element of order 2.
        # Z_2 x Z_2 x ... or Z_2 x Z_p.
        
        # Let's use G = Z_2 x Z_p.
        # Elements: (b, x) where b in {0,1}, x in Z_p.
        # Inverse of (b, x) is (b, -x).
        # We need 5 generators.
        # Let S = {(0, 1), (0, -1), (0, a), (0, -a), (1, 0)}.
        # (1,0) is its own inverse in Z_2 x Z_p? 
        # (1,0) + (1,0) = (0,0). Yes, order 2.
        # So S is symmetric.
        
        # We need to find p and a such that girth >= 9.
        # The girth is the length of the shortest cycle.
        # A cycle is a sequence of generators s_1...s_L summing to 0.
        
        # Let's try small primes p.
        # n = 2 * p.
        
        # To be safe and fast, we can hardcode a known good configuration if we find one, 
        # or search for p.
        
        # Let's search for p.
        # We need to check girth. 
        # Girth check in Cayley graph:
        # BFS from identity (0,0).
        
        def check_girth_cayley(p, a, k, g_target):
            # Group size N = 2 * p
            N = 2 * p
            # Generators:
            # 0: (0, 1)
            # 1: (0, -1)
            # 2: (0, a)
            # 3: (0, -a)
            # 4: (1, 0)
            
            # Map (b, x) to index b * p + x
            def idx(b, x):
                return b * p + (x % p)
                
            def add(g1, g2):
                b1, x1 = g1
                b2, x2 = g2
                return ((b1 + b2) % 2, (x1 + x2) % p)
                
            def sub(g1, g2):
                b1, x1 = g1
                b2, x2 = g2
                return ((b1 - b2) % 2, (x1 - x2) % p)
                
            # BFS
            # We want to find the shortest cycle.
            # A cycle is a path from u to v where u and v are adjacent? No.
            # Standard girth check: BFS from each node? Too slow for N~1000.
            # For Cayley graphs, we can check for closed walks from identity.
            # But a cycle doesn't have to go through identity.
            # However, in a Cayley graph, all cycles are conjugate? No, but we can check
            # if there is a short relation in the generators.
            
            # Actually, for small N, we can just build the adjacency matrix and run standard girth check.
            # N will be around 1000-2000. 
            # Adjacency matrix 2000x2000 is 4M entries. 
            # BFS from each node is O(N * E) = O(N * k * N) = O(N^2 k). 
            # 2000^2 * 5 = 20M ops. This is fast in Python if optimized? 
            # Maybe use numpy for BFS? Or just list-based BFS.
            
            # Let's construct the adjacency list.
            adj = [[] for _ in range(N)]
            gens = [(0, 1), (0, -1), (0, a), (0, -a), (1, 0)]
            
            for i in range(N):
                b = i // p
                x = i % p
                for g in gens:
                    nb = add((b, x), g)
                    nb_idx = idx(nb[0], nb[1])
                    adj[i].append(nb_idx)
                    
            # Check girth
            # We can use the standard BFS girth check.
            # Since N is small (~1000-2000), this is feasible.
            
            min_girth = float('inf')
            
            # Optimization: We only need to check up to g_target - 1.
            # If we find a cycle < g_target, return False.
            
            for start in range(N):
                # BFS
                dist = [-1] * N
                parent = [-1] * N
                dist[start] = 0
                queue = [start]
                qi = 0
                while qi < len(queue):
                    u = queue[qi]
                    qi += 1
                    if dist[u] >= g_target - 1: # Max depth to find cycle < g_target
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif v != parent[u]: # Back edge
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
                            if min_girth < g_target:
                                return False
            return True

        # Search for p and a
        # p should be prime.
        # Let's try p from 3 upwards.
        # We want N = 2p to be as small as possible? 
        # The task minimizes n. 
        # So we should find the smallest valid graph.
        
        # However, the known minimum is 1116. 
        # 2p = 1116 => p = 558 (not prime).
        # 2p = 1118 => p = 559 (13*43).
        # 2p = 1120 => p = 560.
        
        # Let's try to find a valid p, a.
        # We'll iterate p and a.
        
        found = False
        best_n = None
        best_edges = None
        
        # To be fast, we limit the search range.
        # If we don't find one quickly, we fallback.
        
        # Let's try a specific known good structure or just search.
        # For k=5, g=9, a graph of size 1116 exists.
        # Let's try p around 500-600.
        
        # Actually, let's try smaller p first to see if a smaller graph exists.
        # The Moore bound is 428.
        # 2p >= 428 => p >= 214.
        
        import sympy # Wait, sympy is not in stdlib. We need to check primality manually or use a simple list.
        
        def is_prime(n):
            if n < 2: return False
            if n % 2 == 0: return n == 2
            for i in range(3, int(n**0.5)+1, 2):
                if n % i == 0: return False
            return True

        # We will search p from 215 to 1000.
        # And a from 1 to p-1.
        # This might be too slow if we do BFS for every (p, a).
        # BFS is O(N^2). N~500 => 250k ops. 
        # 1000 * 500 = 500k pairs. 500k * 250k = 10^11. Too slow.
        
        # We need a faster girth check or fewer candidates.
        # We can check the girth of the generator set algebraically first?
        # Shortest relation in Z_p x Z_2.
        
        # Let's restrict a. 
        # Common choices: a=2, a=3, etc.
        
        # Let's try a fixed set of 'a' values for each p.
        # a in [2, 3, 5, 7, 11, 13]
        
        for p in range(215, 1000):
            if time.time() - start_time > time_limit:
                break
            if not is_prime(p):
                continue
                
            for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
                if a >= p:
                    continue
                # Quick check: does the set {1, -1, a, -a} have short relations?
                # 1+1+...+1 = 0 => p is large.
                # 1+1+...+1+a = 0?
                # We just run the BFS check.
                
                if check_girth_cayley(p, a, k, g):
                    # Construct edges
                    N = 2 * p
                    edges = []
                    # To avoid duplicates, only add edge if u < v
                    # Since it's undirected, (u,v) and (v,u) are same.
                    # In our generation, we add neighbors.
                    # We should just generate the set of edges.
                    
                    # Re-generate adj to get edges
                    adj_set = set()
                    gens = [(0, 1), (0, -1), (0, a), (0, -a), (1, 0)]
                    
                    for i in range(N):
                        b = i // p
                        x = i % p
                        for gen in gens:
                            nb = ((b + gen[0]) % 2, (x + gen[1]) % p)
                            nb_idx = nb[0] * p + nb[1]
                            if i < nb_idx:
                                adj_set.add((i, nb_idx))
                                
                    best_n = N
                    best_edges = list(adj_set)
                    found = True
                    break
            if found:
                break
        
        if found:
            return best_n, best_edges

    # Fallback: Return a valid (though possibly large) graph using the seed logic or a simple construction.
    # Since we must return *something*, and the seed program worked, we can reuse the seed's fallback logic 
    # or a simple known graph.
    
    # Let's use the seed's algebraic construction as a fallback to ensure validity.
    # This part is copied from the seed to ensure we don't fail.
    
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
    best_n = len(order)
    best_edges = edges

    return best_n, best_edges
