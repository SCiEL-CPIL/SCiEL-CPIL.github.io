import time
import random
import sys

def construct(k, g):
    # Fallback: The original LUW construction
    def luw_construct(k, g):
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

    # Randomized Search
    deadline = time.time() + 110.0
    best_n = None
    best_edges = None
    
    # We search for N starting from a high value down to the Moore bound
    # The Moore bound for k=3, g=16 is 512.
    # We try larger N first as they are more likely to have girth >= 16.
    
    # Helper to generate a random 3-regular graph
    def gen_random_3_reg(n):
        # n must be even for 3-regular? No, 3n must be even, so n must be even.
        if n % 2 != 0:
            return None
        stubs = []
        for i in range(n):
            stubs.extend([i, i, i])
        
        attempts = 0
        while attempts < 20: # Limit attempts for this N
            random.shuffle(stubs)
            adj = [[] for _ in range(n)]
            valid = True
            # Use a set to track edges for multi-edge check
            # For speed, we can just check after building, but set is safer
            # Actually, for large n, set of tuples might be slow.
            # Let's build adjacency and then verify.
            
            edges_list = []
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                edges_list.append((u, v))
                adj[u].append(v)
                adj[v].append(u)
            
            if not valid:
                attempts += 1
                continue
                
            # Check for multi-edges
            # Sort edges and check for duplicates
            # This is O(E log E). E = 1.5 N.
            # For N=100000, E=150000. Sort is fast.
            sorted_edges = sorted(edges_list)
            for i in range(len(sorted_edges)-1):
                if sorted_edges[i] == sorted_edges[i+1]:
                    valid = False
                    break
            if not valid:
                attempts += 1
                continue
                
            return adj, edges_list
            
        return None

    def check_girth(adj, min_girth):
        n = len(adj)
        # BFS from each vertex
        for v in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[v] = 0
            queue = [v]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for w in adj[u]:
                    if w == parent[u]:
                        continue
                    if dist[w] == -1:
                        dist[w] = dist[u] + 1
                        parent[w] = u
                        queue.append(w)
                    else:
                        # Found a cycle
                        cycle_len = dist[u] + dist[w] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    # Search strategy
    # Try N values
    # Start from 200,000 down to 512
    # Step size can be adaptive or fixed.
    # Let's try a few specific ranges.
    
    # If k != 3, we cannot use this specific random 3-reg generator.
    # The task specifies k=3, g=16.
    if k != 3:
        return luw_construct(k, g)
        
    if g != 16:
        # The random search is tuned for g=16.
        # For other g, fallback to LUW.
        return luw_construct(k, g)

    # Range of N to try
    # We know 936 works. We want to find something smaller than LUW (~354k).
    # Let's try N from 100,000 down to 1000.
    
    current_time = time.time()
    
    # We will try N in descending order of likelihood of success?
    # Larger N -> higher girth on average -> easier to find girth >= 16.
    # So start with large N.
    
    # List of N to try
    # We can't try all.
    # Let's try N = 100,000, 50,000, 20,000, 10,000, 5,000, 2,000, 1,000, 512.
    n_candidates = [100000, 80000, 60000, 40000, 20000, 10000, 5000, 2000, 1000, 512]
    
    for n in n_candidates:
        if time.time() > deadline:
            break
        if n % 2 != 0:
            n += 1 # Ensure even
            
        # Try a few random graphs for this N
        max_tries = 50
        for _ in range(max_tries):
            if time.time() > deadline:
                break
                
            res = gen_random_3_reg(n)
            if res is None:
                continue
            adj, edges = res
            
            # Check girth
            if check_girth(adj, g):
                best_n = n
                best_edges = edges
                break # Found one for this N, try smaller N
        
    if best_n is not None:
        return best_n, best_edges
    else:
        return luw_construct(k, g)
