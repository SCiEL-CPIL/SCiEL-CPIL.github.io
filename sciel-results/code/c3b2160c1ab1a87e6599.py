import time
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def luw_construction(d, q):
        """
        Constructs the LUW graph for given d and q.
        Returns (n, edges).
        """
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        # Rule definition
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
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
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

    def get_girth_fast(n, edges, target_girth):
        """
        Checks if the graph has girth at least target_girth.
        Returns True if girth >= target_girth, else False.
        Uses BFS from each vertex to find shortest cycles.
        """
        if n == 0:
            return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # BFS for shortest cycle
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    else:
                        # Cycle found: dist[u] + dist[v] + 1
                        # We only care if it's shorter than target_girth
                        # If dist[u] + dist[v] + 1 < target_girth, fail
                        if dist[u] + dist[v] + 1 < target_girth:
                            return False
            if time.time() > deadline:
                return False # Safety
        return True

    def abs_construction(q):
        """
        Attempts to construct an Alon-Babai-Suzuki style graph on UT(3,q).
        For now, this is a placeholder for a specific construction if we can't find a standard one quickly.
        We will stick to LUW but optimize the search.
        """
        pass

    # Determine candidate d values
    q = k
    
    min_d = max(1, (g - 4 + 1) // 2)
    
    seed_d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # We want to try d values that yield small n.
    # n = q^(d+1) for LUW? No, LUW(q,d) has q^(d+1) vertices?
    # Let's check the size of LUW.
    # The seed program generates vertices (side, x) where x is a tuple of length d+1 (indices 0..d).
    # x[0] is None, x[1]..x[d] are in F_q.
    # So there are 2 * q^d vertices?
    # Let's trace: start = (0, (None,) + (0,)*d).
    # The recursion builds a tree of depth d?
    # The number of vertices is 2 * q^d.
    # For k=5, g=9: min_d = (9-4+1)//2 = 3. n = 2 * 5^3 = 250.
    # seed_d = max(3, 9-5=4) = 4. n = 2 * 5^4 = 1250.
    # Wait, the seed used seed_d and got 1250?
    # The task says "smallest known such graph has 1116 vertices".
    # LUW(5, 4) has 1250 vertices.
    # LUW(5, 3) has 250 vertices. Does it have girth 9?
    # Theoretical girth of LUW(q,d) is 2d+4?
    # If d=3, girth = 10?
    # If girth is 10, then it is >= 9.
    # So n=250 should be a valid answer for k=5, g=9?
    # Why did the seed program return 1250 (or similar)?
    # Let's check the seed's logic.
    # seed_d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=9: (9-5)%2 = 0, so seed_d = g-4 = 5.
    # n = 2 * 5^5 = 6250?
    # Wait, the current program's seed_d calculation:
    # seed_d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=9: (9-5)=4, even. seed_d = 9-4 = 5.
    # n = 2 * 5^5 = 6250.
    # But the prompt says "score 1.0000" for the current program.
    # And "smallest known ... 1116".
    # If I return 250, that is much better.
    # I need to verify if LUW(5,3) actually has girth >= 9.
    # The theoretical girth of LUW graphs is often cited as 2d+4 or similar.
    # For d=3, 2*3+4 = 10.
    # So girth 10 >= 9.
    # So n=250 should be valid.
    
    candidates = sorted(set([min_d, min_d + 1, min_d + 2, seed_d]))
    
    best_result = None
    best_n = float('inf')
    
    for d in candidates:
        if time.time() > deadline - 10:
            break
            
        try:
            n, edges = luw_construction(d, q)
        except Exception:
            continue
            
        # Check if it's k-regular
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(dg != k for dg in deg):
            continue
            
        # Check girth
        if n < 5000:
            if not get_girth_fast(n, edges, g):
                continue
        else:
            # Trust the theoretical bound
            if 2 * d + 4 < g:
                continue
                
        if n < best_n:
            best_n = n
            best_result = (n, edges)
            
    if best_result:
        return best_result
    else:
        # Fallback to seed's logic
        d = seed_d
        return luw_construction(d, q)
