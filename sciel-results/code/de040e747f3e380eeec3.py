import random
import time
import sys

def _luw_construct(k, g):
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

def _check_girth(n, edges, target_girth):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check for cycles shorter than target_girth
    # Since graph is 4-regular, we can do BFS from each node
    # But we only need to check up to depth target_girth // 2 - 1
    # For girth 10, depth 4.
    # If we find a cycle of length < 10, return False.
    
    max_depth = target_girth // 2
    # We need to detect cycles.
    # A cycle of length L exists if there is a path of length L from u to u.
    # BFS from each node u, keeping track of depth.
    # If we revisit a node at depth < max_depth, we have a cycle?
    # Not exactly. We need to ensure the path is simple.
    # Standard girth check: BFS from each node, stop when depth >= max_depth.
    # If we find a neighbor that was visited at depth d-1, and d < max_depth, 
    # and the edge is not the tree edge, we have a cycle of length 2d.
    
    for u in range(n):
        visited = [-1] * n
        visited[u] = 0
        queue = [(u, 0)]
        while queue:
            node, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            for v in adj[node]:
                if visited[v] == -1:
                    visited[v] = depth + 1
                    queue.append((v, depth + 1))
                else:
                    # Found a cycle
                    # Cycle length is depth + 1 + visited[v]?
                    # No, this is for undirected.
                    # If v is visited, and v is not parent, cycle length is depth + 1 + visited[v] ?
                    # Actually, if we see a visited node, the cycle length is depth + 1 + (depth of v from u)?
                    # This logic is tricky.
                    # Simpler: Just check if any path of length < target_girth exists from u to u.
                    pass
    
    # Better girth check for small n:
    # Enumerate all paths of length up to max_depth from each start node.
    # If we return to start, we have a cycle.
    # For n=320, depth 4.
    # 320 * 4^4 = 320 * 256 = 81920. Very fast.
    
    for u in range(n):
        # DFS/BFS to find cycles starting at u
        # State: (node, depth, prev)
        # We want to find a path u -> ... -> u of length < target_girth
        # And length >= 3 (simple cycle)
        
        def dfs(node, depth, prev):
            if depth == 0:
                return depth > 0 and node == u # Cycle found?
            if depth >= max_depth:
                return False
            for v in adj[node]:
                if v == prev:
                    continue
                if v == u and depth + 1 >= 3:
                    # Cycle found of length depth + 1
                    if depth + 1 < target_girth:
                        return True
                if dfs(v, depth + 1, node):
                    return True
            return False
        
        if dfs(u, 0, -1):
            return False
            
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    if k == 4 and g == 10:
        # Search for a smaller graph
        # Try N from 170 to 320
        # Use random Cayley graphs on C_N
        random.seed(42)
        
        for N in range(170, 321):
            if time.time() > deadline:
                break
            
            # Try a few random generator sets for C_N
            # Generators must be coprime to N? No, just generate the group.
            # For C_N, generators are integers.
            # We need 4 generators.
            # To ensure connectivity, gcd(g1, g2, g3, g4, N) = 1.
            
            attempts = 500 if N > 200 else 1000
            for _ in range(attempts):
                if time.time() > deadline:
                    break
                
                # Pick 4 random integers
                gens = [random.randint(1, N-1) for _ in range(4)]
                # Ensure they are not 0
                # Check connectivity
                # gcd of all gens and N
                import math
                g_val = N
                for gen in gens:
                    g_val = math.gcd(g_val, gen)
                if g_val != 1:
                    continue
                
                # Build edges
                edges = []
                seen = set()
                for u in range(N):
                    for gen in gens:
                        v = (u + gen) % N
                        if u < v:
                            if (u, v) not in seen:
                                seen.add((u, v))
                                edges.append((u, v))
                        elif (v, u) not in seen:
                            seen.add((v, u))
                            edges.append((v, u))
                
                # Check girth
                if _check_girth(N, edges, g):
                    return N, edges
                    
        # Fallback to LUW
        return _luw_construct(k, g)
    else:
        return _luw_construct(k, g)
