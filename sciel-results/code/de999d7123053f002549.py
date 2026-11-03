import random
import time
import sys
from collections import deque

def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Fallback: The LUW graph from the seed
    def construct_luw():
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

    # Try to find a better graph
    best_n, best_edges = construct_luw()
    
    # Helper: Check girth
    def check_girth(n, edges, min_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check for cycles shorter than min_girth
        # BFS from each vertex up to depth (min_girth - 2)
        # If we find a neighbor at depth < min_girth - 1, cycle exists
        # Actually, if we find a neighbor at depth d, cycle length is d+1.
        # We need d+1 >= min_girth => d >= min_girth - 1.
        # So we must NOT find any neighbor at depth < min_girth - 1.
        # Max depth to check: min_girth - 2.
        max_depth = min_girth - 2
        
        for start in range(n):
            visited = [-1] * n
            visited[start] = 0
            queue = deque([(start, 0)])
            while queue:
                u, dist = queue.popleft()
                if dist >= max_depth:
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = dist + 1
                        queue.append((v, dist + 1))
                    elif v != start and dist + 1 >= visited[v]:
                        # Found a cycle?
                        # If v is already visited, and we are seeing it again from a different path
                        # The cycle length is dist + 1 + visited[v] ? No.
                        # Simple check: if we see a neighbor of start at depth < min_girth-1
                        # Or more generally, if the BFS tree has a back edge.
                        # For girth, we just need to ensure no short cycles.
                        # If we find a node v at depth d such that v is a neighbor of start,
                        # and d < min_girth - 1, then cycle length d+1 < min_girth.
                        pass
        
        # More efficient check for small girth:
        # Count cycles of length L.
        # For L=4: trace paths of length 3.
        # For L=6: trace paths of length 5.
        # For L=8: trace paths of length 7.
        
        # This BFS approach is O(N * (k-1)^depth).
        # For k=5, depth 7 is 4^7 = 16384. N=1000 => 1.6e7. OK.
        
        # Let's implement a direct cycle check for L < min_girth
        # Since min_girth is 9 or 10, we check 4, 6, 8 (and 5, 7 for non-bipartite).
        
        # Actually, the BFS method above is incomplete for detecting all cycles.
        # A cycle of length 4: u-v-w-x-u.
        # BFS from u: v(1), w(2), x(3). x is neighbor of u.
        # If we check "is neighbor of start at depth < min_girth-1", we catch this.
        # What about cycles not involving start?
        # If a cycle exists, BFS from any vertex on the cycle will detect it.
        # So checking all starts is correct.
        
        # Optimization: Only check starts if we haven't found a cycle yet.
        # And we can stop early.
        
        for start in range(n):
            visited = [-1] * n
            visited[start] = 0
            queue = deque([(start, 0)])
            found_cycle = False
            while queue and not found_cycle:
                u, dist = queue.popleft()
                if dist >= max_depth:
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = dist + 1
                        queue.append((v, dist + 1))
                    else:
                        # v is already visited
                        # If v is a neighbor of start, and dist + 1 < min_girth - 1 ?
                        # No, if v is neighbor of start, cycle length is dist + 1 + 1?
                        # Path start -> ... -> u -> v.
                        # If v is neighbor of start, cycle is start -> ... -> u -> v -> start.
                        # Length = dist + 1 (u to v) + 1 (v to start) = dist + 2?
                        # Wait. dist is distance from start to u.
                        # Path length start to u is dist.
                        # Edge u-v adds 1.
                        # Edge v-start adds 1.
                        # Total length dist + 2.
                        # We need dist + 2 >= min_girth.
                        # So if dist + 2 < min_girth, we have a short cycle.
                        if v in adj[start] and dist + 2 < min_girth:
                            return False
            # Also need to check for cycles not returning to start?
            # No, any cycle contains a vertex. BFS from that vertex will see the cycle.
            # The condition "v is neighbor of start" catches cycles that close back to start.
            # Does it catch all cycles?
            # Yes, if there is a cycle, pick any vertex on it as 'start'.
            # The BFS will traverse the cycle. When it reaches the other neighbor of start,
            # it will detect the closure.
            
        return True

    # Generate random bipartite k-regular graph
    def gen_bipartite(n):
        if n % 2 != 0: return None
        half = n // 2
        # 5 permutations
        perms = [list(range(half)) for _ in range(k)]
        for p in perms:
            random.shuffle(p)
        edges = []
        seen = set()
        for i in range(half):
            for j in range(k):
                v = half + perms[j][i]
                if i == v: return None # Loop (should not happen in bipartite)
                if (i, v) in seen or (v, i) in seen: return None
                seen.add((i, v))
                edges.append((i, v))
        return edges

    # Generate random non-bipartite k-regular graph
    def gen_non_bipartite(n):
        # Configuration model
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        edges = []
        seen = set()
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v: return None
            if (u, v) in seen or (v, u) in seen: return None
            seen.add((u, v))
            edges.append((u, v))
        return edges

    # Search for bipartite girth 10
    # Moore bound for bipartite 5-reg girth 10 is 682.
    # Try N from 700 to 1100
    for N in range(700, min(1116, best_n), 2):
        if time.time() > deadline: break
        if N % 2 != 0: continue
        for _ in range(20):
            if time.time() > deadline: break
            edges = gen_bipartite(N)
            if edges and check_girth(N, edges, 10):
                best_n, best_edges = N, edges
                break
    
    # Search for non-bipartite girth 9
    # Moore bound 427.
    for N in range(450, min(1116, best_n), 1):
        if time.time() > deadline: break
        for _ in range(10):
            if time.time() > deadline: break
            edges = gen_non_bipartite(N)
            if edges and check_girth(N, edges, 9):
                best_n, best_edges = N, edges
                break

    return best_n, best_edges
