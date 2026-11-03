import random
from collections import deque
import time

def construct(k, g):
    def calc_moore(k, g):
        r = (g - 1) // 2
        total_verts = 1
        for i in range(r):
            if i == 0:
                total_verts += k
            else:
                total_verts += k * (k-1)**i
        return total_verts

    M = calc_moore(k, g)
    # Start with a slightly larger n to make finding a high-girth graph easier
    # then we could potentially shrink? No, shrinking is hard. 
    # Just search for a valid n starting from M.
    # The problem asks for "as few vertices as possible".
    # So we should iterate n from M upwards.
    
    # However, local search is better if we have a good starting n.
    # Let's try n = M first. If it fails after some time, try M+1, etc.
    # But 120s limit.
    
    # Let's define a robust random regular graph generator.
    def gen_random_regular(n, k):
        if n * k % 2 != 0:
            return None
        # Configuration model with retry
        for _ in range(50):
            half_edges = []
            for i in range(n):
                half_edges.extend([i] * k)
            random.shuffle(half_edges)
            
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(n * k // 2):
                u = half_edges[2 * i]
                v = half_edges[2 * i + 1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            
            if valid:
                # Verify degree
                if all(len(adj[i]) == k for i in range(n)):
                    return adj
        return None

    def get_shortest_cycle(adj, n, g):
        """Returns a list of nodes forming a cycle of length < g, or None."""
        # BFS from each node to find a back edge
        for start in range(n):
            visited = [-1] * n
            parent = [-1] * n
            visited[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if visited[u] >= g - 1: # Max depth to detect cycle < g
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = visited[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v == parent[u]:
                            continue
                        # Found a cycle
                        # Reconstruct cycle
                        cycle = []
                        curr = u
                        while curr != v:
                            cycle.append(curr)
                            curr = parent[curr]
                        cycle.append(v)
                        cycle.reverse() # v -> ... -> u
                        # The cycle is v -> ... -> u -> v
                        if len(cycle) < g:
                            return cycle
        return None

    def try_swap(adj, n, k, cycle):
        """Try to break the cycle by swapping an edge in it with a random edge elsewhere."""
        if len(cycle) < 2:
            return False
            
        # Pick a random edge in the cycle
        idx = random.randint(0, len(cycle) - 1)
        u = cycle[idx]
        v = cycle[(idx + 1) % len(cycle)]
        
        # We want to remove (u, v) and (x, y) and add (u, x) and (v, y)
        # x, y must be neighbors of each other.
        # x, y must not be u, v.
        # New edges (u, x) and (v, y) must not exist.
        
        # Collect all edges to pick a random one (x, y)
        # To avoid O(E) scan, we can just pick a random vertex x, then a random neighbor y of x.
        # But we need to ensure (x, y) is not in the cycle and swapping is valid.
        
        attempts = 0
        while attempts < 20:
            x = random.randint(0, n - 1)
            if x == u or x == v:
                attempts += 1
                continue
            if not adj[x]:
                attempts += 1
                continue
            y = random.choice(adj[x])
            if y == u or y == v:
                attempts += 1
                continue
            if x == y:
                attempts += 1
                continue
                
            # Check if (x, y) is in the cycle? 
            # If (x, y) is in the cycle, swapping might just rearrange the cycle or create a smaller one.
            # Ideally pick an edge NOT in the cycle.
            # For simplicity, we just check validity of the new edges.
            
            # Check if (u, x) and (v, y) already exist
            if x in adj[u] or y in adj[v]:
                attempts += 1
                continue
            
            # Perform swap
            # Remove u-v
            adj[u].remove(v)
            adj[v].remove(u)
            # Remove x-y
            adj[x].remove(y)
            adj[y].remove(x)
            
            # Add u-x
            adj[u].append(x)
            adj[x].append(u)
            # Add v-y
            adj[v].append(y)
            adj[y].append(v)
            
            return True
        return False

    # Main search loop
    time_limit = 115.0
    start_time = time.time()
    
    # Try increasing n
    for n in range(M, M + 200):
        if time.time() - start_time > time_limit:
            break
            
        # Generate initial graph
        adj = gen_random_regular(n, k)
        if adj is None:
            continue
            
        # Local search
        for _ in range(5000): # Max swaps
            if time.time() - start_time > time_limit:
                break
                
            cycle = get_shortest_cycle(adj, n, g)
            if cycle is None:
                # Valid graph found
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
                
            # Try to break the cycle
            if not try_swap(adj, n, k, cycle):
                # If swap failed, try a different random edge in the cycle or restart?
                # Just try again
                pass
                
    # Fallback: Return a valid graph if found in previous iterations? 
    # Or just return a simple cycle if n is small? No, must be k-regular.
    # If we didn't find one, we must return *something* valid? 
    # The task says "invalid result... scores 0".
    # We should ensure we return a valid graph.
    # If local search fails, we can try to return a known small graph?
    # For k=5, g=9, we might not find one quickly.
    # However, the prompt implies we should improve the score.
    # If we fail, we should probably fall back to the seed's algebraic construction if it was valid?
    # The seed's algebraic construction was a fallback.
    
    # Let's re-add the seed's algebraic construction as a fallback to ensure validity.
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
