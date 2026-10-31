"""Construct a 3-regular graph with girth >= g on as few vertices as possible."""
import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 125  # safety margin before 150s kill
    
    # Moore bound for degree k, girth g (even)
    if g % 2 == 0:
        moore_bound = 2 * sum((k - 1) ** i for i in range(g // 2))
    else:
        moore_bound = 1 + k * sum((k - 1) ** i for i in range((g - 1) // 2))
    
    def verify_girth(n, adj, k, g):
        """Verify girth >= g using BFS from every vertex up to depth r = g//2 - 1."""
        r = g // 2 - 1
        expected = [1] + [k * (k - 1) ** i for i in range(r)]
        
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
            
            if counts != expected:
                return False
        return True
    
    def find_short_cycle(n, adj, g):
        """Find a cycle of length < g. Returns list of nodes or None."""
        r = g // 2 - 1
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= r:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and dist[v] <= dist[u]:
                        # Non-tree edge creating a cycle
                        # Find LCA of u and v
                        au, av = u, v
                        du, dv = dist[au], dist[av]
                        while du > dv:
                            au = parent[au]
                            du -= 1
                        while dv > du:
                            av = parent[av]
                            dv -= 1
                        while au != av:
                            au = parent[au]
                            av = parent[av]
                        # au is LCA
                        path_u = []
                        cur = u
                        while cur != au:
                            path_u.append(cur)
                            cur = parent[cur]
                        path_u.append(au)
                        path_u.reverse()
                        path_v = []
                        cur = v
                        while cur != au:
                            path_v.append(cur)
                            cur = parent[cur]
                        path_v.append(au)
                        path_v.reverse()
                        cycle = path_u[:-1] + path_v[::-1]
                        if 3 <= len(cycle) < g:
                            return cycle
        return None
    
    def generate_random_regular(n, k, seed=None):
        """Generate random k-regular graph via configuration model with retries."""
        if seed is not None:
            random.seed(seed)
        for attempt in range(200):
            stubs = []
            for i in range(n):
                stubs.extend([i] * k)
            random.shuffle(stubs)
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if valid:
                return adj
        return None
    
    def edge_swap(adj, n, k):
        """Perform a random 2-switch to maintain k-regularity."""
        # Pick two random edges (u1,v1) and (u2,v2) that are not adjacent
        # and swap to (u1,u2) and (v1,v2) or (u1,v2) and (v1,u2)
        for _ in range(100):
            u1 = random.randint(0, n - 1)
            v1 = random.choice(adj[u1])
            u2 = random.randint(0, n - 1)
            v2 = random.choice(adj[u2])
            
            # Check no shared vertices
            if len({u1, v1, u2, v2}) < 4:
                continue
            # Check new edges don't already exist
            if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                continue
            
            # Remove old edges
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            # Add new edges
            adj[u1].append(u2)
            adj[u2].append(u1)
            adj[v1].append(v2)
            adj[v2].append(v1)
            return True
        return False
    
    # Strategy: Try random regular graphs with local search
    # Start from Moore bound and go up in steps
    best_n = None
    best_edges = None
    
    # Try vertex counts from Moore bound upward
    for n in range(moore_bound, moore_bound + 100, 2):
        if time.time() > deadline - 20:
            break
        
        # Try several random graphs at this size
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            
            adj = generate_random_regular(n, k, seed=random.randint(0, 10**9))
            if adj is None:
                continue
            
            # Check if already valid
            if verify_girth(n, adj, k, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
            
            # Local search: find short cycles and try to break them
            for iteration in range(200):
                if time.time() > deadline - 20:
                    break
                
                cycle = find_short_cycle(n, adj, g)
                if cycle is None:
                    if verify_girth(n, adj, k, g):
                        edges = []
                        for i in range(n):
                            for j in adj[i]:
                                if i < j:
                                    edges.append((i, j))
                        return n, edges
                    break
                
                # Try to break the cycle by swapping edges
                # Pick two non-adjacent edges in the cycle and try to swap
                if len(cycle) >= 4:
                    # Try several random swaps
                    for _ in range(10):
                        if time.time() > deadline - 20:
                            break
                        # Save state
                        adj_backup = [list(lst) for lst in adj]
                        
                        if edge_swap(adj, n, k):
                            # Check if this helped (reduced number of short cycles)
                            new_cycle = find_short_cycle(n, adj, g)
                            if new_cycle is None or len(new_cycle) > len(cycle):
                                # Keep the change
                                continue
                            else:
                                # Revert
                                adj = adj_backup
                        else:
                            adj = adj_backup
                
                # Also try pure random swaps
                edge_swap(adj, n, k)
            
            # Final check
            if verify_girth(n, adj, k, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
    
    # Fallback: LUW construction
    def luw_construction(k, d):
        q = k
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r_val, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_val]
        
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
    
    for d in [7, 8, 9]:
        if time.time() > deadline - 5:
            break
        try:
            n, edges, adj = luw_construction(k, d)
            if verify_girth(n, adj, k, g):
                return n, edges
        except:
            pass
    
    # Final fallback: larger LUW
    d_orig = max(7, g - 5 if (g - 5) % 2 else g - 4)
    n, edges, adj = luw_construction(k, d_orig)
    return n, edges
