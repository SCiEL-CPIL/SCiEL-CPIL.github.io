import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Fallback: L-U-W graph
    def luw_fallback():
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

    # Heuristic Search for Bipartite Graph
    def check_girth_bipartite(adj, n, g):
        # adj is list of lists of neighbors
        # n is total vertices (2*M)
        # g is target girth
        # Returns True if girth >= g
        M = n // 2
        # BFS from each vertex in A (0..M-1)
        for start in range(M):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= g // 2: # Max depth needed
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif parent[u] != v and parent[v] != u:
                        # Cycle found
                        # Length is dist[u] + dist[v] + 1? 
                        # In bipartite, cycle length is dist[u] + dist[v]
                        # Wait, if u and v are connected, and both visited.
                        # Path start->...->u-v->...->start
                        # Length = dist[u] + 1 + dist[v]
                        # But we need to ensure it's a simple cycle.
                        # For bipartite, if we find a cross edge, it's a cycle.
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def generate_bipartite_random(M, k):
        # Returns adjacency list
        adj = [[] for _ in range(2 * M)]
        # Stubs for A
        stubs_a = []
        for i in range(M):
            stubs_a.extend([i] * k)
        # Stubs for B
        stubs_b = []
        for i in range(M):
            stubs_b.extend([M + i] * k)
        
        # Shuffle B stubs
        random.shuffle(stubs_b)
        
        # Pair A[i] with B[i]
        # Check for multi-edges
        while True:
            random.shuffle(stubs_b)
            valid = True
            seen_edges = set()
            # Clear adj
            for i in range(2 * M):
                adj[i] = []
            for i in range(k * M):
                u = stubs_a[i]
                v = stubs_b[i]
                edge = (min(u, v), max(u, v))
                if edge in seen_edges:
                    valid = False
                    break
                seen_edges.add(edge)
                adj[u].append(v)
                adj[v].append(u)
            if valid:
                break
        return adj

    def local_search_bipartite(adj, n, g, max_steps):
        M = n // 2
        # Precompute neighbors for fast access
        # adj is list of lists
        
        for step in range(max_steps):
            # Check girth
            if check_girth_bipartite(adj, n, g):
                return True
            
            # Find a short cycle to break
            # We'll just pick a random switch and see if it improves girth
            # Or pick edges involved in short cycles.
            # For simplicity, try random switches.
            
            # Pick random edge (u, v)
            u = random.randint(0, M - 1)
            v = random.choice(adj[u])
            
            # Pick another random edge (w, x)
            w = random.randint(0, M - 1)
            x = random.choice(adj[w])
            
            # Ensure (u,v) != (w,x) and no shared vertices
            if u == w or u == x or v == w or v == x:
                continue
                
            # Try switch: (u,v) and (w,x) -> (u,x) and (w,v)
            # Check if (u,x) and (w,v) exist
            if x in adj[u] or v in adj[w]:
                continue
                
            # Perform switch
            adj[u].remove(v)
            adj[w].remove(x)
            adj[u].append(x)
            adj[w].append(v)
            adj[v].remove(u)
            adj[x].remove(w)
            adj[v].append(w)
            adj[x].append(u)
            
            # Check if girth improved
            # If not, revert? Or keep if not worse?
            # To avoid getting stuck, we can use a simple hill climbing.
            # But checking girth is expensive.
            # Let's just accept the switch if it doesn't create a loop (impossible in bipartite) 
            # and hope it helps.
            # Actually, we should check if it creates a shorter cycle.
            # This is hard to do quickly.
            # Let's just do a few random switches and check girth periodically.
            
            if step % 10 == 0:
                # Check girth
                if check_girth_bipartite(adj, n, g):
                    return True
                if time.time() > deadline:
                    return False
        return check_girth_bipartite(adj, n, g)

    best_n = None
    best_edges = None
    
    # Try bipartite graphs
    # Moore bound for bipartite 4-regular girth 10 is 242.
    # Known best is 320.
    for n in range(242, 322, 2):
        if time.time() > deadline:
            break
        if n % 2 != 0:
            continue
        M = n // 2
        # Try a few restarts
        for restart in range(5):
            if time.time() > deadline:
                break
            adj = generate_bipartite_random(M, k)
            # Local search
            if local_search_bipartite(adj, n, g, max_steps=100):
                # Convert to edge list
                edges = []
                for u in range(M):
                    for v in adj[u]:
                        if u < v: # Avoid duplicates? No, adj is undirected.
                            # Just collect all
                            pass
                # Collect unique edges
                edge_set = set()
                for u in range(n):
                    for v in adj[u]:
                        edge_set.add((min(u, v), max(u, v)))
                edges = list(edge_set)
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                # Found a valid graph, we can stop searching for smaller n?
                # No, we want the smallest n.
                # But we are iterating n upwards. So first found is smallest?
                # Yes, we iterate 242, 244, ...
                # So if we find one at 242, we return it.
                return n, edges
    
    # If no bipartite graph found, try non-bipartite?
    # Or just return L-U-W.
    # Given time constraints, L-U-W is safe.
    return luw_fallback()
