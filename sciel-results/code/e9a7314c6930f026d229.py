import random
import time
import sys
from collections import deque

def construct(k, g):
    # Target: 4-regular graph of girth 10 with n=320
    # If k != 4 or g != 10, fallback to LUW or other method
    if k != 4 or g != 10:
        return _construct_luw(k, g)

    n = 320
    deadline = time.time() + 110  # Leave 40s margin for 150s limit
    
    # Helper: Generate a simple random 4-regular graph using Wilson's algorithm approach
    # or configuration model with rejection
    def gen_random_4reg(n, rng=None):
        if rng is None:
            rng = random
        # Use configuration model with retries
        for attempt in range(100):
            stubs = []
            for i in range(n):
                for _ in range(4):
                    stubs.append(i)
            rng.shuffle(stubs)
            
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            
            if valid and all(len(adj[i]) == 4 for i in range(n)):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return edges
        return None

    # Helper: Check girth using BFS from each vertex
    def check_girth(n, edges, g_limit):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        min_cycle = float('inf')
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for nei in adj[node]:
                    if dist[nei] == -1:
                        dist[nei] = dist[node] + 1
                        parent[nei] = node
                        queue.append(nei)
                    elif nei != parent[node]:
                        cycle_len = dist[node] + dist[nei] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                            if min_cycle < g_limit:
                                return min_cycle
        return min_cycle

    # Helper: Find a specific short cycle for local search
    def find_short_cycle(n, adj, g):
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for nei in adj[node]:
                    if dist[nei] == -1:
                        dist[nei] = dist[node] + 1
                        parent[nei] = node
                        queue.append(nei)
                    elif nei != parent[node]:
                        cycle_len = dist[node] + dist[nei] + 1
                        if cycle_len < g:
                            # Reconstruct the cycle
                            # Path from start to node
                            path1 = []
                            cur = node
                            while cur != -1:
                                path1.append(cur)
                                cur = parent[cur]
                            path1.reverse()
                            # Path from start to nei
                            path2 = []
                            cur = nei
                            while cur != -1:
                                path2.append(cur)
                                cur = parent[cur]
                            path2.reverse()
                            # Find LCA
                            # Build sets for faster lookup
                            set1 = set(path1)
                            lca = start
                            for v in path2:
                                if v in set1:
                                    lca = v
                                    break
                            # Extract cycle
                            cycle = []
                            # From lca to node
                            i = path1.index(lca)
                            cycle.extend(path1[i:])
                            # From lca to nei (reversed)
                            j = path2.index(lca)
                            cycle.extend(path2[j:][::-1])
                            # Remove duplicate lca
                            if len(cycle) > 1:
                                cycle.pop()
                            return cycle
        return None

    # Helper: Get edge set and adjacency
    def get_adj(edges, n):
        adj = [[] for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add((min(u,v), max(u,v)))
        return adj, edge_set

    # Local search: try to improve girth by swapping edges
    def local_search(n, edges, g, deadline, rng):
        best_edges = list(edges)
        best_girth = check_girth(n, best_edges, g)
        
        adj, edge_set = get_adj(best_edges, n)
        
        # Build vertex-to-edge mapping for quick lookup
        # For each vertex, store its neighbors and the edge indices
        
        while time.time() < deadline - 5:
            # Find a short cycle
            cycle = find_short_cycle(n, adj, g)
            if cycle is None:
                break
            
            # Try to break this cycle by swapping an edge
            # For each edge in the cycle, try to remove it and add a different edge
            # that maintains 4-regularity
            
            improved = False
            cycle_edges = []
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i+1) % len(cycle)]
                cycle_edges.append((u, v))
            
            for u, v in cycle_edges:
                # Remove edge (u, v)
                # u loses neighbor v, v loses neighbor u
                # We need to find new neighbors for u and v
                # u's other neighbors: adj[u] minus v
                # v's other neighbors: adj[v] minus u
                
                # Find a vertex w that is not u, not v, not adjacent to u, not adjacent to v
                # and such that adding (u,w) and (v,w') doesn't create issues
                # Actually, we need to swap: remove (u,v), add (u,x) and (v,y) where x,y are chosen
                # such that degrees remain 4
                
                # Better approach: 2-switch. Pick two edges (u,v) and (a,b) and replace with (u,a) and (v,b)
                # or (u,b) and (v,a)
                
                # For simplicity, try to find a vertex w adjacent to neither u nor v
                # Remove (u,v), add (u,w) - but then w has degree 5
                # We need to also remove an edge from w
                
                # Let's do a proper 2-switch:
                # Remove edges (u,v) and (a,b), add edges (u,a) and (v,b)
                # Conditions: a != u, a != v, b != u, b != v
                # u not adjacent to a, v not adjacent to b
                # a not adjacent to b (already true since (a,b) is an edge we're removing... wait no)
                # Actually (a,b) must be an existing edge
                # After swap: u adjacent to a, v adjacent to b
                # Need: u not in adj[a] before, v not in adj[b] before
                # And the new edges don't create multi-edges
                
                # Find candidate (a, b)
                candidates = []
                for a in range(n):
                    if a == u or a == v:
                        continue
                    if v in adj[a]:  # a adjacent to v, skip
                        continue
                    for b in adj[a]:
                        if b == u or b == v:
                            continue
                        if u in adj[b]:  # b adjacent to u, skip
                            continue
                        if a == b:
                            continue
                        # Check that (u,a) and (v,b) are not existing edges
                        if (min(u,a), max(u,a)) in edge_set:
                            continue
                        if (min(v,b), max(v,b)) in edge_set:
                            continue
                        candidates.append((a, b))
                
                if not candidates:
                    continue
                
                # Try a random candidate
                a, b = rng.choice(candidates)
                
                # Perform the swap
                # Remove (u,v) and (a,b)
                adj[u].remove(v)
                adj[v].remove(u)
                adj[a].remove(b)
                adj[b].remove(a)
                edge_set.discard((min(u,v), max(u,v)))
                edge_set.discard((min(a,b), max(a,b)))
                
                # Add (u,a) and (v,b)
                adj[u].append(a)
                adj[a].append(u)
                adj[v].append(b)
                adj[b].append(v)
                edge_set.add((min(u,a), max(u,a)))
                edge_set.add((min(v,b), max(v,b)))
                
                # Check girth (quick check)
                new_girth = check_girth(n, list(edge_set), g)
                
                if new_girth > best_girth:
                    best_girth = new_girth
                    best_edges = list(edge_set)
                    improved = True
                else:
                    # Revert
                    adj[u].remove(a)
                    adj[a].remove(u)
                    adj[v].remove(b)
                    adj[b].remove(v)
                    edge_set.discard((min(u,a), max(u,a)))
                    edge_set.discard((min(v,b), max(v,b)))
                    adj[u].append(v)
                    adj[v].append(u)
                    adj[a].append(b)
                    adj[b].append(a)
                    edge_set.add((min(u,v), max(u,v)))
                    edge_set.add((min(a,b), max(a,b)))
            
            if improved:
                adj, edge_set = get_adj(best_edges, n)
            
            if best_girth >= g:
                break
        
        return best_edges, best_girth

    # Main: try local search
    rng = random.Random(42)
    
    # Generate initial random 4-regular graph
    edges = gen_random_4reg(n, rng)
    if edges is None:
        return _construct_luw(k, g)
    
    # Run local search
    edges, girth = local_search(n, edges, g, deadline, rng)
    
    if girth >= g:
        return n, edges
    
    # Fallback: Return LUW graph
    return _construct_luw(k, g)

def _construct_luw(k, g):
    # Original seed code
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
