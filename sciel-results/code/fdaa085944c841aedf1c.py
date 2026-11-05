import time
import random
from collections import defaultdict
import heapq

_T0 = time.time()
_DEADLINE = 150.0
_MARGIN = 10.0

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = set(tuple(sorted(e)) for e in zip(order, order[1:]))
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _solve():
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    for n, p, seed in specs:
        if time.time() > _DEADLINE - _MARGIN:
            # Fallback: all singles
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
            
        edges = _instance(n, p, seed)
        if not edges:
            results.append([])
            continue
            
        # Build adjacency list and degree
        adj = defaultdict(set)
        deg = defaultdict(int)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            
        # Find odd degree vertices
        odd_verts = [u for u in range(n) if deg[u] % 2 == 1]
        
        if not odd_verts:
            # Graph is Eulerian
            cycles = _decompose_eulerian(adj, n)
            results.append(cycles)
            continue
            
        # Compute shortest paths between all vertices (Floyd-Warshall)
        # dist[i][j]
        INF = float('inf')
        dist = [[INF] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        for u, v in edges:
            dist[u][v] = 1
            dist[v][u] = 1
            
        for k in range(n):
            for i in range(n):
                if dist[i][k] == INF: continue
                for j in range(n):
                    if dist[k][j] == INF: continue
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        
        # Find T-Join
        # We want to cover odd_verts with disjoint paths
        # This is Min Weight Perfect Matching on complete graph of odd_verts
        # Edge weight is dist[u][v]
        
        t_join_edges = set()
        if len(odd_verts) <= 20:
            # DP for Min Weight Perfect Matching
            k = len(odd_verts)
            # dp[mask] where mask is bitmask of odd_verts
            size = 1 << k
            dp = [INF] * size
            dp[0] = 0
            parent = [-1] * size # not strictly needed for reconstruction if we store edges
            
            # To reconstruct, we can store the choice or re-run
            # Let's store the pair chosen for each state
            choice = [None] * size
            
            for mask in range(size):
                if dp[mask] == INF: continue
                # Find first unset bit? No, we are building up.
                # Standard DP: pick first set bit i, pair with j.
                # Actually, it's easier to iterate masks and remove pairs.
                # Let's use the standard:
                # dp[mask] = min over i<j in mask of dist[i][j] + dp[mask ^ (1<<i) ^ (1<<j)]
                
                # Optimization: only process if mask has even number of bits?
                # All states with even bits are reachable.
                
                if mask == 0: continue
                
                # Find first set bit
                i = 0
                while i < k and not (mask & (1 << i)):
                    i += 1
                if i == k: continue
                
                # Try pairing i with j > i
                for j in range(i + 1, k):
                    if mask & (1 << j):
                        prev_mask = mask ^ (1 << i) ^ (1 << j)
                        cost = dist[odd_verts[i]][odd_verts[j]] + dp[prev_mask]
                        if cost < dp[mask]:
                            dp[mask] = cost
                            choice[mask] = (i, j)
                            
            # Reconstruct
            mask = (1 << k) - 1
            while mask:
                i, j = choice[mask]
                u, v = odd_verts[i], odd_verts[j]
                # Add shortest path edges
                # We need to reconstruct the path
                path = _get_path(u, v, dist, adj, n)
                for e in path:
                    t_join_edges.add(e)
                mask ^= (1 << i) ^ (1 << j)
                
        else:
            # Greedy Matching
            pairs = []
            for i in range(len(odd_verts)):
                for j in range(i + 1, len(odd_verts)):
                    u, v = odd_verts[i], odd_verts[j]
                    pairs.append((dist[u][v], u, v))
            pairs.sort()
            matched = set()
            for d, u, v in pairs:
                if u in matched or v in matched:
                    continue
                matched.add(u)
                matched.add(v)
                path = _get_path(u, v, dist, adj, n)
                for e in path:
                    t_join_edges.add(e)
                
        # Remove T-Join edges from graph to get Eulerian subgraph F
        adj_f = defaultdict(set)
        for u, v in edges:
            e1 = (min(u, v), max(u, v))
            if e1 not in t_join_edges:
                adj_f[u].add(v)
                adj_f[v].add(u)
                
        # Decompose F into cycles
        cycles = _decompose_eulerian(adj_f, n)
        
        # Add T-Join edges as single pieces
        for u, v in t_join_edges:
            cycles.append([u, v])
            
        results.append(cycles)
        
    return results

def _get_path(u, v, dist, adj, n):
    # Reconstruct shortest path from dist matrix
    # Since we only have dist, we need to find neighbors
    # We can do a simple BFS/DFS on the original graph constrained by dist
    # dist[u][v] is the length.
    # Next step w must satisfy dist[u][w] + dist[w][v] == dist[u][v] and w is neighbor of u.
    
    if u == v:
        return []
        
    path = []
    curr = u
    target = v
    while curr != target:
        next_node = -1
        for w in adj[curr]:
            if dist[curr][w] + dist[w][target] == dist[curr][target]:
                next_node = w
                break
        if next_node == -1:
            # Should not happen if dist is correct
            break
        path.append((min(curr, next_node), max(curr, next_node)))
        curr = next_node
    return path

def _decompose_eulerian(adj, n):
    # adj is a dict of sets
    # Make a copy
    adj_copy = {u: set(v) for u, v in adj.items()}
    cycles = []
    
    # While there are edges
    total_edges = sum(len(v) for v in adj_copy.values()) // 2
    processed = 0
    
    while processed < total_edges:
        # Find a vertex with degree > 0
        start = -1
        for u in adj_copy:
            if len(adj_copy[u]) > 0:
                start = u
                break
        if start == -1:
            break
            
        # Trace cycle
        cycle = [start]
        curr = start
        while True:
            # Find next neighbor
            if not adj_copy[curr]:
                # Should not happen in Eulerian graph unless we are done
                break
            next_node = next(iter(adj_copy[curr]))
            # Remove edge
            adj_copy[curr].remove(next_node)
            adj_copy[next_node].remove(curr)
            processed += 1
            if next_node == start:
                break
            cycle.append(next_node)
            curr = next_node
            
        if len(cycle) >= 3:
            cycles.append(cycle)
            
    return cycles

def construct():
    return _solve()
