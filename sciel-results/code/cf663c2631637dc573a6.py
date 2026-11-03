"""Construct small 5-regular graphs of girth 9 via randomized local search."""
import time
import random
import sys

def construct(k, g):
    deadline = time.time() + 115
    
    def check_girth_fast(n, adj):
        """Check if graph has girth >= g using BFS from each vertex."""
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            for node in queue:
                if dist[node] >= g - 1:
                    continue
                for neighbor in adj[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        parent[neighbor] = node
                        queue.append(neighbor)
                    elif neighbor != parent[node]:
                        # Found a cycle
                        cycle_len = dist[node] + dist[neighbor] + 1
                        if cycle_len < g:
                            return False
        return True
    
    def find_short_cycle(n, adj):
        """Find a cycle shorter than g."""
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            for node in queue:
                if dist[node] >= g - 1:
                    continue
                for neighbor in adj[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        parent[neighbor] = node
                        queue.append(neighbor)
                    elif neighbor != parent[node]:
                        cycle_len = dist[node] + dist[neighbor] + 1
                        if cycle_len < g:
                            # Reconstruct cycle
                            cycle = []
                            a, b = node, neighbor
                            while a != b:
                                cycle.append(a)
                                a = parent[a]
                            cycle.append(b)
                            # Reconstruct back from b
                            path_b = []
                            b = neighbor
                            while b != cycle[-1]:
                                path_b.append(b)
                                b = parent[b]
                            path_b.reverse()
                            return cycle + path_b
        return None
    
    def generate_random_k_regular(n, k):
        """Generate a random k-regular graph."""
        if n * k % 2 != 0:
            return None
        for _ in range(100):
            stubs = []
            for i in range(n):
                stubs.extend([i] * k)
            random.shuffle(stubs)
            edges = set()
            adj = [[] for _ in range(n)]
            valid = True
            i = 0
            while i < len(stubs):
                u = stubs[i]
                v = stubs[i + 1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edges:
                    valid = False
                    break
                edges.add(key)
                adj[u].append(v)
                adj[v].append(u)
                i += 2
            if valid:
                return adj
        return None
    
    def try_rewire(adj, n, u, v, ou, ov):
        """Try to rewire edges (u,v) and (ou,ov) to (u,ou) and (v,ov)."""
        if u == ou or u == ov or v == ou or v == ov:
            return False
        if ov in adj[u] or ou in adj[v]:
            return False
        
        # Make change
        adj[u].remove(v)
        adj[v].remove(u)
        adj[ou].remove(ov)
        adj[ov].remove(ou)
        adj[u].append(ou)
        adj[v].append(ov)
        adj[ou].append(u)
        adj[ov].append(v)
        
        if check_girth_fast(n, adj):
            return True
        else:
            # Revert
            adj[u].remove(ou)
            adj[v].remove(ov)
            adj[ou].remove(u)
            adj[ov].remove(v)
            adj[u].append(v)
            adj[v].append(u)
            adj[ou].append(ov)
            adj[ov].append(ou)
            return False
    
    # Moore bound
    if g % 2 == 1:
        half = (g - 3) // 2
        moore = 1 + k
        for i in range(1, half + 1):
            moore += k * (k - 1) ** i
    else:
        moore = 0
        for i in range(g // 2):
            moore += 2 * (k - 1) ** i
    
    best_n = None
    best_edges = None
    
    # Try starting sizes from Moore bound upward
    for start_n in range(moore, moore + 150):
        if time.time() > deadline - 5:
            break
        
        for attempt in range(30):
            if time.time() > deadline - 5:
                break
            
            adj = generate_random_k_regular(start_n, k)
            if adj is None:
                continue
            
            if check_girth_fast(start_n, adj):
                if best_n is None or start_n < best_n:
                    best_n = start_n
                    best_edges = [(min(u, v), max(u, v)) for u in range(start_n) for v in adj[u] if u < v]
                break
            
            # Try to improve
            for _ in range(500):
                if time.time() > deadline - 5:
                    break
                
                cycle = find_short_cycle(start_n, adj)
                if cycle is None:
                    if best_n is None or start_n < best_n:
                        best_n = start_n
                        best_edges = [(min(u, v), max(u, v)) for u in range(start_n) for v in adj[u] if u < v]
                    break
                
                # Get edges in cycle
                cycle_set = set(cycle)
                cycle_edges = []
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    cycle_edges.append((u, v))
                
                random.shuffle(cycle_edges)
                for u, v in cycle_edges:
                    # Try to find a good rewire
                    for _ in range(10):
                        ou = random.randint(0, start_n - 1)
                        if ou in cycle_set:
                            continue
                        if not adj[ou]:
                            continue
                        ov = random.choice(adj[ou])
                        if ov in cycle_set:
                            continue
                        if (ou, ov) in cycle_edges or (ov, ou) in cycle_edges:
                            continue
                        
                        if try_rewire(adj, start_n, u, v, ou, ov):
                            break
                    else:
                        continue
                    break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback
    n = max(moore, 2 * k)
    adj = generate_random_k_regular(n, k)
    if adj is None:
        return n, []
    edges = [(min(u, v), max(u, v)) for u in range(n) for v in adj[u] if u < v]
    return n, edges
