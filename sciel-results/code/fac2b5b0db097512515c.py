import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    def remaining():
        return deadline - time.time()

    def check_girth(n, edges, g, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        min_cycle = float('inf')
        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g:
                return min_cycle
        return min_cycle

    def random_3_regular(n):
        if n % 2 != 0:
            return None
        half = n // 2
        left = list(range(half))
        right = list(range(half, n))
        random.shuffle(left)
        random.shuffle(right)
        edges = []
        for i in range(half):
            edges.append((left[i], right[i]))
        # Now we need to add 2 more edges per vertex
        # We can use a random perfect matching on the remaining degrees
        # This is a standard way to generate random regular graphs
        # But a simpler way is to use the configuration model
        stubs = []
        for i in range(n):
            stubs.extend([i, i, i])
        random.shuffle(stubs)
        edges = []
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u != v:
                edges.append((u, v))
        # Check for simplicity
        if len(set(tuple(sorted(e)) for e in edges)) != len(edges):
            return None
        # Check for degree
        deg = [0]*n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != 3 for d in deg):
            return None
        return edges

    def local_search(n, edges, g, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        def get_girth():
            min_cycle = float('inf')
            for s in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[s] = 0
                queue = [s]
                qi = 0
                while qi < len(queue):
                    node = queue[qi]
                    qi += 1
                    for nb in adj[node]:
                        if nb == parent[node]:
                            continue
                        if dist[nb] != -1:
                            cycle_len = dist[node] + dist[nb] + 1
                            if cycle_len < min_cycle:
                                min_cycle = cycle_len
                            if cycle_len < g:
                                return min_cycle
                        else:
                            dist[nb] = dist[node] + 1
                            parent[nb] = node
                            queue.append(nb)
                if min_cycle < g:
                    return min_cycle
            return min_cycle

        best_girth = get_girth()
        if best_girth >= g:
            return edges, best_girth

        while time.time() < time_limit:
            # Pick a random edge to swap
            i = random.randint(0, len(edges)-1)
            u, v = edges[i]
            # Find another edge to swap with
            j = random.randint(0, len(edges)-1)
            if i == j:
                continue
            x, y = edges[j]
            # Swap (u,v) and (x,y) with (u,x) and (v,y) or (u,y) and (v,x)
            # Check if new edges are valid (no loops, no multi-edges)
            new_edges_1 = (u, x)
            new_edges_2 = (v, y)
            if new_edges_1[0] == new_edges_1[1] or new_edges_2[0] == new_edges_2[1]:
                continue
            # Check if edges already exist
            if (u, x) in edges or (x, u) in edges or (v, y) in edges or (y, v) in edges:
                continue
            
            # Perform swap
            edges[i] = new_edges_1
            edges[j] = new_edges_2
            
            # Update adjacency
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            
            new_girth = get_girth()
            if new_girth > best_girth:
                best_girth = new_girth
                if best_girth >= g:
                    return edges, best_girth
            else:
                # Revert
                edges[i] = (u, v)
                edges[j] = (x, y)
                adj[u].remove(x)
                adj[x].remove(u)
                adj[v].remove(y)
                adj[y].remove(v)
                adj[u].append(v)
                adj[v].append(u)
                adj[x].append(y)
                adj[y].append(x)
        
        return edges, best_girth

    if k != 3:
        return (0, [])

    n = 936
    best_edges = None
    best_girth = 0
    
    for _ in range(5):
        if remaining() < 20:
            break
        edges = random_3_regular(n)
        if edges is None:
            continue
        edges, girth = local_search(n, edges, g, deadline - 10)
        if girth > best_girth:
            best_girth = girth
            best_edges = edges
            if girth >= g:
                return (n, edges)
    
    if best_edges is not None:
        return (n, best_edges)
    return (0, [])
