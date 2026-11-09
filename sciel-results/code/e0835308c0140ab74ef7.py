import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before the 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_best_cycle(adj, rng):
        """
        Finds a long cycle in the graph using iterative DFS with proper cycle extraction.
        Tries multiple start vertices biased towards high degree.
        """
        candidates = [u for u, neighbors in adj.items() if len(neighbors) > 1]
        if not candidates:
            return None

        best_cycle = None
        best_len = 0

        candidates.sort(key=lambda x: len(adj[x]), reverse=True)
        # Pick top 8 and some random ones for diversity
        starts = candidates[:min(8, len(candidates))]
        if len(candidates) > 8:
            starts += rng.sample(candidates, min(8, len(candidates) - 8))
        starts = list(set(starts))
        rng.shuffle(starts)

        for start in starts:
            color = {}
            parent = {}
            path = []
            path_index = {}
            local_best_cycle = None
            local_best_len = 0

            # Iterative DFS to avoid recursion limits and allow better control
            stack = [(start, iter(adj[start]))]
            color[start] = 1
            parent[start] = -1
            path.append(start)
            path_index[start] = 0

            while stack:
                u, neighbors_iter = stack[-1]
                found_next = False
                for w in neighbors_iter:
                    if w == parent[u]:
                        continue
                    if color.get(w, 0) == 0:
                        color[w] = 1
                        parent[w] = u
                        path.append(w)
                        path_index[w] = len(path) - 1
                        stack.append((w, iter(adj[w])))
                        found_next = True
                        break
                    elif color.get(w, 0) == 1:
                        # Back edge: w is on current path
                        idx = path_index.get(w)
                        if idx is not None:
                            cyc = path[idx:]
                            if len(cyc) > local_best_len:
                                local_best_len = len(cyc)
                                local_best_cycle = list(cyc)
                if not found_next:
                    stack.pop()
                    color[u] = 2
                    if path and path[-1] == u:
                        path.pop()
                        if path_index:
                            del path_index[u]

            if local_best_cycle and local_best_len > best_len:
                best_len = local_best_len
                best_cycle = local_best_cycle
            
            # Early exit if we found a very long cycle
            if best_len >= 25:
                break

        return best_cycle

    def decompose(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        best_pieces = None
        best_count = float('inf')
        
        max_restarts = 200 # Adjust based on speed, iterative DFS is faster
        
        for i in range(max_restarts):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            rng = random.Random(i * 13 + 7)
            # Deep copy adj for this restart
            current_adj = {u: set(v) for u, v in adj.items()}
            current_edges = set(edges)
            pieces = []
            
            while True:
                if time.time() - start_time > DEADLINE - 2:
                    break
                
                # Check if any edges remain
                if not any(len(neigh) > 0 for neigh in current_adj.values()):
                    break
                
                cycle = find_best_cycle(current_adj, rng)
                if cycle is None:
                    break
                
                # Remove cycle edges
                for j in range(len(cycle)):
                    a, b = cycle[j], cycle[(j + 1) % len(cycle)]
                    e = (min(a, b), max(a, b))
                    current_edges.discard(e)
                    if b in current_adj[a]:
                        current_adj[a].discard(b)
                    if a in current_adj[b]:
                        current_adj[b].discard(a)
                
                pieces.append(list(cycle))
            
            # Add remaining edges as single pieces
            for e in current_edges:
                pieces.append([e[0], e[1]])
            
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = [list(p) for p in pieces]
                
        if best_pieces is None:
            return [[u, v] for u, v in edges]
            
        return best_pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        results.append(decompose(edges))
    return results
