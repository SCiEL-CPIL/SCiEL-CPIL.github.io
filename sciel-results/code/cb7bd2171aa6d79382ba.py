import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before the 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_cycle_by_extension(adj, start, rng, max_len=30):
        """
        Try to find a cycle starting from `start` by greedily extending a path
        and looking for a back edge to an earlier vertex in the path.
        """
        if len(adj[start]) < 2:
            return None
        
        # Build a path greedily
        path = [start]
        visited = {start}
        current = start
        
        # We want to avoid going back to parent, so track parent
        parent = {start: -1}
        
        # Try multiple random extensions
        best_cycle = None
        best_len = 0
        
        for attempt in range(20):  # Try 20 different random paths
            path = [start]
            visited = {start}
            current = start
            parent = {start: -1}
            
            # Build a long path
            while len(path) < max_len:
                neighbors = list(adj[current])
                # Filter out parent and already visited
                candidates = [n for n in neighbors if n != parent.get(current, -1) and n not in visited]
                
                if not candidates:
                    break
                
                # Choose next vertex randomly, but prefer higher degree
                rng.shuffle(candidates)
                # Sort by degree descending for bias
                candidates.sort(key=lambda x: len(adj[x]), reverse=True)
                # Mix: take top half and random from rest
                if len(candidates) > 2:
                    top_half = candidates[:len(candidates)//2]
                    rest = candidates[len(candidates)//2:]
                    rng.shuffle(top_half)
                    next_v = top_half[0] if top_half else candidates[0]
                else:
                    next_v = candidates[0]
                
                visited.add(next_v)
                parent[next_v] = current
                path.append(next_v)
                current = next_v
            
            # Now look for a back edge from current or any vertex in path to an earlier vertex
            # Check from the end of the path backwards
            for i in range(len(path) - 1, -1, -1):
                v = path[i]
                for w in adj[v]:
                    if w in visited and w != parent.get(v, -1):
                        # Find index of w in path
                        if w in path:
                            w_idx = path.index(w)
                            if w_idx < i:  # Ensure we form a proper cycle
                                cycle = path[w_idx:i+1]
                                if len(cycle) > best_len:
                                    best_len = len(cycle)
                                    best_cycle = list(cycle)
            
            if best_len >= 15:
                break
        
        return best_cycle

    def find_best_cycle(adj, rng):
        """
        Finds a long cycle in the graph represented by adj.
        Uses multiple strategies: greedy extension and DFS-based.
        """
        # Get vertices with degree > 1
        candidates = [u for u, neighbors in adj.items() if len(neighbors) > 1]
        if not candidates:
            return None

        best_cycle = None
        best_len = 0

        # Sort by degree to bias towards hubs
        candidates.sort(key=lambda x: len(adj[x]), reverse=True)
        
        # Try top candidates and some random ones
        num_to_try = min(12, len(candidates))
        starts = candidates[:num_to_try]
        if len(candidates) > num_to_try:
            starts += rng.sample(candidates, min(4, len(candidates) - num_to_try))
        
        starts = list(set(starts))
        rng.shuffle(starts)

        for start in starts:
            # Strategy 1: Greedy path extension
            cycle = find_cycle_by_extension(adj, start, rng)
            if cycle and len(cycle) > best_len:
                best_len = len(cycle)
                best_cycle = cycle
            
            # Strategy 2: DFS-based cycle finding
            color = {start: 1}
            parent = {start: -1}
            path = [start]
            path_index = {start: 0}
            local_best_cycle = None
            local_best_len = 0
            
            def get_shuffled_iter(neighs):
                lst = list(neighs)
                rng.shuffle(lst)
                return iter(lst)
            
            stack = [(start, get_shuffled_iter(adj[start]))]
            
            while stack:
                u, it = stack[-1]
                found_next = False
                for w in it:
                    if w == parent[u]:
                        continue
                    if color.get(w, 0) == 0:
                        color[w] = 1
                        parent[w] = u
                        path.append(w)
                        path_index[w] = len(path) - 1
                        stack.append((w, get_shuffled_iter(adj[w])))
                        found_next = True
                        break
                    elif color.get(w, 0) == 1:
                        # Back edge found: w is on the current path
                        if w in path_index:
                            idx = path_index[w]
                            cyc = path[idx:]
                            if len(cyc) > local_best_len:
                                local_best_len = len(cyc)
                                local_best_cycle = list(cyc)
                
                if not found_next:
                    stack.pop()
                    color[u] = 2
                    if path and path[-1] == u:
                        path.pop()
                        if u in path_index:
                            del path_index[u]
            
            if local_best_cycle and local_best_len > best_len:
                best_len = local_best_len
                best_cycle = local_best_cycle
                
            # Early exit if we found a very good cycle
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
        
        # Adaptive restarts based on time
        max_restarts = 200
        
        for i in range(max_restarts):
            if time.time() - start_time > DEADLINE:
                break
                
            rng = random.Random(i * 7 + 13)  # Better seed mixing
            # Deep copy adj for this restart
            current_adj = {u: set(v) for u, v in adj.items()}
            current_edges = set(edges)
            pieces = []
            
            while True:
                if time.time() - start_time > DEADLINE:
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
