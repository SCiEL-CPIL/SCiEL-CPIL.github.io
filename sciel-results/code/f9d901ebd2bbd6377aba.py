import time
from collections import defaultdict

def construct():
    deadline = time.time() + 140

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_longest_cycle(adj, edge_set, n):
        best_cycle = None
        best_len = 2
        
        for start in range(n):
            if not adj[start]:
                continue
            
            # DFS
            # We use a stack of (node, path, visited_set)
            # To save memory, we can use a list for path and a set for visited
            # But copying sets is expensive. 
            # For N=30, it's acceptable.
            
            stack = [(start, [start], {start})]
            while stack:
                node, path, visited = stack.pop()
                
                # Check close
                if len(path) >= 3:
                    for neighbor in adj[node]:
                        if neighbor == start and (min(node, start), max(node, start)) in edge_set:
                            if len(path) > best_len:
                                best_len = len(path)
                                best_cycle = path[:]
                            
                # Expand
                # Pruning: if max possible length <= best_len, skip
                # Max possible length is len(path) + (n - len(visited))
                if len(path) + (n - len(visited)) <= best_len:
                    continue
                    
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        if (min(node, neighbor), max(node, neighbor)) in edge_set:
                            # New visited set
                            # Optimization: if we are deep, copying set is O(N)
                            # N=30 is small.
                            new_visited = visited | {neighbor}
                            stack.append((neighbor, path + [neighbor], new_visited))
                            
            if best_len == n:
                break

        return best_cycle

    def decompose(edges, n):
        if not edges:
            return []

        remaining_edges = set(edges)
        pieces = []
        
        while remaining_edges:
            if time.time() > deadline:
                for e in remaining_edges:
                    pieces.append(list(e))
                break
            
            # Build adj for remaining
            adj = defaultdict(list)
            for u, v in remaining_edges:
                adj[u].append(v)
                adj[v].append(u)
                
            cycle = find_longest_cycle(adj, remaining_edges, n)
            
            if cycle:
                cycle_edges = set()
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i+1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    cycle_edges.add(e)
                    
                pieces.append(cycle)
                remaining_edges -= cycle_edges
            else:
                for e in remaining_edges:
                    pieces.append(list(e))
                break
                
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n)
        results.append(pieces)

    return results
