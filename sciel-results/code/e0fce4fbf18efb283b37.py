import random
import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 140.0  # Safety margin

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_path(u, v, edges, rng):
        # Find a path from u to v in graph defined by edges
        # Returns list of vertices [u, ..., v] or None
        adj = {i: [] for i in range(max(u, v) + 1)} # This is inefficient, let's build proper adj
        # Better: build adj once outside or pass it
        # Let's pass adj as argument for efficiency, or build it inside if edges is small
        # Actually, edges changes each step, so we need to rebuild or update adj.
        # Rebuilding adj from edges is O(E). E is small.
        adj = {}
        for e in edges:
            a, b = e
            if a not in adj: adj[a] = []
            if b not in adj: adj[b] = []
            adj[a].append(b)
            adj[b].append(a)
        
        # DFS
        stack = [(u, [u])]
        visited = {u}
        while stack:
            curr, path = stack.pop()
            if curr == v:
                return path
            # Shuffle neighbors for randomness
            neighbors = list(adj.get(curr, []))
            rng.shuffle(neighbors)
            for nb in neighbors:
                if nb not in visited:
                    visited.add(nb)
                    stack.append((nb, path + [nb]))
        return None

    def decompose_once(edges, seed):
        rng = random.Random(seed)
        remaining = set(edges)
        pieces = []
        # Sort edges to ensure deterministic order for random.choice if needed?
        # random.choice on a set is not deterministic in order, but that's fine.
        # However, iterating a set is arbitrary.
        # Let's convert to list for random.choice? Or just pick a random edge from the set.
        # random.sample from list(remaining) is better.
        
        while remaining:
            if time.time() - start_time > TIME_LIMIT:
                # Emergency exit: return current best or simple decomposition
                # For safety, just add all remaining as edges
                for e in remaining:
                    pieces.append(list(e))
                return pieces
            
            # Pick a random edge
            # To avoid list(remaining) overhead, we can just pick an index
            # But list(remaining) is fast for small sets
            edge_list = list(remaining)
            u, v = edge_list[rng.randrange(len(edge_list))]
            remaining.remove((u, v))
            
            path = find_path(u, v, remaining, rng)
            if path:
                # path is [u, ..., v]
                # Cycle vertices: path
                # Remove edges in path
                for i in range(len(path) - 1):
                    e1 = tuple(sorted((path[i], path[i + 1])))
                    if e1 in remaining:
                        remaining.remove(e1)
                pieces.append(path)
            else:
                pieces.append([u, v])
                
        return pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        edges = instance(*spec)
        best_pieces = None
        best_count = float('inf')
        
        # Try multiple restarts
        # Estimate time per restart:
        # Max 400 edges. Each step O(E) for adj + DFS.
        # Very fast. 10 restarts should be fine.
        for restart_seed in range(10):
            if time.time() - start_time > TIME_LIMIT - 5: # Leave 5s buffer
                break
            pieces = decompose_once(edges, 1000 + restart_seed)
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = pieces
                
        if best_pieces is None:
            # Fallback if no restart completed
            best_pieces = [list(e) for e in edges]
            
        results.append(best_pieces)
        
    return results

# Helper for Random class usage in instance generator
from random import Random
