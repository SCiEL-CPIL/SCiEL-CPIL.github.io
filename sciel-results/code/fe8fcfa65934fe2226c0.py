import time
import random
from collections import defaultdict, deque

def construct():
    t0 = time.time()
    DEADLINE = 145.0
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_warnsdorff(edges, deadline):
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        paths = []
        
        # Helper to get available neighbors for Warnsdorff
        def get_next(u, path_vertices, used_edges):
            cands = []
            for v in adj[u]:
                e = (min(u, v), max(u, v))
                if e in used_edges:
                    continue
                if v in path_vertices:
                    continue
                # Calculate flexibility of v
                # Number of unused edges incident to v, excluding e
                flex = 0
                for w in adj[v]:
                    if w == u:
                        continue
                    e2 = (min(v, w), max(v, w))
                    if e2 not in used_edges and w not in path_vertices:
                        flex += 1
                cands.append((flex, v))
            
            if not cands:
                return None
            
            # Warnsdorff: pick min flexibility
            cands.sort(key=lambda x: (x[0], x[1]))
            return cands[0][1]

        while remaining and time.time() < deadline:
            # Pick a start vertex
            # Heuristic: Pick a vertex with the most remaining edges
            # Or random?
            # Let's try picking the vertex with highest remaining degree
            deg_rem = defaultdict(int)
            for u, v in remaining:
                deg_rem[u] += 1
                deg_rem[v] += 1
            
            if not deg_rem:
                break
                
            start = max(deg_rem.keys(), key=lambda x: (deg_rem[x], -x))
            
            path = [start]
            path_vertices = {start}
            used_edges = set()
            
            while True:
                next_v = get_next(path[-1], path_vertices, used_edges)
                if next_v is None:
                    break
                path.append(next_v)
                path_vertices.add(next_v)
                e = (min(path[-2], next_v), max(path[-2], next_v))
                used_edges.add(e)
            
            if len(path) > 1:
                paths.append(path)
                for e in used_edges:
                    remaining.discard(e)
            
            # If we got stuck with a single vertex (no edges), skip
            # But remaining is not empty, so this shouldn't happen unless isolated vertex in remaining?
            # If remaining has edges, deg_rem > 0.
            
        return paths

    result = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        best_paths = None
        best_count = float('inf')
        
        # Try multiple restarts with different random seeds for tie-breaking or start selection
        # Since the graph is fixed, the randomness comes from tie-breaking in Warnsdorff
        # or we can randomize the start vertex selection.
        
        for attempt in range(20): # 20 restarts per graph
            if time.time() > DEADLINE - 5:
                break
            
            # Modify decompose to accept a random seed for tie-breaking
            # Or just run it. The `max` with `-x` is deterministic.
            # Let's add a bit of randomness to start selection in the restart loop.
            
            # Actually, let's just run the deterministic Warnsdorff first.
            # If we want to explore, we can randomize the "flexibility" calculation slightly or the start.
            
            # For now, let's just run the deterministic version once.
            # If it's not good, we need a better heuristic.
            
            paths = decompose_warnsdorff(edges, DEADLINE - 5)
            
            # Verify and count
            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
            
            # If we are already at the theoretical lower bound (1?), we can stop.
            # But lower bound is not 1.
            # If best_count is small, we can stop early.
            if best_count <= 10: # Arbitrary threshold
                break

        if best_paths is None:
            # Fallback to single edges
            best_paths = [list(e) for e in edges]
        
        result.append(best_paths)
        
    return result

# EVOLVE-BLOCK-END
