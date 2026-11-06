"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque, defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []

    for spec in specs:
        if time.time() > deadline - 5.0:
            # Not enough time left; return trivial decomposition for remaining
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue

        edges = instance(*spec)
        n = spec[0]
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # We'll use a greedy approach: repeatedly find a long cycle in the remaining graph,
        # remove its edges, and add it as a piece. Remaining edges become individual pieces.
        
        # To find long cycles efficiently, we use a randomized greedy longest-path heuristic
        # that tries to extend a path as far as possible, then closes it into a cycle.
        
        def find_long_cycle(adj, remaining_edges, max_iter=50):
            """Try to find a long cycle using randomized greedy path extension."""
            if not remaining_edges:
                return None
            
            best_cycle = None
            best_len = 0
            
            # Get all vertices that have at least one remaining edge
            active_vertices = set()
            for u, v in remaining_edges:
                active_vertices.add(u)
                active_vertices.add(v)
            
            if len(active_vertices) < 3:
                return None
            
            # Try multiple random starting points
            vertices_list = list(active_vertices)
            random.shuffle(vertices_list)
            
            for start in vertices_list[:min(20, len(vertices_list))]:
                # Greedy path extension
                path = [start]
                visited = {start}
                current = start
                
                while True:
                    # Find neighbors of current that are in remaining_edges and not visited
                    # (or only visited if it's the start and we can close the cycle)
                    candidates = []
                    for v in adj[current]:
                        if v in visited:
                            # Can only close to start
                            if v == start and len(path) >= 3:
                                candidates.append(('close', v))
                            continue
                        # Check if edge (current, v) is in remaining_edges
                        e = tuple(sorted((current, v)))
                        if e in remaining_edges:
                            candidates.append(('extend', v))
                    
                    if not candidates:
                        break
                    
                    # Prefer extension over closing
                    extends = [c for c in candidates if c[0] == 'extend']
                    closes = [c for c in candidates if c[0] == 'close']
                    
                    if extends:
                        # Pick randomly among extensions, but prefer those with more options
                        # to avoid dead ends
                        def score(v):
                            return len([x for x in adj[v] if x not in visited or x == start])
                        extends.sort(key=lambda x: score(x[1]), reverse=True)
                        # With some probability pick the best, otherwise random
                        if random.random() < 0.7:
                            next_v = extends[0][1]
                        else:
                            next_v = random.choice(extends)[1]
                    elif closes:
                        next_v = closes[0][1]
                    else:
                        break
                    
                    if next_v == start:
                        # Close the cycle
                        cycle = path[:]
                        if len(cycle) >= 3 and all(len(set(cycle[i:i+2])) == 2 for i in range(len(cycle)-1)) and len(set(cycle[-1], cycle[0])) == 2:
                            # Verify all vertices distinct
                            if len(set(cycle)) == len(cycle):
                                if len(cycle) > best_len:
                                    best_len = len(cycle)
                                    best_cycle = cycle
                        break
                    else:
                        path.append(next_v)
                        visited.add(next_v)
                        current = next_v
                
                if time.time() > deadline - 10.0:
                    break
            
            return best_cycle
        
        # Greedy decomposition
        remaining = set(edges)
        pieces = []
        
        # Keep finding cycles until no more can be found
        progress = True
        while progress and time.time() < deadline - 5.0:
            progress = False
            cycle = find_long_cycle(adj, remaining)
            if cycle and len(cycle) >= 3:
                # Remove cycle edges from remaining
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = tuple(sorted((u, v)))
                    if e in remaining:
                        remaining.discard(e)
                pieces.append(cycle)
                progress = True
        
        # Add remaining edges as individual pieces
        for u, v in remaining:
            pieces.append([u, v])
        
        results.append(pieces)
    
    return results

# EVOLVE-BLOCK-END
