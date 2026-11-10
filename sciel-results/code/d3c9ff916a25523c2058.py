"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict
import random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def find_best_decomposition(edges, n, max_iters=200):
        """Find decomposition with minimum pieces using randomized greedy cycle extraction."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        if not edge_set:
            return []
        
        best_pieces = None
        best_count = len(edge_set) + 1
        
        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Build adjacency
            adj = defaultdict(set)
            for u, v in edge_set:
                adj[u].add(v)
                adj[v].add(u)
            
            remaining = set(edge_set)
            pieces = []
            
            # Greedy: repeatedly find a cycle in the remaining graph
            # Try to find longer cycles first by using BFS/DFS with depth preference
            while remaining:
                # Pick a random starting edge to add diversity
                if not remaining:
                    break
                start_edge = random.choice(list(remaining))
                start = start_edge[0]
                
                # Try to find a cycle starting from 'start' using randomized DFS
                # with preference for longer paths
                cycle = _find_cycle_greedy(start, adj, remaining)
                
                if cycle and len(cycle) >= 3:
                    # Remove cycle edges from remaining
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        e = tuple(sorted((u, v)))
                        remaining.discard(e)
                    pieces.append(cycle)
                else:
                    # No cycle found from this start; try another approach
                    # Just take one edge as a piece
                    e = tuple(sorted(start_edge))
                    remaining.discard(e)
                    pieces.append(list(start_edge))
            
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = [p[:] for p in pieces]
        
        return best_pieces if best_pieces else [list(e) for e in edge_set]
    
    def _find_cycle_greedy(start, adj, remaining):
        """Find a cycle starting at 'start' using randomized DFS."""
        # Try multiple random walks to find a cycle back to start
        for attempt in range(50):
            path = [start]
            visited = {start}
            current = start
            found = False
            
            for _ in range(100):  # max path length
                neighbors = [nb for nb in adj[current] if (min(current, nb), max(current, nb)) in remaining]
                if not neighbors:
                    break
                
                # Prefer unvisited neighbors for longer cycles, but allow revisiting start
                unvisited = [nb for nb in neighbors if nb not in visited]
                if unvisited:
                    next_node = random.choice(unvisited)
                elif start in neighbors:
                    # Close the cycle
                    next_node = start
                    found = True
                else:
                    break
                
                if found:
                    if len(path) >= 3:
                        return path
                    break
                
                path.append(next_node)
                visited.add(next_node)
                current = next_node
            
            if found and len(path) >= 3:
                return path
        
        return None
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = find_best_decomposition(edges, n)
        results.append(pieces)
        if time.time() > deadline - 5:
            # Fallback: just use individual edges for remaining graphs
            for spec2 in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))[len(results):]:
                n2, p2, seed2 = spec2
                edges2 = instance(n2, p2, seed2)
                results.append([list(e) for e in edges2])
            break
    
    return results
# EVOLVE-BLOCK-END
