"""Improved program for cycle decomposition."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def decompose(edges):
        remaining = set(edges)
        pieces = []
        
        # Pre-calculate degrees for initial sorting, but we need dynamic degrees
        # We will recalculate degrees or maintain them.
        # For small graphs, recalculating or using a set is fine.
        
        while remaining:
            if time.time() > deadline:
                break
            
            # Strategy: Pick an edge that is likely to be in a cycle.
            # Edges in dense parts are more likely to be in cycles.
            # Calculate degree of each node in 'remaining'
            deg = defaultdict(int)
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1
            
            # Pick edge with max sum of degrees
            # If tie, pick arbitrarily (or randomly)
            best_edge = None
            best_score = -1
            # To make it deterministic but varied, we can just pick the first max
            for e in remaining:
                u, v = e
                score = deg[u] + deg[v]
                if score > best_score:
                    best_score = score
                    best_edge = e
            
            if best_edge is None:
                break

            u, v = best_edge
            
            # Try to find a path from u to v in remaining \ {best_edge}
            # Using DFS
            path = None
            remaining_minus_edge = remaining - {best_edge}
            
            # Build adjacency for remaining_minus_edge
            adj = defaultdict(set)
            for a, b in remaining_minus_edge:
                adj[a].add(b)
                adj[b].add(a)
            
            # DFS to find path from u to v
            # We want a long path, so we can use a heuristic DFS
            # Standard DFS finds *a* path.
            # To encourage longer paths, we can sort neighbors by degree (desc) or randomize.
            # Let's try a few random restarts of DFS if the first path is short?
            # For now, just one DFS with a heuristic: visit neighbors with higher degree first.
            
            visited = {u}
            stack = [(u, [u])]
            found_path = None
            
            # To avoid getting stuck in a short cycle immediately, 
            # we can try to extend. But DFS backtracks.
            # A simple DFS will find the shortest path in terms of edges? 
            # No, DFS finds *a* path. It's not guaranteed to be shortest or longest.
            # However, if we sort neighbors by degree, we tend to go into dense areas.
            
            while stack:
                node, cur_path = stack.pop()
                if node == v:
                    found_path = cur_path
                    break
                # Sort neighbors by degree descending to prioritize dense areas
                neighbors = sorted(adj[node], key=lambda x: deg[x], reverse=True)
                for nb in neighbors:
                    if nb not in visited:
                        visited.add(nb)
                        stack.append((nb, cur_path + [nb]))
            
            if found_path:
                # Cycle is found_path + [v, u] (closing edge is best_edge)
                # Wait, found_path is u -> ... -> v.
                # The cycle is found_path + [v, u]? No, the edge is (u,v).
                # The cycle vertices are found_path + [u] (to close the loop back to start? No)
                # Path: u -> n1 -> ... -> v.
                # Cycle: u, n1, ..., v, u.
                cycle = found_path + [u]
                
                # Verify cycle validity (distinct vertices except start/end)
                # found_path has distinct vertices because of 'visited' set.
                # So cycle has distinct vertices.
                
                pieces.append(cycle)
                
                # Remove all edges in this cycle from remaining
                for i in range(len(cycle) - 1):
                    a, b = cycle[i], cycle[i+1]
                    e = (min(a, b), max(a, b))
                    remaining.discard(e)
                
                # Note: best_edge (u,v) is also removed if it's in the cycle
                # The last edge in cycle is (v, u) which is best_edge.
                # So it gets removed.
            else:
                # No path found, so best_edge is a bridge (in this subgraph)
                # It must be a single edge piece
                pieces.append([u, v])
                remaining.discard(best_edge)
                
        # If we hit deadline, add remaining as single edges
        if remaining:
            for e in sorted(remaining):
                pieces.append([e[0], e[1]])
                
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline:
            # Fallback: all edges as single pieces
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            edges = instance(*spec)
            pieces = decompose(edges)
            results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
