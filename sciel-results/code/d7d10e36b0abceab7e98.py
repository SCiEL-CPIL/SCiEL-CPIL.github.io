"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose_graph(edges, deadline):
    """Decompose edges into cycles and single edges efficiently.
    
    Structural change: instead of DFS per start vertex (exponential in worst case),
    use a systematic edge-pairing approach to find cycles. We look for pairs of
    edges sharing a vertex that can form part of a cycle, then extend greedily.
    This is more predictable and avoids the DFS explosion.
    """
    if not edges:
        return []
    
    edges_set = set(tuple(sorted(e)) for e in edges)
    adj = defaultdict(set)
    for u, v in edges_set:
        adj[u].add(v)
        adj[v].add(u)
    
    pieces = []
    remaining = set(edges_set)
    
    # Phase 1: Find cycles using a greedy extension strategy
    # Strategy: pick an edge, try to extend it into a cycle by finding
    # paths between its endpoints that don't use that edge.
    
    while remaining and time.time() < deadline:
        # Find vertices with degree >= 2
        candidates = [v for v in adj if len(adj[v]) >= 2]
        if not candidates:
            break
        
        # Sort by degree descending
        candidates.sort(key=lambda v: -len(adj[v]))
        
        best_cycle = None
        best_len = 0
        
        # Try to find cycles by looking at pairs of neighbors
        # For each vertex, look at pairs of its neighbors and check
        # if there's a path between them not using the vertex
        for start in candidates[:15]:
            if len(adj[start]) < 2:
                continue
            
            neighbors = list(adj[start])
            
            # For each pair of neighbors, try to find a path between them
            # that doesn't go through start
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    n1, n2 = neighbors[i], neighbors[j]
                    
                    # BFS from n1 to n2, avoiding start
                    # This gives us a path n1 -> ... -> n2
                    # Combined with edges start-n1 and start-n2, we get a cycle
                    path = _bfs_path(n1, n2, adj, remaining, exclude=start, max_depth=8)
                    if path and len(path) >= 2:
                        # Cycle is: start, n1, path[1], ..., path[-1], n2, start
                        cycle = [start] + [n1] + list(path[1:]) + [n2]
                        # Verify all edges are in remaining
                        valid = True
                        cycle_edges = set()
                        for k in range(len(cycle)):
                            a, b = cycle[k], cycle[(k + 1) % len(cycle)]
                            e = tuple(sorted((a, b)))
                            if e not in remaining:
                                valid = False
                                break
                            cycle_edges.add(e)
                        
                        if valid and len(cycle) > best_len:
                            best_len = len(cycle)
                            best_cycle = cycle
                        
                        # If we found a reasonably long cycle, stop
                        if best_len >= 6:
                            break
                if best_len >= 6:
                    break
            
            if best_len >= 6:
                break
        
        if best_cycle and best_len >= 3:
            pieces.append(best_cycle)
            # Remove cycle edges from remaining and adj
            for k in range(len(best_cycle)):
                a, b = best_cycle[k], best_cycle[(k + 1) % len(best_cycle)]
                e = tuple(sorted((a, b)))
                remaining.discard(e)
                adj[a].discard(b)
                adj[b].discard(a)
        else:
            # No more cycles can be found
            break
    
    # Phase 2: Add remaining edges as single-edge pieces
    for e in sorted(remaining):
        pieces.append([e[0], e[1]])
    
    return pieces

def _bfs_path(src, dst, adj, remaining, exclude=None, max_depth=8):
    """Find a path from src to dst using BFS, avoiding the exclude vertex.
    Only uses edges that are in remaining."""
    if src == dst:
        return None
    
    # BFS
    from collections import deque
    visited = {src}
    if exclude is not None:
        visited.add(exclude)
    
    queue = deque()
    queue.append((src, [src]))
    
    while queue:
        node, path = queue.popleft()
        if len(path) - 1 > max_depth:
            continue
        
        for nb in adj[node]:
            if nb == exclude:
                continue
            if nb in visited:
                continue
            # Check if edge (node, nb) is in remaining
            e = tuple(sorted((node, nb)))
            if e not in remaining:
                continue
            
            new_path = path + [nb]
            if nb == dst:
                return new_path
            
            visited.add(nb)
            queue.append((nb, new_path))
    
    return None

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    # Evaluator kills at 150s, leave 10s margin
    deadline = time.time() + 140.0
    
    for i, spec in enumerate(specs):
        # Allocate time: ~45s per graph, with decreasing budget
        remaining_graphs = len(specs) - i
        graph_deadline = deadline - (remaining_graphs - 1) * 5.0
        
        edges = _instance(*spec)
        pieces = _decompose_graph(edges, graph_deadline)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
