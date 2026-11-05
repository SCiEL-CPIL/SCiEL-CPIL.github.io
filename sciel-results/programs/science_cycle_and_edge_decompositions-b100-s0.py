import time
import random
from collections import defaultdict, deque
import heapq

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_long_cycle_greedy(adj, n, rng, max_tries_per_start=8):
    """Find a long cycle using randomized greedy DFS."""
    best_cycle = None
    best_len = 0
    
    # Prioritize vertices with higher degree for better cycles
    starts = [v for v in range(n) if adj[v]]
    starts.sort(key=lambda v: len(adj[v]), reverse=True)
    
    for start in starts:
        if not adj[start]:
            continue
        
        for _ in range(max_tries_per_start):
            path = [start]
            visited = {start}
            current = start
            
            while True:
                neighbors = list(adj[current])
                unvisited = [v for v in neighbors if v not in visited]
                
                if not unvisited:
                    if len(path) >= 3 and start in adj[current]:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                    break
                else:
                    # Prefer unvisited neighbors with lower degree (leaf-like)
                    # to keep options open for longer paths
                    if len(unvisited) > 1:
                        unvisited.sort(key=lambda v: len(adj[v]))
                        # Random choice among lower-degree options
                        idx = rng.randrange(min(3, len(unvisited)))
                        next_node = unvisited[idx]
                    else:
                        next_node = unvisited[0]
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node
        
        # Early stop if we found a very long cycle
        if best_len >= n * 0.6:
            break
    
    return best_cycle

def _polish_pieces(pieces, n, deadline):
    """Try to merge adjacent single-edge pieces into cycles via local search."""
    # Build adjacency from pieces
    edge_to_piece = {}
    for i, piece in enumerate(pieces):
        if len(piece) == 2:
            u, v = piece
            edge_to_piece[(u, v)] = i
            edge_to_piece[(v, u)] = i
    
    # Build graph where nodes are edge-pieces, edges connect if they share a vertex
    # and the union could form a cycle
    edge_pieces = [i for i, p in enumerate(pieces) if len(p) == 2]
    
    if len(edge_pieces) < 3:
        return pieces
    
    # Try to find cycles in the "edge adjacency" graph
    # Two edge-pieces are adjacent if they share a vertex
    adj_edges = defaultdict(set)
    for i in edge_pieces:
        u, v = pieces[i]
        for j in edge_pieces:
            if i == j:
                continue
            a, b = pieces[j]
            # Check if they share a vertex
            if v == a or v == b or u == a or u == b:
                adj_edges[i].add(j)
    
    # Try to find cycles of length 3+ in this adjacency graph
    # that correspond to valid edge-disjoint cycles in the original graph
    visited_edges = set()
    new_pieces = list(pieces)
    modified = False
    
    # Simple approach: try to find triangles in the edge-adjacency graph
    # where three edges form a cycle (u-v, v-w, w-u)
    for i in edge_pieces:
        if i in visited_edges:
            continue
        u1, v1 = pieces[i]
        
        for j in adj_edges.get(i, []):
            if j in visited_edges or j <= i:
                continue
            u2, v2 = pieces[j]
            
            # Find shared vertex
            if v1 == u2 or v1 == v2:
                shared = v1
                other_j = u2 if v1 == u2 else v2
            elif u1 == u2 or u1 == v2:
                shared = u1
                other_j = u2 if u1 == u2 else v2
            else:
                continue
            
            # Look for a third edge connecting other_j to either u1 or v1
            # such that it doesn't use the shared vertex
            target1 = u1 if shared == v1 else v1
            target2 = other_j
            
            for k in adj_edges.get(j, []):
                if k in visited_edges or k <= j or k == i:
                    continue
                u3, v3 = pieces[k]
                
                # Check if this edge connects target2 to target1
                if (u3 == target2 and v3 == target1) or (u3 == target1 and v3 == target2):
                    # Found a triangle: i, j, k form a cycle
                    # The cycle is: shared -> target1 -> target2 -> shared
                    # Wait, let me think about this more carefully
                    # Edge i: u1-v1, Edge j: shared-other_j, Edge k: target1-target2
                    # We need them to form a cycle
                    # Let's just verify: the three edges should form a cycle
                    # Vertices involved: {u1,v1}, {u2,v2}, {u3,v3}
                    # They should share exactly 2 vertices total forming a cycle
                    
                    all_verts = set()
                    for piece in [pieces[i], pieces[j], pieces[k]]:
                        all_verts.update(piece)
                    
                    # For a 3-cycle, we need exactly 3 distinct vertices
                    if len(all_verts) == 3:
                        # Find the cycle order
                        verts = list(all_verts)
                        # Build the cycle by following edges
                        cycle = [verts[0]]
                        current = verts[0]
                        for _ in range(2):
                            for v in verts[1:]:
                                if v != current and (v in adj_edges.get(i, []) or True):
                                    # Check if there's an edge from current to v in our pieces
                                    found = False
                                    for idx in [i, j, k]:
                                        a, b = pieces[idx]
                                        if (a == current and b == v) or (a == v and b == current):
                                            found = True
                                            break
                                    if found:
                                        cycle.append(v)
                                        current = v
                                        break
                        
                        if len(cycle) == 3:
                            # Replace the three edge pieces with one cycle piece
                            visited_edges.update([i, j, k])
                            new_pieces[i] = cycle
                            new_pieces[j] = None
                            new_pieces[k] = None
                            modified = True
                            break
                if modified:
                    break
            if modified:
                break
        if modified:
            break
    
    if modified:
        return [p for p in new_pieces if p is not None]
    
    return pieces

def _decompose_once(edges, n, rng, deadline):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    pieces = []
    remaining = len(edges)
    
    while remaining > 0:
        if time.time() > deadline:
            break
        
        # Find a long cycle using greedy randomized strategy
        best_cycle = _find_long_cycle_greedy(adj, n, rng, max_tries_per_start=8)
        
        if best_cycle:
            m = len(best_cycle)
            for i in range(m):
                a = best_cycle[i]
                b = best_cycle[(i + 1) % m]
                adj[a].discard(b)
                adj[b].discard(a)
            remaining -= m
            pieces.append(best_cycle)
        else:
            edge_removed = False
            for u in range(n):
                for v in list(adj[u]):
                    if u < v:
                        adj[u].discard(v)
                        adj[v].discard(u)
                        remaining -= 1
                        pieces.append([u, v])
                        edge_removed = True
                        break
                if edge_removed:
                    break
            if not edge_removed:
                break
    
    return pieces

def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')
    
    base_seed = 42
    attempt = 0
    
    while time.time() < deadline - 1.0:
        rng = random.Random(base_seed + attempt)
        pieces = _decompose_once(edges, n, rng, deadline - 0.5)
        cnt = len(pieces)
        
        if cnt < best_count:
            best_count = cnt
            best = pieces
            # Try polish on the best solution
            if time.time() < deadline - 2.0:
                polished = _polish_pieces(best, n, deadline - 1.0)
                if len(polished) < best_count:
                    best_count = len(polished)
                    best = polished
        
        attempt += 1
        # Increase tries per start as we go (more exploration)
    
    return best

def construct():
    start_time = time.time()
    total_budget = 135.0  # Safety margin before 150s kill
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    # Allocate time proportionally to graph size (more edges = more work)
    sizes = []
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        sizes.append(len(edges))
    
    total_edges = sum(sizes)
    
    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        
        # Give more time to larger graphs
        frac = sizes[i] / total_edges
        remaining_time = total_budget - (time.time() - start_time)
        # Reserve 2s for remaining graphs
        graphs_left = len(specs) - i
        reserved = 2.0 * graphs_left
        available = max(remaining_time - reserved, 5.0)
        
        graph_deadline = time.time() + available * frac * 1.5  # 1.5x to allow slack
        # Cap at absolute deadline
        graph_deadline = min(graph_deadline, start_time + total_budget - 3.0)
        
        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)
        
        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break
    
    return results
