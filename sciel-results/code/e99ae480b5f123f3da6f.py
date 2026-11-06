"""Decompose three fixed graphs into cycles and edges, minimizing piece count."""
import time
import random
from collections import defaultdict, deque

_DEADLINE_SECONDS = 110.0


def _instance(n, p, seed):
    """Replicate the verifier's instance generator exactly."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _eulerian_decompose(edges, n, time_limit):
    """
    Decompose a graph into cycles and edges using an Eulerian circuit approach.
    This tends to produce longer cycles, reducing the total number of pieces.
    
    Strategy:
    1. Find the largest set of edges that can be decomposed into cycles (Eulerian subgraph)
    2. Use Hierholzer's algorithm to find an Eulerian circuit
    3. Split the circuit into simple cycles at repeated vertices
    4. Remaining edges are added as single-edge pieces
    """
    start_time = time.time()
    
    if not edges:
        return []
    
    # Build adjacency with edge tracking
    adj = defaultdict(list)  # vertex -> list of (neighbor, edge_id)
    edge_map = {}  # edge_id -> (u, v)
    
    for eid, (u, v) in enumerate(edges):
        adj[u].append((v, eid))
        adj[v].append((u, eid))
        edge_map[eid] = (u, v)
    
    # Try to find a large Eulerian subgraph
    # An Eulerian subgraph requires all vertices to have even degree
    # We can find this by:
    # 1. Start with all edges
    # 2. While there's a vertex with odd degree, remove an edge incident to it
    #    that also has another odd-degree endpoint (or just remove one edge)
    
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    
    # Find edges to remove to make all degrees even
    # This is equivalent to finding a T-join where T is the set of odd-degree vertices
    # For simplicity, we'll use a greedy approach
    
    used_edges = set(range(len(edges)))
    odd_vertices = [v for v in range(n) if deg[v] % 2 == 1]
    
    # Try to pair up odd vertices and find paths between them to remove
    # This is complex, so let's use a simpler heuristic:
    # Greedily remove edges to make degrees even
    
    removed_edges = set()
    current_deg = list(deg)
    
    # Sort edges by some criterion - prefer edges between high-degree vertices
    # to preserve structure
    edge_list = list(used_edges)
    
    # Find a maximal even subgraph
    # Use a simple approach: repeatedly find a vertex with odd degree and
    # remove an edge from it to another odd-degree vertex if possible
    
    odd_set = set(odd_vertices)
    
    # Build adjacency for remaining edges
    rem_adj = defaultdict(set)
    for eid in used_edges:
        u, v = edge_map[eid]
        rem_adj[u].add((v, eid))
        rem_adj[v].add((u, eid))
    
    # Try to find paths between pairs of odd vertices to remove
    # This would give us an Eulerian subgraph
    
    # Simpler approach: just use all edges and handle odd degrees by
    # finding cycles in the remaining structure
    
    # Let's try a different strategy:
    # 1. Find all cycles using a modified Hierholzer approach
    # 2. Track which edges are used
    
    pieces = []
    remaining_adj = defaultdict(set)
    for eid in used_edges:
        u, v = edge_map[eid]
        remaining_adj[u].add(v)
        remaining_adj[v].add(u)
    
    remaining_deg = [len(remaining_adj[v]) for v in range(n)]
    
    # Repeatedly find a cycle and remove it
    # Use a heuristic to find good cycles
    
    def find_cycle(start, adj, deg, time_left):
        """Find a cycle starting from start vertex."""
        if time.time() - start_time > time_left:
            return None
        
        if deg[start] < 2:
            return None
        
        # DFS to find a cycle
        stack = [(start, [start])]
        visited = {start}
        
        while stack:
            node, path = stack.pop()
            
            if node == start and len(path) >= 3:
                return path
            
            neighbors = list(adj[node])
            # Sort neighbors by degree (prefer high degree)
            neighbors.sort(key=lambda x: -deg[x])
            
            for nb in neighbors:
                if nb == start and len(path) >= 3:
                    return path
                if nb not in visited:
                    visited.add(nb)
                    stack.append((nb, path + [nb]))
        
        return None
    
    # Try to find long cycles by starting from high-degree vertices
    while True:
        if time.time() - start_time > time_limit:
            break
        
        remaining_edges = sum(len(remaining_adj[u]) for u in range(n)) // 2
        if remaining_edges == 0:
            break
        
        # Find a cycle
        cycle = None
        
        # Try vertices with highest degree first
        candidates = [v for v in range(n) if remaining_deg[v] >= 2]
        candidates.sort(key=lambda v: -remaining_deg[v])
        
        for start in candidates[:10]:  # Try top 10 candidates
            if remaining_deg[start] < 2:
                continue
            
            cycle = find_cycle(start, remaining_adj, remaining_deg, time_limit * 0.1)
            if cycle:
                break
        
        if cycle:
            pieces.append(cycle)
            # Remove cycle edges
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                remaining_adj[u].discard(v)
                remaining_adj[v].discard(u)
                remaining_deg[u] -= 1
                remaining_deg[v] -= 1
        else:
            # No cycle found, take an edge
            edge_found = None
            for u in range(n):
                if remaining_adj[u]:
                    v = next(iter(remaining_adj[u]))
                    edge_found = (u, v)
                    break
            
            if edge_found is None:
                break
            
            u, v = edge_found
            pieces.append([u, v])
            remaining_adj[u].discard(v)
            remaining_adj[v].discard(u)
            remaining_deg[u] -= 1
            remaining_deg[v] -= 1
    
    # Add any remaining edges
    for u in range(n):
        for v in remaining_adj[u]:
            if u < v:
                pieces.append([u, v])
    
    return pieces


def _improve_decomposition(pieces, edges_set, n, time_limit):
    """
    Try to improve the decomposition by merging edge pieces into cycles.
    Focus on finding triangles and 4-cycles among edge pieces.
    """
    start_time = time.time()
    
    if not pieces:
        return pieces
    
    # Separate edge pieces and cycle pieces
    edge_pieces = []  # (index, u, v)
    cycle_pieces = []
    
    for i, piece in enumerate(pieces):
        if len(piece) == 2:
            u, v = piece
            edge_pieces.append((i, u, v))
        else:
            cycle_pieces.append(piece)
    
    if not edge_pieces:
        return pieces
    
    # Build a graph of edge pieces
    # Node = edge piece, edge = shared vertex
    # Try to find cycles in this meta-graph
    
    # For efficiency, focus on finding triangles first
    # A triangle in the original graph corresponds to 3 edge pieces
    
    # Build vertex -> list of edge pieces
    vert_to_edges = defaultdict(list)
    for idx, u, v in edge_pieces:
        vert_to_edges[u].append((idx, v))
        vert_to_edges[v].append((idx, u))
    
    # Track which edge pieces are used
    used = [False] * len(pieces)
    new_cycles = []
    
    changed = True
    while changed:
        if time.time() - start_time > time_limit:
            break
        changed = False
        
        # Find triangles among unused edge pieces
        for v in list(vert_to_edges.keys()):
            edges_at_v = [(idx, u) for idx, u in vert_to_edges[v] if not used[idx]]
            
            if len(edges_at_v) < 2:
                continue
            
            for i in range(len(edges_at_v)):
                if changed:
                    break
                idx1, u1 = edges_at_v[i]
                
                for j in range(i + 1, len(edges_at_v)):
                    if changed:
                        break
                    idx2, u2 = edges_at_v[j]
                    
                    # Check if (u1, u2) is an unused edge piece
                    edge_key = (min(u1, u2), max(u1, u2))
                    
                    # Find if this edge exists
                    for idx3, a, b in edge_pieces:
                        if (min(a, b), max(a, b)) == edge_key and not used[idx3]:
                            # Found a triangle
                            new_cycles.append([v, u1, u2])
                            used[idx1] = True
                            used[idx2] = True
                            used[idx3] = True
                            changed = True
                            break
    
    # Build final piece list
    final_pieces = []
    
    # Add cycle pieces
    final_pieces.extend(cycle_pieces)
    
    # Add new cycles
    final_pieces.extend(new_cycles)
    
    # Add unused edge pieces
    for idx, u, v in edge_pieces:
        if not used[idx]:
            final_pieces.append([u, v])
    
    return final_pieces


def _decompose_graph(edges, n, time_limit):
    """
    Main decomposition function combining multiple strategies.
    """
    start_time = time.time()
    
    if not edges:
        return []
    
    # Strategy 1: Eulerian-style decomposition
    pieces1 = _eulerian_decompose(edges, n, time_limit * 0.5)
    count1 = len(pieces1)
    
    # Strategy 2: Try to improve by merging
    edges_set = set((min(u, v), max(u, v)) for u, v in edges)
    pieces2 = _improve_decomposition(pieces1, edges_set, n, time_limit * 0.3)
    count2 = len(pieces2)
    
    # Choose the better one
    if count2 < count1:
        return pieces2
    else:
        return pieces1


def construct():
    """
    Decompose three fixed graphs into cycles and edges, minimizing piece count.
    Returns three lists of pieces.
    """
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []
    
    for n, p, seed in specs:
        elapsed = time.time() - start_time
        remaining = _DEADLINE_SECONDS - elapsed
        
        if remaining < 5:
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = _instance(n, p, seed)
        
        # Allocate time for this graph
        graph_time = remaining * 0.85
        
        # Try decomposition
        pieces = _decompose_graph(edges, n, graph_time)
        
        results.append(pieces)
    
    return results
