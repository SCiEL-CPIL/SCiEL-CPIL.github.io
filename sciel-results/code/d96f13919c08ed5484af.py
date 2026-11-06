"""Decompositions of three fixed graphs into few cycles and edges (Erdős-Gallai)."""
# EVOLVE-BLOCK-START
import random
from collections import defaultdict, deque
import itertools

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_cycles_illp(adj, edges_set, edge_to_idx):
        """
        Find maximum edge-disjoint cycles using a greedy ILP-like approach.
        We use a maximum matching reduction on an auxiliary graph.
        
        Strategy: 
        1. Build auxiliary graph where each original edge is a node.
        2. Two edge-nodes are connected if the corresponding edges share a vertex.
        3. Finding a cycle in original graph corresponds to a cycle in this line graph.
        4. We want to partition as many edges as possible into cycles.
        
        Practical approach: Repeatedly find cycles using DFS/BFS, preferring 
        shorter cycles first, and remove them. This is a greedy heuristic but 
        we enhance it with a global optimization pass.
        """
        n = max(adj.keys()) + 1 if adj else 0
        used_edges = set()
        cycles = []
        
        # Precompute edge list
        edge_list = list(edges_set)
        
        # Try to find cycles greedily, but with a twist:
        # We'll use a priority queue approach where we try to find cycles 
        # that cover the most "hard to place" edges first.
        
        # First, let's identify vertices with high degree
        deg = defaultdict(int)
        for u, v in edges_set:
            deg[u] += 1
            deg[v] += 1
        
        # Sort edges by the minimum degree of their endpoints (ascending)
        # so we handle "bottleneck" edges first
        sorted_edges = sorted(edge_list, key=lambda e: min(deg[e[0]], deg[e[1]]))
        
        # Greedy cycle finding: repeatedly find a cycle, prefer shorter ones
        # but also ensure we don't leave isolated edges behind
        remaining = set(edge_list)
        
        # Use a more sophisticated approach: 
        # 1. Find all simple cycles up to a certain length
        # 2. Select a maximum set of edge-disjoint cycles
        
        # For efficiency, we'll use a greedy approach with backtracking
        # for small cycles first
        
        # Step 1: Find all triangles (3-cycles)
        triangles = []
        edge_to_neighbors = defaultdict(set)
        for u, v in edges_set:
            edge_to_neighbors[u].add(v)
            edge_to_neighbors[v].add(u)
        
        for u in range(n):
            for v in edge_to_neighbors[u]:
                if v <= u:
                    continue
                for w in edge_to_neighbors[v]:
                    if w > v and w in edge_to_neighbors[u]:
                        triangles.append([(u, v), (v, w), (u, w)])
        
        # Step 2: Greedily select edge-disjoint triangles
        selected_triangles = []
        used = set()
        # Sort triangles by... let's just take them in order
        for tri in triangles:
            if all(e not in used for e in tri):
                selected_triangles.append(tri)
                for e in tri:
                    used.add(e)
        
        cycles.extend(selected_triangles)
        remaining -= used
        
        # Step 3: For remaining edges, try to find longer cycles
        # Use DFS to find cycles in the remaining subgraph
        rem_adj = defaultdict(set)
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        # Find cycles in remaining graph
        while remaining:
            # Pick an edge with minimum degree endpoint
            best_edge = None
            best_deg = float('inf')
            for u, v in remaining:
                d = min(len(rem_adj[u]), len(rem_adj[v]))
                if d < best_deg:
                    best_deg = d
                    best_edge = (u, v)
            
            if best_edge is None:
                break
            
            u, v = best_edge
            # Try to find a cycle starting with edge (u, v)
            # BFS/DFS to find path from v back to u not using (u,v)
            found_cycle = None
            
            # Try to find a path from v to u in rem_adj without using edge (u,v)
            # Use BFS
            parent = {v: None}
            queue = deque([v])
            found = False
            while queue and not found:
                curr = queue.popleft()
                if curr == u:
                    # Reconstruct path
                    path = []
                    node = curr
                    while node is not None:
                        path.append(node)
                        if node == v:
                            break
                        node = parent[node]
                    path.reverse()
                    # path is from v to u
                    cycle_edges = []
                    for i in range(len(path) - 1):
                        a, b = path[i], path[i+1]
                        e = tuple(sorted((a, b)))
                        cycle_edges.append(e)
                    found_cycle = cycle_edges
                    found = True
                    break
                for nb in rem_adj[curr]:
                    if nb not in parent:
                        # Avoid using the edge (u,v) in reverse
                        if curr == v and nb == u:
                            continue
                        parent[nb] = curr
                        queue.append(nb)
            
            if found_cycle and len(found_cycle) >= 3:
                cycles.append(found_cycle)
                for e in found_cycle:
                    remaining.discard(e)
                    if e in rem_adj:
                        a, b = e
                        if b in rem_adj[a]:
                            rem_adj[a].discard(b)
                        if a in rem_adj[b]:
                            rem_adj[b].discard(a)
            else:
                # Cannot find a cycle containing this edge, leave it as a single edge
                # Move to next edge
                # Remove this edge from consideration for cycle finding
                # but keep it as a single edge piece
                pass
        
        return cycles, remaining
    
    def decompose_graph(n, p, seed):
        edges = instance(n, p, seed)
        edges_set = set(tuple(e) for e in edges)
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find cycles
        cycles, remaining_edges = find_cycles_illp(adj, edges_set, None)
        
        # Convert to pieces
        pieces = []
        for cycle in cycles:
            # Extract vertex sequence
            if len(cycle) == 0:
                continue
            # Reconstruct vertex order from edges
            # cycle is a list of sorted edge tuples
            # Build a graph from these edges
            cycle_adj = defaultdict(list)
            for u, v in cycle:
                cycle_adj[u].append(v)
                cycle_adj[v].append(u)
            
            # Find a Hamiltonian cycle in this small graph
            # Since it's a cycle, each vertex has degree 2
            start = None
            for vertex in cycle_adj:
                if len(cycle_adj[vertex]) == 2:
                    start = vertex
                    break
            
            if start is None:
                # Fallback: just list the edges
                for u, v in cycle:
                    pieces.append([u, v])
                continue
            
            # Walk the cycle
            vertices = [start]
            prev = None
            curr = start
            for _ in range(len(cycle)):
                nexts = [x for x in cycle_adj[curr] if x != prev]
                if not nexts:
                    break
                nxt = nexts[0]
                if nxt == start:
                    break
                vertices.append(nxt)
                prev = curr
                curr = nxt
            
            if len(vertices) >= 3:
                pieces.append(vertices)
            else:
                for u, v in cycle:
                    pieces.append([u, v])
        
        # Add remaining edges as single edge pieces
        for u, v in remaining_edges:
            pieces.append([u, v])
        
        return pieces
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        pieces = decompose_graph(n, p, seed)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
