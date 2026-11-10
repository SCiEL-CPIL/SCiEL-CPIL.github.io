"""Rewritten program for Gallai's conjecture path decomposition."""
import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_simple_path(start, end, adj, remaining_edges):
        """Find a simple path from start to end using only edges in remaining_edges."""
        if start == end:
            return [start]
        
        # BFS
        visited = {start}
        parent = {start: None}
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v in visited:
                    continue
                edge = tuple(sorted((u, v)))
                if edge not in remaining_edges:
                    continue
                
                visited.add(v)
                parent[v] = u
                
                if v == end:
                    # Reconstruct path
                    path = []
                    curr = end
                    while curr is not None:
                        path.append(curr)
                        curr = parent[curr]
                    return path[::-1]
                
                queue.append(v)
        
        return None

    def decompose_eulerian(adj, remaining_edges, vertices):
        """
        Decompose an Eulerian graph (all degrees even) into edge-disjoint cycles.
        Returns a list of cycles (list of vertices).
        """
        if not remaining_edges:
            return []
        
        # Build adjacency list with edge references
        # We need to track which edges are used
        edges_list = list(remaining_edges)
        edge_idx = {e: i for i, e in enumerate(edges_list)}
        
        # Adjacency: u -> list of (v, edge_id)
        adj_euler = defaultdict(list)
        for e in edges_list:
            u, v = e
            adj_euler[u].append((v, e))
            adj_euler[v].append((u, e))
        
        used_edges = set()
        cycles = []
        
        # Start from any vertex with remaining edges
        for v in vertices:
            if adj_euler[v]:
                # Find an unused edge
                start_edge = None
                for nb, e in adj_euler[v]:
                    if e not in used_edges:
                        start_edge = e
                        break
                
                if start_edge is None:
                    continue
                
                # Hierholzer's algorithm variant to find a cycle
                # Since all degrees are even, we can always find a cycle
                # We'll do a DFS to find a cycle starting from v
                cycle = []
                curr = v
                cycle.append(curr)
                
                # We need to find a path that returns to v
                # This is tricky with simple DFS if we don't backtrack properly.
                # Standard Hierholzer finds the circuit.
                
                # Let's use a standard Hierholzer implementation
                # It returns the circuit as a list of vertices
                stack = [v]
                circuit = []
                used_in_circuit = set()
                
                while stack:
                    u = stack[-1]
                    # Find an unused edge
                    next_edge = None
                    for nb, e in adj_euler[u]:
                        if e not in used_edges and e not in used_in_circuit:
                            next_edge = e
                            break
                    
                    if next_edge is None:
                        circuit.append(stack.pop())
                    else:
                        used_in_circuit.add(next_edge)
                        u, v_next = next_edge
                        if u == v:
                            stack.append(v_next)
                        else:
                            stack.append(v_next)
                
                # circuit is reversed
                circuit.reverse()
                
                # Remove duplicate start/end if it's a cycle
                # Hierholzer returns v...v
                if len(circuit) > 1 and circuit[0] == circuit[-1]:
                    cycle = circuit[:-1]
                else:
                    cycle = circuit
                
                if cycle:
                    cycles.append(cycle)
                    
                    # Mark edges as used
                    for i in range(len(cycle) - 1):
                        u, v = cycle[i], cycle[i+1]
                        e = tuple(sorted((u, v)))
                        used_edges.add(e)
                    if len(cycle) > 1:
                        u, v = cycle[-1], cycle[0]
                        e = tuple(sorted((u, v)))
                        used_edges.add(e)
                        
        return cycles

    def process_graph(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        adj = defaultdict(set)
        deg = defaultdict(int)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        odd_vertices = [v for v in range(n) if deg[v] % 2 == 1]
        
        paths = []
        remaining_edges = set(edges)
        
        # 1. Pair up odd vertices
        # Simple greedy pairing: sort by vertex id or just pair sequentially
        # Better: find shortest paths between pairs to minimize edge usage?
        # For now, just pair them arbitrarily, then find paths.
        # To be safer, we can try to find paths that are short.
        
        # Let's try to pair them such that paths are found easily.
        # Just pair (0,1), (2,3), etc.
        odd_pairs = []
        for i in range(0, len(odd_vertices), 2):
            if i + 1 < len(odd_vertices):
                odd_pairs.append((odd_vertices[i], odd_vertices[i+1]))
        
        # Find paths for each pair
        for u, v in odd_pairs:
            path = find_simple_path(u, v, adj, remaining_edges)
            if path:
                paths.append(path)
                # Remove edges from remaining
                for i in range(len(path) - 1):
                    e = tuple(sorted((path[i], path[i+1])))
                    remaining_edges.discard(e)
            else:
                # Should not happen if graph is connected and u,v are in it
                # Fallback: add as separate edge paths?
                # This indicates a bug or disconnected graph
                pass
        
        # 2. Decompose remaining Eulerian graph
        # The remaining graph has all even degrees.
        # Find all vertices that still have edges
        active_vertices = set()
        for e in remaining_edges:
            active_vertices.add(e[0])
            active_vertices.add(e[1])
        
        if not active_vertices:
            return paths
        
        # Rebuild adj for remaining edges
        rem_adj = defaultdict(set)
        for u, v in remaining_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
            
        cycles = decompose_eulerian(rem_adj, remaining_edges, active_vertices)
        
        # 3. Convert cycles to 2 simple paths each
        for cycle in cycles:
            if len(cycle) < 2:
                continue
            # Cycle is [v1, ..., vk]
            # Path 1: v1 ... vk
            # Path 2: vk ... v1 (just the edge? No, vk-v1 is one edge)
            # If cycle is v1-v2-...-vk-v1
            # Path 1: [v1, ..., vk] covers (v1,v2)..(vk-1,vk)
            # Path 2: [vk, v1] covers (vk, v1)
            
            p1 = cycle[:]
            p2 = [cycle[-1], cycle[0]]
            
            paths.append(p1)
            paths.append(p2)
            
        return paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        if time.time() > deadline - 2:
            # Fallback to trivial if time is up
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        paths = process_graph(*spec)
        results.append(paths)
        
    return results
