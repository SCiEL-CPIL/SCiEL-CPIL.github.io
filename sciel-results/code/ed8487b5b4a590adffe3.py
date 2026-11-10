"""Decompose each instance graph into edge-disjoint cycles and edges, minimizing pieces."""
import time
import random
from collections import defaultdict, deque
import numpy as np

def construct():
    deadline = time.time() + 140

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_2_core(edge_set):
        if not edge_set:
            return set()
        deg = defaultdict(int)
        adj = defaultdict(set)
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        queue = deque([v for v in deg if deg[v] < 2])
        in_core = set(deg.keys())
        while queue:
            v = queue.popleft()
            if v not in in_core:
                continue
            in_core.discard(v)
            for nb in adj[v]:
                if nb in in_core:
                    deg[nb] -= 1
                    if deg[nb] < 2:
                        queue.append(nb)
        core_edges = set()
        for u, v in edge_set:
            if u in in_core and v in in_core:
                core_edges.add(tuple(sorted((u, v))))
        return core_edges

    def max_flow_cycles(rem_edges, n_vertices):
        """
        Find maximum set of edge-disjoint cycles using max flow.
        
        Network:
        - For each vertex v: node v_in (id = v) and v_out (id = n + v)
        - Arc v_in -> v_out with capacity 1 (vertex capacity)
        - For each undirected edge (u,v): arcs u_out -> v_in (cap 1) and v_out -> u_in (cap 1)
        - Super source S connected to all v_in with cap 1
        - Super sink T connected from all v_out with cap 1
        
        A unit of flow S -> u_in -> u_out -> v_in -> v_out -> T represents using edge (u,v).
        The flow decomposes into paths S->...->T, but since each vertex has in-cap 1 and out-cap 1,
        the flow in the original graph forms a collection of vertex-disjoint paths and cycles.
        
        Wait - this doesn't directly give cycles. Let me think again.
        
        Actually, for finding edge-disjoint cycles, I need a different approach.
        
        Alternative: Use the fact that in the 2-core, we can find cycles by looking for
        directed cycles in a flow network. 
        
        Better approach for edge-disjoint cycle packing:
        - Each edge has capacity 1
        - We want to find a maximum collection of edge-disjoint cycles
        - This is equivalent to finding a maximum flow in a network where:
          * Each vertex v is split into v_in and v_out with arc v_in->v_out cap 1
          * Each undirected edge (u,v) gives arcs u_out->v_in and v_out->u_in with cap 1
          * Source S connects to all v_in with cap 1
          * All v_out connect to sink T with cap 1
        
        The max flow value equals the maximum number of edge-disjoint paths from S to T,
        but that's not what we want.
        
        Let me use a different formulation. To find edge-disjoint cycles:
        - The problem of packing edge-disjoint cycles is NP-hard in general.
        - But for sparse graphs, we can use a heuristic: repeatedly find cycles.
        
        Actually, let me use a smarter greedy: find the cycle that removes the most "useful" edges.
        
        New structural approach: Use a randomized local search / simulated annealing on the
        decomposition to minimize piece count.
        
        Let me instead focus on a better cycle extraction:
        1. Find 2-core
        2. In the 2-core, use a modified DFS that tries to find the longest possible cycle
           by backtracking when stuck
        3. Use multiple random restarts with different orderings
        """
        if not rem_edges:
            return []
        
        adj = defaultdict(set)
        for u, v in rem_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        vertices = set()
        for u, v in rem_edges:
            vertices.add(u)
            vertices.add(v)
        
        if len(vertices) < 3:
            return []
        
        # Try to find many edge-disjoint cycles
        cycles = []
        used_edges = set()
        
        # Sort vertices by degree descending for better cycle finding
        deg = {v: len(adj[v]) for v in vertices}
        sorted_verts = sorted(vertices, key=lambda v: deg[v], reverse=True)
        
        max_attempts = 200
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            if time.time() > deadline - 3:
                break
            
            # Find a cycle not using used_edges
            available = set()
            for u, v in rem_edges:
                e = tuple(sorted((u, v)))
                if e not in used_edges:
                    available.add(e)
            
            if not available:
                break
            
            # BFS-based cycle finding with preference for long cycles
            # Try from high-degree vertices
            found = False
            for start in sorted_verts:
                if len(adj[start]) < 2:
                    continue
                # Check if start has at least 2 available edges
                avail_neighbors = [nb for nb in adj[start] if tuple(sorted((start, nb))) in available]
                if len(avail_neighbors) < 2:
                    continue
                
                # Try to find a cycle through start using DFS with backtracking
                cycle = _find_long_cycle_dfs(start, adj, available, n_vertices)
                if cycle and len(cycle) >= 3:
                    cycles.append(cycle)
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        used_edges.add(tuple(sorted((u, v))))
                    found = True
                    break
            
            if not found:
                break
        
        return cycles

    def _find_long_cycle_dfs(start, adj, available, n_vertices, max_depth=50):
        """Find a long cycle starting from start using DFS with backtracking."""
        best_cycle = None
        best_len = 0
        
        def dfs(node, path, visited):
            nonlocal best_cycle, best_len
            if len(path) > max_depth:
                return
            if len(path) >= 3 and start in adj[node]:
                # Check if edge (node, start) is available
                e = tuple(sorted((node, start)))
                if e in available and all(
                    tuple(sorted((path[i], path[(i+1) % len(path)]))) in available
                    for i in range(len(path))
                ):
                    if len(path) > best_len:
                        best_len = len(path)
                        best_cycle = path[:]
            
            for nb in adj[node]:
                e = tuple(sorted((node, nb)))
                if e not in available:
                    continue
                if nb == start and len(path) >= 3:
                    # Check all edges in path are available
                    valid = True
                    for i in range(len(path)):
                        pe = tuple(sorted((path[i], path[(i+1) % len(path)])))
                        if pe not in available:
                            valid = False
                            break
                    if valid and len(path) > best_len:
                        best_len = len(path)
                        best_cycle = path[:]
                    continue
                if nb in visited:
                    continue
                visited.add(nb)
                path.append(nb)
                dfs(nb, path, visited)
                path.pop()
                visited.discard(nb)
                if best_cycle and len(best_cycle) >= 20:
                    return  # Good enough
        
        dfs(start, [start], {start})
        return best_cycle

    def find_cycle_bfs(rem_edges, used_edges, n_vertices):
        """Find any cycle using BFS, avoiding used edges."""
        available = set()
        for u, v in rem_edges:
            e = tuple(sorted((u, v)))
            if e not in used_edges:
                available.add(e)
        
        if not available:
            return None
        
        adj = defaultdict(set)
        for u, v in available:
            adj[u].add(v)
            adj[v].add(u)
        
        vertices = set()
        for u, v in available:
            vertices.add(u)
            vertices.add(v)
        
        for start in vertices:
            if len(adj[start]) < 2:
                continue
            # BFS to find a cycle
            visited = {start: None}  # node -> parent
            parent_edge = {}  # node -> edge used to reach it
            queue = deque([(start, None)])
            
            while queue:
                curr, prev = queue.popleft()
                for nb in adj[curr]:
                    e = tuple(sorted((curr, nb)))
                    if nb == start and curr != prev:
                        # Found a cycle
                        # Reconstruct path from curr back to start
                        path = [start]
                        node = curr
                        while node != start:
                            path.append(node)
                            node = parent_edge[node][0]
                        path.reverse()
                        if len(path) >= 3:
                            return path
                    elif nb not in visited:
                        visited[nb] = curr
                        parent_edge[nb] = (curr, e)
                        queue.append((nb, curr))
        
        return None

    def decompose(edges, n, time_budget):
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        best_pieces = None
        best_count = len(edge_set) + 1
        
        def decompose_once(rem_set):
            rem = set(rem_set)
            pieces = []
            
            core_edges = get_2_core(rem)
            forest_edges = rem - core_edges
            
            for e in forest_edges:
                pieces.append(list(e))
            
            rem = core_edges
            if not rem:
                return pieces
            
            # Try multiple strategies for cycle extraction
            strategy = random.randint(0, 2)
            
            if strategy == 0:
                # Use DFS-based long cycle finding
                used_edges = set()
                while rem:
                    if time.time() > deadline - 3:
                        break
                    available = set()
                    for u, v in rem:
                        e = tuple(sorted((u, v)))
                        if e not in used_edges:
                            available.add(e)
                    if not available:
                        break
                    
                    adj = defaultdict(set)
                    for u, v in available:
                        adj[u].add(v)
                        adj[v].add(u)
                    
                    vertices = set()
                    for u, v in available:
                        vertices.add(u)
                        vertices.add(v)
                    
                    deg = {v: len(adj[v]) for v in vertices}
                    sorted_verts = sorted(vertices, key=lambda v: deg[v], reverse=True)
                    
                    cycle = None
                    for start in sorted_verts[:15]:
                        if len(adj[start]) < 2:
                            continue
                        avail_neighbors = [nb for nb in adj[start] if tuple(sorted((start, nb))) in available]
                        if len(avail_neighbors) < 2:
                            continue
                        c = _find_long_cycle_dfs(start, adj, available, n, max_depth=40)
                        if c and len(c) >= 3:
                            cycle = c
                            break
                    
                    if not cycle:
                        # Fall back to BFS
                        cycle = find_cycle_bfs(rem, used_edges, n)
                    
                    if not cycle or len(cycle) < 3:
                        e = tuple(next(iter(rem)))
                        rem.discard(e)
                        pieces.append(list(e))
                    else:
                        for i in range(len(cycle)):
                            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                            e = tuple(sorted((u, v)))
                            rem.discard(e)
                        pieces.append(cycle[:])
                        
            else:
                # Greedy: always find a cycle, prefer longer ones via multiple attempts
                while rem:
                    if time.time() > deadline - 3:
                        break
                    
                    # Try to find a long cycle with multiple random attempts
                    best_c = None
                    for _ in range(5):
                        c = find_cycle_bfs(rem, set(), n)
                        if c and len(c) >= 3:
                            if not best_c or len(c) > len(best_c):
                                best_c = c
                        if best_c and len(best_c) >= 15:
                            break
                    
                    if not best_c or len(best_c) < 3:
                        e = tuple(next(iter(rem)))
                        rem.discard(e)
                        pieces.append(list(e))
                    else:
                        for i in range(len(best_c)):
                            u, v = best_c[i], best_c[(i + 1) % len(best_c)]
                            e = tuple(sorted((u, v)))
                            rem.discard(e)
                        pieces.append(best_c[:])
            
            return pieces
        
        # Multiple restarts
        max_restarts = 60
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
            random.seed(r * 12345 + 7)
            pieces = decompose_once(edge_set)
            cnt = len(pieces)
            if cnt < best_count:
                best_count = cnt
                best_pieces = [p[:] for p in pieces]
        
        if best_pieces is None:
            best_pieces = [list(e) for e in edge_set]
        return best_pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    for spec in specs:
        if time.time() > deadline - 3:
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n, deadline - time.time())
        results.append(pieces)

    return results
