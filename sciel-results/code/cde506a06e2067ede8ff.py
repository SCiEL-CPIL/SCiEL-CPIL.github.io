"""Decompose edges into minimum number of cycles and single edges."""
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def edge_key(u, v):
        return (min(u, v), max(u, v))

    def decompose(edges, time_limit):
        """
        Decompose edges into cycles and single edges, minimizing total pieces.
        Strategy:
        1. Use Eulerian decomposition on the whole graph to get edge-disjoint cycles.
        2. For remaining odd-degree structure, extract as many cycles as possible.
        3. Remaining edges become single-edge pieces.
        
        Key insight: In any graph, the minimum number of edge-disjoint cycles + edges
        to cover all edges is at most the number of edges, but cycles save pieces.
        A graph with all even degrees can be decomposed entirely into cycles.
        """
        remaining = set()
        for e in edges:
            remaining.add(edge_key(e[0], e[1]))
        
        if not remaining:
            return []
        
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        pieces = []
        
        def commit_cycle(cycle_vertices):
            """Remove cycle edges from remaining and record the piece."""
            k = len(cycle_vertices)
            for i in range(k):
                u = cycle_vertices[i]
                v = cycle_vertices[(i + 1) % k]
                ek = edge_key(u, v)
                remaining.discard(ek)
                adj[u].discard(v)
                adj[v].discard(u)
            pieces.append(cycle_vertices[:])
        
        def commit_single_edge(u, v):
            """Remove single edge from remaining and record the piece."""
            ek = edge_key(u, v)
            remaining.discard(ek)
            adj[u].discard(v)
            adj[v].discard(u)
            pieces.append([u, v])
        
        # Phase 1: Extract cycles from even-degree subgraphs (Eulerian components)
        # Repeatedly find connected components where all vertices have even degree
        # within the remaining graph, and decompose each into cycles.
        
        while time.time() < time_limit and remaining:
            # Find vertices with even degree in remaining graph
            even_degree_vertices = [v for v in adj if len(adj[v]) % 2 == 0 and len(adj[v]) > 0]
            
            if not even_degree_vertices:
                break
            
            # Find connected components among even-degree vertices
            # Use BFS/DFS restricted to edges where both endpoints have even degree
            visited = set()
            found_component = None
            
            for start in even_degree_vertices:
                if start in visited:
                    continue
                # BFS to find component
                component = []
                queue = [start]
                visited.add(start)
                while queue:
                    curr = queue.pop()
                    component.append(curr)
                    for nb in adj[curr]:
                        if nb not in visited and len(adj[nb]) % 2 == 0 and len(adj[nb]) > 0:
                            visited.add(nb)
                            queue.append(nb)
                
                if len(component) >= 3:
                    found_component = component
                    break
            
            if found_component is None:
                # No valid even-degree component, try to find any cycle
                break
            
            # Decompose this component into cycles using Hierholzer's algorithm
            # Build a subgraph of just this component
            comp_edges = set()
            for v in found_component:
                for nb in adj[v]:
                    if nb in found_component:
                        ek = edge_key(v, nb)
                        comp_edges.add(ek)
            
            if not comp_edges:
                continue
            
            # Hierholzer's algorithm to find all cycles
            # We need to work with the subgraph
            comp_adj = defaultdict(list)
            for u, v in comp_edges:
                comp_adj[u].append(v)
                comp_adj[v].append(u)
            
            # Use a stack-based approach
            used_comp_edges = set()
            
            def find_eulerian_cycles():
                """Find all cycles in the even-degree subgraph using Hierholzer."""
                cycles = []
                remaining_comp = set(comp_edges)
                
                while remaining_comp:
                    if time.time() > time_limit:
                        break
                    
                    # Find a starting vertex with unused edges
                    start = None
                    for e in remaining_comp:
                        u, v = e
                        if comp_adj[u] and any(edge_key(u, x) in remaining_comp for x in comp_adj[u]):
                            start = u
                            break
                    if start is None:
                        break
                    
                    # Hierholzer's algorithm
                    stack = [start]
                    path = []
                    while stack:
                        curr = stack[-1]
                        # Find an unused edge from curr
                        next_vertex = None
                        for nb in comp_adj[curr]:
                            ek = edge_key(curr, nb)
                            if ek in remaining_comp:
                                next_vertex = nb
                                break
                        if next_vertex is None:
                            path.append(stack.pop())
                        else:
                            ek = edge_key(curr, next_vertex)
                            remaining_comp.discard(ek)
                            stack.append(next_vertex)
                    
                    # The path is in reverse order
                    path.reverse()
                    
                    if len(path) >= 3:
                        # Extract the cycle
                        # path[0] should equal path[-1] for a proper cycle
                        if path[0] == path[-1]:
                            cycle = path[:-1]
                            if len(cycle) >= 3:
                                cycles.append(cycle)
                        else:
                            # This shouldn't happen in an Eulerian circuit,
                            # but handle gracefully
                            if len(path) >= 3:
                                cycles.append(path)
                
                return cycles
            
            cycles = find_eulerian_cycles()
            
            # Commit the cycles
            for cycle in cycles:
                if time.time() > time_limit:
                    break
                # Verify the cycle is still valid (edges still in remaining)
                valid = True
                k = len(cycle)
                for i in range(k):
                    u = cycle[i]
                    v = cycle[(i + 1) % k]
                    if edge_key(u, v) not in remaining:
                        valid = False
                        break
                if valid and len(cycle) >= 3:
                    commit_cycle(cycle)
            
            if not cycles:
                # Couldn't extract any cycles from this component
                break
        
        # Phase 2: Extract remaining cycles (from mixed-degree parts)
        while time.time() < time_limit and remaining:
            # Find any cycle in the remaining graph
            best_cycle = None
            best_len = 0
            
            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                break
            
            random.shuffle(candidates)
            
            for start in candidates[:30]:
                if time.time() > time_limit:
                    break
                
                # DFS to find a cycle through start
                path = [start]
                visited = {start}
                
                def dfs_find_cycle(current, depth, max_depth=20):
                    if time.time() > time_limit:
                        return None
                    if depth >= max_depth:
                        return None
                    neighbors = list(adj[current])
                    random.shuffle(neighbors)
                    for nb in neighbors:
                        if nb == start and len(path) >= 3:
                            return path[:]
                        if nb not in visited:
                            visited.add(nb)
                            path.append(nb)
                            result = dfs_find_cycle(nb, depth + 1, max_depth)
                            if result is not None:
                                return result
                            path.pop()
                            visited.discard(nb)
                    return None
                
                cycle = dfs_find_cycle(start, 0)
                if cycle is not None and len(cycle) > best_len:
                    best_cycle = cycle
                    best_len = len(cycle)
                    if best_len >= 10:  # Good enough
                        break
            
            if best_cycle is None:
                break
            
            # Verify cycle is valid
            valid = True
            k = len(best_cycle)
            for i in range(k):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % k]
                if edge_key(u, v) not in remaining:
                    valid = False
                    break
            
            if valid:
                commit_cycle(best_cycle)
            else:
                # Cycle is no longer valid, try again
                continue
        
        # Phase 3: Remaining edges become single-edge pieces
        for u, v in list(remaining):
            if time.time() > time_limit:
                break
            commit_single_edge(u, v)
        
        return pieces

    # Main: decompose each instance
    results = []
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    
    for spec in specs:
        if time.time() > deadline - 15:
            # Not enough time, use trivial solution
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        
        # Try multiple random seeds and keep the best
        best_pieces = None
        best_count = float('inf')
        
        time_for_this = min(40, deadline - time.time() - 10)
        if time_for_this < 3:
            time_for_this = 3
        
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            random.seed(seed * 7 + 13)
            pieces = decompose(edges, time.time() + time_for_this / 20)
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = [p[:] for p in pieces]
        
        if best_pieces is None:
            best_pieces = [list(e) for e in edges]
        
        results.append(best_pieces)
    
    return results
