"""Decompose three fixed Erdős-Gallai instance graphs into few cycles and edges."""
import random
from random import Random

def construct():
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edge_list):
        edge_set = {tuple(sorted(e)) for e in edge_list}
        n = max(max(e) for e in edge_set) + 1 if edge_set else 0
        if n == 0:
            return []
        
        adj = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)

        pieces = []

        def remove_cycle(cycle):
            nonlocal edge_set, adj
            m = len(cycle)
            if m < 3:
                return
            for i in range(m):
                a = cycle[i]
                b = cycle[(i + 1) % m]
                key = (min(a, b), max(a, b))
                if key in edge_set:
                    edge_set.discard(key)
                    adj[a].discard(b)
                    adj[b].discard(a)

        def find_longest_cycle(start):
            # DFS to find the longest cycle in the connected component of start
            # Returns a list of vertices representing the cycle, or None
            visited = {start}
            parent = {start: None}
            depth = {start: 0}
            best_cycle = None
            best_len = 0
            
            # Iterative DFS to avoid recursion limits (though N is small)
            # We need to handle back edges carefully.
            # Stack stores (node, iterator over neighbors)
            stack = [(start, iter(adj[start]))]
            
            while stack:
                node, neighbors = stack[-1]
                advanced = False
                for nb in neighbors:
                    if nb not in visited:
                        visited.add(nb)
                        parent[nb] = node
                        depth[nb] = depth[node] + 1
                        stack.append((nb, iter(adj[nb])))
                        advanced = True
                        break
                    elif nb != parent[node]:
                        # Back edge to an ancestor
                        # The cycle is from nb to node, plus the edge node-nb
                        # Length is depth[node] - depth[nb] + 1
                        # We need to reconstruct the path
                        curr_len = depth[node] - depth[nb] + 1
                        if curr_len > best_len:
                            # Reconstruct path from node back to nb
                            path = []
                            curr = node
                            while curr != nb:
                                path.append(curr)
                                curr = parent[curr]
                            path.append(nb)
                            # The cycle is path (which is nb -> ... -> node) 
                            # Wait, path is node -> ... -> nb.
                            # Cycle is nb -> ... -> node -> nb.
                            # So the list of vertices is path reversed?
                            # path: [node, ..., nb]
                            # Cycle vertices: [nb, ..., node]
                            # The edges are (nb, next), ..., (last, node), (node, nb).
                            # So the cycle representation [nb, ..., node] works.
                            best_cycle = path[::-1]
                            best_len = curr_len
                if not advanced:
                    stack.pop()
            
            return best_cycle

        # Main loop
        # We iterate until no edges with degree >= 2 remain
        # To find the "best" cycle to remove, we try all candidates?
        # Or just pick one?
        # Removing a long cycle is good.
        # Let's try to find the longest cycle in the entire graph at each step.
        # This might be expensive if we do it naively.
        # But N is small.
        
        # Optimization: Only consider components with cycles.
        # A component has a cycle if it has any vertex with degree >= 2.
        
        while edge_set:
            # Find all vertices with degree >= 2
            candidates = [v for v in range(n) if len(adj[v]) >= 2]
            if not candidates:
                break
            
            # Try to find the longest cycle among all candidates
            # Since the graph might be disconnected, we should check each candidate's component.
            # However, if we pick a candidate, find_longest_cycle explores its whole component.
            # So we just need to pick one candidate per component?
            # Or just pick the best among all candidates?
            # If we pick candidate v1 in comp1 and v2 in comp2, we should compare the best cycle in comp1 vs comp2.
            # But find_longest_cycle(v1) only explores comp1.
            # So we can group candidates by component?
            # Or just run find_longest_cycle on each unique component representative.
            
            # Simple approach:
            # 1. Identify connected components.
            # 2. For each component with a cycle, find the longest cycle.
            # 3. Pick the global longest cycle.
            
            # Find components
            visited_comp = set()
            components = []
            for v in range(n):
                if v not in visited_comp and adj[v]:
                    # BFS/DFS to get component
                    comp = []
                    stack = [v]
                    visited_comp.add(v)
                    while stack:
                        curr = stack.pop()
                        comp.append(curr)
                        for nb in adj[curr]:
                            if nb not in visited_comp:
                                visited_comp.add(nb)
                                stack.append(nb)
                    components.append(comp)
            
            best_global_cycle = None
            best_global_len = 0
            
            for comp in components:
                # Check if this component has a cycle
                has_cycle = False
                for v in comp:
                    if len(adj[v]) >= 2:
                        has_cycle = True
                        break
                if not has_cycle:
                    continue
                
                # Find longest cycle in this component
                # Pick any node in comp as start
                start_node = comp[0]
                cycle = find_longest_cycle(start_node)
                if cycle is not None and len(cycle) > best_global_len:
                    best_global_len = len(cycle)
                    best_global_cycle = cycle
            
            if best_global_cycle is None:
                # Should not happen if has_cycle was true, but safety check
                break
            
            pieces.append(best_global_cycle)
            remove_cycle(best_global_cycle)
        
        # Remaining edges
        for u, v in edge_set:
            pieces.append([u, v])

        return pieces

    return [decompose(instance(*spec)) for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))]
