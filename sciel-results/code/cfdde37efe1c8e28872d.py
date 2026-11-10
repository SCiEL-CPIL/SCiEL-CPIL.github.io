"""Decompose random graphs into cycles and edges to minimize piece count."""
import time
import random
from random import Random
from collections import defaultdict, deque

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_connected_components(edges):
        """Find connected components in the graph defined by edges."""
        if not edges:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        visited = set()
        components = []
        
        for node in list(adj.keys()):
            if node in visited:
                continue
            
            # BFS to find component
            component_nodes = set()
            queue = deque([node])
            visited.add(node)
            
            while queue:
                curr = queue.popleft()
                component_nodes.add(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            # Collect all edges within this component
            component_edges = set()
            for u, v in edges:
                if u in component_nodes and v in component_nodes:
                    component_edges.add((u, v))
            
            components.append((component_nodes, component_edges))
        
        return components

    def decompose_component(component_edges):
        """Decompose a connected component into cycles and edges."""
        if not component_edges:
            return []
        
        remaining = set(component_edges)
        pieces = []
        
        # Build adjacency
        def build_adj(rem):
            adj = defaultdict(set)
            for u, v in rem:
                adj[u].add(v)
                adj[v].add(u)
            return adj
        
        adj = build_adj(remaining)
        
        # Find bridges to identify 2-edge-connected components
        def find_bridges_and_2ecc(adj, edges_set):
            """Find bridges and 2-edge-connected components using Tarjan's algorithm."""
            disc = {}
            low = {}
            parent = {}
            bridges = set()
            timer = 0
            visited = set()
            
            def dfs(u):
                nonlocal timer
                disc[u] = low[u] = timer
                timer += 1
                visited.add(u)
                
                for v in sorted(adj[u]):
                    if (min(u, v), max(u, v)) not in edges_set:
                        continue
                    
                    if v == parent.get(u):
                        continue
                    
                    if v not in visited:
                        parent[v] = u
                        dfs(v)
                        low[u] = min(low[u], low[v])
                        
                        if low[v] > disc[u]:
                            bridges.add((min(u, v), max(u, v)))
                    else:
                        low[u] = min(low[u], disc[v])
            
            for node in adj:
                if node not in visited:
                    dfs(node)
            
            # Now find 2-edge-connected components by removing bridges
            non_bridge_edges = set()
            for e in edges_set:
                if e not in bridges:
                    non_bridge_edges.add(e)
            
            # Find connected components of non-bridge edges
            nb_adj = defaultdict(set)
            for u, v in non_bridge_edges:
                nb_adj[u].add(v)
                nb_adj[v].add(u)
            
            visited_2ecc = set()
            two_edge_components = []
            
            for node in nb_adj:
                if node in visited_2ecc:
                    continue
                
                comp_nodes = set()
                queue = deque([node])
                visited_2ecc.add(node)
                
                while queue:
                    curr = queue.popleft()
                    comp_nodes.add(curr)
                    for neighbor in nb_adj[curr]:
                        if neighbor not in visited_2ecc:
                            visited_2ecc.add(neighbor)
                            queue.append(neighbor)
                
                comp_edges = set()
                for u, v in non_bridge_edges:
                    if u in comp_nodes and v in comp_nodes:
                        comp_edges.add((u, v))
                
                two_edge_components.append(comp_edges)
            
            return bridges, two_edge_components
        
        # Extract cycles from 2-edge-connected components
        def extract_cycles_from_2ecc(edges_2ecc, max_iterations=50):
            """Extract edge-disjoint cycles from a 2-edge-connected subgraph."""
            if len(edges_2ecc) < 3:
                return [], edges_2ecc
            
            cycles = []
            remaining_2ecc = set(edges_2ecc)
            
            for _ in range(max_iterations):
                if not remaining_2ecc:
                    break
                
                # Find a cycle using DFS
                def find_cycle_in_subgraph(rem_edges):
                    if len(rem_edges) < 3:
                        return None
                    
                    # Build local adjacency
                    local_adj = defaultdict(set)
                    for u, v in rem_edges:
                        local_adj[u].add(v)
                        local_adj[v].add(u)
                    
                    # Try to find a cycle starting from each vertex
                    for start in sorted(local_adj.keys()):
                        if len(local_adj[start]) < 2:
                            continue
                        
                        # DFS to find cycle back to start
                        def dfs_find_cycle(node, path, visited):
                            if len(path) >= 3 and node == start:
                                return path[:]
                            
                            for neighbor in sorted(local_adj[node]):
                                edge = (min(node, neighbor), max(node, neighbor))
                                if edge not in rem_edges:
                                    continue
                                
                                if neighbor == start and len(path) >= 3:
                                    return path + [start]
                                
                                if neighbor not in visited:
                                    result = dfs_find_cycle(neighbor, path + [neighbor], visited | {neighbor})
                                    if result:
                                        return result
                            
                            return None
                        
                        cycle = dfs_find_cycle(start, [start], {start})
                        if cycle and len(cycle) >= 4:  # at least 3 edges
                            return cycle[:-1]  # Remove the repeated start at end
                    
                    return None
                
                cycle = find_cycle_in_subgraph(remaining_2ecc)
                
                if cycle and len(cycle) >= 3:
                    # Remove cycle edges from remaining
                    for i in range(len(cycle) - 1):
                        u, v = cycle[i], cycle[i + 1]
                        edge = (min(u, v), max(u, v))
                        remaining_2ecc.discard(edge)
                    
                    # Also remove the closing edge
                    u, v = cycle[-1], cycle[0]
                    edge = (min(u, v), max(u, v))
                    remaining_2ecc.discard(edge)
                    
                    cycles.append(cycle)
                else:
                    break
            
            return cycles, remaining_2ecc
        
        # Main decomposition loop
        while remaining:
            if time.time() > deadline - 15:
                break
            
            # Find 2-edge-connected components
            bridges, two_edge_components = find_bridges_and_2ecc(adj, remaining)
            
            found_cycle = False
            for comp_edges in two_edge_components:
                if len(comp_edges) < 3:
                    continue
                
                cycles, leftover = extract_cycles_from_2ecc(comp_edges)
                
                for cycle in cycles:
                    if len(cycle) >= 3:
                        # Remove cycle edges from remaining
                        for i in range(len(cycle) - 1):
                            u, v = cycle[i], cycle[i + 1]
                            edge = (min(u, v), max(u, v))
                            remaining.discard(edge)
                        
                        # Remove closing edge
                        u, v = cycle[-1], cycle[0]
                        edge = (min(u, v), max(u, v))
                        remaining.discard(edge)
                        
                        pieces.append(cycle[:])
                        found_cycle = True
                        break
                
                if found_cycle:
                    break
            
            if not found_cycle:
                # No more cycles found; add remaining edges as singles
                for e in sorted(remaining):
                    pieces.append([e[0], e[1]])
                remaining = set()
                break
            
            # Rebuild adjacency
            adj = build_adj(remaining)
        
        # Add any remaining edges as singles
        for e in sorted(remaining):
            pieces.append([e[0], e[1]])
        
        return pieces

    def decompose(edges):
        """Decompose a set of edges into cycles and single edges, minimizing piece count."""
        if not edges:
            return []
        
        # Decompose each connected component independently
        components = find_connected_components(edges)
        
        all_pieces = []
        for _, comp_edges in components:
            if time.time() > deadline - 10:
                # Not enough time; add remaining as singles
                for e in sorted(comp_edges):
                    all_pieces.append([e[0], e[1]])
                continue
            
            pieces = decompose_component(comp_edges)
            all_pieces.extend(pieces)
        
        return all_pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline - 5:
            # Not enough time; use trivial decomposition
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        result = decompose(edges)
        results.append(result)

    return results
