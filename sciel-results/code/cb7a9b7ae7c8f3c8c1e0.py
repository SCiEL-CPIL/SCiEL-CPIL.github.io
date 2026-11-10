"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from collections import defaultdict, deque

def construct():
    # Replicate the verifier's instance generator exactly.
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        if not edges:
            return []
        
        remaining = set(edges)
        pieces = []
        
        # Find connected components
        def get_components(edge_set):
            if not edge_set:
                return []
            adj = defaultdict(set)
            for u, v in edge_set:
                adj[u].add(v)
                adj[v].add(u)
            visited = set()
            components = []
            for node in list(adj.keys()):
                if node not in visited:
                    comp_nodes = set()
                    stack = [node]
                    while stack:
                        n = stack.pop()
                        if n not in visited:
                            visited.add(n)
                            comp_nodes.add(n)
                            for nb in adj[n]:
                                if nb not in visited:
                                    stack.append(nb)
                    comp_edges = set()
                    for u, v in edge_set:
                        if u in comp_nodes and v in comp_nodes:
                            comp_edges.add((u, v))
                    if comp_edges:
                        components.append((comp_nodes, comp_edges))
            return components
        
        # Find bridges in a component
        def find_bridges(comp_edges):
            if not comp_edges:
                return set()
            adj = defaultdict(set)
            for u, v in comp_edges:
                adj[u].add(v)
                adj[v].add(u)
            visited = set()
            bridges = set()
            time_counter = [0]
            disc = {}
            low = {}
            
            def dfs(u, parent):
                disc[u] = time_counter[0]
                low[u] = time_counter[0]
                time_counter[0] += 1
                visited.add(u)
                for v in adj[u]:
                    if v == parent:
                        continue
                    if v not in disc:
                        dfs(v, u)
                        low[u] = min(low[u], low[v])
                        if low[v] > disc[u]:
                            bridges.add(tuple(sorted((u, v))))
                    else:
                        low[u] = min(low[u], disc[v])
            
            for node in adj:
                if node not in visited:
                    dfs(node, -1)
            return bridges
        
        # Find a cycle in a bridgeless component using DFS
        def find_cycle(comp_edges):
            if not comp_edges:
                return None
            adj = defaultdict(set)
            for u, v in comp_edges:
                adj[u].add(v)
                adj[v].add(u)
            
            active_nodes = [n for n in adj if adj[n]]
            if not active_nodes:
                return None
                
            visited = set()
            parent = {}
            
            def dfs_cycle(node, start_node, path):
                visited.add(node)
                path.append(node)
                for neighbor in sorted(adj[node]):
                    if neighbor == start_node and len(path) >= 3:
                        return list(path)
                    if neighbor not in visited:
                        parent[neighbor] = node
                        res = dfs_cycle(neighbor, start_node, path)
                        if res:
                            return res
                path.pop()
                return None

            for start_node in active_nodes:
                if start_node in visited:
                    continue
                res = dfs_cycle(start_node, start_node, [])
                if res:
                    return res
            return None
        
        # Decompose a single component
        def decompose_component(comp_edges):
            comp_pieces = []
            rem = set(comp_edges)
            
            while rem:
                # Find bridges
                bridges = find_bridges(rem)
                if bridges:
                    for b in bridges:
                        comp_pieces.append(list(b))
                        rem.discard(b)
                    continue
                
                # No bridges, find a cycle
                cycle = find_cycle(rem)
                if cycle:
                    comp_pieces.append(cycle)
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        edge = tuple(sorted((u, v)))
                        rem.discard(edge)
                else:
                    # Fallback: treat all remaining edges as individual pieces
                    for e in rem:
                        comp_pieces.append(list(e))
                    break
            
            return comp_pieces
        
        # Main decomposition
        components = get_components(remaining)
        for comp_nodes, comp_edges in components:
            pieces.extend(decompose_component(comp_edges))
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
