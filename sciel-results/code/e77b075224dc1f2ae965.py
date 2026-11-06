"""Decompose graphs into minimum number of cycles and single edges."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    from collections import defaultdict, deque
    import time

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        if not edges:
            return []
        
        # Build adjacency for the full graph
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # State: set of edges in cycles
        S = set()
        
        # Helper: Get cycles in a graph defined by edge set
        def get_cycles(edge_set):
            if not edge_set:
                return []
            cycles = []
            visited_edges = set()
            
            # Find connected components
            # For each component, if it's a cycle, add it
            # Note: A component with even degrees is a union of cycles.
            # We need to decompose it into simple cycles.
            
            # Simple approach: DFS to find cycles
            # But we need to return a list of edge-disjoint cycles that cover all edges in edge_set
            # This is only possible if edge_set is a union of cycles.
            
            # We will maintain S as a union of cycles.
            # So we just need to extract the cycles from S.
            
            # To find cycles in a general graph (E\S), we use DFS.
            # To decompose S into cycles, we can use the fact that all degrees are even.
            
            return [] # Placeholder, logic below

        # Function to decompose a set of edges (with even degrees) into simple cycles
        def decompose_even_graph(edge_set):
            if not edge_set:
                return []
            
            # Build adj for this subgraph
            sub_adj = defaultdict(set)
            for u, v in edge_set:
                sub_adj[u].add(v)
                sub_adj[v].add(u)
                
            cycles = []
            remaining = set(edge_set)
            
            while remaining:
                # Find a cycle in remaining
                # Pick an arbitrary edge
                u, v = next(iter(remaining))
                # DFS to find cycle containing u-v
                visited = set()
                path = [u]
                parent = {u: None}
                stack = [u]
                
                # We want a cycle that includes the edge (u,v) if possible? 
                # Or just any cycle.
                
                # Standard cycle finding in undirected graph
                def find_cycle_in_subgraph(start, adj_set):
                    visited = set()
                    stack = [(start, -1, [start])]
                    visited.add(start)
                    while stack:
                        node, par, path = stack.pop()
                        for nb in sorted(adj_set[node]):
                            if nb == par:
                                continue
                            if nb in visited:
                                if nb in path:
                                    idx = path.index(nb)
                                    return path[idx:]
                            else:
                                visited.add(nb)
                                stack.append((nb, node, path + [nb]))
                    return None

                # Try to find a cycle
                # We can just pick any node and run DFS
                start_node = next(iter(remaining))
                cycle = find_cycle_in_subgraph(start_node, sub_adj)
                
                if cycle is None:
                    # Should not happen if all degrees are even and connected components are cycles
                    # But if the graph is not connected, we process components
                    # Actually, if all degrees are even, every component contains a cycle.
                    # If we failed, something is wrong.
                    break
                
                # Extract edges of this cycle
                cycle_edges = set()
                for i in range(len(cycle) - 1):
                    a, b = cycle[i], cycle[i+1]
                    cycle_edges.add((a, b) if a < b else (b, a))
                    
                # Remove these edges from remaining
                remaining -= cycle_edges
                cycles.append(cycle)
                
                # Update sub_adj for remaining? 
                # It's easier to rebuild or just track remaining.
                # For finding next cycle, we need adj of remaining.
                # Let's rebuild sub_adj from remaining each time? O(E^2) worst case.
                # Given small size, it's fine.
                sub_adj = defaultdict(set)
                for u, v in remaining:
                    sub_adj[u].add(v)
                    sub_adj[v].add(u)
                    
            return cycles

        # Function to calculate score of S
        def score(S):
            if not S:
                return 0
            cycles = decompose_even_graph(S)
            # Score is sum(len(c) - 1)
            return sum(len(c) - 1 for c in cycles)

        # Initial S: empty
        S = set()
        best_score = 0
        
        # Greedy initialization: Add cycles that increase score
        # We need a way to find cycles in E \ S
        def find_cycle_in_complement():
            comp_edges = set(edges) - S
            if not comp_edges:
                return None
            comp_adj = defaultdict(set)
            for u, v in comp_edges:
                comp_adj[u].add(v)
                comp_adj[v].add(u)
                
            # Find any cycle
            for start in comp_adj:
                visited = set()
                stack = [(start, -1, [start])]
                visited.add(start)
                while stack:
                    node, par, path = stack.pop()
                    for nb in sorted(comp_adj[node]):
                        if nb == par:
                            continue
                        if nb in visited:
                            if nb in path:
                                idx = path.index(nb)
                                return path[idx:]
                        else:
                            visited.add(nb)
                            stack.append((nb, node, path + [nb]))
            return None

        # Greedy add
        while True:
            cycle = find_cycle_in_complement()
            if cycle is None:
                break
            # Check if adding this cycle improves score
            new_S = S.union(set((u, v) if u < v else (v, u) for u, v in zip(cycle, cycle[1:])))
            new_score = score(new_S)
            if new_score > best_score:
                S = new_S
                best_score = new_score
            else:
                # If adding doesn't improve, maybe we should skip this cycle?
                # But if we skip it, we might never find a better combination.
                # For now, just break if no single cycle improves.
                # This is a weakness. We need swaps.
                break

        # Local Search: Swaps
        # Try removing one cycle and adding another
        # This is complex. Let's try a simpler heuristic:
        # Randomly remove a cycle, then greedily add cycles back?
        
        # Given time, let's stick to a robust greedy with a twist:
        # Prioritize longer cycles.
        
        S = set()
        best_score = 0
        remaining_edges = set(edges)
        
        # Sort edges? No, sort cycles by length.
        # We will repeatedly find the LONGEST cycle in remaining_edges.
        
        while remaining_edges:
            # Find longest cycle
            best_cycle = None
            # Use DFS to find cycles, keep the longest
            # This is expensive if done naively.
            
            # Heuristic: Find cycles, pick the longest one found
            # To find "longest" cycle, we can do a DFS and track the longest path that closes.
            # Or just find all cycles up to a limit?
            
            # Let's just find ANY cycle, but try to make it long by preferring high-degree nodes.
            # Actually, the previous program did this.
            
            # Let's try a different strategy:
            # 1. Find a cycle.
            # 2. Check if removing it allows for a much better configuration?
            
            # Given the complexity, let's optimize the greedy by trying multiple starting points
            # and picking the longest cycle found.
            
            adj = defaultdict(set)
            for u, v in remaining_edges:
                adj[u].add(v)
                adj[v].add(u)
                
            if not adj:
                break
                
            # Find longest cycle
            longest_cycle = None
            # Try starting from each node? Too slow.
            # Try starting from high degree nodes.
            nodes = sorted(adj.keys(), key=lambda x: -len(adj[x]))
            
            for start in nodes[:10]:
                # DFS to find cycles
                visited = set()
                stack = [(start, -1, [start])]
                visited.add(start)
                while stack:
                    node, par, path = stack.pop()
                    for nb in sorted(adj[node], key=lambda x: -len(adj[x])):
                        if nb == par:
                            continue
                        if nb in visited:
                            if nb in path:
                                idx = path.index(nb)
                                cyc = path[idx:]
                                if len(cyc) >= 3:
                                    if longest_cycle is None or len(cyc) > len(longest_cycle):
                                        longest_cycle = cyc
                        else:
                            visited.add(nb)
                            stack.append((nb, node, path + [nb]))
                            
            if longest_cycle:
                # Remove edges
                for i in range(len(longest_cycle) - 1):
                    u, v = longest_cycle[i], longest_cycle[i+1]
                    e = (u, v) if u < v else (v, u)
                    remaining_edges.discard(e)
                # Add to pieces? We collect cycles first
                # We'll store cycles in a list
                # But we need to return pieces.
                # Let's store the cycles found.
                # Wait, the variable 'pieces' is not defined in this scope yet.
                # Let's restructure.
                
            else:
                break
                
        # This is getting messy. Let's implement the "Even Subgraph Optimization" properly.
        
        # Reset
        S = set()
        
        # Helper to get cycles from S
        def get_cycles_from_S(S):
            if not S:
                return []
            cycles = []
            remaining = set(S)
            while remaining:
                # Find a cycle in remaining
                adj_temp = defaultdict(set)
                for u, v in remaining:
                    adj_temp[u].add(v)
                    adj_temp[v].add(u)
                    
                start = next(iter(remaining))
                visited = set()
                stack = [(start, -1, [start])]
                visited.add(start)
                cycle = None
                while stack:
                    node, par, path = stack.pop()
                    for nb in sorted(adj_temp[node]):
                        if nb == par:
                            continue
                        if nb in visited:
                            if nb in path:
                                idx = path.index(nb)
                                cycle = path[idx:]
                                break
                        else:
                            visited.add(nb)
                            stack.append((nb, node, path + [nb]))
                    if cycle:
                        break
                
                if not cycle:
                    # Should not happen
                    break
                    
                cycle_edges = set()
                for i in range(len(cycle) - 1):
                    u, v = cycle[i], cycle[i+1]
                    cycle_edges.add((u, v) if u < v else (v, u))
                    
                remaining -= cycle_edges
                cycles.append(cycle)
                
            return cycles

        def evaluate(S):
            cycles = get_cycles_from_S(S)
            return sum(len(c) - 1 for c in cycles)

        # Greedy Initialization
        S = set()
        best_score = 0
        remaining = set(edges)
        
        while remaining:
            # Find longest cycle in remaining
            adj_temp = defaultdict(set)
            for u, v in remaining:
                adj_temp[u].add(v)
                adj_temp[v].add(u)
                
            if not adj_temp:
                break
                
            longest_cycle = None
            nodes = sorted(adj_temp.keys(), key=lambda x: -len(adj_temp[x]))
            
            for start in nodes[:5]:
                visited = set()
                stack = [(start, -1, [start])]
                visited.add(start)
                while stack:
                    node, par, path = stack.pop()
                    for nb in sorted(adj_temp[node], key=lambda x: -len(adj_temp[x])):
                        if nb == par:
                            continue
                        if nb in visited:
                            if nb in path:
                                idx = path.index(nb)
                                cyc = path[idx:]
                                if len(cyc) >= 3:
                                    if longest_cycle is None or len(cyc) > len(longest_cycle):
                                        longest_cycle = cyc
                        else:
                            visited.add(nb)
                            stack.append((nb, node, path + [nb]))
                            
            if longest_cycle:
                cycle_edges = set()
                for i in range(len(longest_cycle) - 1):
                    u, v = longest_cycle[i], longest_cycle[i+1]
                    cycle_edges.add((u, v) if u < v else (v, u))
                
                new_S = S.union(cycle_edges)
                new_score = evaluate(new_S)
                
                if new_score > best_score:
                    S = new_S
                    best_score = new_score
                    remaining -= cycle_edges
                else:
                    # If it doesn't improve, try next cycle?
                    # We can't easily "skip" this cycle in the remaining set without changing the graph.
                    # So we just break or add it anyway?
                    # Adding it might block better cycles.
                    # Let's add it anyway to avoid infinite loop, but it's suboptimal.
                    # Actually, if new_score <= best_score, it means this cycle reduces the potential.
                    # We should probably not add it.
                    # But if we don't add it, we are stuck.
                    # Let's just add it.
                    S = new_S
                    best_score = new_score
                    remaining -= cycle_edges
            else:
                break

        # Local Search: Try to swap
        # This is hard. Let's just return the result.
        
        cycles = get_cycles_from_S(S)
        pieces = []
        for c in cycles:
            pieces.append(c)
            
        # Add remaining edges
        for u, v in remaining:
            pieces.append([u, v])
            
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
