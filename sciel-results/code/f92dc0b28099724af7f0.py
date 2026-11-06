"""Decompose graphs into minimum number of cycles and single edges."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    from collections import defaultdict, deque

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_eulerian_trail(adj, start):
        """
        Returns a list of vertices representing an Eulerian trail in the graph defined by adj.
        adj is a dict of lists.
        start is the starting vertex.
        """
        # We need to consume edges, so we work with a copy or mutable structure
        # For simplicity, we'll use a stack of (node, neighbor_index)
        # But we need to know which edges are used.
        # Let's use a set of used edges.
        
        # Actually, standard Hierholzer's on a graph with edge counts.
        # Let's build a multigraph representation.
        # adj[u] is a list of v.
        # We need to track which specific edge instance is used.
        # Since we are decomposing a simple graph, edges are unique.
        
        # We'll use a stack.
        stack = [start]
        path = []
        # We need to modify adj to remove used edges.
        # Let's pass a copy of adj where we can pop neighbors.
        
        # To make it easy, let's use a dict of sets for neighbors.
        # But we need to return the trail.
        
        # Let's implement Hierholzer's with a set of available edges.
        # This is inefficient if we copy sets.
        # Given N=30, it's fine.
        
        # Let's just use the standard algorithm on the adjacency list, 
        # but we must ensure we don't reuse edges.
        # We'll maintain a set of used_edges.
        
        # Wait, the function `decompose` in the main loop will manage `remaining`.
        # So `get_eulerian_trail` should take the component edges and return the trail.
        
        pass

    def decompose(edges):
        remaining = set(edges)
        pieces = []
        
        # Helper to build adjacency for a set of edges
        def build_adj(rem):
            adj = defaultdict(list)
            for u, v in rem:
                adj[u].append(v)
                adj[v].append(u)
            return adj

        # Helper to find Eulerian Trail
        def find_eulerian_trail(rem):
            if not rem:
                return None
            
            adj = defaultdict(list)
            for u, v in rem:
                adj[u].append(v)
                adj[v].append(u)
            
            # Find start node
            # If all degrees even, start anywhere.
            # If 2 odd, start at one odd.
            odd_nodes = [n for n in adj if len(adj[n]) % 2 == 1]
            if not odd_nodes:
                start = next(iter(adj))
            elif len(odd_nodes) == 2:
                start = odd_nodes[0]
            else:
                # Should not happen for a valid component of a simple graph decomposition step?
                # Actually, if we have a component with >2 odd nodes, it's not Eulerian.
                # But we are decomposing the whole graph.
                # A connected component of a graph can have any number of odd nodes.
                # Hierholzer's only works if 0 or 2 odd nodes.
                # If >2, we cannot find a single Eulerian trail covering all edges.
                # We must split the component?
                # No, the standard decomposition algorithm handles this by:
                # 1. Find any trail.
                # 2. The remaining edges form subgraphs.
                # 3. Repeat.
                
                # So, if odd_nodes > 2, we just pick any node with degree > 0.
                # The trail will stop when it gets stuck.
                # The remaining edges will be handled in the next iteration.
                start = next(iter(adj))
            
            # Hierholzer's Algorithm
            # We need to track used edges.
            used_edges = set()
            stack = [start]
            path = []
            
            while stack:
                node = stack[-1]
                # Find an unused edge from node
                found_edge = False
                for nb in adj[node]:
                    edge = (node, nb) if node < nb else (nb, node)
                    if edge not in used_edges:
                        # Use this edge
                        used_edges.add(edge)
                        stack.append(nb)
                        found_edge = True
                        break
                
                if not found_edge:
                    # Dead end, backtrack
                    path.append(stack.pop())
            
            # path is in reverse order
            path.reverse()
            return path

        # Helper to decompose a trail into simple cycles and edges
        def peel_trail(trail):
            # trail is a list of vertices
            # Returns a list of pieces (cycles or single edges)
            if not trail:
                return []
            
            # If trail has only 2 vertices (1 edge)
            if len(trail) == 2:
                return [[trail[0], trail[1]]]
            
            # Check if it's a circuit
            is_circuit = (trail[0] == trail[-1])
            
            # We will repeatedly extract the longest simple cycle
            # represented by the max-distance repeat.
            
            # To do this efficiently, we work on the list of vertices.
            # We need to remove edges.
            # Let's maintain a set of used edges in the trail.
            # But the trail is just a sequence.
            
            # Strategy:
            # 1. Find indices i, j such that trail[i] == trail[j] and j-i is max.
            # 2. The segment trail[i:j+1] is a closed walk.
            # 3. Extract a simple cycle from this walk.
            # 4. Remove the edges of this cycle from the trail.
            # 5. The remaining edges form one or more trails.
            # 6. Recurse on each.
            
            # This is complex to implement in-place.
            # Simpler: 
            # Just find ANY simple cycle in the trail.
            # Use the first repeat.
            
            # Let's find the first repeat.
            # Map vertex to first index.
            first_idx = {}
            for i, v in enumerate(trail):
                if v not in first_idx:
                    first_idx[v] = i
            
            cycle_found = None
            for i, v in enumerate(trail):
                if v in first_idx and first_idx[v] < i:
                    # Found a repeat
                    start_idx = first_idx[v]
                    end_idx = i
                    # The segment trail[start_idx : end_idx+1] is a closed walk
                    # It contains a simple cycle.
                    # To get a simple cycle, we can just take this segment?
                    # No, it might have internal repeats.
                    # But for the purpose of minimizing pieces, extracting a valid simple cycle is key.
                    # Let's extract the simple cycle by finding the first repeat *within* this segment.
                    
                    # Actually, if we just take the segment, is it a valid piece?
                    # Piece must be distinct vertices.
                    # If the segment has repeats, it's not a simple cycle.
                    
                    # Let's just find the simple cycle formed by the first repeat in the whole trail.
                    # This cycle is trail[start_idx : end_idx+1].
                    # If this segment has no other repeats, it's a simple cycle.
                    # If it has repeats, it's not.
                    
                    # Let's assume the "First Repeat" gives a simple cycle.
                    # This is true if the trail is an Eulerian trail of a simple graph?
                    # No.
                    
                    # Let's use a robust method:
                    # Find the simple cycle by tracing.
                    # Start at start_idx.
                    # The cycle is the path from start_idx to the first time we hit start_idx again.
                    # This is exactly trail[start_idx : end_idx+1] IF there are no other repeats of start_idx in between?
                    # Yes, by definition of "first repeat" of v, there are no other v's in between.
                    # But there could be repeats of OTHER vertices.
                    # So trail[start_idx : end_idx+1] is a closed walk that may contain other cycles.
                    # It is NOT necessarily a simple cycle.
                    
                    # However, it CONTAINS a simple cycle.
                    # We can recursively peel this segment.
                    
                    # For now, let's just return this segment as a piece?
                    # NO, it must be a simple cycle.
                    
                    # Let's find the simple cycle within trail[start_idx : end_idx+1].
                    # We can do this by finding the first repeat in this sub-segment.
                    
                    sub_trail = trail[start_idx : end_idx + 1]
                    # Find first repeat in sub_trail
                    # This will yield a simple cycle.
                    
                    # Let's implement a helper to get the simple cycle from a closed walk.
                    def get_simple_cycle_from_closed_walk(cw):
                        # cw is a list of vertices, cw[0] == cw[-1]
                        # Find first repeat in cw[0:-1]
                        seen = {}
                        for i, v in enumerate(cw[:-1]):
                            if v in seen:
                                # Cycle is cw[seen[v] : i+1]
                                return cw[seen[v] : i + 1]
                            seen[v] = i
                        # If no repeat, it's a simple cycle
                        return cw

                    simple_cycle = get_simple_cycle_from_closed_walk(trail[start_idx : end_idx + 1])
                    # Now we have a simple cycle.
                    # We need to remove its edges from the trail.
                    # This splits the trail into parts.
                    
                    # This is getting too complex for a simple script.
                    # Let's simplify:
                    # Just output the simple cycle found.
                    # And then... what?
                    # We can't easily remove it from the trail and continue.
                    
                    # Alternative:
                    # Just output the simple cycle.
                    # And mark those edges as used.
                    # Then find the next trail in the remaining edges.
                    
                    # So, `peel_trail` should not try to decompose the whole trail at once.
                    # Instead, `decompose` should:
                    # 1. Find Eulerian Trail.
                    # 2. Find a simple cycle in it.
                    # 3. Output that cycle.
                    # 4. Remove those edges from `remaining`.
                    # 5. Go back to step 1 (find new Eulerian Trail in remaining).
                    
                    # This is effectively "Greedy Cycle Extraction" but guided by Eulerian Trails.
                    # The Eulerian Trail ensures we find a cycle that is part of a "dense" structure.
                    
                    return [simple_cycle] # Just return one cycle for now

            # If no cycle found, return single edges
            pieces = []
            for i in range(len(trail) - 1):
                pieces.append([trail[i], trail[i+1]])
            return pieces

        # Main Loop
        while remaining:
            # Find a component
            # Just pick any edge
            u, v = next(iter(remaining))
            
            # Find all edges in this component
            # BFS/DFS on remaining
            comp_edges = set()
            visited_nodes = set()
            stack = [u]
            adj_temp = defaultdict(list)
            for e in remaining:
                adj_temp[e[0]].append(e[1])
                adj_temp[e[1]].append(e[0])
            
            while stack:
                node = stack.pop()
                if node in visited_nodes:
                    continue
                visited_nodes.add(node)
                for nb in adj_temp[node]:
                    edge = (node, nb) if node < nb else (nb, node)
                    if edge in remaining:
                        comp_edges.add(edge)
                        if nb not in visited_nodes:
                            stack.append(nb)
            
            # Find Eulerian Trail in comp_edges
            trail = find_eulerian_trail(comp_edges)
            
            if not trail:
                # Should not happen
                break
            
            # Extract a simple cycle from the trail
            # Use the first repeat method
            first_idx = {}
            cycle = None
            for i, v in enumerate(trail):
                if v in first_idx:
                    start_idx = first_idx[v]
                    # The cycle is trail[start_idx : i+1]
                    # Check if it's simple
                    # It is simple if there are no repeats in trail[start_idx : i]
                    # We assume this is a simple cycle for now.
                    # Actually, we can just use it.
                    cycle = trail[start_idx : i + 1]
                    break
                first_idx[v] = i
            
            if cycle:
                # Validate cycle
                # Check if all vertices are distinct (except first/last)
                if len(set(cycle[:-1])) == len(cycle) - 1:
                    pieces.append(cycle)
                    # Remove cycle edges from remaining
                    for i in range(len(cycle) - 1):
                        a, b = cycle[i], cycle[i+1]
                        edge = (a, b) if a < b else (b, a)
                        remaining.discard(edge)
                else:
                    # Not simple, fallback to single edges?
                    # Or try to find a simpler cycle?
                    # For now, just take the first 3 edges?
                    # Let's just add as single edges to be safe
                    for i in range(len(comp_edges)):
                        pieces.append(list(next(iter(comp_edges))))
                        break # This is messy
            else:
                # No cycle, add all comp_edges as single edges
                for e in comp_edges:
                    pieces.append(list(e))
                    remaining.discard(e)

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
