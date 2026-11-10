"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

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
        
        # Build adjacency list
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(edges)
        pieces = []
        
        # Helper to extract cycles from an Eulerian subgraph (all degrees even)
        def extract_cycles_from_eulerian(graph_edges):
            if not graph_edges:
                return []
            
            # Build local adjacency for this subgraph
            local_adj = defaultdict(list)
            for u, v in graph_edges:
                local_adj[u].append(v)
                local_adj[v].append(u)
                
            cycles = []
            used_edges = set()
            
            # Hierholzer's algorithm
            # We need to find all edge-disjoint cycles that partition the graph
            # Since it's Eulerian, we can just run Hierholzer on the whole component
            # But wait, Hierholzer finds ONE Eulerian circuit.
            # If the graph is not connected, we need to handle components.
            
            # Let's find connected components in the subgraph
            visited_nodes = set()
            for start_node in list(local_adj.keys()):
                if start_node in visited_nodes:
                    continue
                
                # BFS to find component nodes
                component_nodes = []
                q = deque([start_node])
                visited_nodes.add(start_node)
                while q:
                    node = q.popleft()
                    component_nodes.append(node)
                    for nb in local_adj[node]:
                        if nb not in visited_nodes:
                            visited_nodes.add(nb)
                            q.append(nb)
                
                # Now run Hierholzer on this component
                # We need to be careful with edge usage
                # Let's use a stack-based Hierholzer
                stack = [start_node]
                circuit = []
                
                # To ensure we don't get stuck in infinite loops or miss edges,
                # we track used edges. But Hierholzer naturally uses each edge once
                # if we remove them as we go.
                
                # Actually, simpler approach for Eulerian:
                # Just find the circuit.
                
                # Reset local state for this component
                # We need to be able to pick an unused edge.
                # Let's maintain a set of available edges for this component
                comp_edges = set()
                for u, v in graph_edges:
                    if u in visited_nodes and v in visited_nodes:
                        # Ensure both are in this specific component
                        # Since we just built visited_nodes, this check is a bit loose
                        # if there are multiple components. 
                        # Better: filter edges where both endpoints are in component_nodes
                        pass
                
                # Re-do component edge filtering properly
                comp_node_set = set(component_nodes)
                comp_edges = {e for e in graph_edges if e[0] in comp_node_set and e[1] in comp_node_set}
                
                if not comp_edges:
                    continue

                # Build adjacency for this component only
                c_adj = defaultdict(list)
                for u, v in comp_edges:
                    c_adj[u].append(v)
                    c_adj[v].append(u)
                
                # Hierholzer
                stack = [start_node]
                circuit = []
                
                while stack:
                    u = stack[-1]
                    # Find an unused edge
                    # We need to efficiently find an unused neighbor.
                    # Since n is small, linear scan is fine.
                    found_next = False
                    for i, v in enumerate(c_adj[u]):
                        # Check if edge (u,v) is used
                        # We can't easily mark "used" in the list without removal.
                        # Let's pop from the list.
                        if (min(u,v), max(u,v)) not in used_edges:
                            # Use this edge
                            used_edges.add((min(u,v), max(u,v)))
                            # Remove from adjacency lists to avoid re-checking
                            # This is O(deg) which is fine.
                            c_adj[u].remove(v)
                            c_adj[v].remove(u)
                            stack.append(v)
                            found_next = True
                            break
                    if not found_next:
                        circuit.append(stack.pop())
                
                # The circuit is in reverse order
                circuit.reverse()
                # Remove the repeated start node at the end if present
                # Hierholzer typically returns a list where the first and last are the same?
                # No, standard implementation:
                # stack: [start]
                # if no edge, pop to circuit.
                # So circuit ends with start.
                # Example: 1-2-3-1.
                # Stack: [1] -> [1,2] -> [1,2,3] -> [1,2,3,1]
                # Pop 1 -> circuit [1]
                # Pop 3 -> circuit [1,3]
                # Pop 2 -> circuit [1,3,2]
                # Pop 1 -> circuit [1,3,2,1]
                # So yes, it closes the loop.
                
                if len(circuit) > 2: # At least a triangle (3 nodes + 1 close) -> len 4 in this list format?
                    # Wait, the piece format is [v1, v2, ... vk] where (vk, v1) is the closing edge.
                    # The circuit list from Hierholzer is [v1, ..., vk, v1].
                    # We should strip the last element to match the format "closing edge is implied".
                    # So if circuit is [1, 2, 3, 1], the piece is [1, 2, 3].
                    if len(circuit) > 1:
                        piece = circuit[:-1]
                        # Ensure distinct vertices?
                        # Hierholzer on a simple graph produces a trail.
                        # If the graph has multiple edges, it might repeat nodes.
                        # But the input is a simple graph (set of edges).
                        # So no multi-edges.
                        # However, a node can appear multiple times in the circuit if it's a "figure-8" shape?
                        # No, an Eulerian circuit visits each edge exactly once.
                        # It can visit a node multiple times.
                        # The problem statement says: "a cycle given by at least three distinct vertices".
                        # This implies a simple cycle (no repeated vertices).
                        # Hierholzer's circuit is NOT necessarily a simple cycle. It's an Eulerian circuit.
                        # An Eulerian circuit can be decomposed into simple cycles.
                        # But we want to MINIMIZE pieces.
                        # If we output the entire Eulerian circuit as one "piece", is it valid?
                        # "A piece is ... a cycle given by at least three distinct vertices".
                        # Usually, a "cycle" in graph theory context for decomposition implies a simple cycle.
                        # If I provide [1, 2, 3, 2, 4, 1], vertices 2 is repeated.
                        # Is that a valid "cycle" piece?
                        # The prompt says: "a cycle given by at least three distinct vertices (the closing edge is implied)."
                        # This phrasing is slightly ambiguous. Does it mean the list of vertices must contain distinct elements?
                        # "at least three distinct vertices" usually refers to the set of vertices in the cycle.
                        # If I have a bowtie graph (two triangles sharing a vertex), an Eulerian circuit is 1-2-3-1-4-5-1.
                        # Vertices: {1,2,3,4,5}. Distinct count is 5.
                        # But the sequence has 1 repeated.
                        # If the verifier checks `len(set(piece)) == len(piece)`, then this fails.
                        # If the verifier checks that the edges form a cycle in the graph, it might pass.
                        # However, "cycle" strongly suggests a simple cycle.
                        # Let's assume the verifier requires simple cycles (no repeated vertices in the list).
                        # Therefore, we MUST decompose the Eulerian circuit into simple cycles.
                        
                        # How to decompose an Eulerian circuit into simple cycles?
                        # We can traverse the circuit and break it whenever we revisit a node.
                        
                        simple_cycles = []
                        current_cycle = [circuit[0]]
                        seen_in_cycle = {circuit[0]}
                        
                        for i in range(1, len(circuit)):
                            v = circuit[i]
                            if v in seen_in_cycle:
                                # We found a simple cycle: current_cycle + [v] (closing to start of this sub-cycle?)
                                # Wait, the circuit is a sequence.
                                # If we see a repeated node, the segment between the two occurrences is a simple cycle.
                                # Let's say circuit is 1, 2, 3, 1, 4, 5, 1.
                                # i=3, v=1. 1 is in seen_in_cycle (from index 0).
                                # The segment 1, 2, 3, 1 is a cycle.
                                # Piece: [1, 2, 3].
                                simple_cycles.append(current_cycle[:-1]) # [1, 2, 3]
                                # Reset for next cycle starting at v (which is 1)
                                current_cycle = [v]
                                seen_in_cycle = {v}
                            else:
                                current_cycle.append(v)
                                seen_in_cycle.add(v)
                        
                        # The last part
                        if len(current_cycle) > 1:
                            # The last node in circuit is the same as the first node of the whole circuit?
                            # Yes, Eulerian circuit closes.
                            # So the last segment closes back to the start of the segment.
                            simple_cycles.append(current_cycle[:-1])
                        
                        pieces.extend(simple_cycles)
                        # Remove these edges from remaining_edges
                        for cyc in simple_cycles:
                            for i in range(len(cyc)):
                                u = cyc[i]
                                v = cyc[(i + 1) % len(cyc)]
                                remaining_edges.discard((min(u, v), max(u, v)))
            return pieces

        # Main Loop
        while remaining_edges:
            if time.time() > deadline:
                # Fallback: add all remaining edges as singletons
                for u, v in remaining_edges:
                    pieces.append([u, v])
                break
            
            # 1. Find connected components in the remaining graph
            visited = set()
            components = []
            
            for u in list(remaining_edges.keys() if hasattr(remaining_edges, 'keys') else [e[0] for e in remaining_edges]):
                if u in visited:
                    continue
                # BFS
                comp_nodes = set()
                q = deque([u])
                visited.add(u)
                while q:
                    node = q.popleft()
                    comp_nodes.add(node)
                    for nb in adj[node]:
                        if (min(node, nb), max(node, nb)) in remaining_edges:
                            if nb not in visited:
                                visited.add(nb)
                                q.append(nb)
                
                if len(comp_nodes) > 1:
                    components.append(comp_nodes)
            
            if not components:
                # No edges left? Or isolated nodes?
                # If remaining_edges is not empty, there must be edges.
                # If components is empty, maybe we missed something?
                # Let's just add all remaining edges as singletons to be safe.
                for u, v in remaining_edges:
                    pieces.append([u, v])
                break
            
            # 2. For each component, try to find a large Eulerian subgraph
            # We want to maximize the number of edges in the Eulerian subgraph.
            # Heuristic: 
            # - Start with the component.
            # - Find nodes with odd degree.
            # - The minimum number of edges to remove to make all degrees even is related to the pairing of odd-degree nodes.
            # - A simpler heuristic: 
            #   a. Find the "core" by iteratively removing leaves (degree 1).
            #   b. The remaining graph has min degree >= 2. It might still have odd degrees.
            #   c. If min degree >= 2, it contains a cycle.
            #   d. To make it Eulerian, we need to remove edges to fix odd degrees.
            
            # Let's try a simpler, more robust approach:
            # Just find ANY large cycle? No, that's the old strategy.
            # Let's try to extract a maximal Eulerian subgraph.
            
            # Strategy:
            # 1. Identify all nodes in the component.
            # 2. Calculate degrees within the component.
            # 3. If all degrees are even, the whole component is Eulerian. Extract it.
            # 4. If not, we need to remove edges.
            #    Removing an edge (u,v) flips the parity of deg(u) and deg(v).
            #    We want to remove a minimum set of edges such that all degrees become even.
            #    This is equivalent to finding a T-join where T is the set of odd-degree nodes.
            #    For small graphs, we can try to pair up odd nodes and remove shortest paths between them?
            #    Actually, removing a path between two odd nodes makes them even (and intermediate nodes flip twice, so stay even).
            #    So, we can pair up odd nodes and remove the edges along the shortest path between each pair.
            
            # Let's implement this pairing heuristic.
            
            best_subgraph_edges = set()
            best_cycle_count = 0
            
            for comp in components:
                # Build subgraph
                comp_edges = set()
                for u, v in remaining_edges:
                    if u in comp and v in comp:
                        comp_edges.add((u, v))
                
                if not comp_edges:
                    continue
                
                # Calculate degrees
                deg = {u: 0 for u in comp}
                for u, v in comp_edges:
                    deg[u] += 1
                    deg[v] += 1
                
                odd_nodes = [u for u in comp if deg[u] % 2 == 1]
                
                if not odd_nodes:
                    # Already Eulerian
                    best_subgraph_edges = comp_edges
                    break # This is perfect, use it
                else:
                    # Need to remove edges to fix odd nodes.
                    # Pair up odd nodes.
                    # To minimize removed edges, we should remove short paths.
                    # Let's find shortest paths between all pairs of odd nodes?
                    # That's O(K^2 * (V+E)) where K is number of odd nodes.
                    # K is small (<= V).
                    
                    # Let's just greedily pair them? Or find minimum weight perfect matching on odd nodes?
                    # Given time constraints, let's try a simple heuristic:
                    # Pair up odd nodes arbitrarily and remove the direct edge if it exists, 
                    # or a short path.
                    
                    # Actually, a simpler approach:
                    # Just find a large cycle and remove it?
                    # No, let's stick to the Eulerian extraction.
                    
                    # Let's try to find a set of edges to remove.
                    # We can model this as: we want to keep a subset of edges that is Eulerian.
                    # This is equivalent to finding a maximum Eulerian subgraph.
                    # 
                    # Heuristic:
                    # 1. Start with all edges.
                    # 2. While there are odd nodes:
                    #    a. Pick two odd nodes u, v.
                    #    b. Find a path between u and v in the current graph.
                    #    c. Remove the edges of this path.
                    #    d. This makes u and v even. Intermediate nodes flip twice (even).
                    #    e. Other nodes unchanged.
                    # 
                    # To maximize remaining edges, we should remove SHORT paths.
                    
                    current_edges = set(comp_edges)
                    current_deg = dict(deg)
                    current_odd = set(odd_nodes)
                    
                    # Build adjacency for current_edges
                    curr_adj = defaultdict(list)
                    for u, v in current_edges:
                        curr_adj[u].append(v)
                        curr_adj[v].append(u)
                    
                    # While there are odd nodes
                    while current_odd:
                        if time.time() > deadline:
                            break
                            
                        # Pick two odd nodes
                        # To be efficient, pick two that are close?
                        # Let's just pick the first two.
                        u = next(iter(current_odd))
                        current_odd.remove(u)
                        v = next(iter(current_odd))
                        current_odd.remove(v)
                        
                        # Find shortest path between u and v
                        # BFS
                        prev = {u: None}
                        q = deque([u])
                        found = False
                        while q:
                            node = q.popleft()
                            if node == v:
                                found = True
                                break
                            for nb in curr_adj[node]:
                                if nb not in prev and (min(node, nb), max(node, nb)) in current_edges:
                                    prev[nb] = node
                                    q.append(nb)
                        
                        if not found:
                            # Should not happen if connected
                            break
                            
                        # Reconstruct path
                        path = []
                        curr = v
                        while curr is not None:
                            path.append(curr)
                            curr = prev[curr]
                        path.reverse()
                        
                        # Remove edges along path
                        for i in range(len(path) - 1):
                            e = (min(path[i], path[i+1]), max(path[i], path[i+1]))
                            if e in current_edges:
                                current_edges.remove(e)
                                curr_adj[path[i]].remove(path[i+1])
                                curr_adj[path[i+1]].remove(path[i])
                                # Update degrees
                                current_deg[path[i]] -= 1
                                current_deg[path[i+1]] -= 1
                                # Update odd set
                                # This logic is getting complex. 
                                # Let's just recalculate odd set after each removal? 
                                # Or just check at the end?
                                # Since we are removing a path, u and v become even.
                                # Intermediate nodes become even (flip twice).
                                # So we can just remove u and v from current_odd.
                                # But what if u or v was already even? No, we picked them from current_odd.
                                # What if an intermediate node was odd? It becomes even.
                                # What if an intermediate node was even? It becomes odd?
                                # Wait. 
                                # Path: u - a - b - v.
                                # deg(u) flips. deg(v) flips.
                                # deg(a) flips (edge u-a removed) and flips (edge a-b removed)? 
                                # No, we remove the ENTIRE path.
                                # So deg(a) decreases by 2. Parity unchanged.
                                # So only u and v change parity.
                                # So current_odd update: remove u and v.
                                # Add any node that became odd? No, intermediate nodes keep parity.
                                # So just remove u and v.
                        
                        # Update current_odd
                        # We already removed u and v from the set.
                        # But we need to check if any other node became odd?
                        # No, as argued above, only u and v change parity.
                        # So the set of odd nodes shrinks by 2.
                    
                    # Now current_edges should be Eulerian (or empty)
                    if len(current_edges) > len(best_subgraph_edges):
                        best_subgraph_edges = current_edges
                        
            if best_subgraph_edges:
                # Extract cycles from this Eulerian subgraph
                extract_cycles_from_eulerian(best_subgraph_edges)
            else:
                # No good Eulerian subgraph found?
                # Just take any cycle?
                # Or add edges as singletons?
                # If we can't find an Eulerian subgraph, it means the component is a tree?
                # But a tree has no cycles.
                # If the component has a cycle, we should have found one.
                # Maybe the heuristic failed?
                # Fallback: find any cycle.
                found_cycle = False
                for u in comp:
                    # DFS to find cycle
                    visited_local = set()
                    stack = [(u, iter([nb for nb in adj[u] if (min(u, nb), max(u, nb)) in remaining_edges]))]
                    parent = {u: None}
                    path = [u]
                    
                    while stack:
                        node, neighbors = stack[-1]
                        found_next = False
                        for nb in neighbors:
                            if (min(node, nb), max(node, nb)) not in remaining_edges:
                                continue
                            if nb not in visited_local:
                                visited_local.add(nb)
                                parent[nb] = node
                                path.append(nb)
                                stack.append((nb, iter([n2 for n2 in adj[nb] if (min(nb, n2), max(nb, n2)) in remaining_edges])))
                                found_next = True
                                break
                            elif nb != parent[node]:
                                # Cycle found
                                idx = path.index(nb)
                                cycle = path[idx:]
                                pieces.append(cycle)
                                for i in range(len(cycle)):
                                    eu = cycle[i]
                                    ev = cycle[(i + 1) % len(cycle)]
                                    remaining_edges.discard((min(eu, ev), max(eu, ev)))
                                found_cycle = True
                                break
                        if found_cycle:
                            break
                        if not found_next:
                            stack.pop()
                            if path:
                                path.pop()
                    if found_cycle:
                        break
                if not found_cycle:
                    # No cycle in this component. It's a tree.
                    for u, v in remaining_edges:
                        if u in comp and v in comp:
                            pieces.append([u, v])
                            remaining_edges.discard((u, v))

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
        
    return results
# EVOLVE-BLOCK-END
