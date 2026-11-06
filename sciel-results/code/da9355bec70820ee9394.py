import time
import sys
from collections import defaultdict
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges):
        if not edges:
            return []
        
        # Build adjacency and degree
        adj = defaultdict(list)
        deg = defaultdict(int)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
            
        # 1. Identify odd-degree vertices
        odd_vertices = [v for v in deg if deg[v] % 2 == 1]
        
        # 2. If no odd vertices, graph is Eulerian. One trail (cycle) covers all edges.
        #    We can break it into one path.
        #    If odd vertices exist, we need to pair them.
        
        # We will try to find a good matching of odd vertices.
        # The number of odd vertices is always even.
        
        # Heuristic: We want the "virtual edges" (matching pairs) to be "long" in terms of graph distance?
        # Or maybe "short"?
        # If we connect two odd vertices with a virtual edge, the Eulerian circuit will traverse
        # the path between them in the original graph as part of the trail.
        # Actually, the virtual edge is NOT in the graph. The Eulerian circuit in G + Virtual
        # will use the virtual edge. When we remove it, we split the circuit into trails.
        # The trails will start/end at the odd vertices.
        # The length of the trail depends on the structure.
        
        # Let's just use a simple greedy matching first, then try to improve.
        # Or, since n is small (<=30), we can try all perfect matchings if count is small?
        # Max odd vertices is 30. 29! is too big.
        
        # Strategy: 
        # 1. Generate a random matching.
        # 2. Compute the resulting trails.
        # 3. Calculate the cost: sum of (number of repeated vertices in each trail) + 1.
        # 4. Try to improve the matching by swapping pairs (2-opt on matching).
        
        def get_trails(matching_pairs):
            """
            Returns a list of trails (list of vertices) covering all edges.
            matching_pairs: list of (u, v) representing virtual edges.
            """
            # Build multigraph adjacency including virtual edges
            # We need to find an Eulerian circuit in G + Virtual.
            
            # Since we only add virtual edges to make degrees even, the multigraph is Eulerian.
            # We can use Hierholzer's algorithm.
            
            # adj_multi: dict of list of (neighbor, is_virtual)
            # But we don't want to output virtual edges.
            # We just want the sequence of original edges.
            
            # Let's construct the multigraph explicitly.
            # Edges in G are unique. Virtual edges are unique.
            # We need to track which edges are used.
            
            # For Hierholzer, we need to be able to remove edges.
            # Let's represent edges by index.
            
            orig_edges = list(edges)
            edge_index = {tuple(sorted(e)): i for i, e in enumerate(orig_edges)}
            
            # Build adjacency for Hierholzer
            # adj_h[v] = list of (neighbor, edge_id, is_virtual)
            adj_h = defaultdict(list)
            
            for i, (u, v) in enumerate(orig_edges):
                adj_h[u].append((v, i, False))
                adj_h[v].append((u, i, False))
                
            for u, v in matching_pairs:
                # Virtual edge
                adj_h[u].append((v, -1, True))
                adj_h[v].append((u, -1, True))
                
            # Hierholzer's Algorithm
            # Stack-based
            stack = []
            circuit = []
            
            # Start from any vertex with edges
            start_node = None
            for v in adj_h:
                if adj_h[v]:
                    start_node = v
                    break
            if start_node is None:
                return []
                
            stack.append(start_node)
            
            while stack:
                v = stack[-1]
                if not adj_h[v]:
                    circuit.append(v)
                    stack.pop()
                else:
                    # Pop an edge
                    nb, eid, is_virt = adj_h[v].pop()
                    # Remove the reverse edge
                    # Find the edge (v, nb) in adj_h[nb]
                    # This is O(deg^2) but n is small.
                    # To make it faster, we can use a dict of lists or just linear search.
                    # Given small n, linear search is fine.
                    
                    # Remove (nb, eid, is_virt) from adj_h[nb]
                    # Note: if eid != -1, the reverse edge has the same eid.
                    # If is_virt, the reverse edge is also virtual.
                    
                    found = False
                    for i, (n_nb, n_eid, n_is_virt) in enumerate(adj_h[nb]):
                        if n_eid == eid:
                            # Remove this specific edge
                            adj_h[nb].pop(i)
                            found = True
                            break
                    if not found:
                        # This shouldn't happen if we built it correctly
                        continue
                        
                    stack.append(nb)
                    
            # circuit is in reverse order
            circuit.reverse()
            
            # Now, circuit is a sequence of vertices.
            # We need to extract the trails.
            # The virtual edges connect the trails.
            # A trail starts after a virtual edge is "crossed" in the circuit?
            # Actually, the circuit is one big loop.
            # The virtual edges are "breaks".
            # Let's identify the positions of virtual edges in the circuit.
            
            # We need to map consecutive vertices in the circuit to edges.
            # If (u, v) is a virtual edge, it's a break.
            # If (u, v) is an original edge, it's part of a trail.
            
            trails = []
            current_trail = []
            
            # The circuit is a cycle.
            # We need to break it at virtual edges.
            
            # Let's iterate through the circuit edges.
            # circuit[i] to circuit[i+1] (mod len)
            
            L = len(circuit)
            if L == 0:
                return []
                
            # Identify which transitions are virtual.
            # We don't have the edge info directly in the circuit list of vertices.
            # We need to reconstruct which edge was used.
            # Since the graph is simple (except virtual), and virtual edges are distinct from original,
            # we can check if (u,v) is in the matching.
            
            # However, if there are multiple edges between u,v in original? No, simple graph.
            # So if (u,v) is in matching_pairs, it's virtual.
            # Else, it's original.
            
            # But wait, what if an original edge (u,v) exists AND a virtual edge (u,v) exists?
            # In that case, in the multigraph, there are two edges.
            # Our Hierholzer implementation above used `eid` to distinguish.
            # But in the `circuit` list, we only have vertices.
            # We lost the information of WHICH edge was traversed.
            
            # We need to modify Hierholzer to store the edges used, or just the type.
            
            # Let's redo Hierholzer to store the sequence of (u, v, is_virtual).
            
            # Reset adj_h
            adj_h = defaultdict(list)
            for i, (u, v) in enumerate(orig_edges):
                adj_h[u].append((v, i, False))
                adj_h[v].append((u, i, False))
                
            for u, v in matching_pairs:
                adj_h[u].append((v, -1, True))
                adj_h[v].append((u, -1, True))
                
            stack = [(start_node, None)] # (node, incoming_edge_info)
            circuit_edges = [] # list of (u, v, is_virtual)
            
            # Standard Hierholzer with edge tracking is tricky with stack of nodes.
            # Let's use the recursive version or a different iterative one.
            
            # Iterative with stack of (node, iterator_index) is complex.
            # Let's just use the vertex circuit and assume we can map back.
            # Since the graph is small, we can just check:
            # For each pair (circuit[i], circuit[i+1]), if it's in matching, it's virtual.
            # Is it possible that an original edge matches a virtual edge?
            # Yes. If we have original edge (1,2) and virtual edge (1,2).
            # Then in the circuit, the transition 1->2 could be either.
            # This ambiguity is a problem.
            
            # To avoid this, we can ensure that virtual edges are NOT in the original graph?
            # No, we can't change the graph.
            
            # Solution: When building adj_h, if we have both original and virtual,
            # we must track them separately.
            
            # Let's try a different approach for the decomposition:
            # We know the set of trails.
            # We can just find the Eulerian circuit in the multigraph by explicitly tracking edges.
            
            # Let's implement a simple DFS for Eulerian circuit that returns the edges.
            
            def find_eulerian_circuit():
                # Returns a list of (u, v, is_virtual)
                # We need to ensure we visit all edges.
                
                # adj_edge: dict v -> list of (nb, edge_id, is_virt)
                # We need to remove edges as we go.
                
                # To handle the ambiguity of parallel edges (orig + virt),
                # we use unique IDs.
                
                # Rebuild adj with unique IDs
                adj_e = defaultdict(list)
                edge_info = {} # id -> (u, v, is_virt)
                
                eid_counter = 0
                for u, v in orig_edges:
                    adj_e[u].append((v, eid_counter, False))
                    adj_e[v].append((u, eid_counter, False))
                    edge_info[eid_counter] = (u, v, False)
                    eid_counter += 1
                    
                for u, v in matching_pairs:
                    adj_e[u].append((v, eid_counter, True))
                    adj_e[v].append((u, eid_counter, True))
                    edge_info[eid_counter] = (u, v, True)
                    eid_counter += 1
                    
                # Hierholzer
                # stack of nodes
                # path of edges
                
                start = start_node
                stack = [start]
                edge_path = []
                
                while stack:
                    v = stack[-1]
                    if not adj_e[v]:
                        stack.pop()
                        if edge_path:
                            # The last edge added was leading to v?
                            # No, we add edges when we traverse.
                            pass
                    else:
                        nb, eid, is_virt = adj_e[v].pop()
                        # Remove reverse
                        # Find eid in adj_e[nb]
                        for i, (n_nb, n_eid, n_is_virt) in enumerate(adj_e[nb]):
                            if n_eid == eid:
                                adj_e[nb].pop(i)
                                break
                                
                        stack.append(nb)
                        edge_path.append((v, nb, is_virt, eid))
                        
                # edge_path is in reverse order?
                # In Hierholzer, the circuit is formed by popping from stack.
                # The edges are added to edge_path in the order they are traversed.
                # But the final circuit is the reverse of the stack pop order?
                # Actually, the standard algorithm builds the circuit in reverse.
                # Let's just reverse edge_path.
                
                edge_path.reverse()
                return edge_path
                
            circuit_edges = find_eulerian_circuit()
            
            # Now, split circuit_edges into trails based on virtual edges.
            # A virtual edge marks the end of one trail and start of another?
            # Actually, the virtual edges connect the odd vertices.
            # The circuit is: ... -> u --(virt)--> v -> ...
            # The segment from v to u (via original edges) is a trail?
            # No. The circuit is a loop.
            # If we remove virtual edges, we get disjoint paths/trails.
            # The virtual edges were the "bridges" that made it a loop.
            
            # So, we iterate through circuit_edges.
            # Whenever we encounter a virtual edge, we start a new trail.
            # The trail consists of the subsequent original edges until the next virtual edge.
            
            trails = []
            current_trail_vertices = []
            
            # The circuit is a cycle.
            # We need to handle the wrap-around.
            
            # Let's find the indices of virtual edges.
            virt_indices = [i for i, (u, v, is_v, eid) in enumerate(circuit_edges) if is_v]
            
            if not virt_indices:
                # No virtual edges, it's a single Eulerian cycle.
                # It's a closed trail. We can make it a path by dropping the last vertex (which is same as first).
                # Or just keep it as a cycle? The task asks for paths.
                # A cycle is a path if we consider it open? No, path usually implies distinct vertices?
                # "Simple path" usually implies distinct vertices.
                # A cycle is NOT a simple path if it has > 2 vertices.
                # So we must break the cycle.
                # We can just take the whole cycle as one path? No, endpoints must be distinct for a path?
                # Usually, a path is a sequence of vertices $v_1, \dots, v_k$ with $v_i \neq v_j$.
                # A cycle $v_1, \dots, v_k, v_1$ has $v_1 = v_k$.
                # So we should output $v_1, \dots, v_k$ as a path.
                
                verts = [circuit_edges[0][0]]
                for u, v, is_v, eid in circuit_edges:
                    verts.append(v)
                # verts[0] == verts[-1]
                trails.append(verts[:-1])
                return trails
                
            # If there are virtual edges:
            # The trails are the segments of original edges between virtual edges.
            
            # Let's extract the segments.
            # The virtual edges are at indices virt_indices.
            # The segment after virt_indices[i] and before virt_indices[i+1] is a trail.
            
            # Note: The circuit is a list of edges.
            # Edge i connects circuit_vertices[i] to circuit_vertices[i+1].
            
            # Let's reconstruct the vertex sequence for the whole circuit.
            circuit_verts = [circuit_edges[0][0]]
            for u, v, is_v, eid in circuit_edges:
                circuit_verts.append(v)
                
            # Now, for each interval between virtual edges:
            # The virtual edge connects circuit_verts[i] to circuit_verts[i+1].
            # The next original edges start from circuit_verts[i+1].
            
            # Let's iterate through the virtual edge indices.
            # We need to be careful with the circular nature.
            
            # Let's just iterate through the circuit edges and build trails.
            # We start a new trail when we see a virtual edge?
            # Actually, the virtual edge is the "connector".
            # The trail is the sequence of ORIGINAL edges.
            
            # Let's collect the original edges in order.
            # And note where the virtual edges are.
            
            # Since the graph is Eulerian in the multigraph, the sequence of original edges
            # forms trails that connect the odd vertices.
            
            # Let's just group the original edges.
            # We traverse the circuit.
            # When we hit a virtual edge, we "close" the current trail?
            # No, the virtual edge connects the end of one trail to the start of another.
            
            # Let's say the circuit is:
            # Orig Edges A -> Virt Edge 1 -> Orig Edges B -> Virt Edge 2 -> ...
            
            # Then Trail 1 is A, Trail 2 is B, etc.
            
            trails = []
            current_trail = []
            
            # We need to know the vertices of the trail.
            # Let's track the current vertex.
            
            # Start at the beginning of the circuit.
            # We don't know if the circuit starts with an original or virtual edge.
            
            # Let's just iterate and split.
            
            i = 0
            L = len(circuit_edges)
            
            # Find a starting point: after a virtual edge.
            # If the circuit starts with original edges, we might have a trail that wraps around.
            
            # Let's find the index of the first virtual edge.
            start_idx = -1
            for idx, (u, v, is_v, eid) in enumerate(circuit_edges):
                if is_v:
                    start_idx = idx
                    break
                    
            if start_idx == -1:
                # No virtual edges, handled above.
                pass
            else:
                # Start collecting after start_idx.
                # The edge at start_idx is virtual.
                # The next edge (start_idx + 1) % L is the start of a trail.
                
                # We will collect L edges.
                # But we only want original edges.
                
                # Let's just iterate L times from start_idx + 1.
                
                for step in range(1, L + 1):
                    idx = (start_idx + step) % L
                    u, v, is_v, eid = circuit_edges[idx]
                    
                    if is_v:
                        # End of current trail
                        if current_trail:
                            trails.append(current_trail)
                            current_trail = []
                    else:
                        # Original edge
                        if not current_trail:
                            current_trail = [u]
                        current_trail.append(v)
                        
                # If we ended on an original edge, we might have a trail left?
                # No, because we started after a virtual edge, and we went full circle.
                # The last edge before the first virtual edge (which is at start_idx)
                # must be an original edge?
                # The sequence is: ... Orig -> Virt -> Orig ...
                # So yes, the segment before the first virt edge is a trail.
                # Did we capture it?
                # We iterated from start_idx+1 to start_idx (inclusive of the one before start_idx).
                # The last edge processed is at index start_idx - 1 (mod L).
                # If that edge was original, it was added to current_trail.
                # But we didn't append current_trail to trails if it wasn't closed by a virtual edge?
                # Wait, the loop ends.
                # The trail that ends at the vertex BEFORE the virtual edge at start_idx
                # is the one we just built.
                # It is NOT closed by a virtual edge in our loop (because the virtual edge
                # that closes it is at start_idx, which we skipped as the "start").
                # So we need to append it.
                
                if current_trail:
                    trails.append(current_trail)
                    current_trail = []
                    
            # Now we have a list of trails (lists of vertices).
            # Each trail uses original edges.
            # Are they simple?
            # No, they might have repeated vertices.
            
            # We need to convert these trails into simple paths.
            # We will do this in a separate step.
            
            return trails
            
        def decompose_trails_into_simple_paths(trails):
            """
            Takes a list of trails (vertex lists) and returns a list of simple paths.
            """
            simple_paths = []
            for trail in trails:
                # If trail is empty, skip
                if not trail:
                    continue
                    
                # If trail has no repeated vertices, it's a simple path.
                if len(trail) == len(set(trail)):
                    simple_paths.append(trail)
                    continue
                    
                # Otherwise, we need to split it.
                # We can use a greedy approach:
                # Find the first repeated vertex.
                # Split the trail into two parts:
                # 1. The cycle (from first occurrence to second occurrence).
                # 2. The remaining path.
                # Then recursively decompose the cycle and the remaining path.
                # A cycle can be made into a simple path by removing one vertex (the repeated one)?
                # No, a cycle $v_1, \dots, v_k, v_1$ can be made into a path $v_1, \dots, v_k$.
                
                # Let's implement a simple splitter.
                
                def split_trail(t):
                    # Returns a list of simple paths
                    if not t:
                        return []
                    
                    # Check if simple
                    if len(t) == len(set(t)):
                        return [t]
                    
                    # Find first repeated vertex
                    seen = {}
                    repeat_idx = -1
                    repeat_vert = None
                    for i, v in enumerate(t):
                        if v in seen:
                            repeat_idx = i
                            repeat_vert = v
                            break
                        seen[v] = i
                        
                    if repeat_idx == -1:
                        return [t] # Should be simple
                    
                    # The cycle is t[seen[repeat_vert] : repeat_idx + 1]
                    # The path before is t[:seen[repeat_vert]]
                    # The path after is t[repeat_idx:]
                    
                    # Actually, the segment t[seen[repeat_vert] : repeat_idx + 1] is a cycle.
                    # We can convert this cycle to a simple path by dropping the last vertex (which is the same as the first).
                    # Cycle: [A, B, C, A] -> Path: [A, B, C]
                    
                    start_idx = seen[repeat_vert]
                    cycle_part = t[start_idx : repeat_idx + 1]
                    path_part_after = t[repeat_idx:]
                    path_part_before = t[:start_idx]
                    
                    # Wait, the trail is continuous.
                    # t = P_before + Cycle + P_after?
                    # No.
                    # t = [v0, v1, ..., vk, ..., vm]
                    # If vk is repeated, and it first appeared at vi.
                    # Then t[vi...vk] is a cycle.
                    # The rest is t[0...vi] and t[vk...m].
                    # But t[0...vi] ends at vi. t[vk...m] starts at vk.
                    # Since vi and vk are the same vertex, we can merge?
                    # No, the edges are distinct.
                    
                    # Let's just extract the cycle as a path.
                    # Cycle Path: cycle_part[:-1]
                    # Remaining Trail: path_part_before + path_part_after?
                    # No, the edges in the cycle are used.
                    # The remaining edges are in path_part_before (excluding the last edge to vi?)
                    # and path_part_after (excluding the first edge from vk?).
                    
                    # This is getting complicated.
                    # Simpler: Just output the cycle as a path and the rest as another trail,
                    # and recursively process the rest.
                    
                    # But we must cover all edges.
                    # The cycle uses edges (vi, vi+1) ... (vk-1, vk).
                    # The remaining edges are (v0, v1) ... (vi-1, vi) and (vk, vk+1) ... (vm-1, vm).
                    # Note that vi == vk.
                    # So the remaining graph is two paths sharing a vertex vi.
                    # We can treat them as two separate trails?
                    # Yes, because they are edge-disjoint.
                    
                    # So:
                    # 1. Simple path from cycle: cycle_part[:-1]
                    # 2. Trail 1: path_part_before (ends at vi)
                    # 3. Trail 2: path_part_after (starts at vi)
                    
                    # Recursively decompose Trail 1 and Trail 2.
                    
                    res = [cycle_part[:-1]]
                    res.extend(split_trail(path_part_before))
                    res.extend(split_trail(path_part_after))
                    
                    return res
                    
                simple_paths.extend(split_trail(trail))
                
            return simple_paths
            
        # Main Logic:
        
        # 1. Get initial matching
        # Random matching of odd vertices
        rng = Random(42)
        
        def random_matching(verts):
            verts = list(verts)
            rng.shuffle(verts)
            pairs = []
            for i in range(0, len(verts), 2):
                pairs.append((verts[i], verts[i+1]))
            return pairs
            
        # 2. Local Search on Matching
        # We want to minimize the number of simple paths.
        # Cost = number of simple paths.
        
        best_cost = float('inf')
        best_paths = None
        best_matching = None
        
        # Initial solution
        init_matching = random_matching(odd_vertices)
        trails = get_trails(init_matching)
        simple_paths = decompose_trails_into_simple_paths(trails)
        curr_cost = len(simple_paths)
        curr_matching = init_matching
        
        if curr_cost < best_cost:
            best_cost = curr_cost
            best_paths = simple_paths
            best_matching = curr_matching
            
        # Local Search
        # Time limit per graph: 30s?
        # Total deadline 145s. 3 graphs. ~45s each.
        
        import time
        start_search = time.time()
        
        while time.time() - start_search < 40:
            # Try to improve matching by swapping two pairs
            if len(curr_matching) < 2:
                break
                
            i, j = rng.sample(range(len(curr_matching)), 2)
            u1, v1 = curr_matching[i]
            u2, v2 = curr_matching[j]
            
            # Try swapping to (u1, u2), (v1, v2)
            new_matching = list(curr_matching)
            new_matching[i] = (u1, u2)
            new_matching[j] = (v1, v2)
            
            trails = get_trails(new_matching)
            simple_paths = decompose_trails_into_simple_paths(trails)
            new_cost = len(simple_paths)
            
            if new_cost < curr_cost:
                curr_cost = new_cost
                curr_matching = new_matching
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_paths = simple_paths
                    best_matching = curr_matching
                    
            # Also try (u1, v2), (v1, u2)
            new_matching2 = list(curr_matching)
            new_matching2[i] = (u1, v2)
            new_matching2[j] = (v1, u2)
            
            trails = get_trails(new_matching2)
            simple_paths = decompose_trails_into_simple_paths(trails)
            new_cost2 = len(simple_paths)
            
            if new_cost2 < curr_cost:
                curr_cost = new_cost2
                curr_matching = new_matching2
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_paths = simple_paths
                    best_matching = curr_matching
                    
        return best_paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            break
        edges = instance(*spec)
        paths = decompose_graph(edges)
        results.append(paths)
    
    # If we ran out of time, fill remaining with trivial edge paths
    while len(results) < 3:
        spec = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])
    
    return results
