import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    graphs = [instance(*spec) for spec in specs]

    def decompose_graph(edges):
        if not edges:
            return []
        
        # Build adjacency list with edge IDs
        # We need to track edges to remove them from the trail
        edge_map = {} # (u, v) -> (id, u, v)
        adj = defaultdict(list) # u -> [(v, edge_id), ...]
        
        for i, (u, v) in enumerate(edges):
            adj[u].append((v, i))
            adj[v].append((u, i))
            edge_map[(u, v)] = (i, u, v)
            edge_map[(v, u)] = (i, u, v)

        # Find connected components
        visited_nodes = set()
        components = []
        
        for node in adj:
            if node in visited_nodes:
                continue
            # BFS to find component
            comp_nodes = []
            queue = deque([node])
            visited_nodes.add(node)
            while queue:
                u = queue.popleft()
                comp_nodes.append(u)
                for v, _ in adj[u]:
                    if v not in visited_nodes:
                        visited_nodes.add(v)
                        queue.append(v)
            components.append(comp_nodes)

        pieces = []
        
        for comp in components:
            if not comp:
                continue
            
            # Check if component is a tree (Eulerian trail exists if 0 or 2 odd degree vertices)
            # For Chinese Postman to produce a single closed tour (Eulerian circuit),
            # all vertices must have even degree.
            # If there are odd degree vertices, we must add "virtual" edges to make them even.
            # However, we are decomposing the *existing* edges.
            # A graph has an Eulerian circuit iff all vertices have even degree and it is connected.
            # If not all even, we cannot have a single Eulerian circuit covering all edges exactly once.
            # We must decompose into paths/cycles.
            
            # Actually, the problem allows "single edge" or "cycle".
            # If we just find an Eulerian trail in a component that has 2 odd vertices, 
            # the trail starts at one odd and ends at the other.
            # The trail is a sequence of edges. 
            # Can we partition an Eulerian trail into cycles?
            # Not directly if it's not closed.
            
            # Let's look at the degrees.
            # If all degrees are even, we have an Eulerian circuit.
            # An Eulerian circuit is a closed walk that visits every edge exactly once.
            # It can be decomposed into edge-disjoint cycles.
            # How many cycles? 
            # If the circuit is a single simple cycle, 1 piece.
            # If it self-intersects, we can split it.
            # Standard result: Any Eulerian graph can be decomposed into cycles.
            # The number of cycles in a decomposition can be as low as 1 (if the graph is a single cycle) 
            # or higher.
            # Specifically, if we have an Eulerian circuit of length L, and it visits a vertex v 
            # multiple times, we can split the circuit at those occurrences.
            
            # Strategy:
            # 1. If component has all even degrees:
            #    Find Eulerian circuit.
            #    The circuit is a list of vertices [v0, v1, ..., vk, v0].
            #    We can output this as a single "cycle" piece ONLY IF all vertices in the circuit are distinct.
            #    If vertices repeat, we must split.
            #    Splitting an Eulerian circuit into simple cycles:
            #    While the current path is not a simple cycle (has repeated vertices),
            #    find the first repeated vertex, split the path into a cycle and a remaining path.
            #    The remaining path is still a valid trail (starts/ends at the split point) but might not be closed?
            #    Wait. An Eulerian circuit is closed. If we split at a repeated vertex v, 
            #    we get a cycle C (v...v) and a remaining walk W (v...v).
            #    W is also a closed walk.
            #    So we can recursively decompose W.
            #    This yields a set of edge-disjoint simple cycles covering all edges.
            
            # 2. If component has odd degrees:
            #    It cannot have an Eulerian circuit.
            #    We can find an Eulerian trail (starts at odd u, ends at odd v).
            #    This trail is not closed.
            #    We can add a "virtual" edge (v, u) to make it a circuit, decompose into cycles,
            #    and then handle the virtual edge.
            #    The cycle containing the virtual edge will become a path.
            #    A path can be decomposed into... well, a path of length 1 is an edge.
            #    A path of length > 1 is not a cycle or a single edge.
            #    BUT, we are allowed to output "single edge" pieces.
            #    So if we have a path u-v-w, we can output [u,v] and [v,w] as two edge pieces?
            #    Or is there a better way?
            #    Actually, if we have an Eulerian trail, we can just output the edges along the trail?
            #    No, we want to minimize pieces.
            #    If we have a path of length L, we can cover it with L single-edge pieces.
            #    Or if we can find cycles within the path? No, a path has no cycles.
            #    So for the "odd" part, we are stuck with edge pieces for the path segments.
            
            # Let's refine:
            # Count odd degree vertices in component.
            odd_vertices = [v for v in comp if len(adj[v]) % 2 != 0]
            
            if len(odd_vertices) == 0:
                # Eulerian circuit exists
                # Find circuit
                circuit = find_eulerian_circuit(comp, adj, edge_map)
                # Decompose circuit into simple cycles
                cycles = decompose_circuit_into_cycles(circuit)
                pieces.extend(cycles)
            else:
                # There are 2, 4, ... odd vertices.
                # We can pair them up and add virtual edges to make all degrees even.
                # Then find Eulerian circuit in the augmented graph.
                # Decompose into cycles.
                # The cycles that contain virtual edges become paths.
                # We convert these paths into single-edge pieces.
                
                # To minimize pieces, we want to minimize the number of virtual edges?
                # Each virtual edge creates one path in the final decomposition (the cycle containing it).
                # That path will be broken into single edges.
                # If we have k odd vertices, we need k/2 virtual edges.
                # This is fixed by the graph structure.
                
                # Pair odd vertices arbitrarily?
                # Better to pair them such that the virtual edges are "short" or "useful"?
                # Actually, the number of single-edge pieces will be the sum of lengths of the paths formed by virtual edges.
                # Wait, if a cycle contains a virtual edge (u,v), the rest of the cycle is a path from v to u.
                # If that path has length L, we output L single edges.
                # Total pieces = (number of cycles without virtual edges) + (sum of lengths of paths from virtual edges).
                
                # Let's just do it.
                # 1. Create virtual edges
                virtual_edges = []
                for i in range(0, len(odd_vertices), 2):
                    u = odd_vertices[i]
                    v = odd_vertices[i+1]
                    virtual_edges.append((u, v))
                    # Add to adj temporarily? Or handle in circuit finding?
                    # Easier to add to adj and edge_map with negative IDs?
                    # Let's use a separate structure or just modify adj and restore.
                    
                # Since we need to be fast and safe, let's just find an Eulerian trail directly?
                # Hierholzer's algorithm works for Eulerian trails too.
                # If we find a trail, it's a sequence of edges.
                # It starts at u, ends at v.
                # We can output the edges? No, we want cycles.
                # But a trail is just a path.
                # Can we decompose a trail into cycles? No.
                # So if the graph has odd degrees, we MUST output some single edges.
                
                # Strategy for odd components:
                # Find an Eulerian trail.
                # The trail is a path P.
                # We can output all edges in P as single-edge pieces.
                # This gives |E| pieces for this component.
                # Is there a better way?
                # We could find cycles in the graph first, remove them, and then the remaining edges form a path (or disjoint paths).
                # Since the graph is connected and has 2 odd vertices, removing any cycle leaves a connected graph with the same odd vertices (or 0 if the cycle included them? No, removing a cycle changes degrees by 2, so parity is preserved).
                # So we can always extract cycles until we are left with a path connecting the two odd vertices.
                # This path must be output as single edges.
                # This is equivalent to the virtual edge method.
                
                # So, algorithm for odd component:
                # 1. Find all cycles (using Hierholzer or DFS) and output them.
                # 2. The remaining edges form a path (or forest of paths, but since connected and 2 odds, it's a single path).
                # 3. Output each edge in the remaining path as a single-edge piece.
                
                # How to do this efficiently?
                # Just run Hierholzer's to find an Eulerian trail.
                # The trail visits all edges.
                # If the graph is Eulerian (all even), the trail is a circuit.
                # If not, it's a path.
                # We can decompose the trail into cycles + one path?
                # Yes. Hierholzer's algorithm naturally builds cycles.
                # If we modify it to track the "main" path, we can separate the cycles from the final path.
                
                # Let's implement a unified Hierholzer that returns a list of cycles and a final path.
                # Actually, it's easier to just find the Eulerian trail/circuit and then decompose it.
                
                # Let's stick to the "Virtual Edge" approach for simplicity and correctness.
                # 1. Add virtual edges to make all degrees even.
                # 2. Find Eulerian circuit.
                # 3. Decompose into simple cycles.
                # 4. For each cycle, if it contains a virtual edge, remove that edge and output the remaining path as single edges.
                #    If it doesn't, output the cycle.
                
                # Implementation detail:
                # We need to know which edges are virtual.
                
                # Let's modify adj and edge_map to include virtual edges.
                # Virtual edge IDs will be negative.
                
                orig_adj = {u: list(vs) for u, vs in adj.items()}
                orig_edge_map = dict(edge_map)
                
                next_id = -1
                for u, v in virtual_edges:
                    eid = next_id
                    next_id -= 1
                    adj[u].append((v, eid))
                    adj[v].append((u, eid))
                    edge_map[(u, v)] = (eid, u, v, True) # is_virtual
                    edge_map[(v, u)] = (eid, u, v, True)
                    
                # Find Eulerian circuit
                circuit = find_eulerian_circuit(comp, adj, edge_map)
                
                # Restore adj and edge_map
                adj = orig_adj
                edge_map = orig_edge_map
                
                # Decompose circuit into simple cycles
                # But we need to know if a cycle contains a virtual edge.
                # The circuit is a list of vertices.
                # We need to know the edges used.
                # find_eulerian_circuit should return the list of edges (or vertex sequence with edge IDs).
                
                # Let's re-implement find_eulerian_circuit to return edge sequence.
                # Or vertex sequence, and we can look up edges.
                # Since we restored edge_map, we can look up (u, v) in orig_edge_map.
                # But virtual edges are not in orig_edge_map.
                # So if (u, v) is not in orig_edge_map, it's a virtual edge.
                
                cycles = decompose_circuit_into_cycles(circuit)
                for cycle in cycles:
                    # Check if cycle contains virtual edge
                    has_virtual = False
                    for i in range(len(cycle) - 1):
                        u, v = cycle[i], cycle[i+1]
                        if (u, v) not in edge_map:
                            has_virtual = True
                            break
                        if (v, u) not in edge_map:
                            has_virtual = True
                            break
                    if not has_virtual:
                        # Check closing edge
                        u, v = cycle[-1], cycle[0]
                        if (u, v) not in edge_map:
                            has_virtual = True
                        if (v, u) not in edge_map:
                            has_virtual = True
                    
                    if has_virtual:
                        # This cycle contains a virtual edge.
                        # The real edges form a path.
                        # We need to extract the path of real edges.
                        # The cycle is [v0, v1, ..., vk, v0].
                        # One of the edges is virtual.
                        # Find the virtual edge position.
                        # Then the rest is a path.
                        # Output each edge in the path as a single-edge piece.
                        
                        # Find virtual edge
                        virt_idx = -1
                        for i in range(len(cycle) - 1):
                            u, v = cycle[i], cycle[i+1]
                            if (u, v) not in edge_map and (v, u) not in edge_map:
                                virt_idx = i
                                break
                        if virt_idx == -1:
                            u, v = cycle[-1], cycle[0]
                            if (u, v) not in edge_map and (v, u) not in edge_map:
                                virt_idx = len(cycle) - 1 # closing edge
            
                        if virt_idx != -1:
                            # The path is from cycle[virt_idx+1] to cycle[virt_idx]
                            # Going forward in the cycle list?
                            # Cycle: v0 -> v1 -> ... -> vk -> v0
                            # If edge (v_i, v_{i+1}) is virtual:
                            # The real path is v_{i+1} -> v_{i+2} -> ... -> v_k -> v_0 -> ... -> v_i
                            # This is a valid path in the original graph.
                            
                            path_vertices = []
                            # Start after virtual edge
                            start_idx = (virt_idx + 1) % len(cycle)
                            # End at virt_idx
                            # We traverse the cycle from start_idx to virt_idx (wrapping around if needed)
                            # Since it's a cycle, we can just take the sequence.
                            
                            # Let's just iterate and skip the virtual edge.
                            # We want the sequence of vertices of the real path.
                            # If virtual edge is (A, B), the path is B -> ... -> A.
                            
                            # Reconstruct path
                            path = []
                            current = cycle[(virt_idx + 1) % len(cycle)]
                            path.append(current)
                            # Next vertex in cycle
                            next_idx = (virt_idx + 2) % len(cycle)
                            while next_idx != virt_idx:
                                current = cycle[next_idx]
                                path.append(current)
                                next_idx = (next_idx + 1) % len(cycle)
                                if next_idx == virt_idx:
                                    break
                            
                            # Wait, if virt_idx is the last edge (closing), i.e. (vk, v0).
                            # Then start is v0. End is vk.
                            # Path: v0 -> v1 -> ... -> vk.
                            # My loop:
                            # virt_idx = len(cycle)-1.
                            # start_idx = 0.
                            # next_idx = 1.
                            # while 1 != len(cycle)-1:
                            #   append cycle[1]
                            #   next_idx = 2
                            #   ...
                            #   append cycle[len(cycle)-1] (which is vk)
                            #   next_idx = 0.
                            #   if 0 == len(cycle)-1? No.
                            #   This logic is flawed if it wraps.
                            
                            # Simpler:
                            # The cycle is a list of vertices.
                            # Find the virtual edge.
                            # If edge (u,v) is virtual, remove it.
                            # The remaining edges form a path.
                            # We can just output the edges of the path.
                            
                            # Let's identify the virtual edge endpoints.
                            # And then the path is the rest of the cycle.
                            
                            # To be safe, let's just output the edges of the cycle, excluding the virtual one, as single-edge pieces.
                            for i in range(len(cycle) - 1):
                                u, v = cycle[i], cycle[i+1]
                                if (u, v) in edge_map or (v, u) in edge_map:
                                    pieces.append([u, v])
                                # else: virtual, skip
                                
                            u, v = cycle[-1], cycle[0]
                            if (u, v) in edge_map or (v, u) in edge_map:
                                pieces.append([u, v])
                    else:
                        pieces.append(cycle)
                        
        return pieces

    def find_eulerian_circuit(start_node, adj, edge_map):
        # Hierholzer's algorithm
        # Returns a list of vertices representing the circuit
        # adj: u -> [(v, eid), ...]
        
        # We need to consume edges.
        # Make a copy of adj to avoid modifying original?
        # Or modify in place and restore?
        # Given the small size, copying is fine.
        
        adj_copy = {u: list(vs) for u, vs in adj.items()}
        
        stack = [start_node]
        circuit = []
        
        while stack:
            u = stack[-1]
            if adj_copy[u]:
                v, eid = adj_copy[u].pop()
                stack.append(v)
            else:
                circuit.append(stack.pop())
        
        circuit.reverse()
        return circuit

    def decompose_circuit_into_cycles(circuit):
        # circuit is a list of vertices [v0, v1, ..., vk, v0]
        # Decompose into simple cycles
        if not circuit:
            return []
            
        cycles = []
        current_path = circuit[:]
        
        while len(current_path) > 2: # A cycle has at least 3 vertices (v0, v1, v2, v0)
            # Check if current_path is a simple cycle
            # i.e., all vertices distinct (except v0 == vk)
            if len(set(current_path[:-1])) == len(current_path) - 1:
                cycles.append(current_path)
                break
            
            # Find first repeated vertex
            # The path is v0, v1, ..., vk
            # v0 == vk is the closure.
            # We look for a vertex vi (i>0) that appears earlier.
            # Actually, in a closed walk, if it's not simple, some vertex appears > 1 time.
            
            seen = {}
            split_idx = -1
            for i, v in enumerate(current_path):
                if v in seen:
                    # We found a repeat of v at index i, previously at seen[v]
                    # The cycle is current_path[seen[v]:i+1]
                    # The remaining path is current_path[i+1:] + current_path[:seen[v]] ?
                    # No.
                    # The walk is v0...v_{seen[v]} ... v_i ... v_k
                    # Since v_{seen[v]} == v_i, the segment v_{seen[v]}...v_i is a cycle.
                    # The remaining walk is v_i...v_k...v_{seen[v]}?
                    # Wait, the walk is closed.
                    # If we extract the cycle v_{seen[v]}...v_i, we are left with the walk v_i...v_k (which is v_{seen[v]})...v_{seen[v]}?
                    # No, the original walk is v0...v_{seen[v]}...v_i...v_k(=v0).
                    # If we remove the cycle v_{seen[v]}...v_i, the remaining edges form a walk from v_i to v_{seen[v]}?
                    # But v_i == v_{seen[v]}. So the remaining walk is closed?
                    # Yes.
                    
                    # Let's verify.
                    # Walk: A - B - C - B - D - A
                    # Repeated B at index 1 and 3.
                    # Cycle: B - C - B.
                    # Remaining: B - D - A - B?
                    # Original: A-B (edge 1), B-C (2), C-B (3), B-D (4), D-A (5).
                    # Cycle uses 2, 3.
                    # Remaining edges: 1, 4, 5.
                    # Path: A-B, B-D, D-A. This is a cycle A-B-D-A.
                    # So the remaining walk is a cycle too.
                    
                    # So we can split.
                    # Cycle 1: current_path[seen[v]:i+1]
                    # Cycle 2 (remaining): current_path[i+1:] + current_path[:seen[v]]
                    # Wait, the remaining walk starts at v_i (which is v) and ends at v_{seen[v]} (which is v).
                    # The edges are from i+1 to end, and from start to seen[v].
                    # So the vertex sequence is current_path[i+1:] + current_path[:seen[v]+1]?
                    # Let's trace:
                    # Path: [0, 1, 2, 1, 3, 0]
                    # Repeat 1 at 1 and 3.
                    # Cycle 1: [1, 2, 1]
                    # Remaining: [1, 3, 0, 1] -> [1, 3, 0] + [1]?
                    # Edges: (1,3), (3,0), (0,1).
                    # Vertices: 1, 3, 0, 1.
                    # So remaining path is current_path[i+1:] + [current_path[0]]?
                    # No, current_path[:seen[v]] is [0, 1].
                    # current_path[i+1:] is [3, 0].
                    # Concatenate: [3, 0, 0, 1]? No.
                    
                    # The remaining walk is formed by the edges not in the cycle.
                    # The cycle uses edges (v_{seen[v]}, v_{seen[v]+1}) ... (v_{i-1}, v_i).
                    # The remaining edges are (v_0, v_1) ... (v_{seen[v]-1}, v_{seen[v]}) AND (v_{i+1}, v_{i+2}) ... (v_{k-1}, v_k).
                    # Since v_k = v_0 and v_i = v_{seen[v]},
                    # The remaining walk is v_{seen[v]} -> v_{seen[v]+1} ... NO, those are in the cycle.
                    # The remaining walk is v_i (=v_{seen[v]}) -> v_{i+1} ... -> v_k (=v_0) -> v_1 ... -> v_{seen[v]}.
                    # So the vertex sequence is:
                    # [v_i, v_{i+1}, ..., v_k, v_1, ..., v_{seen[v]}]
                    # Note: v_k is v_0.
                    # So: current_path[i+1:] + current_path[1:seen[v]+1]
                    
                    cycle1 = current_path[seen[v]:i+1]
                    cycle2 = current_path[i+1:] + current_path[1:seen[v]+1]
                    
                    cycles.append(cycle1)
                    current_path = cycle2
                    break
            if split_idx == -1 and len(set(current_path[:-1])) == len(current_path) - 1:
                # It was simple all along
                cycles.append(current_path)
                break
            if split_idx == -1:
                # Should not happen if not simple
                break
                
        return cycles

    results = []
    for i, edges in enumerate(graphs):
        if time.time() > deadline:
            # Fallback
            results.append([list(e) for e in edges])
            continue
            
        pieces = decompose_graph(edges)
        results.append(pieces)
        
    return results

if __name__ == "__main__":
    res = construct()
    for i, spec in enumerate([(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]):
        n, p, seed = spec
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        edge_set = set(edges)
        
        used_edges = set()
        valid = True
        for piece in res[i]:
            if len(piece) == 2:
                u, v = piece
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            elif len(piece) >= 3:
                if len(set(piece)) != len(piece):
                    valid = False
                    break
                for k in range(len(piece) - 1):
                    u, v = piece[k], piece[k+1]
                    e = tuple(sorted((u, v)))
                    if e not in edge_set or e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break
                u, v = piece[-1], piece[0]
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            else:
                valid = False
                break
                
        if valid and used_edges != edge_set:
            valid = False
            
        print(f"Graph {i}: Valid={valid}, Pieces={len(res[i])}")

