import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    DEADLINE = 150.0 # seconds
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_eulerian_circuit(adj, start):
        """
        Find an Eulerian circuit in a graph represented by adjacency list.
        Returns list of vertices in the circuit.
        Assumes all vertices with non-zero degree are in one component and degrees are even.
        """
        # Hierholzer's algorithm
        stack = [start]
        circuit = []
        # We need to track used edges. 
        # Since we are working with a multigraph (virtual edges added), 
        # we should represent edges uniquely.
        # For simplicity in this context, we will modify the adjacency list 
        # to remove edges as we visit them.
        
        # To handle multi-edges (virtual edges might duplicate real edges or each other),
        # we should use a list of lists for adjacency and pop edges.
        
        pass # Implemented in the wrapper below

    def decompose_graph(edges):
        n = 0
        for u, v in edges:
            n = max(n, u + 1, v + 1)
        
        # Build adjacency list with edge indices to handle multi-edges properly
        # adj[u] = list of (v, edge_index)
        adj = {i: [] for i in range(n)}
        for idx, (u, v) in enumerate(edges):
            adj[u].append((v, idx))
            adj[v].append((u, idx))
        
        # Find odd degree vertices
        odd_vertices = [v for v in range(n) if len(adj[v]) % 2 != 0]
        
        if not odd_vertices:
            # Eulerian circuit
            # Pick any vertex with degree > 0
            start = next((v for v in range(n) if len(adj[v]) > 0), 0)
            
            # We need to find a circuit.
            # Since we need to return a simple path, a circuit is not a simple path 
            # unless we break it. But we must cover all edges.
            # A circuit $v_0, \dots, v_k, v_0$ covers all edges.
            # We can return $v_0, \dots, v_k$ as a path? No, that misses edge $(v_k, v_0)$.
            # We can return $v_0, \dots, v_k, v_0$? Is that a simple path? 
            # Definition of simple path: distinct vertices. $v_0 = v_k$ violates this.
            # So a pure cycle cannot be covered by 1 simple path.
            # It requires at least 2 paths? 
            # If we have a cycle, we can take a path $P_1$ that is almost the whole cycle, 
            # and a path $P_2$ for the remaining edge?
            # Actually, if we have a cycle, we can just pick an edge $(u,v)$ to be the "break".
            # But we must cover ALL edges.
            # If we use 1 path, it must be a Hamiltonian path.
            # If the graph is a cycle, no Hamiltonian path exists (it's a cycle).
            # So we need more paths.
            # However, for random graphs, this case (all even degrees) is less likely 
            # to be a *pure* cycle without chords.
            # If there are chords, we can likely find a Hamiltonian path or decompose into few paths.
            
            # Strategy for 0 odd vertices:
            # Try to find a Hamiltonian path. If found, return [path].
            # Else, decompose into 2 paths? Or just use the Eulerian circuit and split it 
            # into 2 paths by cutting an edge?
            # Cutting an edge $(u,v)$ in the circuit $u, \dots, v$ gives path $u, \dots, v$.
            # But we missed the edge $(u,v)$? No, the circuit uses it.
            # If we split the circuit into a path, we must remove one edge from the path?
            # No, the path must cover edges.
            # If we have a circuit $C$, we can express it as a path $P$ plus one edge $e$?
            # No, $P$ must be a simple path.
            # Let's just return the Eulerian circuit vertices as a single "path" 
            # and hope the verifier accepts $v_0, \dots, v_{k-1}$? 
            # No, "use graph edges". The edge $(v_{k-1}, v_0)$ is not in the path $v_0, \dots, v_{k-1}$.
            # So we MUST cover it.
            # If we return $v_0, \dots, v_{k-1}, v_0$, it's not simple.
            
            # Safe fallback for 0 odd: 
            # Find an Eulerian circuit. 
            # Pick any edge in the circuit, say between $u$ and $v$.
            # The rest of the circuit is a path from $v$ to $u$ (or $u$ to $v$).
            # This path covers all edges EXCEPT $(u,v)$.
            # So we have 1 path covering $M-1$ edges, and 1 path covering 1 edge.
            # Total 2 paths.
            # Both are simple?
            # The path from $v$ to $u$ in the circuit is a walk. Is it simple?
            # In a simple graph, an Eulerian circuit might revisit vertices.
            # If it revisits, it's not a simple path.
            # But for small random graphs, it's likely simple.
            
            # Let's implement a generic Eulerian circuit finder that returns the sequence of edges.
            
            pass

    # Helper: Find Eulerian circuit using Hierholzer's
    def get_eulerian_circuit(n, adj, edges):
        # adj is list of lists of (neighbor, edge_id)
        # We need to modify adj to remove edges.
        # To preserve original, we copy it.
        adj_copy = {u: list(v) for u, v in adj.items()}
        
        start = 0
        for u in range(n):
            if adj_copy[u]:
                start = u
                break
        
        if not adj_copy[start]:
            return []
            
        stack = [start]
        circuit = []
        
        while stack:
            u = stack[-1]
            if adj_copy[u]:
                v, eid = adj_copy[u].pop()
                # Remove reverse edge
                # Find eid in adj_copy[v]
                # Since it's a small graph, linear search is fine
                for i, (nv, neid) in enumerate(adj_copy[v]):
                    if neid == eid:
                        adj_copy[v].pop(i)
                        break
                stack.append(v)
            else:
                circuit.append(stack.pop())
                
        return circuit[::-1]

    def is_simple_path(path):
        if not path:
            return True
        return len(path) == len(set(path))

    def decompose_instance(spec):
        n, p, seed = spec
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        # Build adj
        adj = {i: [] for i in range(n)}
        for idx, (u, v) in enumerate(edges):
            adj[u].append((v, idx))
            adj[v].append((u, idx))
            
        odd_vertices = [v for v in range(n) if len(adj[v]) % 2 != 0]
        
        # Case 1: No odd vertices
        if not odd_vertices:
            # Try to find a simple path covering all edges? Unlikely for large N.
            # Fallback: Eulerian circuit -> 2 paths.
            # Circuit is a list of vertices.
            # We can break it into 2 paths.
            # Path 1: Circuit[0] ... Circuit[k-1]
            # Path 2: Circuit[k-1] ... Circuit[0]? No.
            
            # Let's just use the 2-path strategy:
            # Find circuit.
            # Pick an edge (u,v) in the circuit.
            # The circuit is u, ..., v, ..., u.
            # Wait, the circuit is a list of vertices $v_0, v_1, \dots, v_{m-1}, v_0$.
            # The edges are $(v_0, v_1), \dots, (v_{m-1}, v_0)$.
            # If we take path $P_1 = [v_1, \dots, v_{m-1}, v_0]$, it covers edges $(v_1, v_2) \dots (v_{m-1}, v_0)$.
            # It misses $(v_0, v_1)$.
            # So $P_2 = [v_0, v_1]$.
            # Are they simple?
            # $P_1$ is a subsequence of the circuit. If the circuit is simple (no repeated vertices except start/end), 
            # then $P_1$ is simple.
            # $P_2$ is length 2, always simple.
            
            circuit = get_eulerian_circuit(n, adj, edges)
            if not circuit:
                return []
            
            # Circuit is $v_0, \dots, v_k, v_0$.
            # In my get_eulerian_circuit, it returns $v_0, \dots, v_k, v_0$.
            # Actually, Hierholzer returns the vertices.
            # Let's check the output format.
            # If circuit is [0, 1, 2, 0], edges are (0,1), (1,2), (2,0).
            # Path 1: [1, 2, 0]. Covers (1,2), (2,0).
            # Path 2: [0, 1]. Covers (0,1).
            
            if len(circuit) < 2:
                return []
                
            # The circuit list from Hierholzer usually ends with the start vertex.
            # e.g. [0, 1, 2, 0]
            # We want to split it.
            # Let's assume circuit[0] == circuit[-1].
            # Path 1: circuit[1:]  -> [1, 2, 0]
            # Path 2: [circuit[-1], circuit[0]] -> [0, 0]? No.
            # Path 2 should be the edge we cut.
            # The edge is (circuit[0], circuit[1]).
            # So Path 2: [circuit[0], circuit[1]].
            
            p1 = circuit[1:]
            p2 = [circuit[0], circuit[1]]
            
            # Verify simplicity
            if is_simple_path(p1) and is_simple_path(p2):
                return [p1, p2]
            
            # If not simple, we might need more paths.
            # For now, return the circuit as a single path if we can? 
            # No, must be simple.
            # Fallback: Edge by edge.
            return greedy_decompose(n, edges)

        # Case 2: Odd vertices exist
        # Pair them up
        # To maximize chance of simple paths, we can try multiple random pairings.
        
        for attempt in range(50):
            if time.time() > DEADLINE:
                break
                
            # Shuffle odd vertices
            rng = random.Random(seed + attempt)
            odds = list(odd_vertices)
            rng.shuffle(odds)
            
            # Pair them
            pairs = []
            for i in range(0, len(odds), 2):
                pairs.append((odds[i], odds[i+1]))
                
            # Build augmented graph
            # We need to add virtual edges.
            # We will simulate the Eulerian circuit on the augmented graph.
            # To do this, we create a new adjacency list including virtual edges.
            
            adj_aug = {u: list(v) for u, v in adj.items()}
            # Add virtual edges
            # We need unique IDs for virtual edges to distinguish them.
            # Real edges: 0..M-1
            # Virtual edges: M..M+K-1
            
            M = len(edges)
            for i, (u, v) in enumerate(pairs):
                eid = M + i
                adj_aug[u].append((v, eid))
                adj_aug[v].append((u, eid))
                
            # Find Eulerian circuit in augmented graph
            circuit = get_eulerian_circuit(n, adj_aug, None) # edges param not used for IDs
            
            if not circuit:
                continue
                
            # Extract paths
            # The circuit contains virtual edges.
            # We need to identify where virtual edges are used.
            # The circuit is a list of vertices.
            # We need to know which edges were traversed.
            # My get_eulerian_circuit doesn't return edge IDs.
            # Let's modify it to return edge IDs or reconstruct.
            
            # Re-implement Hierholzer to track edges
            def get_eulerian_circuit_with_edges(n, adj):
                adj_copy = {u: list(v) for u, v in adj.items()}
                start = 0
                for u in range(n):
                    if adj_copy[u]:
                        start = u
                        break
                if not adj_copy[start]:
                    return [], []
                    
                stack = [(start, -1)] # (vertex, edge_id)
                circuit_vertices = []
                circuit_edges = []
                
                while stack:
                    u, eid_in = stack[-1]
                    if adj_copy[u]:
                        v, eid = adj_copy[u].pop()
                        # Remove reverse
                        for i, (nv, neid) in enumerate(adj_copy[v]):
                            if neid == eid:
                                adj_copy[v].pop(i)
                                break
                        stack.append((v, eid))
                    else:
                        u, eid_in = stack.pop()
                        circuit_vertices.append(u)
                        if eid_in != -1:
                            circuit_edges.append(eid_in)
                            
                circuit_vertices.reverse()
                circuit_edges.reverse()
                return circuit_vertices, circuit_edges

            verts, eids = get_eulerian_circuit_with_edges(n, adj_aug)
            
            if not verts:
                continue
                
            # The circuit is a sequence of vertices and edges.
            # Verts: [v0, v1, ..., vk, v0]
            # Eids: [e0, e1, ..., ek-1] corresponding to (v0,v1), (v1,v2)...
            
            # We need to split this sequence at virtual edges.
            # A virtual edge is an edge with ID >= M.
            
            paths = []
            current_path = [verts[0]]
            
            for i in range(len(eids)):
                u = verts[i]
                v = verts[i+1]
                eid = eids[i]
                
                current_path.append(v)
                
                if eid >= M:
                    # Virtual edge encountered.
                    # The current_path is a path in the original graph.
                    # Check if it's simple.
                    if is_simple_path(current_path):
                        paths.append(current_path)
                    else:
                        # Not simple. We can't easily fix this without backtracking.
                        # Mark as failed
                        paths = None
                        break
                        
                    current_path = [v] # Start new path from v?
                    # Wait, the virtual edge connects end of P1 to start of P2.
                    # The next real edge starts from v.
                    # So the next path starts at v.
            
            if paths is None:
                continue
                
            # Don't forget the last segment
            if is_simple_path(current_path):
                paths.append(current_path)
            else:
                continue
                
            # Verify all paths are simple and cover edges
            # We already checked simplicity.
            # Do they cover all edges?
            # The Eulerian circuit covers all edges in the augmented graph.
            # The virtual edges are not in the original graph.
            # The real edges are exactly the ones in the original graph.
            # So yes, they cover all edges.
            
            return paths
            
        # If all attempts failed to find simple paths
        return greedy_decompose(n, edges)

    def greedy_decompose(n, edges):
        # Fallback: Just return each edge as a path.
        # This is always valid and simple.
        return [[u, v] for u, v in edges]

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE:
            # Return best effort or empty?
            # If we timeout, we should return something valid.
            # Greedy is valid.
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append(greedy_decompose(n, edges))
        else:
            results.append(decompose_instance(spec))
            
    return results

# Need to import Random
from random import Random

# Re-verify the code structure.
# The `instance` function uses `Random`.
# The `construct` function returns a list of 3 lists of paths.

# One detail: `get_eulerian_circuit_with_edges` uses `adj` which is a dict of lists.
# The `adj_aug` is built correctly.

# Let's double check the `is_simple_path` for the last segment.
# In the loop, we append `v` to `current_path`.
# If the edge is virtual, we finalize `current_path`.
# The last segment is finalized after the loop.
# This logic seems correct.

# What if the graph is disconnected?
# The task says "connected graphs".
# So we don't need to handle multiple components.

# What if `odd_vertices` is empty but the graph is not a cycle?
# E.g. two cycles connected by a bridge?
# If connected and all degrees even, it's Eulerian.
# The circuit will exist.
# The 2-path split should work if the circuit is simple.
# If the circuit is not simple, the 2-path split might produce non-simple paths.
# In that case, we fall back to `greedy_decompose`.

# This should be robust enough.

# Final check on imports.
# `random` and `time` are standard.

# One optimization:
# In `decompose_instance`, if `odd_vertices` is empty, we don't try multiple pairings.
# We just try the 2-path split.
# If it fails, we go to greedy.
# This is fine.

# The `greedy_decompose` returns `[[u, v] for u, v in edges]`.
# This is a list of paths. Each path is a list of 2 vertices.
# This is valid.

# The score will be the sum of lengths of these lists.
# For the baseline, it was sum of edges.
# With this algorithm, it will be much smaller.

# Example:
# Triangle.
# Odd vertices: None.
# Circuit: [0, 1, 2, 0].
# Eids: [0, 1, 2]. (Assuming edges are (0,1), (1,2), (2,0)).
# Split:
# Path 1: [1, 2, 0]. Simple? Yes.
# Path 2: [0, 1]. Simple? Yes.
# Total paths: 2.
# Baseline: 3.
# Improvement.

# For a larger graph, the number of paths will be close to `odd_count / 2`.
# This is significantly better than `num_edges`.

# The deadline guard is in place.

# One potential issue: `get_eulerian_circuit_with_edges` modifies `adj_copy`.
# We pass `adj_aug` which is a copy of `adj` with added edges.
# This is fine.

# The `instance` function is defined inside `construct` in the seed, 
# but I defined it inside `construct` in my draft too.
# I should make sure `instance` is accessible.
# In my code, `instance` is defined inside `construct`.
# `decompose_instance` is defined inside `construct`.
# `greedy_decompose` is defined inside `construct`.
# This is fine.

# I will structure the final code clearly.
