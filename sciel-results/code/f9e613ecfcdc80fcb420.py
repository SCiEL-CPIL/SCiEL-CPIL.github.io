"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    # STRATEGY: Double-ring (Möbius ladder) construction
    # Ring E (even vertices): 0-2-4-...-38-0 (20 edges)
    # Ring O (odd vertices): 1-3-5-...-39-1 (20 edges)
    # Rungs: (2i, 2i+1) for i in 0..19 (20 edges)
    # Total: 60 edges
    #
    # Cycles can be extracted structurally:
    # - Length 4: (2i, 2i+1, 2(i+1)+1, 2(i+1)) using rungs and ring edges
    # - Length 3: (2i, 2i+1, 2i-1) if rung (2i, 2i-1) exists
    # - General: paths on E and O connected by rungs
    #
    # We'll build the base graph, extract cycles structurally,
    # then try to delete edges while maintaining pancyclicity.

    def build_base_graph():
        """Build the 60-edge double-ring graph."""
        edges = set()
        # Ring E: 0-2-4-...-38-0
        for i in range(20):
            u = 2 * i
            v = 2 * ((i + 1) % 20)
            edges.add((min(u, v), max(u, v)))
        # Ring O: 1-3-5-...-39-1
        for i in range(20):
            u = 2 * i + 1
            v = 2 * ((i + 1) % 20) + 1
            edges.add((min(u, v), max(u, v)))
        # Rungs: (2i, 2i+1)
        for i in range(20):
            u = 2 * i
            v = 2 * i + 1
            edges.add((min(u, v), max(u, v)))
        return edges

    def build_adjacency(edges):
        """Build adjacency list from edge set."""
        adj = {i: set() for i in range(N)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def extract_cycles_structured(adj, edges):
        """
        Extract one cycle of each length 3..40 using the double-ring structure.
        Returns dict {length: [vertex_list]} or None if any length is missing.
        
        Key insight: In the double-ring graph, a cycle can be formed by:
        - Taking a path of length a on ring E from vertex 2i to 2j
        - Taking a path of length b on ring O from vertex 2i+1 to 2j+1
        - Connecting with rungs (2i, 2i+1) and (2j, 2j+1)
        - Total cycle length = a + b + 2
        - We need a+b+2 = L, so a+b = L-2
        - a can range from 1 to 19 (path on E), b from 1 to 19 (path on O)
        - This gives cycle lengths from 4 to 40
        - For length 3: need a special triangle
        
        Actually, let's think more carefully:
        - Path on E from 2i to 2j: goes through vertices 2i, 2(i+1), ..., 2j (mod 20)
          Number of edges = distance on E ring = d_E
          Number of vertices in path = d_E + 1
        - Path on O from 2i+1 to 2j+1: similarly d_O edges, d_O+1 vertices
        - Rungs: (2i, 2i+1) and (2j, 2j+1)
        - Cycle: 2i -> 2i+1 (rung) -> ... -> 2j+1 (on O) -> 2j (rung) -> ... -> 2i (on E)
        - Total edges in cycle = d_O + d_E + 2
        - Total vertices = d_O + d_E + 2 (since it's a cycle, edges = vertices)
        
        So for cycle length L: d_E + d_O = L - 2
        d_E in [1, 19], d_O in [1, 19]
        L-2 in [2, 38]
        
        For L=3: d_E + d_O = 1, impossible with d_E,d_O >= 1.
        So length 3 needs a different approach.
        
        For length 3: triangle (2i, 2i+1, 2i+3) needs edges (2i,2i+1), (2i+1,2i+3), (2i+3,2i)
        (2i+1, 2i+3) is on ring O. (2i+3, 2i) is NOT on ring E (distance 3 on E).
        So we need a chord or rung to make a triangle.
        
        Alternative for length 3: (2i, 2i+1, 2(i-1)+1) = (2i, 2i+1, 2i-1)
        Edges: (2i, 2i+1) rung, (2i+1, 2i-1) on ring O? No, 2i+1 and 2i-1 are adjacent on O.
        (2i-1, 2i) on ring E? Yes, adjacent.
        So triangle: (2i, 2i+1, 2i-1, 2i) - wait that's 3 edges: (2i,2i+1), (2i+1,2i-1), (2i-1,2i)
        Vertices: 2i, 2i+1, 2i-1. That's a triangle! Length 3.
        
        Let me verify: vertices 2i, 2i+1, 2i-1
        Edge (2i, 2i+1): rung ✓
        Edge (2i+1, 2i-1): on ring O (consecutive odd vertices) ✓
        Edge (2i-1, 2i): on ring E (consecutive even vertices) ✓
        Yes! Triangle exists in the base graph.
        
        For L >= 4: use the two-ring path construction.
        For L=4: d_E + d_O = 2, so d_E=1, d_O=1.
        Cycle: (2i, 2i+1, 2i+3, 2i+2, 2i) 
        Edges: (2i,2i+1) rung, (2i+1,2i+3) ring O, (2i+3,2i+2) rung, (2i+2,2i) ring E.
        Wait, 2i+3 and 2i+2: is that a rung? Rung is (2k, 2k+1). 
        2i+2 = 2(i+1), 2i+3 = 2(i+1)+1. Yes! Rung (2(i+1), 2(i+1)+1).
        So cycle: 2i -> 2i+1 (rung) -> 2i+3 (ring O) -> 2i+2 (rung) -> 2i (ring E).
        Vertices: 2i, 2i+1, 2i+3, 2i+2. Four distinct vertices. ✓
        
        General: For length L (4 <= L <= 40):
        Choose d_E and d_O such that d_E + d_O = L-2, 1 <= d_E <= 19, 1 <= d_O <= 19.
        Start at 2i on ring E. Go d_E steps on E to reach 2(i+d_E mod 20).
        Rung to 2(i+d_E mod 20)+1.
        Go d_O steps on O to reach 2(i+d_E mod 20)+d_O+1... 
        Wait, I need to be more careful.
        
        Let me define:
        - Start at vertex 2i on ring E.
        - Traverse d_E edges on ring E: vertices 2i, 2(i+1), ..., 2(i+d_E) [mod 20 for indices]
          End at vertex 2*((i+d_E) % 20)
        - Take rung to 2*((i+d_E) % 20) + 1
        - Traverse d_O edges on ring O: vertices 2*((i+d_E)%20)+1, 2*((i+d_E+1)%20)+1, ..., 2*((i+d_E+d_O)%20)+1
          End at vertex 2*((i+d_E+d_O) % 20) + 1
        - But we need to get back to 2i!
        - For this to close, we need 2*((i+d_E+d_O) % 20) + 1 to connect to 2i.
        - That would be a rung if ((i+d_E+d_O) % 20) == i, i.e., d_E+d_O ≡ 0 (mod 20).
        - But d_E+d_O = L-2, and L ranges from 4 to 40, so L-2 ranges from 2 to 38.
        - This only works when L-2 ≡ 0 (mod 20), i.e., L=22 or L=42 (out of range).
        
        So the simple "go out and back" doesn't close properly for all lengths.
        
        Alternative: Use TWO rungs at different positions.
        - Start at 2i.
        - Go d_E steps on E to 2(i+d_E).
        - Rung to 2(i+d_E)+1.
        - Go d_O steps on O to 2(i+d_E+d_O)+1.
        - Rung to 2(i+d_E+d_O).
        - Go back on E from 2(i+d_E+d_O) to 2i.
        - Distance on E from 2(i+d_E+d_O) to 2i is (20 - d_E - d_O) mod 20.
        - Total edges: d_E + 1 + d_O + 1 + (20 - d_E - d_O) = 22.
        - That's always 22! Not useful.
        
        Let me reconsider. The cycle must be simple (no repeated vertices).
        
        Better approach: 
        - Take a contiguous arc of length a on ring E (a edges, a+1 vertices)
        - Take a contiguous arc of length b on ring O (b edges, b+1 vertices)
        - Connect with two rungs
        - The two arcs must be at "matching" positions
        
        Specifically:
        - Arc on E: from 2i to 2(i+a) [a edges, vertices 2i, 2(i+1), ..., 2(i+a)]
        - Arc on O: from 2i+1 to 2(i+b)+1 [b edges, vertices 2i+1, 2(i+1)+1, ..., 2(i+b)+1]
        - Rung 1: (2i, 2i+1)
        - Rung 2: (2(i+a), 2(i+b)+1) -- this needs to be an edge!
          But rung 2 is (2(i+a), 2(i+a)+1) in the base graph.
          So we need 2(i+b)+1 = 2(i+a)+1, i.e., a=b.
          Then cycle length = a + b + 2 = 2a + 2.
          This only gives even lengths: 4, 6, 8, ..., 40.
        
        For odd lengths, we need a different structure.
        
        For odd length L: 
        - Use one rung and a path that goes around
        - Or use a "twisted" connection
        
        Alternative for odd L:
        - Arc on E: from 2i to 2(i+a) [a edges]
        - Arc on O: from 2(i+a)+1 to 2(i+a+b)+1 [b edges]
        - Rung at start: (2i, 2i+1) -- but 2i+1 is not the start of the O arc
        - This is getting complicated.
        
        Let me try a completely different structural approach for cycle extraction.
        
        Actually, let's just use a simple backtracking search for cycle extraction,
        but optimize it heavily. The graph has ~60 edges and 40 vertices, so
        backtracking for a specific cycle length should be fast enough.
        
        But the current program's backtracking is too slow. Let me use a 
        more efficient approach: for each length L, do a DFS with pruning.
        """
        cycles = {}
        
        # Length 3: triangle (2i, 2i+1, 2i-1) for any i
        # Verify edges exist
        i = 0
        v1, v2, v3 = 0, 1, 39  # 2*0, 2*0+1, 2*(-1 mod 20)+1 = 39
        if (v1, v2) in edges or (v2, v1) in edges:
            pass
        # Check triangle 0, 1, 39
        e1 = (min(0,1), max(0,1))
        e2 = (min(1,39), max(1,39))
        e3 = (min(39,0), max(39,0))
        if e1 in edges and e2 in edges and e3 in edges:
            cycles[3] = [0, 1, 39]
        else:
            # Try other triangles
            for i in range(20):
                v1 = 2*i
                v2 = 2*i+1
                v3 = 2*((i-1)%20)+1
                e1 = (min(v1,v2), max(v1,v2))
                e2 = (min(v2,v3), max(v2,v3))
                e3 = (min(v3,v1), max(v3,v1))
                if e1 in edges and e2 in edges and e3 in edges:
                    cycles[3] = [v1, v2, v3]
                    break
            if 3 not in cycles:
                return None
        
        # For lengths 4..40: use backtracking with good pruning
        for L in range(4, N+1):
            if time.time() - start_time > deadline - 5:
                return None
            
            found = False
            cycle = None
            
            # Try to find a cycle of length L starting from vertex 0
            # Use DFS with pruning
            adj = build_adjacency(edges)
            
            def dfs(path, visited, target_len):
                if len(path) == target_len:
                    if path[0] in adj[path[-1]]:
                        return list(path)
                    return None
                remaining = target_len - len(path)
                if remaining > N - len(visited):
                    return None
                for nxt in adj[path[-1]]:
                    if nxt not in visited and nxt != path[0]:
                        path.append(nxt)
                        visited.add(nxt)
                        result = dfs(path, visited, target_len)
                        if result:
                            return result
                        path.pop()
                        visited.remove(nxt)
                return None
            
            # Try from vertex 0 first (most likely to succeed in structured graph)
            path = [0]
            visited = {0}
            cycle = dfs(path, visited, L)
            
            if cycle is None:
                # Try from other vertices
                for start in range(1, N):
                    if time.time() - start_time > deadline - 5:
                        return None
                    path = [start]
                    visited = {start}
                    cycle = dfs(path, visited, L)
                    if cycle is not None:
                        break
            
            if cycle is None:
                return None
            
            cycles[L] = cycle
        
        return cycles

    def verify_and_extract(edges):
        """Verify pancyclicity and extract cycles. Returns cycles dict or None."""
        return extract_cycles_structured(build_adjacency(edges), edges)

    def construct_inner():
        start_time = time.time()
        deadline = 85.0
        
        # Start with 60-edge double-ring
        edges = build_base_graph()
        
        # Verify base graph is pancyclic
        cycles = verify_and_extract(edges)
        if cycles is None:
            # Fallback: use Hamiltonian cycle + chords (59 edges)
            edges = set()
            for i in range(N):
                edges.add((min(i, (i+1)%N), max(i, (i+1)%N)))
            for k in range(2, 21):
                edges.add((0, k))
            cycles = verify_and_extract(edges)
            if cycles is None:
                # Shouldn't happen
                return None
        
        # Try to delete edges to reduce count
        # Non-critical edges are more likely to be deletable
        # Try deleting rungs first (they might be redundant)
        rung_edges = [(2*i, 2*i+1) for i in range(20)]
        ring_e_edges = [(2*i, 2*((i+1)%20)) for i in range(20)]
        ring_o_edges = [(2*i+1, 2*((i+1)%20)+1) for i in range(20)]
        
        # Try deleting each rung
        for rung in rung_edges:
            if time.time() - start_time > deadline - 15:
                break
            if rung not in edges:
                continue
            edges_copy = set(edges)
            edges_copy.discard(rung)
            
            cycles_test = verify_and_extract(edges_copy)
            if cycles_test is not None:
                edges = edges_copy
        
        # Try deleting ring edges
        all_ring_edges = ring_e_edges + ring_o_edges
        for ring_edge in all_ring_edges:
            if time.time() - start_time > deadline - 15:
                break
            if ring_edge not in edges:
                continue
            edges_copy = set(edges)
            edges_copy.discard(ring_edge)
            
            cycles_test = verify_and_extract(edges_copy)
            if cycles_test is not None:
                edges = edges_copy
        
        # Second pass: try deleting more edges after first pass
        all_edges = list(edges)
        random.seed(42)
        random.shuffle(all_edges)
        
        for edge in all_edges:
            if time.time() - start_time > deadline - 10:
                break
            if edge not in edges:
                continue
            edges_copy = set(edges)
            edges_copy.discard(edge)
            
            cycles_test = verify_and_extract(edges_copy)
            if cycles_test is not None:
                edges = edges_copy
        
        # Final verification
        final_cycles = verify_and_extract(edges)
        if final_cycles is None:
            # Fallback to base
            edges = build_base_graph()
            final_cycles = verify_and_extract(edges)
            if final_cycles is None:
                return None
        
        edge_list = sorted(set(edges))
        cycles_out = []
        for L in range(3, N+1):
            cycles_out.append(final_cycles[L])
        
        return {'edges': edge_list, 'cycles': cycles_out}

    result = construct_inner()
    if result is None:
        # Ultimate fallback: Hamiltonian cycle + chords
        edges = set()
        for i in range(N):
            edges.add((min(i, (i+1)%N), max(i, (i+1)%N)))
        for k in range(2, 21):
            edges.add((0, k))
        edge_list = sorted(edges)
        cycles = []
        for L in range(3, N+1):
            if L == N:
                cycles.append(list(range(N)))
            else:
                k1 = L - 1
                if 2 <= k1 <= 20:
                    cycles.append(list(range(0, k1+1)))
                else:
                    k2 = N - L + 1
                    if 2 <= k2 <= 20:
                        cycles.append([0] + list(range(k2, N)))
                    else:
                        cycles.append(list(range(N)))  # Shouldn't happen
        return {'edges': edge_list, 'cycles': cycles}
    
    return result
# EVOLVE-BLOCK-END

# Call the construct function
construct()
