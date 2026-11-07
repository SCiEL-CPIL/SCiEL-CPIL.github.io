"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs an undirected simple graph on vertices 0..11 with minimum
    distinct cycle lengths.
    
    Uses a single K_{3,3} component on vertices 0..5, and attaches a
    4-cycle (6-7-8-9-6) with a cross edge 10-11.
    
    Wait - vertex 10 and 11 need degree >= 3. Let me reconsider.
    
    Alternative: Use a single K_{3,3} on vertices 0..5 (6 vertices, degree 3 each),
    then for vertices 6..11, create a structure where all have degree >= 3.
    
    Actually, the simplest improvement: try to find a graph where all cycles
    have the same length. The only way to get 1 distinct cycle length is if
    every simple cycle has the same length.
    
    For a bipartite graph with parts of size 3 and 3 (K_{3,3}), cycles are 4 and 6.
    
    What about a 4-regular bipartite graph? K_{4,4} has cycles of lengths 4, 6, 8.
    
    What about a graph where all cycles are 4-cycles? That would require no 6-cycles.
    A graph where every pair of vertices in the same part has exactly one common neighbor
    in the other part would only have 4-cycles. But with 12 vertices and min degree 3...
    
    Let me think differently. The current score is 2 (two distinct cycle lengths: 4 and 6).
    Can we get to 1?
    
    A graph where all simple cycles have the same length L. If L=4, then no 6-cycles.
    If L=6, then no 4-cycles (but we need girth >= 4, so no 3-cycles, and no 4-cycles either).
    
    For L=4: every simple cycle is a 4-cycle. This means the graph is a "4-cycle graph" 
    in the sense that there are no longer cycles. But any connected bipartite graph with 
    min degree >= 2 will have cycles. If we ensure no 6-cycles exist...
    
    Actually, let me try a different approach: use a search to find a graph with 
    minimum distinct cycle lengths, verifying constraints.
    """
    import itertools
    import time
    
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill
    
    n = 12
    
    # We need to find a graph on 12 vertices with:
    # 1. min degree >= 3
    # 2. triangle-free (girth >= 4)
    # 3. contains at least one cycle
    # 4. minimize number of distinct cycle lengths
    
    # Strategy: Try known constructions first, then search if time permits.
    
    # Construction 1: Two K_{3,3} components (current best: 2 distinct lengths {4,6})
    def two_k33():
        edges = []
        for u in range(3):
            for v in range(3, 6):
                edges.append([u, v])
        for u in range(6, 9):
            for v in range(9, 12):
                edges.append([u, v])
        return edges
    
    # Construction 2: Single K_{3,3} + another structure
    # Let's try: K_{3,3} on 0-5, then 6-11 form a 4-regular bipartite subgraph
    # K_{3,3} on 6-8 and 9-11? That's another K_{3,3}, same as above.
    
    # Construction 3: Try to make all cycles length 4.
    # A graph where every cycle is a 4-cycle is called a "4-cycle graph" or has girth 4
    # and no cycles longer than 4. But in a connected graph with min degree >= 3,
    # there will be longer cycles unless the graph is very small.
    
    # Actually, for a connected bipartite graph with parts A and B, a 4-cycle exists
    # iff two vertices in A share a common neighbor in B. A 6-cycle exists iff there's
    # a path of length 5 between two adjacent vertices.
    
    # Let me try: Can we construct a graph where all cycles are exactly length 4?
    # This requires that for any two vertices u, v in the same bipartition, 
    # they have at most one common neighbor (otherwise multiple 4-cycles, but that's ok,
    # still length 4). The issue is 6-cycles: a 6-cycle u1-v1-u2-v2-u3-v3-u1 exists
    # if u1~v1, v1~u2, u2~v2, v2~u3, u3~v3, v3~u1, with all distinct.
    
    # To avoid 6-cycles in a bipartite graph, we need that for any three distinct 
    # vertices u1, u2, u3 in A, there don't exist v1, v2, v3 in B such that
    # u1~v1, v1~u2, u2~v2, v2~u3, u3~v3, v3~u1.
    
    # This is hard to guarantee. Let me just try a computational search for small cases.
    
    # Actually, let me think about what's achievable. With 12 vertices and min degree 3,
    # we have at least 18 edges. 
    
    # Let me try a different construction: the Petersen graph has 10 vertices, 
    # is 3-regular, girth 5. Not bipartite, has triangles? No, girth 5 means no triangles.
    # But it has 5-cycles and other cycles. Not helpful directly.
    
    # Let me try: 3-regular bipartite graph on 12 vertices.
    # If we can find one where all cycles are the same length...
    
    # For a 3-regular bipartite graph, the number of 4-cycles and 6-cycles depends on structure.
    
    # Let me just do a quick search over 3-regular bipartite graphs.
    
    # Bipartition: A = {0,1,2,3,4,5}, B = {6,7,8,9,10,11}
    # Each vertex in A connects to exactly 3 vertices in B, and vice versa.
    
    # There are C(6,3) = 20 possible neighbor sets for each vertex in A.
    # Total possible graphs: 20^6 = 64,000,000. Too many to brute force naively.
    
    # But we can be smarter: fix the first few vertices' neighbors and prune.
    
    # Actually, let me try a simpler approach: just verify the current construction
    # and see if we can do better with a targeted search.
    
    # Let me try: can we get 1 distinct cycle length?
    # The only way is if all cycles have the same length.
    # In a bipartite graph, cycle lengths are even. So all cycles could be 4, or all 6, etc.
    
    # For all cycles to be 4: no 6-cycles. 
    # For all cycles to be 6: no 4-cycles, but girth must be >= 4, so girth = 6.
    
    # A 3-regular bipartite graph with girth 6 on 12 vertices: this is the 
    # "cubic bipartite graph of girth 6". The smallest such is the Heawood graph 
    # (14 vertices). For 12 vertices, it might not exist.
    
    # Let me check: a 3-regular bipartite graph on 12 vertices with girth 6.
    # This would mean no 4-cycles. Two vertices in A cannot share a neighbor in B.
    # So the neighborhoods of vertices in A are disjoint pairs... wait, each vertex 
    # in A has 3 neighbors in B. If no two vertices in A share a neighbor, then 
    # 6*3 = 18 > 6, impossible. So 4-cycles must exist in any 3-regular bipartite 
    # graph on 12 vertices.
    
    # So girth 6 is impossible. The best we can hope for is girth 4 with all cycles 
    # being 4-cycles, which means no 6-cycles.
    
    # Let me search for a 3-regular bipartite graph on 12 vertices with no 6-cycles.
    
    # If we can find one, it has 1 distinct cycle length (4), giving score 1.0 * seed.
    
    # Let me do a bounded search.
    
    from itertools import combinations
    
    # A = {0,1,2,3,4,5}, B = {6,7,8,9,10,11}
    # Each vertex in A chooses 3 neighbors from B.
    # We need the resulting graph to be 3-regular on B as well (each B vertex has degree 3).
    
    # Let's represent the adjacency as a 6x6 binary matrix M where M[i][j] = 1 
    # if vertex i in A is connected to vertex j in B.
    # Row sums = 3, column sums = 3.
    
    # We want to avoid 6-cycles. A 6-cycle in bipartite graph corresponds to a 
    # 3x3 submatrix with a specific pattern... actually, a 6-cycle exists if there's
    # a cycle of length 6 in the bipartite graph, which means there exist distinct
    # a1,a2,a3 in A and b1,b2,b3 in B such that a1-b1, b1-a2, a2-b2, b2-a3, a3-b3, b3-a1.
    
    # This is equivalent to: in the bipartite graph, there's a path of length 5 
    # between two adjacent vertices.
    
    # Let me just try random 3-regular bipartite graphs and check.
    
    import random
    
    def is_3reg_bipartite_valid(M):
        """Check if M is a valid 3-regular bipartite adjacency matrix."""
        for i in range(6):
            if sum(M[i]) != 3:
                return False
        for j in range(6):
            col_sum = sum(M[i][j] for i in range(6))
            if col_sum != 3:
                return False
        return True
    
    def has_6cycle(M):
        """Check if the bipartite graph has a 6-cycle."""
        # A 6-cycle: a1-b1-a2-b2-a3-b3-a1
        # Enumerate triples of A vertices and triples of B vertices
        A_verts = list(range(6))
        B_verts = list(range(6))
        for a1, a2, a3 in itertools.permutations(A_verts, 3):
            for b1, b2, b3 in itertools.permutations(B_verts, 3):
                if M[a1][b1] and M[a2][b1] and M[a2][b2] and M[a3][b2] and M[a3][b3] and M[a1][b3]:
                    return True
        return False
    
    def cycle_lengths(M):
        """Find all distinct cycle lengths in the bipartite graph."""
        # Build full graph
        adj = {i: set() for i in range(12)}
        for i in range(6):
            for j in range(6):
                if M[i][j]:
                    adj[i].add(6 + j)
                    adj[6 + j].add(i)
        
        # Find all simple cycles using DFS
        lengths = set()
        
        def dfs_cycles(start, current, visited, path):
            if time.time() - start_time > DEADLINE:
                return
            for neighbor in sorted(adj[current]):
                if neighbor == start and len(path) >= 4:
                    lengths.add(len(path))
                elif neighbor not in visited and neighbor > start:  # only count each cycle once
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs_cycles(start, neighbor, visited, path)
                    path.pop()
                    visited.discard(neighbor)
        
        for s in range(12):
            visited = {s}
            dfs_cycles(s, s, visited, [s])
        
        return lengths
    
    # Try random 3-regular bipartite graphs
    best_result = None
    best_score = float('inf')
    
    # Generate random 3-regular bipartite adjacency matrices
    def random_3reg_bipartite():
        """Generate a random 3-regular bipartite 6x6 binary matrix."""
        # Start with random assignment
        while True:
            M = [[0]*6 for _ in range(6)]
            # Assign neighbors for each A vertex
            for i in range(6):
                neighbors = random.sample(range(6), 3)
                for j in neighbors:
                    M[i][j] = 1
            # Check column sums
            col_sums = [sum(M[i][j] for i in range(6)) for j in range(6)]
            if all(c == 3 for c in col_sums):
                return M
            # Retry
    
    # Try a limited number of random graphs
    for trial in range(50000):
        if time.time() - start_time > DEADLINE:
            break
        M = random_3reg_bipartite()
        if not has_6cycle(M):
            # Check if it has any cycles (it should, since min degree 3)
            cl = cycle_lengths(M)
            if cl:
                score = len(cl)
                if score < best_score:
                    best_score = score
                    best_result = M
                    if score == 1:
                        break
    
    # If we found a 1-distinct-length graph, use it
    if best_result is not None and best_score == 1:
        edges = []
        for i in range(6):
            for j in range(6):
                if best_result[i][j]:
                    edges.append([int(i), int(6 + j)])
        return edges
    
    # Otherwise, fall back to two K_{3,3} (score 2)
    return two_k33()

# EVOLVE-BLOCK-END
