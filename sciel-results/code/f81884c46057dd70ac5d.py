"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 150s kill

    n = 16
    # All possible edges
    all_edges = list(itertools.combinations(range(n), 2))
    edge_index = {e: i for i, e in enumerate(all_edges)}
    m_edges = len(all_edges)  # 120

    # Generate all Q3 subgraphs (embeddings)
    # A Q3 is a cube: 8 vertices, 12 edges.
    # We need to find all sets of 8 vertices and 12 edges that form a Q3.
    # Instead of enumerating all 8-vertex subsets (which is C(16,8)=12870) and checking,
    # we can use the fact that Q3 is bipartite with parts of size 4.
    # A Q3 is isomorphic to the hypercube. Two vertices are adjacent if they differ in exactly 1 bit.
    # So we can represent each vertex of the Q3 as a 3-bit binary number.
    # The edges connect vertices that differ in exactly 1 bit.
    
    # Let's enumerate all possible Q3 subgraphs by choosing 8 vertices and checking if they induce a Q3.
    # But that's expensive. Better: use the structure.
    # A Q3 has 8 vertices and 12 edges. It is 3-regular.
    # Let's just enumerate all 8-vertex subsets and check if the induced subgraph is a Q3.
    
    def is_q3(vertices, adj):
        """Check if the induced subgraph on vertices is isomorphic to Q3."""
        if len(vertices) != 8:
            return False
        verts = list(vertices)
        # Build adjacency within these vertices
        sub_adj = {v: set() for v in verts}
        for u in verts:
            for w in adj.get(u, []):
                if w in sub_adj:
                    sub_adj[u].add(w)
        # Check 3-regular
        for v in verts:
            if len(sub_adj[v]) != 3:
                return False
        # Check bipartite with parts of size 4
        # Try to 2-color
        color = {}
        for start in verts:
            if start not in color:
                color[start] = 0
                queue = [start]
                while queue:
                    u = queue.pop()
                    for w in sub_adj[u]:
                        if w not in color:
                            color[w] = 1 - color[u]
                            queue.append(w)
                        elif color[w] == color[u]:
                            return False
        part0 = [v for v in verts if color[v] == 0]
        part1 = [v for v in verts if color[v] == 1]
        if len(part0) != 4 or len(part1) != 4:
            return False
        # Check that each vertex in part0 is connected to exactly 3 vertices in part1
        # and the structure is a cube.
        # For a Q3, the bipartite adjacency matrix (4x4) has row sums 3 and col sums 3.
        # Also, the graph should be connected and have girth 4.
        # Let's just check that it's isomorphic to the standard Q3.
        # Standard Q3: vertices 0..7, edges between i and j if popcount(i^j)==1.
        std_edges = set()
        for i in range(8):
            for j in range(i+1, 8):
                if bin(i ^ j).count('1') == 1:
                    std_edges.add((min(i,j), max(i,j)))
        # Map our vertices to 0..7 and check
        # This is a small isomorphism check
        from collections import defaultdict
        # Use canonical form: sort vertices by their neighborhood pattern
        # Simpler: try all 8! permutations? No, too slow.
        # Use a simple backtracking isomorphism check.
        
        def isomorphic(g1, g2):
            """Check if g1 and g2 are isomorphic. Both are 8-vertex 3-regular graphs."""
            n = len(g1)
            if n != len(g2):
                return False
            # Try all permutations? 8! = 40320, feasible.
            # But we can prune.
            # Use degree sequence (all 3, so no help).
            # Use adjacency matrix.
            def to_matrix(adj):
                mat = [[0]*n for _ in range(n)]
                for i in range(n):
                    for j in adj[i]:
                        mat[i][j] = 1
                        mat[j][i] = 1
                return mat
            m1 = to_matrix(g1)
            m2 = to_matrix(g2)
            # Check if m1 is a permutation of m2
            # Try all permutations of vertices
            for perm in itertools.permutations(range(n)):
                # Check if perm maps m1 to m2
                ok = True
                for i in range(n):
                    for j in range(n):
                        if m1[i][j] != m2[perm[i]][perm[j]]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    return True
            return False
        
        # Build g1 as adjacency list indexed 0..7
        idx = {v: i for i, v in enumerate(verts)}
        g1 = [set() for _ in range(8)]
        for u in verts:
            for w in sub_adj[u]:
                g1[idx[u]].add(idx[w])
        
        # Build g2 (standard Q3)
        g2 = [set() for _ in range(8)]
        for i in range(8):
            for j in range(i+1, 8):
                if bin(i ^ j).count('1') == 1:
                    g2[i].add(j)
                    g2[j].add(i)
        
        return isomorphic(g1, g2)
    
    # This is too slow. Let's use a different approach.
    # Precompute all Q3 subgraphs by enumerating all 8-vertex subsets.
    # C(16,8) = 12870. For each, check if it's a Q3.
    # The isomorphism check above is 40320 permutations * 64 comparisons = ~2.5M ops per subset.
    # 12870 * 2.5M = 32 billion ops. Too slow.
    
    # Better approach: use the fact that Q3 is the Cartesian product of three K2s.
    # Or use a faster isomorphism check for 8-vertex 3-regular graphs.
    # Actually, there are only a few 8-vertex 3-regular graphs. Let's enumerate them.
    # Or use a canonical labeling.
    
    # For now, let's use a heuristic: if the induced subgraph is 3-regular, bipartite with parts of size 4,
    # and has exactly 12 edges, it's likely a Q3. But there might be other graphs.
    # The only 8-vertex 3-regular bipartite graphs with parts of size 4 are Q3 and the complete bipartite K4,4 minus a perfect matching? No, K4,4 has 16 edges.
    # Let's just check: 3-regular, bipartite, 4+4, 12 edges.
    # The number of such graphs is small. Let's just assume it's Q3 if it satisfies these conditions.
    # This might give false positives, but for the purpose of forbidding Q3, false positives are conservative (we forbid more graphs than necessary).
    
    q3_subgraphs = []
    for subset in itertools.combinations(range(n), 8):
        # Build adjacency within subset
        sub_adj = {v: set() for v in subset}
        for u in subset:
            for w in all_edges:
                if w[0] == u and w[1] in sub_adj:
                    sub_adj[u].add(w[1])
                elif w[1] == u and w[0] in sub_adj:
                    sub_adj[u].add(w[0])
        # Check 3-regular
        if any(len(sub_adj[v]) != 3 for v in subset):
            continue
        # Check bipartite
        color = {}
        ok_bip = True
        for start in subset:
            if start not in color:
                color[start] = 0
                queue = [start]
                while queue and ok_bip:
                    u = queue.pop()
                    for w in sub_adj[u]:
                        if w not in color:
                            color[w] = 1 - color[u]
                            queue.append(w)
                        elif color[w] == color[u]:
                            ok_bip = False
                            break
        if not ok_bip:
            continue
        part0 = [v for v in subset if color[v] == 0]
        part1 = [v for v in subset if color[v] == 1]
        if len(part0) != 4 or len(part1) != 4:
            continue
        # This is a Q3 candidate
        # Extract the 12 edges
        edges_in_sub = []
        for u in part0:
            for w in sub_adj[u]:
                if w in part1:
                    edges_in_sub.append((min(u,w), max(u,w)))
        if len(edges_in_sub) == 12:
            q3_subgraphs.append(set(edges_in_sub))
    
    # Deduplicate
    q3_subgraphs = list(set(tuple(sorted(s)) for s in q3_subgraphs))
    q3_subgraphs = [list(s) for s in q3_subgraphs]
    
    # Now we have a list of Q3 subgraphs, each a set of 12 edges.
    # We want to find a maximum subset of edges that does not contain any Q3.
    
    # Greedy local search:
    # Start with all edges.
    # While there is a Q3 violation, remove the edge that is in the most violating Q3s.
    
    # For efficiency, let's use a different approach:
    # Start with a dense graph and iteratively remove edges.
    
    # Let's try a simulated annealing approach.
    # State: set of edges (bitmask of 120 bits)
    # Objective: maximize number of edges, subject to no Q3.
    
    # For speed, let's use a greedy approach first.
    
    # Greedy: start with all edges, remove edges that are in many Q3s.
    
    # Count how many Q3s each edge is in
    edge_q3_count = [0] * m_edges
    for q3 in q3_subgraphs:
        for e in q3:
            edge_q3_count[edge_index[e]] += 1
    
    # Start with all edges
    edges_present = [True] * m_edges
    edge_count = m_edges
    
    # Iteratively remove the edge that is in the most remaining Q3s
    # This is a simple heuristic
    
    # Let's use a more sophisticated local search.
    # Start with a random dense graph.
    
    best_edges = None
    best_count = 0
    
    # Try multiple random restarts
    for restart in range(10):
        if time.time() > deadline - 10:
            break
        
        # Start with a random subset of edges
        random.seed(42 + restart)
        prob = 0.8  # start dense
        edges_present = [random.random() < prob for _ in range(m_edges)]
        edge_count = sum(edges_present)
        
        # Local search: iteratively try to add edges or swap edges
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try to add each missing edge
            for i in range(m_edges):
                if not edges_present[i]:
                    # Check if adding edge i creates a Q3
                    # An edge i is in Q3s that contain it.
                    # If all other edges in any such Q3 are present, adding i creates a Q3.
                    creates_q3 = False
                    for q3 in q3_subgraphs:
                        if edge_index[all_edges[i]] in [edge_index[e] for e in q3]:
                            # Check if all other edges in q3 are present
                            all_present = True
                            for e in q3:
                                if e == all_edges[i]:
                                    continue
                                if not edges_present[edge_index[e]]:
                                    all_present = False
                                    break
                            if all_present:
                                creates_q3 = True
                                break
                    if not creates_q3:
                        edges_present[i] = True
                        edge_count += 1
                        improved = True
            
            # Try to swap: remove an edge and add another
            # This is more complex. Let's skip for now.
        
        if edge_count > best_count:
            best_count = edge_count
            best_edges = edges_present[:]
    
    # Convert to list of edges
    result = []
    for i in range(m_edges):
        if best_edges[i]:
            result.append(list(all_edges[i]))
    
    return result
# EVOLVE-BLOCK-END
