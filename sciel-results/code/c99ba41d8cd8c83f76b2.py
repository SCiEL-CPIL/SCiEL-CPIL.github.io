"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Q3 adjacency: vertices 0-7, edges between i and j if they differ in exactly 1 bit
    q3_vertices = list(range(8))
    q3_edges = set()
    for i in range(8):
        for j in range(i+1, 8):
            diff = bin(i ^ j).count('1')
            if diff == 1:
                q3_edges.add((i, j))
    
    def is_q3_subgraph(adj_matrix, vertices_subset):
        """Check if the subgraph on vertices_subset contains Q3 as a subgraph."""
        n = len(vertices_subset)
        if n < 8:
            return False
        
        # Map subset vertices to 0..n-1
        idx_map = {v: i for i, v in enumerate(vertices_subset)}
        
        # Build local adjacency for the 8 vertices we'll map Q3 into
        # We need to find an injection from Q3's 8 vertices to these n vertices
        # such that all Q3 edges are preserved.
        
        # Use backtracking: assign Q3 vertices to subset vertices
        q3_adj = [set() for _ in range(8)]
        for (a, b) in q3_edges:
            q3_adj[a].add(b)
            q3_adj[b].add(a)
        
        # Order Q3 vertices by degree (all degree 3, so any order)
        # Use a canonical ordering
        order = list(range(8))
        
        assignment = [-1] * 8  # q3_vertex -> subset_vertex
        used = set()
        
        def backtrack(q3_idx):
            if q3_idx == 8:
                return True
            for sv in vertices_subset:
                if sv in used:
                    continue
                # Check if sv is compatible with already assigned neighbors
                compatible = True
                for prev_q3 in range(q3_idx):
                    if prev_q3 in q3_adj[order[q3_idx]]:
                        prev_sv = assignment[prev_q3]
                        if not adj_matrix[sv][prev_sv]:
                            compatible = False
                            break
                if compatible:
                    used.add(sv)
                    assignment[order[q3_idx]] = sv
                    if backtrack(q3_idx + 1):
                        return True
                    used.remove(sv)
                    assignment[order[q3_idx]] = -1
            return False
        
        return backtrack(0)
    
    def count_q3_subgraphs(edges_list):
        """Count number of Q3 subgraphs in the graph."""
        n = 16
        adj_matrix = [[0]*n for _ in range(n)]
        for (u, v) in edges_list:
            adj_matrix[u][v] = 1
            adj_matrix[v][u] = 1
        
        count = 0
        for subset in itertools.combinations(range(n), 8):
            if is_q3_subgraph(adj_matrix, subset):
                count += 1
        return count
    
    def verify_q3_free(edges_list):
        """Verify the graph is Q3-free."""
        return count_q3_subgraphs(edges_list) == 0
    
    # Candidate 1: K_{3,13} - known Q3-free (39 edges)
    cand1 = [[a, b] for a in range(3) for b in range(3, 16)]
    
    # Candidate 2: Try odd-parity hypercube (connect if differ in odd number of bits)
    # Vertices 0-15 as 4-bit strings
    cand2 = []
    for i in range(16):
        for j in range(i+1, 16):
            diff = bin(i ^ j).count('1')
            if diff % 2 == 1:
                cand2.append([i, j])
    # This has 64 edges
    
    # Candidate 3: Even-parity hypercube (connect if differ in even number of bits, >0)
    cand3 = []
    for i in range(16):
        for j in range(i+1, 16):
            diff = bin(i ^ j).count('1')
            if diff % 2 == 0 and diff > 0:
                cand3.append([i, j])
    # This has C(4,2) + C(4,4) = 6+1 = 7 edges per vertex? No...
    # diff=2: C(4,2)=6, diff=4: 1. So degree 7, total edges 16*7/2=56
    
    # Candidate 4: Complete bipartite K_{4,12} minus some edges to break Q3
    # Actually let's try a different approach: start dense and remove edges
    
    # Candidate 5: Use a Turan-like construction with 3 parts
    # Parts of sizes 5,5,6. Connect all between parts, none within.
    # This is K_{5,5,6} complete 3-partite
    cand5 = []
    part1 = list(range(5))
    part2 = list(range(5, 10))
    part3 = list(range(10, 16))
    for a in part1:
        for b in part2:
            cand5.append([a, b])
        for b in part3:
            cand5.append([a, b])
    for a in part2:
        for b in part3:
            cand5.append([a, b])
    # Edges: 5*5 + 5*6 + 5*6 = 25+30+30 = 85
    
    # Try candidates, prefer more edges, must be Q3-free
    candidates = [
        (len(cand5), cand5),
        (len(cand2), cand2),
        (len(cand3), cand3),
        (len(cand1), cand1),
    ]
    
    candidates.sort(key=lambda x: -x[0])
    
    best = cand1
    for edge_count, edges in candidates:
        if time.time() > deadline - 30:
            break
        try:
            if verify_q3_free(edges):
                best = edges
                break  # Found the densest Q3-free graph
        except Exception:
            continue
    
    return best
# EVOLVE-BLOCK-END
