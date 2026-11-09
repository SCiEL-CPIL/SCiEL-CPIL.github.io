"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START

# Q3 edges (the 3-cube): vertices 0..7, edges connecting vertices that differ in exactly one bit
Q3_EDGES = []
for v in range(8):
    for b in range(3):
        u = v ^ (1 << b)
        if u > v:
            Q3_EDGES.append((v, u))
# Q3_EDGES has 12 edges

# Precompute Q3 edge list as a set of frozensets for fast lookup
Q3_EDGE_SET = set(frozenset(e) for e in Q3_EDGES)

def has_q3_subgraph(adj, edge_set):
    """
    Check if the graph (adj: list of sets, edge_set: set of frozenset pairs) contains Q3 as a subgraph.
    A subgraph means: there exists an injection f: {0..7} -> V such that for every edge (u,v) in Q3,
    (f(u), f(v)) is in the graph.
    
    We check all 8-vertex subsets and all bijections from Q3 vertices to them,
    but that's 8! * C(16,8) which is too many. Instead, use a smarter approach.
    
    Actually, for a general subgraph check: we need to find 8 vertices v0..v7 such that
    for each Q3 edge (a,b), the pair (va, vb) is an edge in G.
    
    A faster approach: try all ordered 8-tuples of distinct vertices? That's 16P8 = 16!/(8!) ≈ 5e8, too many.
    
    Better: use the structure. Q3 is bipartite with parts {0,3,5,6} and {1,2,4,7} (even/odd parity).
    For each choice of 4 vertices for the even part and 4 for the odd part, check if the bipartite
    graph between them contains Q3 as a subgraph.
    
    Even part of Q3: {0,3,5,6} (vertices with even popcount)
    Odd part of Q3: {1,2,4,7} (vertices with odd popcount)
    
    Q3 bipartite edges (even-odd):
    0-1, 0-2, 0-4
    3-1, 3-5, 3-7
    5-2, 5-4, 5-7
    6-2, 6-4, 6-7
    
    Wait, let me recompute. Q3 vertices 0..7, edge if Hamming distance 1.
    0 (000) connects to 1(001), 2(010), 4(100)
    1 (001) connects to 0, 3(011), 5(101)
    2 (010) connects to 0, 3(011), 6(110)
    3 (011) connects to 1, 2, 7(111)
    4 (100) connects to 0, 5(101), 6(110)
    5 (101) connects to 1, 4, 7(111)
    6 (110) connects to 2, 4, 7(111)
    7 (111) connects to 3, 5, 6
    
    Even parity (popcount even): 0(0), 3(2), 5(2), 6(2)
    Odd parity (popcount odd): 1(1), 2(1), 4(1), 7(3)
    
    Bipartite edges (even, odd):
    0-1, 0-2, 0-4
    3-1, 3-2, 3-7
    5-1, 5-4, 5-7
    6-2, 6-4, 6-7
    """
    even_q3 = [0, 3, 5, 6]
    odd_q3 = [1, 2, 4, 7]
    
    # Bipartite adjacency in Q3: for each even vertex, which odd vertices it connects to
    q3_bip_adj = {
        0: [1, 2, 4],
        3: [1, 2, 7],
        5: [1, 4, 7],
        6: [2, 4, 7]
    }
    
    n = len(adj)
    if n < 8:
        return False
    
    # Try all choices of 4 vertices for even side, 4 for odd side
    # C(16,4) * C(12,4) = 1820 * 495 = ~900k, and for each we check 4! * 4! = 576 bijections
    # Total: ~520 million, too slow in Python.
    
    # Better: iterate over 8-vertex subsets, then check if Q3 is a subgraph of the induced subgraph.
    # C(16,8) = 12870 subsets. For each, check if Q3 is a subgraph.
    # For a fixed 8-vertex set, we need to check if there's an injection from Q3 to these 8 vertices
    # preserving edges. Since Q3 has 8 vertices and we're mapping to exactly 8 vertices, it's a bijection.
    # So we need to check if the induced subgraph on these 8 vertices contains Q3 as a spanning subgraph.
    # That means: is there a bijection f: {0..7} -> S such that for all Q3 edges (a,b), (f(a),f(b)) in E?
    
    # For each 8-subset, we can try to match. But 8! = 40320 per subset, times 12870 = 519M, too slow.
    
    # Alternative: use the bipartite structure. For each 8-subset S, split into two 4-sets A, B (the bipartition).
    # But the bipartition of Q3 is fixed (even/odd). So we need to assign 4 vertices of S to even positions and 4 to odd.
    # There are C(8,4) = 70 ways to choose which 4 go to even. For each, check if the bipartite graph between
    # the chosen 4 (as even) and the other 4 (as odd) contains Q3's bipartite graph as a subgraph.
    # Checking if a 4x4 bipartite graph contains a specific 4x4 bipartite graph as a subgraph:
    # We need a bijection from even_q3 to A and odd_q3 to B such that all Q3 bipartite edges are present.
    # That's 4! * 4! = 576 checks per (A,B) split, times 70 splits per 8-subset, times 12870 subsets = 519M. Still too slow.
    
    # Let's use a different approach: random restarts with local search, and for the Q3 check,
    # use a randomized or heuristic check that is fast but might miss some Q3s.
    # Actually, for correctness we need exact check. Let me think of a faster exact check.
    
    # Key insight: Q3 is a specific graph. We can precompute all 8-vertex subsets that could contain Q3
    # by checking the degree sequence and other invariants first.
    
    # Actually, let's just do the full check but optimize with early termination.
    # For each 8-subset S:
    #   For each way to assign the 8 Q3 vertices to S (i.e., each permutation of 8 elements applied to S):
    #     Check if all 12 Q3 edges map to edges in G.
    #     Early terminate if any Q3 edge is missing.
    #   
    # 12870 * 40320 = 519M permutations, each checking up to 12 edges. Too slow in pure Python.
    
    # Alternative: use numpy or a more clever approach.
    
    # Let me try a different strategy: instead of checking all subsets, use the fact that Q3 has specific
    # properties. For instance, every vertex in Q3 has degree 3. So if we find 8 vertices where at least
    # one has degree < 3 in the induced subgraph, it can't be Q3.
    
    # Actually, the most practical approach for 16 vertices: use a SAT-like or backtracking search
    # to find a Q3 embedding, with pruning.
    
    # Let me implement a backtracking search for Q3 embedding:
    # - Order Q3 vertices in a way that maximizes pruning (e.g., by degree, all degree 3).
    # - Try to assign Q3 vertices to graph vertices, maintaining consistency.
    
    # For 16 vertices, this might be feasible with good pruning.
    
    # Actually, let me just use the brute force but with numpy for speed, or accept that the check
    # is expensive and limit the number of edges we add.
    
    # Given time constraints, let me use a greedy approach with a fast (but potentially incomplete) Q3 check,
    # and then verify the final result with a complete check.
    
    # For the fast check: only check Q3 embeddings where the 8 vertices form a "natural" cube structure,
    # i.e., use the fact that Q3 is a Cayley graph of Z_2^3. So any Q3 subgraph corresponds to choosing
    # 3 "directions" (edges) and 8 vertices that are affine combinations.
    
    # Actually, the cleanest fast check: for each choice of 4 vertices (one for each "corner" of a potential
    # cube in a specific orientation), verify. But this is getting complicated.
    
    # Let me just do the full check with early termination and hope it's fast enough for 16 vertices.
    
    # Actually, I realize: for the greedy algorithm, we don't need to check Q3 after every edge addition.
    # We can check at the end, and if Q3 is found, remove edges. But that's not optimal.
    
    # Let me use a different approach: start with a known Q3-free graph (like K_{3,13} with 39 edges)
    # and try to add edges one by one, checking Q3 after each addition. If the check is too slow,
    # we can batch-check.
    
    # For now, let me implement the check and see.
    
    # Actually, I'll use a randomized approach: try many random 8-subsets and permutations, and if we
    # find a Q3, return True. If we don't find one after many tries, assume it's Q3-free (with some error).
    # But the task requires exact search, so we need a complete check.
    
    # Let me implement the complete check with numpy for speed.
    
    import numpy as np
    
    # Build adjacency matrix
    n = len(adj)
    adj_mat = np.zeros((n, n), dtype=bool)
    for u in range(n):
        for v in adj[u]:
            adj_mat[u, v] = True
            adj_mat[v, u] = True
    
    # Check all 8-subsets
    vertices = list(range(n))
    for subset in combinations(vertices, 8):
        # For this subset, check if Q3 is a subgraph
        # Try all 8! permutations? No, too slow.
        # Instead, use the bipartite structure.
        # Even Q3 vertices: 0,3,5,6; Odd: 1,2,4,7
        # For each way to assign 4 subset vertices to even positions:
        even_positions = [0, 3, 5, 6]
        odd_positions = [1, 2, 4, 7]
        
        # Q3 bipartite adjacency (even -> odd)
        q3_e2o = {
            0: [1, 2, 4],
            3: [1, 2, 7],
            5: [1, 4, 7],
            6: [2, 4, 7]
        }
        
        # For each choice of 4 vertices from subset for even side:
        for even_assign in combinations(subset, 4):
            odd_assign = tuple(v for v in subset if v not in even_assign)
            
            # Check all 4! * 4! = 576 bijections
            from itertools import permutations
            for perm_e in permutations(even_assign):
                # perm_e[i] is the graph vertex assigned to Q3 even vertex even_positions[i]
                e_map = {even_positions[i]: perm_e[i] for i in range(4)}
                for perm_o in permutations(odd_assign):
                    o_map = {odd_positions[i]: perm_o[i] for i in range(4)}
                    
                    # Check all Q3 bipartite edges
                    valid = True
                    for e in even_positions:
                        for o in q3_e2o[e]:
                            if not adj_mat[e_map[e], o_map[o]]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        return True
    
    return False


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10-second safety margin
    
    n = 16
    # Start with K_{3,13}
    adj = [set() for _ in range(n)]
    edges = []
    
    # Add K_{3,13} edges: vertices 0,1,2 connect to 3..15
    for a in range(3):
        for b in range(3, 16):
            adj[a].add(b)
            adj[b].add(a)
            edges.append([a, b])
    
    # Build edge set for fast lookup
    edge_set = set()
    for e in edges:
        edge_set.add(frozenset(e))
    
    # Try to add edges between vertices 0,1,2 (the "top" of K_{3,13})
    # and between vertices 3..15 (the "bottom")
    
    # First, try edges among {0,1,2}
    for a in range(3):
        for b in range(a+1, 3):
            if time.time() > deadline:
                return edges
            # Check if adding (a,b) creates Q3
            adj[a].add(b)
            adj[b].add(a)
            new_edge_set = edge_set | {frozenset((a,b))}
            if has_q3_subgraph(adj, new_edge_set):
                # Remove the edge
                adj[a].discard(b)
                adj[b].discard(a)
            else:
                edges.append([a, b])
                edge_set.add(frozenset((a,b)))
    
    # Try edges among {3..15}
    for a in range(3, 16):
        for b in range(a+1, 16):
            if time.time() > deadline:
                return edges
            adj[a].add(b)
            adj[b].add(a)
            new_edge_set = edge_set | {frozenset((a,b))}
            if has_q3_subgraph(adj, new_edge_set):
                adj[a].discard(b)
                adj[b].discard(a)
            else:
                edges.append([a, b])
                edge_set.add(frozenset((a,b)))
    
    return edges

# EVOLVE-BLOCK-END
