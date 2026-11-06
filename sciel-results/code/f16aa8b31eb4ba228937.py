import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_edges_set = set(all_edges)
    
    # Helper to check if a set of 8 vertices induces a Q3 subgraph
    # This is the core constraint.
    # We will use a fast check: 
    # 1. The induced subgraph on these 8 vertices must have at least 12 edges? No, that's not sufficient.
    # 2. It must contain a subgraph isomorphic to Q3.
    # Since we are working with small graphs, we can check isomorphism to Q3 for the induced subgraph?
    # No, the induced subgraph might have more edges.
    # We need to check if there EXISTS a mapping.
    
    # Let's precompute the Q3 structure.
    # Q3 vertices: 0..7
    # Edges: (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)
    # Let's verify this is Q3.
    # 0 is connected to 1,2,4.
    # 1 is connected to 0,3,5.
    # 2 is connected to 0,3,6.
    # 3 is connected to 1,2,7.
    # 4 is connected to 0,5,6.
    # 5 is connected to 1,4,7.
    # 6 is connected to 2,4,7.
    # 7 is connected to 3,5,6.
    # Yes, this is Q3.
    
    q3_edges_canonical = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7), (6, 7)
    ]
    
    # Precompute all permutations of 8 vertices? 
    # No, that's 40320.
    # We can iterate over all 8-vertex subsets (12870).
    # For each subset, we need to check if the induced subgraph contains a Q3.
    # This is equivalent to checking if there is a bijection from {0..7} to the subset 
    # such that all q3_edges_canonical are present in the graph.
    
    # To speed this up, we can use a backtracking search for the mapping.
    # Or, we can just check all 40320 permutations? 12870 * 40320 is too much.
    
    # Better: 
    # Iterate over all 8-vertex subsets S.
    # Check if G[S] contains a Q3.
    # We can do this by trying to map the vertices of Q3 to S.
    # Since Q3 is 3-regular, we can start with a vertex of degree 3 in Q3 (all are).
    
    # Let's implement a function `contains_q3(adj, S)` that returns True if G[S] contains a Q3.
    # We will use a simple backtracking search.
    
    def contains_q3(adj, S):
        # S is a tuple of 8 vertices
        # We want to map q3_vertices 0..7 to S such that edges are preserved.
        # Let's try all permutations of S? 8! = 40320.
        # 12870 * 40320 = 500M. Too slow in Python.
        
        # We need a smarter check.
        # Observation: Q3 is the unique 3-regular bipartite graph on 8 vertices.
        # So, if G[S] contains a Q3, then G[S] must have a subgraph that is 3-regular and bipartite.
        # This implies G[S] must have at least 12 edges.
        # Also, the degrees in G[S] must be at least 3 for the vertices involved in the Q3.
        
        # Let's check if G[S] has a 3-regular subgraph.
        # This is still hard.
        
        # Alternative:
        # Since we are doing local search, we can just check for violations when we add an edge.
        # When we add an edge (u,v), we only need to check if this edge is part of a NEW Q3.
        # A Q3 containing (u,v) must use 6 other vertices.
        # We can enumerate the possible Q3s containing (u,v).
        
        # How many Q3s contain a specific edge in K16?
        # Total Q3s in K16: C(16,8) * 840 = 10.8M.
        # Each Q3 has 12 edges. Total edge-Q3 incidences: 10.8M * 12 = 130M.
        # Average Q3s per edge: 130M / 120 = ~1.1M.
        # This is still too many to check per edge addition.
        
        # Let's go back to the "find any Q3" approach but make it faster.
        # The current program's `find_q3` is likely the bottleneck.
        # It iterates over vertices, then triples of neighbors, etc.
        
        # Let's try a different strategy:
        # 1. Start with a random graph with p=0.6.
        # 2. While Q3 exists:
        #    - Find a Q3 using a fast heuristic.
        #    - Remove an edge from it.
        # 3. Hill climb: try adding edges.
        
        # To make `find_q3` faster, we can use a precomputed list of "candidate Q3s".
        # Or, we can use the fact that Q3 is bipartite.
        # A Q3 has two parts of size 4.
        # So, we can iterate over all pairs of 4-sets (S1, S2) from the 16 vertices.
        # Check if there is a perfect matching between S1 and S2 that forms a Q3?
        # No, Q3 is not just a matching.
        
        # Let's stick to the backtracking search for a specific Q3 embedding, but optimize it.
        # We will iterate over all 8-vertex subsets S.
        # For each S, we check if G[S] contains a Q3.
        # To check if G[S] contains a Q3, we can use the following:
        # - Check if G[S] has at least 12 edges.
        # - Check if G[S] is bipartite. (If not, no Q3).
        # - If bipartite, check if it contains a 3-regular subgraph.
        
        # Actually, the simplest robust method is to just check all 40320 permutations for a fixed S,
        # but only if G[S] has at least 12 edges.
        # How many S have >= 12 edges?
        # If p=0.6, expected edges in G[S] is 28 * 0.6 = 16.8.
        # So most S will be checked.
        
        # Let's try to optimize the permutation check.
        # Instead of 40320, we can fix the mapping of the first few vertices.
        # Map 0 -> s0.
        # Map 1,2,4 -> s1,s2,s3 (neighbors of 0 in Q3).
        # They must be neighbors of s0 in G.
        # And they must be pairwise non-adjacent.
        
        def check_q3_in_subset(adj, S):
            # S is a list of 8 vertices
            if len(S) != 8: return False
            
            # Count edges in G[S]
            edge_count = 0
            for i in range(8):
                for j in range(i+1, 8):
                    if S[j] in adj[S[i]]:
                        edge_count += 1
            if edge_count < 12:
                return False
            
            # Backtracking search
            # q3_adj = {0: {1,2,4}, 1: {0,3,5}, ...}
            q3_adj = [set() for _ in range(8)]
            for u, v in q3_edges_canonical:
                q3_adj[u].add(v)
                q3_adj[v].add(u)
                
            mapping = [-1] * 8
            # Map 0 to each vertex in S
            for s0 in S:
                mapping[0] = s0
                # Map 1,2,4 to neighbors of s0 in G that are in S
                neighbors_in_s = [x for x in S if x in adj[s0] and x != s0]
                # We need to choose 3 from neighbors_in_s for 1,2,4
                # And they must be pairwise non-adjacent in G
                for comb in combinations(neighbors_in_s, 3):
                    s1, s2, s3 = comb
                    # Check pairwise non-adjacency
                    if s2 in adj[s1] or s3 in adj[s1] or s3 in adj[s2]:
                        continue
                    
                    mapping[1] = s1
                    mapping[2] = s2
                    mapping[4] = s3
                    
                    # Now map 3,5,6,7
                    # 3 is neighbor of 1 and 2 in Q3
                    # 5 is neighbor of 1 and 4
                    # 6 is neighbor of 2 and 4
                    # 7 is neighbor of 3,5,6
                    
                    # Candidates for 3: neighbors of s1 and s2 in G, in S, not used
                    used = {s0, s1, s2, s3}
                    cand3 = [x for x in S if x in adj[s1] and x in adj[s2] and x not in used]
                    cand5 = [x for x in S if x in adj[s1] and x in adj[s3] and x not in used]
                    cand6 = [x for x in S if x in adj[s2] and x in adj[s3] and x not in used]
                    
                    for s3v in cand3:
                        mapping[3] = s3v
                        # Update used
                        used2 = used | {s3v}
                        
                        # Recalculate cand5, cand6 excluding s3v
                        c5 = [x for x in cand5 if x != s3v]
                        c6 = [x for x in cand6 if x != s3v]
                        
                        for s5 in c5:
                            mapping[5] = s5
                            used3 = used2 | {s5}
                            
                            c6_new = [x for x in c6 if x != s5]
                            
                            for s6 in c6_new:
                                mapping[6] = s6
                                used4 = used3 | {s6}
                                
                                # 7 must be the remaining vertex
                                remaining = [x for x in S if x not in used4]
                                if len(remaining) != 1: continue
                                s7 = remaining[0]
                                mapping[7] = s7
                                
                                # Check edges for 7
                                # 7 must be connected to 3, 5, 6 in G
                                if s3v in adj[s7] and s5 in adj[s7] and s6 in adj[s7]:
                                    return True
                                
                                mapping[6] = -1
                            mapping[5] = -1
                        mapping[3] = -1
                    mapping[4] = -1
                    mapping[2] = -1
                    mapping[1] = -1
                mapping[0] = -1
            return False
        
        # This is still potentially slow.
        # Let's use a simpler heuristic for `find_q3`:
        # Just pick a random 8-vertex subset and check it.
        # Repeat until we find a Q3 or we've tried many.
        
        def find_q3_fast(adj, max_tries=100):
            for _ in range(max_tries):
                S = random.sample(range(n), 8)
                if check_q3_in_subset(adj, S):
                    # We found a Q3, but we need the edges to remove one.
                    # The mapping is lost in `check_q3_in_subset`.
                    # Let's modify it to return the edges.
                    pass
            return None
        
        # This approach is getting complicated.
        # Let's go back to the previous program's `find_q3` which was working, 
        # but optimize the removal heuristic and the search loop.
        
        # The previous program scored 1.359.
        # The inspiration 1 scored 1.385 using a static K3,13? No, it said "static construction (K3,13) which yields 39 edges".
        # Wait, 39 edges is low. The seed was 29.
        # 1.359 * 29 = 39.4.
        # So the current program finds ~39-40 edges.
        
        # Is there a known better construction?
        # ex(16, Q3).
        # Turan graph T_4(16) has 4 parts of size 4.
        # Edges: 4 * C(4,2) * 3? No.
        # T_4(16) edges = 16^2/4 - 4 = 64 - 4 = 60? No.
        # Number of edges in T_r(n) is (1 - 1/r) n^2 / 2.
        # For r=4, n=16: (3/4) * 256 / 2 = 96.
        # But T_4(16) is K_{4,4,4,4}.
        # Does K_{4,4,4,4} contain a Q3?
        # Q3 is bipartite. A bipartite subgraph of a complete multipartite graph?
        # Yes, it can.
        # Pick 4 vertices from Part 1 and 4 from Part 2.
        # The induced subgraph is K_{4,4}.
        # K_{4,4} contains a Q3?
        # Q3 is a subgraph of K_{4,4}.
        # Yes, Q3 is a 3-regular subgraph of K_{4,4}.
        # So T_4(16) contains Q3.
        
        # What about T_5(16)?
        # Parts: 4,3,3,3,3.
        # Edges: sum_{i<j} n_i n_j.
        # Total = (16^2 - (16+9+9+9+9))/2 = (256 - 52)/2 = 102.
        # Still likely contains Q3.
        
        # The maximum Q3-free graph is likely much sparser.
        
        # Let's stick to the local search but improve the speed.
        # The bottleneck is `find_q3`.
        # We will use the `check_q3_in_subset` logic but integrated into the search.
        
        # Actually, let's use the exact `find_q3` from the previous program, 
        # but add a "cooldown" or randomization to escape local minima.
        
        def build_adj(edge_set):
            adj = {i: set() for i in range(n)}
            for u, v in edge_set:
                adj[u].add(v)
                adj[v].add(u)
            return adj

        def find_q3(graph_adj):
            # Same as before
            for v in range(n):
                neighbors = list(graph_adj.get(v, set()))
                if len(neighbors) < 3:
                    continue
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        for k in range(j + 1, len(neighbors)):
                            a, b, c = neighbors[i], neighbors[j], neighbors[k]
                            if (a, b) in graph_adj or (b, a) in graph_adj: continue
                            if (a, c) in graph_adj or (c, a) in graph_adj: continue
                            if (b, c) in graph_adj or (c, b) in graph_adj: continue
                            
                            na = graph_adj.get(a, set())
                            nb = graph_adj.get(b, set())
                            nc = graph_adj.get(c, set())
                            
                            forbidden = {v, a, b, c}
                            xs = (na & nb) - forbidden
                            ys = (na & nc) - forbidden
                            zs = (nb & nc) - forbidden
                            
                            for x in xs:
                                for y in ys:
                                    for z in zs:
                                        if x == y or y == z or x == z: continue
                                        if (x, y) in graph_adj or (y, x) in graph_adj: continue
                                        if (x, z) in graph_adj or (z, x) in graph_adj: continue
                                        if (y, z) in graph_adj or (z, y) in graph_adj: continue
                                        
                                        nx = graph_adj.get(x, set())
                                        ny = graph_adj.get(y, set())
                                        nz = graph_adj.get(z, set())
                                        ws = (nx & ny & nz) - forbidden - {x, y, z}
                                        
                                        for w in ws:
                                            if (w, a) in graph_adj or (a, w) in graph_adj: continue
                                            if (w, b) in graph_adj or (b, w) in graph_adj: continue
                                            if (w, c) in graph_adj or (c, w) in graph_adj: continue
                                            
                                            q3_edges = [
                                                (v, a), (v, b), (v, c),
                                                (a, x), (b, x),
                                                (a, y), (c, y),
                                                (b, z), (c, z),
                                                (x, w), (y, w), (z, w)
                                            ]
                                            return q3_edges
            return None

        def remove_edge_heuristic(graph_adj, q3_edges):
            # Remove edge with lowest degree sum
            best_edge = None
            min_score = float('inf')
            for u, v in q3_edges:
                score = len(graph_adj[u]) + len(graph_adj[v])
                if score < min_score:
                    min_score = score
                    best_edge = (u, v)
            
            if best_edge[0] > best_edge[1]:
                best_edge = (best_edge[1], best_edge[0])
            return best_edge

        best_edges = []
        best_count = 0
        random.seed(42)
        
        # Try different initial densities
        densities = [0.4, 0.5, 0.6, 0.7]
        
        for p in densities:
            for restart in range(20):
                if time.time() > deadline - 5:
                    break
                    
                current_edges = set()
                for e in all_edges:
                    if random.random() < p:
                        current_edges.add(e)
                        
                adj = build_adj(current_edges)
                
                # Phase 1: Remove Q3s
                steps = 0
                while steps < 2000:
                    if time.time() > deadline - 5:
                        break
                    q3_edges = find_q3(adj)
                    if q3_edges is None:
                        break
                    e_to_remove = remove_edge_heuristic(adj, q3_edges)
                    if e_to_remove in current_edges:
                        current_edges.remove(e_to_remove)
                        u, v = e_to_remove
                        adj[u].remove(v)
                        adj[v].remove(u)
                    else:
                        break
                    steps += 1
                    
                # Phase 2: Hill Climbing (Add edges)
                missing = [e for e in all_edges if e not in current_edges]
                random.shuffle(missing)
                
                for e in missing:
                    if time.time() > deadline - 5:
                        break
                    u, v = e
                    current_edges.add(e)
                    adj[u].add(v)
                    adj[v].add(u)
                    
                    q3_edges = find_q3(adj)
                    
                    if q3_edges is not None:
                        # Check if e is in the Q3
                        e_norm = (u, v) if u < v else (v, u)
                        contains_e = False
                        for qe in q3_edges:
                            if (qe[0] == u and qe[1] == v) or (qe[0] == v and qe[1] == u):
                                contains_e = True
                                break
                        
                        if contains_e:
                            current_edges.remove(e)
                            adj[u].remove(v)
                            adj[v].remove(u)
                        else:
                            # If e is not in the found Q3, the Q3 was already there?
                            # This shouldn't happen if the graph was Q3-free before adding e.
                            # But if it does, we must remove e to be safe?
                            # Or we can try to remove an edge from the Q3 instead?
                            # For simplicity, remove e.
                            current_edges.remove(e)
                            adj[u].remove(v)
                            adj[v].remove(u)
                    
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = list(current_edges)
                    
        return [[u, v] for u, v in best_edges]

# EVOLVE-BLOCK-END
