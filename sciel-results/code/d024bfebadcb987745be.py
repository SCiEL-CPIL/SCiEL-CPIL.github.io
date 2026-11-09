import time
import sys
from itertools import combinations
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    n = 16
    # Total possible edges in K16
    all_edges = []
    edge_to_idx = {}
    for u in range(n):
        for v in range(u + 1, n):
            edge_to_idx[(u, v)] = len(all_edges)
            all_edges.append((u, v))
    
    m = len(all_edges) # 120

    # Precompute Q3 patterns
    # Q3 vertices can be labeled 0..7.
    # Edges of Q3:
    # (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)
    q3_edges_local = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7), (6, 7)
    ]
    
    # We need to find all sets of 12 global edges that form a Q3.
    # A Q3 is determined by a mapping of its 8 local vertices to 8 distinct global vertices.
    # There are C(16, 8) = 12870 ways to choose the 8 vertices.
    # For each set of 8 vertices, there are 8! / (2^3 * 3!) = 240 automorphisms? 
    # Actually, the number of distinct embeddings of Q3 into K8 is 8! / |Aut(Q3)|.
    # |Aut(Q3)| = 48. So 40320 / 48 = 840 distinct edge sets per 8-vertex subset?
    # Wait, the edge set is determined by the isomorphism. 
    # Two isomorphisms yield the same edge set iff they differ by an automorphism of Q3.
    # So yes, 840 distinct edge sets (subgraphs isomorphic to Q3) for each 8-vertex subset.
    # Total Q3 subgraphs in K16: 12870 * 840 = 10,810,800.
    # This is too many to precompute explicitly in a list of sets of 12 indices without significant memory/time.
    
    # Alternative: Build the conflict graph on the fly or use a smarter representation.
    # Conflict graph G_conf has m=120 vertices.
    # Two edges e1, e2 conflict if there exists a Q3 containing both.
    # Instead of precomputing all Q3s, we can check conflicts pair by pair?
    # Pairs of edges: C(120, 2) = 7140.
    # For each pair (e1, e2), do they co-occur in a Q3?
    # This is much faster.
    
    # Let's build the conflict graph adjacency bitsets.
    # adj_conf[i] is a bitset of vertices j such that edge i and edge j conflict.
    
    adj_conf = [0] * m
    
    # Helper to check if two edges (u1,v1) and (u2,v2) can be in the same Q3.
    # They must be disjoint or share exactly 1 vertex?
    # In Q3, edges can be disjoint or share a vertex. They cannot be the same.
    # If they share 2 vertices, they are the same edge.
    
    # Condition for two edges e1, e2 to be in a Q3:
    # There exists a set of 8 vertices S containing the endpoints of e1 and e2 such that the induced subgraph on S contains a Q3 that includes e1 and e2.
    # Actually, it's simpler: 
    # Q3 is bipartite with parts of size 4.
    # Any edge in Q3 connects the two parts.
    # So if e1 and e2 are in the same Q3, their 4 endpoints (if disjoint) or 3 endpoints (if sharing 1) must be embeddable.
    
    # Let's just iterate through all 840 patterns of Q3 on vertices 0..7.
    # Generate all 840 distinct edge sets of Q3 on 8 vertices.
    # We can generate them by iterating over all permutations of 0..7? No, that's 40320.
    # We can iterate over all subsets of 4 vertices for one part (35 ways), and then the edges are determined?
    # No, the edges are determined by the structure.
    # Let's generate the 840 edge sets for the canonical Q3 on 0..7.
    
    def get_q3_edge_sets():
        sets = set()
        # Iterate over all permutations of 0..7 is too slow if done naively, but 40320 is small.
        # We can just iterate over all 8! permutations and map the canonical edges.
        from itertools import permutations
        canonical_edges = q3_edges_local
        for perm in permutations(range(8)):
            # perm[i] is the global vertex for local vertex i
            mapped_edges = frozenset((min(perm[i], perm[j]), max(perm[i], perm[j])) for i, j in canonical_edges)
            sets.add(mapped_edges)
        return list(sets)
        
    # This takes a few seconds.
    # 40320 permutations * 12 edges * set ops.
    # Let's try to be smarter.
    # The 840 sets are the orbits of the action of S8 on the edges? No.
    # Let's just run it. It's a one-time cost.
    
    t0 = time.time()
    q3_sets_8 = get_q3_edge_sets()
    t1 = time.time()
    # print(f"Generated {len(q3_sets_8)} Q3 patterns in {t1-t0}s")
    
    # Now, for each pair of global edges (i, j), check if they conflict.
    # They conflict if there exists a subset S of 8 vertices containing their endpoints,
    # and a Q3 pattern P on S such that the images of the edges in P include e_i and e_j.
    
    # This is still complex to check pair by pair without precomputation.
    
    # Better approach:
    # For each 8-vertex subset S (12870 of them),
    # and for each of the 840 Q3 patterns on S,
    # mark the 12 edges in that pattern as mutually conflicting.
    
    # Total operations: 12870 * 840 * (12 choose 2) / 2 * (marking cost)
    # 12870 * 840 = 10.8M Q3s.
    # For each Q3, we have 12 edges. We need to add 66 pairs to the conflict graph.
    # 10.8M * 66 = 700M operations. This is too slow in Python.
    
    # We need a faster conflict check or a different MIS heuristic that doesn't require the full conflict graph precomputation.
    # Or, we can use the fact that the conflict graph is dense?
    
    # Let's reconsider the local search but with a much faster Q3 check.
    # The previous local search was slow because `contains_q3_in_subset_fast` was called C(14,6)=3003 times per edge addition.
    # 3003 * 35 splits * inner loop.
    
    # Can we optimize the Q3 check?
    # A Q3 is an 8-vertex subgraph with 12 edges, bipartite 4-4, girth 4.
    # Maybe we can use a different local search strategy:
    # Start with a random graph, and use simulated annealing?
    
    # Let's try a different MIS approach:
    # The conflict graph is sparse enough?
    # Two edges conflict if they are in a Q3.
    # How many edges conflict with a given edge (u,v)?
    # An edge (u,v) is in a Q3. The Q3 has 12 edges.
    # How many Q3s contain (u,v)?
    # (u,v) uses 2 vertices. We need 6 more.
    # The 6 more vertices must form a structure with u,v that is a Q3.
    # The number of Q3s containing a specific edge is C(14, 6) * (number of Q3s on 8 vertices containing a specific edge).
    # Number of Q3s on 8 vertices containing a specific edge:
    # Total Q3s on 8 vertices = 840.
    # Each Q3 has 12 edges. Total edge occurrences = 840 * 12 = 10080.
    # By symmetry, each of the 28 edges in K8 is in 10080 / 28 = 360 Q3s.
    # So each global edge is in 3003 * 360 = 1,081,080 Q3s.
    # Each such Q3 contributes 11 other edges that conflict with it.
    # So the degree in the conflict graph is roughly 1M?
    # No, the conflicts are unique edges.
    # The number of distinct edges conflicting with (u,v) is at most 119.
    # Actually, almost all other edges might conflict?
    # If (u,v) and (x,y) are disjoint, do they conflict?
    # Yes, if they can be in the same Q3.
    # If they share a vertex, yes.
    # So the conflict graph is very dense.
    # MIS on a dense graph is hard.
    
    # Let's go back to optimizing the local search.
    # The bottleneck was the Q3 check.
    # We can precompute for every pair of edges (e1, e2) if they conflict.
    # If we have a boolean matrix `conflict[i][j]`, we can check if adding edge e to the current set E creates a Q3?
    # No, Q3 requires 12 edges.
    # Adding one edge e creates a Q3 if there exist 11 other edges in E that form a "Q3 minus e" with e.
    # This is still complex.
    
    # Let's try a different heuristic:
    # Start with a complete graph and remove edges to break Q3s?
    # Or start with a sparse graph and add edges.
    
    # Let's try to speed up the `contains_q3_in_subset` function significantly.
    # Instead of checking 35 splits, can we check if the 8-vertex subgraph is Q3?
    # Q3 is the unique bipartite graph with 8 vertices, 12 edges, and girth 4?
    # No, there are other bipartite graphs with 8 vertices and 12 edges.
    # But Q3 is the unique 3-regular bipartite graph on 8 vertices?
    # Yes, Q3 is the unique 3-regular bipartite graph with 8 vertices.
    # So, for a set of 8 vertices, check if the induced subgraph is 3-regular and bipartite.
    # This is much faster than checking 35 splits and matchings.
    
    def is_q3_subgraph(v_list):
        # v_list: tuple of 8 vertices
        # Check if induced subgraph is Q3
        # 1. Check degrees: all must be 3.
        # 2. Check if bipartite.
        
        deg = [0] * 8
        adj_local = [[0]*8 for _ in range(8)]
        
        for i in range(8):
            vi = v_list[i]
            for j in range(i+1, 8):
                vj = v_list[j]
                if adj[vi][vj]:
                    adj_local[i][j] = 1
                    adj_local[j][i] = 1
                    deg[i] += 1
                    deg[j] += 1
        
        for i in range(8):
            if deg[i] != 3:
                return False
        
        # Check bipartite
        # BFS/DFS
        color = [-1] * 8
        for start in range(8):
            if color[start] == -1:
                color[start] = 0
                queue = deque([start])
                while queue:
                    u = queue.popleft()
                    for v in range(8):
                        if adj_local[u][v]:
                            if color[v] == -1:
                                color[v] = 1 - color[u]
                                queue.append(v)
                            elif color[v] == color[u]:
                                return False
        return True

    # This check is O(1) per 8-vertex subset (constant 8).
    # C(14, 6) = 3003 subsets.
    # 3003 * O(1) is very fast.
    # Let's use this in the local search.

    n = 16
    adj = [[0] * n for _ in range(n)]
    edges = set()

    def add_edge(u, v):
        if u == v or adj[u][v]:
            return
        adj[u][v] = 1
        adj[v][u] = 1
        edges.add((min(u, v), max(u, v)))

    def remove_edge(u, v):
        if not adj[u][v]:
            return
        adj[u][v] = 0
        adj[v][u] = 0
        edges.discard((min(u, v), max(u, v)))

    # Initialize with a good seed
    # K_{8,8} is bipartite, so Q3-free?
    # Q3 is bipartite. K_{8,8} contains Q3?
    # Yes, Q3 is a subgraph of K_{8,8}.
    # So K_{8,8} is not Q3-free.
    
    # What is a good Q3-free graph?
    # The current seed was K_{3,13} + K3.
    # Let's try a random initialization or a different structured one.
    # Maybe a random graph with p=0.5 and then repair?
    
    # Let's stick to the local search with the fast check.
    
    # To improve the search, we can try multiple random restarts.
    # And use a "best improvement" or "first improvement" strategy.
    
    # We will run local search until deadline.
    
    best_edges_count = 0
    best_edge_set = set()
    
    for restart in range(50): # Limit restarts to ensure we finish
        if time.time() > deadline - 10:
            break
            
        # Initialize
        adj = [[0] * n for _ in range(n)]
        edges = set()
        
        # Random initialization
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.5:
                    add_edge(u, v)
        
        # Local Search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Shuffle edges to avoid bias
            # We want to try adding missing edges.
            missing = [(u, v) for u in range(n) for v in range(u+1, n) if not adj[u][v]]
            random.shuffle(missing)
            
            for u, v in missing:
                if time.time() > deadline - 5:
                    break
                    
                add_edge(u, v)
                
                # Check if Q3 created
                found_q3 = False
                other_vertices = [i for i in range(n) if i != u and i != v]
                
                # Optimize: We only need to check subsets that include u and v.
                # But we must check if the NEW edge is part of the Q3.
                # Since the graph was Q3-free before, any new Q3 must use (u,v).
                # So we check all 8-vertex subsets containing u and v.
                
                for combo in combinations(other_vertices, 6):
                    vertices = (u, v) + tuple(combo)
                    if is_q3_subgraph(vertices):
                        found_q3 = True
                        break
                        
                if found_q3:
                    remove_edge(u, v)
                else:
                    improved = True
                    if len(edges) > best_edges_count:
                        best_edges_count = len(edges)
                        best_edge_set = set(edges)
        
    return [list(e) for e in best_edge_set]

if __name__ == "__main__":
    res = construct()
    print(len(res))
