"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import sys

# Deadline guard: evaluator kills after 150s, keep safety margin.
_DEADLINE = 120.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = 120.0

    n = 16
    # Precompute all 8-vertex subsets that form a Q3 (3-cube).
    # Q3 has 8 vertices and 12 edges. We need to find all induced or subgraph copies.
    # Actually, "subgraph isomorphic to Q3" means there exists an injection f: V(Q3) -> V(G)
    # such that every edge of Q3 maps to an edge of G. It does NOT need to be induced.
    # So we need: for every 8-vertex subset S, check if the subgraph G[S] contains Q3 as a subgraph.
    
    # Q3 structure: vertices are 3-bit strings {0,1}^3. Edges connect vertices differing in exactly 1 bit.
    q3_vertices = list(itertools.product([0,1], repeat=3))
    q3_edges = []
    for i, v in enumerate(q3_vertices):
        for j, w in enumerate(q3_vertices):
            if i < j:
                diff = sum(1 for a, b in zip(v, w) if a != b)
                if diff == 1:
                    q3_edges.append((i, j))
    
    # Number of ways to map Q3 vertices to a given 8-vertex set: 8! = 40320
    # But we can use a smarter check.
    
    # Alternative: precompute all 8-subsets of {0..15} and for each, check if Q3 is a subgraph.
    # There are C(16,8) = 12870 such subsets. For each, we need to check if there's a bijection
    # from Q3 vertices to the 8 chosen vertices such that all 12 Q3 edges map to edges in G.
    
    # A faster approach: for a given 8-vertex set S, the subgraph on S must have at least 12 edges,
    # and must contain Q3 as a subgraph. We can check this via backtracking.
    
    # Even faster: precompute the list of all 8-subsets. For a given graph G (as an adjacency bitmask
    # or adjacency matrix), check each 8-subset.
    
    # Let's use bitmasks for efficiency.
    # Represent graph as a list of 16 bitmasks, adj[i] is a 16-bit integer with bit j set iff (i,j) is an edge.
    
    def make_q3_check(adj):
        """Return a function that checks if the graph with adjacency list 'adj' contains Q3."""
        # Precompute all 8-subsets
        all_8_subsets = list(itertools.combinations(range(n), 8))
        
        # For Q3, we need to find an isomorphic copy.
        # Q3 has a specific structure. Let's label Q3 vertices 0..7 corresponding to 000,001,010,011,100,101,110,111.
        # Edges of Q3:
        q3_adj = [0]*8
        for u, v in q3_edges:
            q3_adj[u] |= (1 << v)
            q3_adj[v] |= (1 << u)
        
        # For a given 8-subset S = (s0, s1, ..., s7), we need to check if there's a permutation
        # of S such that the induced mapping is a graph homomorphism from Q3 to G[S].
        # i.e., for each edge (u,v) in Q3, (perm[u], perm[v]) must be an edge in G.
        
        # We can use backtracking with pruning.
        
        def contains_q3_in_subset(S):
            # S is a tuple of 8 vertex indices
            # Try to assign Q3 vertices 0..7 to S[0..7] (permutation)
            # such that all Q3 edges are preserved.
            
            # Order Q3 vertices by some heuristic? For now, try all permutations with backtracking.
            # 8! = 40320, times 12870 subsets = ~518 million. Too slow in Python.
            
            # Better: use the structure of Q3. Q3 is bipartite with parts of size 4.
            # Let's try a different approach: for the 8-subset, check if the subgraph on S
            # has the right properties.
            
            # Actually, let's precompute for each 8-subset whether it can host Q3 given a specific edge set.
            # This is hard to precompute independently of G.
            
            # Alternative approach: instead of checking all 8-subsets, we can check if Q3 exists
            # by trying all injections from Q3 vertices to graph vertices. There are 16*15*...*9 = 16!/8! ≈ 5.18e8.
            # Too slow.
            
            # Let's use a smarter method: for each 8-subset S, build the adjacency matrix on S,
            # then check if Q3 is a subgraph using a backtracking search that exploits Q3's structure.
            
            # Map S to 0..7
            s_to_local = {s: i for i, s in enumerate(S)}
            
            # Build local adjacency: local_adj[i] is bitmask of neighbors in S
            local_adj = [0]*8
            for i in range(8):
                for j in range(i+1, 8):
                    if adj[S[i]] & (1 << S[j]):
                        local_adj[i] |= (1 << j)
                        local_adj[j] |= (1 << i)
            
            # Now check if local graph contains Q3 as subgraph.
            # Q3 has 8 vertices, 12 edges. The local graph has 8 vertices.
            # We need a bijection f: {0..7} -> {0..7} such that for each Q3 edge (u,v), local_adj[f[u]] & (1<<f[v]) != 0.
            
            # Backtracking: assign Q3 vertices one by one to local vertices.
            # Order Q3 vertices: 0,1,2,3,4,5,6,7
            # For each, try unassigned local vertices.
            
            assigned = [-1]*8  # assigned[q3_vertex] = local_vertex
            local_used = [False]*8
            
            def backtrack(q3_v):
                if q3_v == 8:
                    return True
                for lv in range(8):
                    if local_used[lv]:
                        continue
                    # Check if lv can be assigned to q3_v:
                    # For all already-assigned Q3 neighbors of q3_v, the local neighbor must be consistent.
                    ok = True
                    for q3_u in range(8):
                        if assigned[q3_u] != -1 and (q3_adj[q3_v] & (1 << q3_u)):
                            # q3_u is a neighbor of q3_v in Q3, and is already assigned
                            if not (local_adj[lv] & (1 << assigned[q3_u])):
                                ok = False
                                break
                    if ok:
                        assigned[q3_v] = lv
                        local_used[lv] = True
                        if backtrack(q3_v + 1):
                            return True
                        assigned[q3_v] = -1
                        local_used[lv] = False
                return False
            
            return backtrack(0)
        
        def has_q3():
            for S in all_8_subsets:
                if contains_q3_in_subset(S):
                    return True
            return False
        
        return has_q3
    
    # This is too slow. Let's think differently.
    
    # Better approach: greedy edge addition with Q3 check.
    # Start with empty graph, add edges one by one in a good order, checking Q3-freeness.
    # Or start from a known good graph and try to add edges.
    
    # The seed gives K3,13 with 39 edges. Let's try to improve.
    
    # Key insight: Q3 is 3-regular on 8 vertices. A necessary condition for Q3 to exist in G
    # is that G has a subgraph on 8 vertices with minimum degree at least 3 (since Q3 is 3-regular).
    # More precisely, each vertex in the Q3 copy has degree 3 within the copy.
    
    # Let's use a backtracking search that starts from K3,13 and tries to add edges.
    
    # First, let's build K3,13: vertices 0,1,2 on one side, 3..15 on the other.
    # Edges: all (a,b) for a in {0,1,2}, b in {3..15}. That's 3*13 = 39 edges.
    
    # Now, can we add more edges? Edges within {0,1,2} or within {3..15}.
    # Adding an edge within {0,1,2}: say (0,1). Does this create a Q3?
    # For a Q3 to exist, we need 8 vertices with the Q3 structure.
    # In K3,13, the only edges are between {0,1,2} and {3..15}.
    # If we add edges within {3..15}, we might create Q3.
    
    # Let's implement a fast Q3 check and use greedy/backtracking.
    
    # Fast Q3 check using bitmasks:
    # For each 8-subset, we can precompute whether the Q3 pattern matches.
    # But the pattern depends on G's edges.
    
    # Let's try: precompute all 8-subsets (12870 of them). For each, precompute the list of
    # "Q3 patterns" - i.e., the 12 edge pairs that must be present.
    # For a given G, an 8-subset S contains Q3 iff there exists a permutation p of S such that
    # all 12 Q3 edges (under p) are in G.
    
    # To speed this up: for each 8-subset S, the number of Q3 embeddings is at most 8! = 40320.
    # But we can prune: Q3 is bipartite. Let's partition Q3 vertices into two sets of 4 (by parity of bit count).
    # Part A: 000, 011, 101, 110 (even parity)
    # Part B: 001, 010, 100, 111 (odd parity)
    # Edges only go between A and B.
    
    # So for a candidate 8-subset S, we need to split S into two 4-sets (A', B') such that
    # there's a bijection from A to A' and B to B' preserving all 12 edges.
    
    # This is still complex. Let's just do backtracking with good pruning.
    
    # Actually, let's try a completely different approach:
    # Use the fact that Q3-free graphs on 16 vertices - what's the max edge count?
    # Known results: ex(n, Q3) for n=16. The seed gives 39.
    # Let's try to find a denser graph by construction.
    
    # Idea: take K16 and remove edges to destroy all Q3s.
    # Or: use a random graph and check.
    
    # Let's do a randomized local search:
    # 1. Start with a random dense graph.
    # 2. Check if it contains Q3.
    # 3. If yes, remove a random edge from a Q3.
    # 4. Try to add random edges.
    
    # But checking Q3 is the bottleneck.
    
    # Let's implement a reasonably fast Q3 checker.
    
    def graph_has_q3(adj):
        """Check if graph with adjacency bitmasks 'adj' (list of 16 ints) contains Q3 as a subgraph."""
        # Use the 8-subset approach with backtracking, but with early termination.
        
        # Q3 vertices and edges (0-indexed)
        # Q3: 8 vertices, 12 edges
        # Vertex i has degree 3.
        # Let's enumerate all 8-subsets and check each.
        
        # To speed up: first check if the subgraph on S has at least 12 edges.
        # If not, skip.
        
        # Precompute Q3 structure
        q3_adj = [0]*8
        q3_edges = []
        for u in range(8):
            for v in range(u+1, 8):
                # u and v differ in exactly 1 bit
                diff = bin(u ^ v).count('1')
                if diff == 1:
                    q3_edges.append((u, v))
                    q3_adj[u] |= (1 << v)
                    q3_adj[v] |= (1 << u)
        
        # For each 8-subset
        for S in itertools.combinations(range(n), 8):
            # Count edges in G[S]
            edge_count = 0
            for i in range(8):
                for j in range(i+1, 8):
                    if adj[S[i]] & (1 << S[j]):
                        edge_count += 1
            if edge_count < 12:
                continue
            
            # Build local adjacency
            local_adj = [0]*8
            for i in range(8):
                for j in range(i+1, 8):
                    if adj[S[i]] & (1 << S[j]):
                        local_adj[i] |= (1 << j)
                        local_adj[j] |= (1 << i)
            
            # Check if local graph contains Q3 as subgraph
            # Backtracking with pruning
            assigned = [-1]*8
            local_used = [False]*8
            
            # Order Q3 vertices by degree in the remaining unassigned part? 
            # For simplicity, just try 0,1,2,...,7
            
            def bt(q3_v):
                if q3_v == 8:
                    return True
                for lv in range(8):
                    if local_used[lv]:
                        continue
                    ok = True
                    for q3_u in range(q3_v):
                        if assigned[q3_u] != -1 and (q3_adj[q3_v] & (1 << q3_u)):
                            if not (local_adj[lv] & (1 << assigned[q3_u])):
                                ok = False
                                break
                    if ok:
                        assigned[q3_v] = lv
                        local_used[lv] = True
                        if bt(q3_v + 1):
                            return True
                        assigned[q3_v] = -1
                        local_used[lv] = False
                return False
            
            if bt(0):
                return True
        
        return False
    
    # This might still be slow. Let's test on K3,13 first.
    
    # Build K3,13
    def build_k313():
        adj = [0]*16
        for a in range(3):
            for b in range(3, 16):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        return adj
    
    adj = build_k313()
    
    # Verify K3,13 is Q3-free
    # (It should be, since Q3 needs 4 vertices on each side of the bipartition,
    #  but K3,13 only has 3 on one side.)
    
    # Now try to add edges greedily.
    # Candidate edges: all edges not in K3,13.
    # These are: edges within {0,1,2} (3 edges) and edges within {3..15} (C(13,2)=78 edges).
    
    candidate_edges = []
    for i in range(16):
        for j in range(i+1, 16):
            if not (adj[i] & (1 << j)):
                candidate_edges.append((i, j))
    
    # Shuffle for randomness? Or try in a fixed order?
    # Let's try all candidate edges, and for each, check if adding it preserves Q3-freeness.
    
    # To make this faster, let's use a smarter check: when adding edge (u,v),
    # we only need to check 8-subsets that contain both u and v.
    # There are C(14,6) = 3003 such subsets.
    
    # But even that might be slow. Let's try.
    
    added_edges = []
    remaining = list(candidate_edges)
    
    # Sort candidate edges: maybe try edges within the larger part first?
    # Or random order.
    
    random.seed(42)
    random.shuffle(remaining)
    
    for u, v in remaining:
        if _time_left() < 5.0:
            break
        
        # Try adding edge (u, v)
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        if graph_has_q3(adj):
            # Revert
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            added_edges.append((u, v))
    
    # Build final edge list
    edges = []
    for i in range(16):
        for j in range(i+1, 16):
            if adj[i] & (1 << j):
                edges.append([i, j])
    
    return edges

# EVOLVE-BLOCK-END
