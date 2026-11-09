import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    n = 16
    # 1. Generate all edges of K16
    all_edges = []
    edge_to_idx = {}
    for u in range(n):
        for v in range(u + 1, n):
            edge_to_idx[(u, v)] = len(all_edges)
            all_edges.append((u, v))
    
    m = len(all_edges) # 120

    # 2. Precompute Q3 subgraphs and their edge sets
    # A Q3 is an 8-vertex subgraph isomorphic to the 3-cube.
    # We iterate over all 8-sets of vertices.
    # For each 8-set, we check all possible bijections to the 8 vertices of Q3 (0..7).
    # There are 8! = 40320 permutations. 
    # However, Q3 has 48 automorphisms. We can just generate all edge sets.
    
    # Q3 vertices: 0..7 (binary 000..111)
    # Q3 edges: pairs differing in exactly 1 bit
    q3_edges = set()
    for i in range(8):
        for j in range(i+1, 8):
            # Check if i and j differ in exactly 1 bit
            diff = i ^ j
            if diff and (diff & (diff - 1)) == 0:
                q3_edges.add((i, j))
    
    # We need to map these local Q3 edges to global vertex sets.
    # For a given 8-set S = {v0, ..., v7}, a permutation p maps local i to global v_{p(i)}.
    # The set of global edges is { (v_{p(i)}, v_{p(j)}) | (i,j) in q3_edges }
    
    # To speed up, we can precompute the "signature" of Q3 edge sets on 8-sets?
    # No, the graph structure is fixed, but the embedding varies.
    # Total 8-sets: C(16,8) = 12870.
    # Permutations per 8-set: 40320.
    # Total checks: 12870 * 40320 ~ 500 million. Too slow in Python if done naively.
    
    # Optimization: 
    # Instead of iterating permutations, we can iterate over the 12 edges of Q3.
    # A set of 8 vertices forms a Q3 if there exists a bijection.
    # Actually, we can just generate all subsets of edges of K16 that form a Q3.
    # Let's use a smarter enumeration.
    # A Q3 is determined by its 8 vertices and the pairing of opposite vertices?
    # No.
    
    # Alternative: Iterate over all 8-sets. For each 8-set, count the number of edges in the induced subgraph of K16? No, we are looking for subgraphs isomorphic to Q3, not induced.
    # So we need to find any 12 edges among the C(8,2)=28 possible edges in the 8-set that form a Q3.
    
    # Let's precompute the set of edge-index sets (bitmasks of size 120) that correspond to a Q3.
    # Wait, 2^120 is impossible.
    # But the number of Q3s is finite.
    # Number of Q3s in K16:
    # Choose 8 vertices: C(16,8) = 12870.
    # Number of non-isomorphic embeddings?
    # Actually, for a fixed set of 8 vertices, how many distinct sets of 12 edges form a Q3?
    # This is equivalent to the number of labeled Q3s on 8 vertices.
    # The number of labeled copies of Q3 in K8 is 8! / 48 = 840.
    # So total Q3s = 12870 * 840 = 10,810,800.
    # Storing 10 million bitmasks of 120 bits is ~ 10M * 16 bytes = 160MB. This is feasible.
    # However, generating them might be slow.
    
    # Let's try to generate them efficiently.
    # For each 8-set S, we can generate the 840 edge sets.
    # We can use the fact that Q3 is the Cartesian product of three K2s.
    
    # Let's try a different approach: 
    # Conflict graph construction.
    # Two edges e1, e2 conflict if there exists a Q3 containing both.
    # If we find all Q3s, we can build the conflict graph.
    # Node i in conflict graph corresponds to edge i in K16.
    # Edge (i,j) in conflict graph if edge i and edge j are in some common Q3.
    
    # Let's generate all Q3s first.
    q3_edge_masks = set()
    
    # Map local Q3 edges to a canonical form? No, we need global edge indices.
    
    # To speed up generation:
    # Precompute all permutations of 0..7? No, too many.
    # We can iterate over the 840 labeled Q3s on vertices 0..7.
    # How to generate the 840 sets of edges?
    # We can iterate over all 12-edge subsets of the 28 edges of K8? C(28,12) is huge.
    
    # Better: 
    # A Q3 is defined by a perfect matching of 4 pairs? No.
    # A Q3 has 4 "squares" (4-cycles) in each of 3 directions.
    
    # Let's just iterate over all 8-sets and all 8! permutations?
    # 12870 * 40320 = 520M iterations. In Python, this is too slow (minutes).
    
    # We need a faster way to find if a set of 8 vertices contains a Q3.
    # Or, we can construct the conflict graph directly without enumerating all Q3s explicitly as masks, 
    # but by checking pairs of edges.
    
    # Pair of edges (e1, e2). Do they belong to a common Q3?
    # e1 = (u1, v1), e2 = (u2, v2).
    # If they share a vertex, or are disjoint.
    # If they are disjoint, they can be opposite edges in a square, or parallel edges in a cube?
    # In Q3, any two edges are either:
    # 1. Adjacent (share a vertex)
    # 2. Disjoint but in a common square (distance 2 in some sense)
    # 3. Disjoint and "far" (e.g., opposite edges of the cube, not in a common face)
    
    # Actually, two edges e1, e2 are in a common Q3 iff the subgraph induced by the union of their vertices (plus potentially others) can be embedded.
    
    # Let's go back to generating Q3s but optimize the permutation loop.
    # We can fix the first vertex to be 0 (local) to reduce permutations by 8.
    # Fix the second vertex to be 1? No, 0 and 1 must be connected in Q3? No, they are connected.
    # In Q3, 0 is connected to 1, 2, 4.
    
    # Let's precompute the 840 edge sets for the "standard" Q3 on vertices 0..7.
    # Then for each 8-set S, we apply all 8! permutations? Still slow.
    
    # Alternative Strategy: 
    # Use the fact that we only need to know if two edges conflict.
    # e1 and e2 conflict if there is a Q3 containing both.
    # This is equivalent to: the graph formed by adding e1 and e2 to some set of 10 other edges is a Q3?
    
    # Let's try a heuristic construction of the conflict graph.
    # For each pair of edges (e1, e2) in K16:
    #   Let V = {u1, v1, u2, v2}.
    #   If |V| > 8, they can't be in a Q3? No, Q3 has 8 vertices.
    #   If |V| < 8, we need to add vertices.
    
    # Actually, there is a known characterization.
    # Two edges e1, e2 are in a common Q3 iff:
    # 1. If they share a vertex, the other endpoints must be compatible.
    # 2. If they are disjoint, they must be "parallel" in some sense?
    
    # Let's stick to the enumeration but make it faster.
    # We can use NumPy to vectorize the check?
    # Or, we can precompute the 840 edge masks for the standard Q3.
    
    standard_q3_edges = []
    # Generate all 840 permutations of 0..7 that map 0->0? No, we need all 40320/48 = 840 unique edge sets.
    # We can generate all 40320 permutations, compute the edge mask, and add to a set.
    # 40320 iterations is fast.
    
    std_edge_set = set()
    # Map local edge (i,j) to a bitmask? No, we need to map to global later.
    # Let's just store the set of local edges (i,j) with i<j.
    for p in itertools.permutations(range(8)):
        # Check if this permutation generates a unique edge set?
        # We can just generate the edge set for this permutation.
        edges = set()
        for i in range(8):
            for j in range(i+1, 8):
                if (i, j) in q3_edges:
                    u, v = p[i], p[j]
                    if u > v: u, v = v, u
                    edges.add((u, v))
        # To avoid duplicates, we can use a frozenset or sorted tuple
        std_edge_set.add(frozenset(edges))
        
    # There should be 840 unique frozensets.
    # Now, for each 8-set S in K16, we map these 840 local edge sets to global edge indices.
    
    # Precompute index of global edge
    # edge_to_idx is already defined.
    
    conflict_graph = [0] * m  # Bitmask of conflicts for each edge node
    
    # Iterate over all 8-sets
    # C(16,8) = 12870
    # For each 8-set, 840 Q3s.
    # Total 10.8M Q3s.
    # For each Q3, we have 12 edges.
    # For each pair of edges in the Q3, they conflict.
    # Number of pairs in Q3: C(12,2) = 66.
    # Total pair updates: 10.8M * 66 ~ 700M.
    # This is still heavy for Python.
    
    # Optimization: 
    # Instead of updating pair by pair, we can build a list of edge indices for each Q3.
    # Then for each edge in the Q3, it conflicts with all other edges in the Q3.
    
    # Let's try to run this and see if it fits in time.
    # 10.8M iterations * (mapping 12 edges + 66 conflict updates)
    # Mapping: 12 lookups.
    # Updates: 66 OR operations.
    # ~ 100 operations per Q3.
    # 10.8M * 100 = 1B operations. Python might take 10-30 seconds.
    
    # Let's try to optimize the inner loop.
    
    # Precompute the 840 local edge sets as lists of 12 (u,v) pairs.
    local_q3s = []
    for es in std_edge_set:
        local_q3s.append(list(es))
        
    # Now iterate 8-sets
    # To speed up mapping, we can create a local map for the 8-set.
    
    # Let's check time.
    if time.time() > deadline - 30:
        # Fallback to simple heuristic if precomputation is too slow
        # But let's try to finish precomputation first.
        pass
        
    # We need to be careful with the deadline.
    # Let's do the precomputation.
    
    # Optimization: Use a dictionary to map local vertex to global vertex for the current 8-set.
    
    for combo in itertools.combinations(range(n), 8):
        # Map local 0..7 to global vertices
        # local_to_global = {i: combo[i] for i in range(8)}
        # This is fast enough.
        
        for lq3 in local_q3s:
            # Map local edges to global edge indices
            global_edge_indices = []
            for (u, v) in lq3:
                gu = combo[u]
                gv = combo[v]
                if gu > gv:
                    gu, gv = gv, gu
                idx = edge_to_idx[(gu, gv)]
                global_edge_indices.append(idx)
                
            # Update conflict graph
            # For each edge in global_edge_indices, it conflicts with all others
            for i in range(12):
                ei = global_edge_indices[i]
                mask = 0
                for j in range(12):
                    if i != j:
                        ej = global_edge_indices[j]
                        mask |= (1 << ej)
                conflict_graph[ei] |= mask
                
    # 3. Solve MIS on the conflict graph
    # Nodes: 0..119
    # We want to find a set of edges (nodes) such that no two conflict.
    # This is a Maximum Independent Set.
    
    # Branch and Bound
    # Order nodes by degree (descending) for better pruning?
    # Or just use a simple recursive search with bitmasks.
    
    # Let's use a standard MIS solver.
    # Since m=120, we can use a bitmask of 120 bits (Python int).
    
    # Precompute neighbor masks
    # conflict_graph[i] already contains the mask of neighbors.
    
    # We need to find the maximum set of non-adjacent nodes.
    
    best_size = 0
    best_set = 0
    
    # Sort nodes by degree? 
    # Let's stick to a simple DFS with pruning.
    
    # To speed up, we can use a heuristic: 
    # Start with a greedy solution to get a lower bound.
    
    # Greedy MIS
    remaining = (1 << m) - 1
    greedy_set = 0
    greedy_size = 0
    while remaining:
        # Pick a node with lowest degree in remaining? Or just first one.
        # Let's pick the node with the fewest neighbors in remaining to maximize independence?
        # Actually, for MIS, picking a node with high degree removes more candidates, which is bad for size?
        # No, we want to keep many nodes. 
        # Standard greedy for MIS: pick a node, add to set, remove it and its neighbors.
        # To get a good lower bound, we can try a few heuristics.
        
        # Simple heuristic: pick node 0, then next available, etc.
        # Better: sort by degree ascending?
        
        # Let's just use a simple random restart greedy for lower bound.
        import random
        random.seed(42)
        
        for _ in range(100):
            rem = (1 << m) - 1
            g_set = 0
            g_size = 0
            while rem:
                # Pick a random node from rem
                # To pick random node from bitmask:
                # count bits, pick random index, clear that bit?
                # Or just iterate.
                # For speed, pick the first set bit?
                # Let's pick the node with min degree in rem.
                best_node = -1
                min_deg = m
                r = rem
                while r:
                    # Get lowest set bit
                    lsb = r & (-r)
                    idx = lsb.bit_length() - 1
                    # Count neighbors in rem
                    deg = (conflict_graph[idx] & rem).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        best_node = idx
                    r &= r - 1
                    
                if best_node == -1:
                    break
                    
                g_set |= (1 << best_node)
                g_size += 1
                # Remove best_node and its neighbors
                rem &= ~(conflict_graph[best_node] | (1 << best_node))
                
            if g_size > greedy_size:
                greedy_size = g_size
                greedy_set = g_set
                
    best_size = greedy_size
    best_set = greedy_set
    
    # Now Branch and Bound
    # We can use the greedy solution as a lower bound.
    
    # To make B&B efficient, we should order the vertices.
    # Let's order by degree descending?
    
    # We will use a recursive function.
    # To avoid recursion depth issues (120 is fine), and overhead, we can use a stack?
    # Python recursion is fine for depth 120.
    
    # We need a good upper bound for pruning.
    # Upper bound: 1 + size of MIS of neighbors? No.
    # Simple UB: number of remaining vertices.
    # Better UB: 1 + sum of 1/(1+deg) ? (Tomita bound)
    
    # Let's implement a simple B&B with the Tomita bound.
    
    # We need to maintain the set of available vertices.
    
    # Let's reorder vertices by degree descending to improve pruning?
    # Actually, for MIS, high degree vertices are "expensive" to include?
    # No, including a high degree vertex removes many others.
    
    # Let's just use the standard order 0..119.
    
    # To speed up, we can use a precomputed list of vertices sorted by some heuristic.
    
    # Let's try to run B&B.
    
    # We need to be careful with time.
    
    def solve_mis():
        nonlocal best_size, best_set
        
        # Vertices to consider
        # We use a bitmask 'avail'
        
        # Recursive helper
        def dfs(avail, current_set, current_size):
            nonlocal best_size, best_set
            
            if time.time() > deadline - 5:
                return
                
            if current_size + avail.bit_count() <= best_size:
                return
                
            if avail == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set
                return
                
            # Heuristic: pick a vertex
            # To improve pruning, we can pick a vertex with high degree in avail?
            # Or low degree?
            # Let's pick the vertex with the highest degree in avail to prune more?
            # Actually, standard MIS algorithms often pick a vertex and branch on include/exclude.
            
            # Let's pick the first available vertex for simplicity, but ordered by degree?
            # Let's pre-sort indices by degree descending.
            
            # Get list of available vertices
            # To avoid list creation overhead, we can iterate bits.
            
            # Pick a vertex v
            # Let's use the vertex with the maximum degree in 'avail'
            v = -1
            max_deg = -1
            r = avail
            while r:
                lsb = r & (-r)
                idx = lsb.bit_length() - 1
                deg = (conflict_graph[idx] & avail).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = idx
                r &= r - 1
                
            if v == -1:
                return
                
            # Branch 1: Include v
            # New avail = avail & ~neighbors(v) & ~(1<<v)
            new_avail = avail & ~(conflict_graph[v] | (1 << v))
            dfs(new_avail, current_set | (1 << v), current_size + 1)
            
            # Branch 2: Exclude v
            # New avail = avail & ~(1<<v)
            # But we can also prune: if we exclude v, we don't need to consider it.
            # Also, if we include v, we removed its neighbors.
            # If we exclude v, we keep its neighbors.
            
            # Optimization: If we exclude v, we can also remove any vertex u that is only connected to v? No.
            
            # Simple exclude
            dfs(avail & ~(1 << v), current_set, current_size)
            
        dfs((1 << m) - 1, 0, 0)
        
    solve_mis()
    
    # 4. Convert best_set (bitmask of edge indices) to list of edges
    result_edges = []
    for i in range(m):
        if (best_set >> i) & 1:
            result_edges.append(list(all_edges[i]))
            
    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(len(edges))
