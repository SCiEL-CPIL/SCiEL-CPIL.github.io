import numpy as np
import itertools
import time
import sys
from collections import defaultdict

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 570
    d = 11
    
    # Fallback: D11 minimal vectors (Norm 2) - 220 vectors
    # Type: two entries are +/-1, rest 0
    fallback_list = []
    for idxs in itertools.combinations(range(d), 2):
        i, j = idxs
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                fallback_list.append(v)
    fallback_points = np.array(fallback_list)
    
    # We want to find a maximum subset of integer vectors with norm^2 = 4
    # such that pairwise squared distance >= 4.
    # dist^2 = |u-v|^2 = |u|^2 + |v|^2 - 2u.v = 8 - 2u.v
    # Constraint: 8 - 2u.v >= 4  =>  2u.v <= 4  =>  u.v <= 2.
    # So we need a set of vectors with norm^2=4 and pairwise dot product <= 2.
    
    # Candidate generation: All integer vectors with norm^2 = 4.
    # Types:
    # 1. One entry is +/-2 (11 * 2 = 22 vectors)
    # 2. Two entries are +/-1? No, 1^2+1^2=2.
    # 3. Four entries are +/-1 (C(11,4) * 2^4 = 330 * 16 = 5280 vectors)
    # Total candidates: 5302.
    
    candidates_list = []
    
    # Type 1: +/-2
    for i in range(d):
        for s in [2, -2]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = s
            candidates_list.append(v)
            
    # Type 2: +/-1 on 4 coords
    for idxs in itertools.combinations(range(d), 4):
        i, j, k, l = idxs
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    for sl in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        candidates_list.append(v)
                        
    candidates = np.array(candidates_list)
    n_cands = len(candidates)
    
    # Compute dot product matrix
    dot_matrix = candidates @ candidates.T
    
    # Conflict graph: Edge exists if dot > 2.
    # We want Maximum Independent Set in the conflict graph.
    # Equivalently, Maximum Clique in the compatibility graph (dot <= 2).
    
    # Let's build the compatibility graph adjacency list for MIS/Branch and Bound.
    # Actually, standard MIS algorithms work on the graph where edges represent conflicts.
    # Let's define `conflict` matrix: True if dot > 2.
    conflict = dot_matrix > 2
    np.fill_diagonal(conflict, False) # No self-conflict
    
    # Convert to adjacency lists for faster traversal in B&B
    adj = [set(np.where(conflict[i])[0]) for i in range(n_cands)]
    
    # To speed up B&B, we can order vertices.
    # Greedy Coloring for Upper Bound:
    # We use a greedy coloring of the subgraph induced by candidate vertices.
    # The number of colors is an upper bound on the size of the MIS.
    
    def greedy_coloring(candidates_indices):
        # candidates_indices is a list of vertex ids
        # We want to color this subgraph.
        # Order by degree in the subgraph (highest first) for better bounds?
        # Or just iterate.
        # For efficiency, we'll use a simple greedy coloring.
        colors = {}
        # Sort by degree in the original graph? Or just use the given order.
        # A better ordering helps the bound. Let's sort by degree in the conflict graph.
        # Degree in conflict graph: len(adj[v])
        # But we only care about the subgraph.
        # For now, simple greedy:
        # To make it a valid upper bound, we just need a valid coloring.
        # The number of colors used is the bound.
        
        # We process vertices in a specific order to minimize colors (which tightens the bound? No, we want a TIGHT upper bound, so we want MIN colors? 
        # Wait. In MIS, if we color the graph, any independent set can contain at most one vertex of each color.
        # So the number of colors is an upper bound on the size of the MIS.
        # To make the bound TIGHT (small), we want to use FEW colors.
        # Greedy coloring usually produces a reasonable number of colors.
        
        # Let's order vertices by degree in the conflict graph (highest degree first) to color them.
        # This is a heuristic for coloring.
        sorted_indices = sorted(candidates_indices, key=lambda x: len(adj[x]), reverse=True)
        
        color_map = {}
        for v in sorted_indices:
            # Find the smallest color not used by neighbors in the subgraph
            neighbor_colors = set()
            for u in adj[v]:
                if u in color_map: # u is in the subgraph and already colored
                    neighbor_colors.add(color_map[u])
            c = 0
            while c in neighbor_colors:
                c += 1
            color_map[v] = c
            
        return max(color_map.values()) + 1 if color_map else 0
    
    # Branch and Bound for Maximum Independent Set
    # We will use a recursive search.
    # State: current independent set `S`, set of candidate vertices `P` (potential additions), set of excluded vertices `X` (not needed if we just pass P).
    # Actually, standard B&B:
    # `mis(P)`: P is the set of vertices to consider.
    # We pick a vertex v from P.
    # Branch 1: Include v. Then we must exclude all neighbors of v. New P = P \ (N(v) U {v}).
    # Branch 2: Exclude v. New P = P \ {v}.
    # We use the greedy coloring bound on P to prune.
    
    # To handle Python recursion limits and speed, we'll implement a depth-first search with explicit stack or optimized recursion.
    # Given n=5302, recursion depth could be an issue if we go deep, but MIS size is likely < 300.
    # However, the branching factor is high.
    # We need a good heuristic to pick the next vertex.
    # Heuristic: Pick vertex with highest degree in the current subgraph P.
    
    best_size = 0
    best_set = []
    
    # Initial ordering for the root call
    # We can start with all vertices.
    
    # Optimization: We can pre-compute a good initial solution using Greedy.
    # Greedy MIS:
    def greedy_mis(adj, n):
        # Pick vertex with highest degree? Or lowest?
        # For MIS, picking high degree vertices early might block many others.
        # Actually, a common heuristic is to pick the vertex with the highest degree in the conflict graph?
        # No, that's for Vertex Cover.
        # For MIS, we want to pick vertices that don't conflict.
        # Let's try: Sort by degree (descending). Pick first, remove neighbors.
        # Or: Sort by degree (ascending). Pick first (least constrained?), remove neighbors.
        # Let's try a few random restarts of a simple greedy to get a good lower bound.
        
        best_greedy = []
        for _ in range(10):
            available = set(range(n))
            current_set = []
            # Randomize order slightly?
            # Let's use degree in the current available set.
            # This is O(n^2) if done naively.
            # Simpler: Static order by global degree.
            order = np.argsort(-[len(adj[i]) for i in range(n)])
            for v in order:
                if v in available:
                    current_set.append(v)
                    # Remove v and its neighbors from available
                    to_remove = {v} | (adj[v] & available)
                    available -= to_remove
            if len(current_set) > len(best_greedy):
                best_greedy = current_set
        return best_greedy
    
    initial_sol = greedy_mis(adj, n_cands)
    best_size = len(initial_sol)
    best_set = set(initial_sol)
    
    # Now B&B to improve.
    # We need to be careful with time.
    
    # We will use a recursive function.
    # To avoid deep recursion overhead, we can use a stack.
    # But let's try recursion first with sys.setrecursionlimit.
    sys.setrecursionlimit(10000)
    
    def bb_mis(P, S):
        nonlocal best_size, best_set
        if time.time() > deadline - 5:
            return
        
        # P is a set of vertices
        if not P:
            return
            
        # Upper bound
        # We need to color P.
        # If len(S) + colors(P) <= best_size, prune.
        
        # To make coloring fast, we only color a subset? No, must be valid bound.
        # Greedy coloring is O(|P|^2) worst case.
        # With |P| up to 5000, this is 25M ops. Too slow per node.
        
        # Alternative bound: 
        # If we just use len(P) as bound, it's weak.
        # We need a tighter bound.
        
        # Let's use a simpler heuristic bound:
        # Max degree in P is d_max. 
        # This doesn't help much.
        
        # Maybe we should use a different algorithm?
        # Given the time constraint and the size (5302 nodes), a full B&B with coloring might be too slow in Python.
        
        # Let's try a different structural approach:
        # The problem is finding a code in the graph.
        # We can use "Local Search" with Simulated Annealing or Tabu Search?
        # Or, we can exploit the geometry.
        
        # The vectors are from the D11 lattice (and its cosets?).
        # Actually, the set of norm 4 vectors in Z^11 is a specific graph.
        
        # Let's try a fast heuristic: 
        # 1. Start with the 220 norm-2 vectors (which form a valid set of size 220? 
        #    Wait, the fallback is 220 norm-2 vectors.
        #    Are they a valid kissing configuration?
        #    Norm^2 = 2.
        #    Dist^2 between two norm-2 vectors u, v:
        #    If they share 1 coord (e.g., e1+e2 and e1+e3), dot = 1. Dist^2 = 2+2-2 = 2 < 2? No, min dist^2 must be >= max norm^2 = 2.
        #    So Dist^2 >= 2 is required.
        #    If dot = 1, Dist^2 = 2. OK.
        #    If dot = 0, Dist^2 = 4. OK.
        #    If dot = -1, Dist^2 = 6. OK.
        #    If dot = 2 (same vector), Dist^2 = 0. Not allowed (distinct).
        #    So any set of distinct norm-2 vectors is a valid kissing configuration?
        #    Wait, the condition is min_dist^2 >= max_norm^2.
        #    max_norm^2 = 2.
        #    min_dist^2 for norm-2 vectors:
        #    Closest pair: share 1 index. u=(1,1,0..), v=(1,-1,0..). Dot = 0. Dist^2 = 4.
        #    u=(1,1,0..), v=(1,1,0..). Same.
        #    u=(1,1,0..), v=(1,0,1..). Dot = 1. Dist^2 = 2.
        #    So min_dist^2 = 2.
        #    So the 220 norm-2 vectors form a valid configuration of size 220.
        
        # Can we do better with norm-4 vectors?
        # If we use norm-4 vectors, max_norm^2 = 4.
        # We need min_dist^2 >= 4.
        # This is a stricter constraint on distances, but the vectors are "longer".
        # In the current program, it tries to find a subset of norm-4 vectors with dot <= 2.
        # The score is k.
        # If we find a set of 300 norm-4 vectors, score is 300.
        # If we stick to 220 norm-2 vectors, score is 220.
        
        # So the goal is to find > 220 norm-4 vectors.
        
        # Let's try a faster heuristic for the norm-4 pool.
        # The conflict graph is sparse?
        # Dot > 2.
        # Max dot is 4 (self).
        # Dot 4: Self.
        # Dot 3: Impossible for integer vectors with norm 4?
        #   u.v = 3. u=(2,0..), v=(1,1,1,1..). 2*1 = 2.
        #   u=(2,0..), v=(2,0..). Dot 4.
        #   u=(1,1,1,1..), v=(1,1,1,-1..). Dot = 3.
        #   So Dot 3 is possible.
        #   If dot 3, dist^2 = 8 - 6 = 2 < 4. Conflict.
        #   So we must avoid pairs with dot 3 or 4.
        #   Dot 4 is self.
        #   So we just need to avoid dot 3.
        #   Wait, is dot 3 the only conflict?
        #   Dot 2: Dist^2 = 8 - 4 = 4. OK.
        #   Dot 1: Dist^2 = 6. OK.
        #   Dot 0: Dist^2 = 8. OK.
        #   Dot -1: Dist^2 = 10. OK.
        #   Dot -2: Dist^2 = 12. OK.
        #   Dot -4: Dist^2 = 16. OK.
        
        # So the conflict is ONLY if dot product is 3 (or 4, but that's self).
        # Wait, can dot be > 4? No, max norm 4, so max dot 4.
        # So we just need to find a subset of norm-4 vectors with NO pair having dot product 3.
        
        # This simplifies the problem significantly!
        # We just need to avoid "dot 3" pairs.
        
        # Let's count how many vectors have dot 3 with a given vector.
        # u = (2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        # v must have dot 3 with u.
        # 2 * v[0] = 3 => v[0] = 1.5. Impossible for integer v.
        # So vectors of type (2,0..) have NO conflicts with dot 3?
        # Let's check.
        # u=(2,0..). v=(1,1,1,1..). Dot = 2. OK.
        # v=(2,0..). Dot = 4. Conflict (self or distinct? If distinct, v != u, but v=(2,0..) is the same type. If v=(2,0..) and u=(2,0..), they are the same vector if indices match. If indices differ, dot 0.)
        # So type (2,0..) vectors only conflict with themselves (dot 4).
        # So we can include ALL 22 vectors of type (2,0..)?
        # Yes, because any two distinct (2,0..) vectors have dot 0.
        
        # Now consider u = (1,1,1,1,0..).
        # We need v such that u.v = 3.
        # v must be a norm-4 vector.
        # If v is type (2,0..), say v=(2,0..).
        #   If v[0]=2, dot = 2*1 = 2.
        #   If v[1]=2, dot = 2*1 = 2.
        #   If v[4]=2, dot = 0.
        #   So no type (2,0..) vector has dot 3 with u.
        
        # If v is type (1,1,1,1..).
        #   u.v = sum(u[i]*v[i]).
        #   u has four 1s. v has four 1s.
        #   Max dot is 4 (if same support and signs).
        #   Dot 3: One sign mismatch or one different index?
        #   If same support, 3 matches, 1 mismatch: 1+1+1-1 = 2.
        #   If 3 common indices, 1 different: 1+1+1+0 = 3.
        #   So if v shares exactly 3 indices with u and has the same signs, dot is 3.
        #   Example: u=(1,1,1,1,0..), v=(1,1,1,-1,0..) -> dot 2.
        #   v=(1,1,1,0,1,0..) -> dot 3.
        
        # So the conflict graph is defined by:
        # 1. Type (2,0..) nodes are isolated (except self).
        # 2. Type (1,1,1,1..) nodes conflict with other Type (1,1,1,1..) nodes if they share exactly 3 indices with matching signs.
        
        # This is a MUCH simpler structure!
        # We can include ALL 22 Type (2,0..) vectors.
        # Now we need to find the Maximum Independent Set in the graph of the 5280 Type (1,1,1,1..) vectors, where edges connect pairs sharing 3 indices with same signs.
        
        # Let's build this specific graph and solve MIS.
        # The graph is 4-regular? Or something?
        # For a fixed u (4 indices), how many v share 3 indices?
        # Choose 3 of the 4 indices of u: C(4,3)=4.
        # For each choice of 3, the 4th index of v must be different from u's 4th index.
        # There are 11-4=7 other indices.
        # So 4 * 7 = 28 neighbors?
        # Plus, what about the sign?
        # The 3 shared indices must have the SAME sign as u.
        # The new index can be +1 or -1.
        # So for each of the 4 choices of 3 indices, and 7 choices of the new index, and 2 choices of sign for the new index:
        # 4 * 7 * 2 = 56 neighbors.
        
        # So the graph is 56-regular on 5280 nodes.
        # This is a dense-ish graph but small enough for B&B?
        # 5280 nodes, degree 56.
        
        # We can try to solve MIS on this subgraph.
        # Or, we can use a greedy approach.
        
        # Let's try Greedy on this specific graph.
        # Since we can include all 22 (2,0..) vectors, we start with them.
        # Then we add as many (1,1,1,1..) vectors as possible.
        
        # Let's identify the (1,1,1,1..) candidates.
        # Indices 22 to 22+5280-1.
        
        # Build adjacency for the subgraph of (1,1,1,1..) vectors.
        # We can do this efficiently.
        
        # Map each (1,1,1,1..) vector to an ID.
        # Key: (sorted_indices, signs_tuple)
        # Actually, we can just use the index in the `candidates` array.
        
        # Let's extract the (1,1,1,1..) vectors.
        # They are candidates[22:]
        
        # We will build the conflict list for these.
        # To do this fast, we can iterate.
        
        # But wait, we can just use the dot product matrix we already computed!
        # `conflict` matrix is already computed.
        # We just need to run MIS on the subgraph induced by nodes 22..5301.
        
        # Let's try a simple Greedy on this subgraph.
        # Order by degree?
        # All have degree 56 in the full graph, but in the subgraph it's the same.
        
        # Let's try multiple random Greedy runs on the subgraph.
        
        subgraph_nodes = list(range(22, n_cands))
        n_sub = len(subgraph_nodes)
        
        # Create a local adjacency for the subgraph
        # Map global index to local index
        global_to_local = {g: i for i, g in enumerate(subgraph_nodes)}
        
        local_adj = [set() for _ in range(n_sub)]
        
        for i, gi in enumerate(subgraph_nodes):
            for gj in adj[gi]:
                if gj in global_to_local:
                    local_adj[i].add(global_to_local[gj])
                    
        # Now solve MIS on local_adj
        best_sub_size = 0
        best_sub_set = []
        
        # Greedy
        for _ in range(20):
            available = set(range(n_sub))
            current_set = []
            # Random order
            order = np.random.permutation(n_sub)
            for v in order:
                if v in available:
                    current_set.append(v)
                    # Remove v and neighbors
                    to_remove = {v} | (local_adj[v] & available)
                    available -= to_remove
            if len(current_set) > best_sub_size:
                best_sub_size = len(current_set)
                best_sub_set = current_set
                
        # Total size = 22 + best_sub_size
        total_size = 22 + best_sub_size
        
        if total_size > best_size:
            best_size = total_size
            # Construct the set
            best_set_global = set(range(22)) # All (2,0..) vectors
            for v in best_sub_set:
                best_set_global.add(subgraph_nodes[v])
            best_set = best_set_global
            
        # If we have time, try B&B on the subgraph?
        # The subgraph is 5280 nodes.
        # B&B might be too slow.
        
        # Let's try to improve the greedy with a simple local search (swap).
        # But given time, let's just return the best found.
        
        if best_size > len(fallback_points):
            best_points = candidates[list(best_set)]
            return np.around(best_points).astype(np.int64)
        else:
            return np.around(fallback_points).astype(np.int64)

# Helper to ensure no crash
if __name__ == "__main__":
    res = kissing_number11()
    print(res.shape)
