"""
Find a K4-free graph on 18 vertices with small max triangle-free induced subgraph.
Strategy: Start with a good base graph (e.g., a random K4-free graph or specific construction),
then use local search (edge flips) to minimize the size of the largest triangle-free induced subgraph.
"""

import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 115.0  # safety margin
    
    n = 18
    # Adjacency matrix
    # adj[i][j] = 1 if edge exists
    
    # Helper: Check if graph is K4-free
    def is_k4_free(adj):
        # Check all 4-cliques
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]: continue
                for w in range(v+1, n):
                    if not adj[v][w]: continue
                    if not adj[u][w]: continue
                    for x in range(w+1, n):
                        if adj[u][x] and adj[v][x] and adj[w][x]:
                            return False
        return True

    # Helper: Find max size of triangle-free induced subgraph
    # This is the "independence number" of the triangle hypergraph? No.
    # A set S is triangle-free if it contains no triangle.
    # We want max |S| such that G[S] has no K3.
    # This is equivalent to finding the largest set of vertices with no induced K3?
    # No, just that the subgraph induced by S has no triangles.
    
    # Since n=18 is small, we can use a clever DP or Branch and Bound.
    # Let's use a simple Branch and Bound.
    # State: (current_set, next_vertex_to_consider, current_size)
    # We can prune if current_size + (n - next_vertex) <= best
    
    def max_tf_size(adj):
        best = 0
        # Precompute for each vertex, its neighbors
        # For a set to be triangle-free, it cannot contain a triangle.
        
        # A faster check:
        # If we pick a vertex v, we can't pick any edge from N(v) in the remaining set.
        
        # Let's use a recursive solver.
        # vertices = list(range(n))
        
        def solve(remaining, size):
            nonlocal best
            if not remaining:
                if size > best:
                    best = size
                return
            # Bound
            if size + len(remaining) <= best:
                return
            
            # Heuristic: pick a vertex with high degree in remaining subgraph?
            # Or just iterate.
            # To speed up, order vertices.
            
            v = remaining[0]
            rest = remaining[1:]
            
            # Option 1: Include v
            # We must remove all neighbors of v that form a triangle with v?
            # No, if we include v, we can still include neighbors, as long as they don't form a triangle AMONG themselves or with v.
            # Wait. If we include v, and we include two neighbors u, w of v that are adjacent to each other, that's a triangle.
            # So if we include v, the rest of the set must be an independent set in the graph induced by N(v)?
            # No. The rest of the set (from rest) must be triangle-free, AND no two of them can be adjacent neighbors of v.
            # Let A = {u in rest | adj[v][u]}.
            # We can pick a subset S from rest.
            # S must be triangle-free.
            # Also, S cannot contain an edge u-w where u, w in A and adj[u][w]=1.
            # Actually, if we pick v, and u, w from rest, triangle v-u-w exists if adj[u][w] is true.
            # So any edge in rest that has both endpoints in N(v) is forbidden.
            
            # This is getting complicated. Since n=18, maybe a simple search is fast enough?
            # Number of subsets is 2^18 = 262144.
            # For each subset, check if triangle-free.
            # Checking triangle-free: O(k^3) where k is size.
            # Total: 2^18 * 18^3 ~ 150M ops. Too slow in Python.
            
            # Better: Precompute which subsets are triangle-free?
            # Or use the property:
            # A graph is triangle-free iff it contains no K3.
            
            # Let's try a different angle.
            # The condition "G[S] is triangle-free" is monotone decreasing.
            # If S is not triangle-free, no superset of S is triangle-free.
            
            # Branch and Bound with bitmasks.
            # n=18 fits in an int.
            
            # Precompute for each vertex mask of neighbors.
            # For a set S, it is triangle-free if for every v in S, S \ {v} contains no edge in N(v) ∩ S.
            
            # Let's just use the simple recursive one but with strong pruning.
            
            # Actually, 18 vertices is small enough for a DP over masks if we optimize?
            # No, 2^18 states is fine, but transitions?
            
            # Let's just iterate all masks and check.
            # Check triangle-free for mask:
            # Iterate through set bits.
            
            # Even better:
            # Find all triangles in G.
            # S is triangle-free iff S does not contain any triangle.
            # This is a Set Packing problem (max set avoiding forbidden triples).
            
            # Since n=18, we can find all triangles first.
            # Max triangles in K18 is C(18,3) = 816.
            # Check all 2^18 masks against these?
            
            pass
        
        # Let's optimize the check.
        # Find all triangles in the graph.
        triangles = []
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]: continue
                for k in range(j+1, n):
                    if adj[i][k] and adj[j][k]:
                        triangles.append((i, j, k))
        
        # If no triangles, return n.
        if not triangles:
            return n
            
        # Now check all masks?
        # 262144 masks.
        # For each mask, check if it contains any triangle.
        # A mask contains triangle (i,j,k) if (mask >> i & 1) and ...
        
        # To speed up:
        # For each vertex i, let T_i be the set of triangles containing i.
        
        # Let's just iterate masks.
        best = 0
        # Sort triangles by ...?
        
        # Optimization: If a mask contains a triangle, it's invalid.
        # We can use a trie or just check.
        
        # Given 120s limit, and we call this many times in local search, it must be fast.
        # C-optimized numpy might help?
        # Or just Python loops. 262k * 816 checks is ~200M ops. Too slow.
        
        # Pruning:
        # If we pick a vertex v, we cannot pick any edge (u,w) from N(v) in the remaining set.
        # This suggests a recursive search is better.
        
        def search(remaining_mask, size):
            nonlocal best
            if not remaining_mask:
                if size > best: best = size
                return
            if size + bin(remaining_mask).count('1') <= best:
                return
            
            # Pick a vertex
            # Choose vertex with max degree in remaining subgraph to prune faster?
            # Or just lowest index.
            v_bit = remaining_mask & -remaining_mask
            v = v_bit.bit_length() - 1
            rest = remaining_mask ^ v_bit
            
            # Option 1: Take v
            # If we take v, we cannot take any edge (u,w) where u,w in rest and both are neighbors of v.
            # Let forbidden = {u in rest | adj[v][u]}
            # If u in forbidden, we can take u, but then we cannot take any neighbor of u that is also in forbidden.
            # Basically, the set of vertices from 'forbidden' that we pick must be an independent set in G[forbidden].
            # And the set of vertices from 'rest \ forbidden' must be triangle-free, AND have no edges connecting to 'forbidden' picked?
            # No, edges to 'forbidden' picked don't form triangles with v unless the other endpoint is also in 'forbidden'.
            # Wait. Triangle v-u-w requires u,w neighbors of v AND u-w edge.
            # So if we pick v, we just need to ensure that the subgraph induced by N(v) ∩ S has no edges.
            # i.e., S ∩ N(v) is an independent set.
            # AND S \ N(v) must be triangle-free (in the original graph? No, in the remaining graph).
            # And edges between S∩N(v) and S\N(v) don't form triangles involving v.
            # Triangles not involving v are just triangles in the rest of the graph.
            
            # So:
            # Let A = N(v) ∩ rest
            # Let B = rest \ A
            # We need to pick S_A subset of A such that S_A is independent in G[A].
            # We need to pick S_B subset of B such that S_B is triangle-free in G[B].
            # AND no triangle formed by 1 vertex in S_A and 2 in S_B?
            # Triangle a-b-c where a in A, b,c in B.
            # This is a triangle in the original graph.
            # So we need the whole set S = S_A ∪ S_B to be triangle-free.
            
            # This is still complex.
            
            # Let's just use the simple "take v" and "skip v" but with the constraint that if we take v,
            # we must remove from 'rest' all vertices that form a triangle with v and another vertex in rest?
            # No.
            
            # Given time, I will use a simple heuristic search or a precomputed table?
            # No, let's use the fact that n is small.
            
            # Actually, 18 vertices allows for a full search if implemented in C. In Python, it's tight.
            # Let's try to improve the bound.
            
            # Heuristic: Pick vertex with max degree in remaining.
            
            # Find max degree vertex in remaining
            max_deg = -1
            best_v = -1
            temp = remaining_mask
            while temp:
                bit = temp & -temp
                u = bit.bit_length() - 1
                deg = bin(remaining_mask & adj_mask[u]).count('1')
                if deg > max_deg:
                    max_deg = deg
                    best_v = u
                temp ^= bit
            
            v = best_v
            v_bit = 1 << v
            rest = remaining_mask ^ v_bit
            
            # Take v
            # If we take v, we cannot take any edge from N(v) in the rest.
            # So from the subgraph induced by N(v) ∩ rest, we can only pick an independent set.
            # Let's just branch on picking v, and then in the recursive call, we "remove" all edges between N(v) vertices?
            # No, we must enforce the constraint.
            
            # Simpler:
            # Branch 1: Include v.
            # The rest of the vertices to consider are 'rest'.
            # But we have a constraint: we cannot pick a pair (u,w) from N(v) if they are adjacent.
            # This is equivalent to: in the graph induced by N(v) ∩ rest, we can only pick an independent set.
            # Let's denote this set of constraints.
            
            # This is getting too complicated to implement correctly and fast in Python within the time limit for EVERY step of local search.
            
            # Alternative:
            # The maximum triangle-free subgraph size is at most the independence number of the "triangle graph"?
            
            # Let's go with a simple randomized local search that uses a fast approximate check or a cached check?
            # No, correctness is required.
            
            # What if we use `networkx`? Not allowed (standard lib + numpy + scipy).
            
            # Let's use a simple DFS with the "take v" branch restricting neighbors.
            # If we take v, we add v to the set.
            # We then cannot take any vertex u such that u is adjacent to two other vertices in the set?
            
            # Let's stick to the simplest correct method:
            # Iterate all 2^18 subsets.
            # Optimize the check.
            
            # Precompute: for each triangle (i,j,k), a bitmask.
            tri_masks = [ (1<<i) | (1<<j) | (1<<k) for i,j,k in triangles ]
            
            # For a mask to be triangle-free, (mask & tri_mask) != tri_mask for all tri_masks.
            
            # This is still O(T) per mask.
            
            # Let's just implement the local search and hope the graph structures are simple enough (e.g. sparse)
            # that the number of triangles is small, or the search space is pruned effectively.
            
            # Actually, for a K4-free graph, the number of triangles can be large.
            
            # I will implement the exact solver but try to optimize it with numpy if possible?
            # No, bit operations are faster in Python integers.
            
            # Let's try a different approach for the local search:
            # Instead of minimizing max TF size exactly, maybe we can find a specific known graph?
            # The Ramsey number R(3,4)=9.
            # R(3,5)=14.
            # R(3,6)=18.
            # So a K4-free graph on 18 vertices must contain a triangle?
            # Yes, R(3,4)=9 means any graph on 9 vertices has a triangle or an independent set of size 4.
            # This doesn't help directly.
            
            # Let's assume the local search will work if the solver is fast enough.
            # I will add a timeout check inside the solver? No, it must be exact.
            
            # Let's refine the solver:
            # If the graph is bipartite, TF size is n.
            # If we add edges to break TF-ness, we create triangles.
            
            # I will implement the solver and if it's too slow, the program will timeout.
            # To mitigate, I will start with a graph that is likely to have small TF size.
            # Random K4-free graph.
            
            return best

        # The recursive solver above was incomplete. I need a working one.
        
        # Let's use a simple exhaustive search with pruning based on the maximum possible size.
        
        # Since 2^18 is small, let's just do it.
        # But 2^18 * T might be too slow if T is large.
        
        # Let's try to use the property:
        # If we have a triangle (u,v,w), we can't include all three.
        
        # I will implement a simple backtracking.
        
        # ... (Implementation omitted for brevity in thought, will be in code)
        
        # Given the complexity, I will use a heuristic:
        # The max TF size is likely related to the independence number.
        
        # For the purpose of this challenge, I will provide a local search that uses a fast
        # approximation or a very optimized exact solver.
        
        # Let's write the exact solver using bitmasks and precomputed triangle masks.
        
        best = 0
        # To speed up, iterate masks in descending order of popcount?
        # Or just all.
        
        # Optimization: If we find a mask of size k that is TF, we don't need to check masks of size < k.
        # But we need to check if any larger exists.
        
        # Let's just iterate all.
        # To make it faster, we can skip masks that are subsets of known invalid masks?
        
        # I'll use a simple recursive branch and bound.
        
        def solve_tf(mask):
            nonlocal best
            if not mask:
                if bin(mask).count('1') > best: best = bin(mask).count('1')
                return
            # Heuristic: pick a vertex involved in many triangles?
            
            # Simple bound
            # If size + remaining <= best: return
            
            # Pick v
            v_bit = mask & -mask
            v = v_bit.bit_length() - 1
            rest = mask ^ v_bit
            
            # Check if taking v is possible
            # If we take v, we cannot take any triangle containing v.
            # So we must remove from 'rest' any set of vertices that completes a triangle with v.
            # i.e., if (u,w) is an edge in rest and both u,w are neighbors of v, we can't take both.
            
            # This is hard to prune simply.
            
            # Let's just do:
            # 1. Take v
            # 2. Skip v
            
            # If we take v:
            # We add v.
            # The remaining vertices 'rest' must form a TF set, AND
            # no two vertices in 'rest' that are both neighbors of v can be picked if they are adjacent.
            
            # This is still complex.
            
            # I will fall back to the exhaustive check of all masks, but optimized with early exit.
            
            pass
        
        # Let's use the exhaustive mask check but in a tighter loop.
        # 262144 iterations.
        # Inside, check against triangles.
        
        # To speed up, if the number of triangles is small, it's fast.
        
        max_size = 0
        for m in range(1 << n):
            # Pruning: if popcount(m) <= max_size: continue
            if bin(m).count('1') <= max_size:
                continue
                
            valid = True
            for tm in tri_masks:
                if (m & tm) == tm:
                    valid = False
                    break
            if valid:
                max_size = bin(m).count('1')
        
        return max_size

    # Initialize graph
    # Start with a random K4-free graph.
    # Or start with K_{6,6,6} and perturb?
    # K_{6,6,6} is K4-free.
    
    def random_k4_free():
        # Random graph, then remove edges to break K4s.
        adj = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i][j] = adj[j][i] = 1
        # Remove edges to make K4-free
        # This is hard.
        # Better: Start with empty, add edges randomly if it doesn't form K4.
        
        adj = [[0]*n for _ in range(n)]
        for _ in range(100): # attempts
            for i in range(n):
                for j in range(i+1, n):
                    if not adj[i][j]:
                        # Check if adding (i,j) creates K4
                        # K4 requires i,j and two common neighbors k,l such that k-l is edge.
                        common = [k for k in range(n) if k!=i and k!=j and adj[i][k] and adj[j][k]]
                        conflict = False
                        for k in common:
                            for l in common:
                                if k < l and adj[k][l]:
                                    conflict = True
                                    break
                            if conflict: break
                        if not conflict:
                            if random.random() < 0.8: # density
                                adj[i][j] = adj[j][i] = 1
        return adj

    # Local Search
    best_adj = random_k4_free()
    best_score = max_tf_size(best_adj)
    
    current_adj = [row[:] for row in best_adj]
    current_score = best_score
    
    while time.time() - start_time < timeout:
        # Random edge flip
        i, j = random.sample(range(n), 2)
        if random.random() < 0.5:
            # Add edge
            if not current_adj[i][j]:
                # Check K4
                common = [k for k in range(n) if k!=i and k!=j and current_adj[i][k] and current_adj[j][k]]
                conflict = False
                for k in common:
                    for l in common:
                        if k < l and current_adj[k][l]:
                            conflict = True
                            break
                    if conflict: break
                if not conflict:
                    current_adj[i][j] = current_adj[j][i] = 1
                    score = max_tf_size(current_adj)
                    if score < best_score:
                        best_score = score
                        best_adj = [row[:] for row in current_adj]
                else:
                    # Revert
                    pass
            else:
                # Remove edge
                current_adj[i][j] = current_adj[j][i] = 0
                score = max_tf_size(current_adj)
                if score < best_score:
                    best_score = score
                    best_adj = [row[:] for row in current_adj]
                else:
                    # Revert
                    current_adj[i][j] = current_adj[j][i] = 1

    # Convert best_adj to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    return edges

# Note: The max_tf_size implementation above is likely too slow for local search steps.
# I need to optimize it or change strategy.
# Given the constraints, I will provide a simpler, more robust solution.
# I will use a fixed known graph or a faster heuristic.

def construct():
    # Use a specific construction known to be good.
    # Or run a short search.
    
    # For now, return the seed result if search is too slow.
    # But we must try.
    
    # I will implement a faster TF size check using NumPy if available?
    # No, standard library.
    
    # I will return a K4-free graph from a small search.
    
    # Due to time, I will return the K_{6,6,6} graph if the search doesn't finish.
    # But I'll include the search logic above, fixed.
    
    # Fixed Logic:
    # The max_tf_size function in the previous thought was incomplete.
    # I will implement a simple exhaustive search for max_tf_size which is correct but slow.
    # To make it fast enough, I will limit the search or use a heuristic.
    
    # Actually, 18 vertices is small.
    # Let's try.
    
    pass

# Final Attempt: Provide a working program.

def construct():
    import time
    import random
    
    n = 18
    start_time = time.time()
    timeout = 115.0
    
    # Graph representation: list of sets of neighbors
    # Or bitmask.
    
    # Initialize with K_{6,6,6}
    # Parts 0-5, 6-11, 12-17
    adj = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            # Edge if in different parts
            pi = i // 6
            pj = j // 6
            if pi != pj:
                adj[i][j] = adj[j][i] = 1
                
    def is_k4_free(adj):
        # Fast check using bitmasks
        adj_masks = [0]*n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i][j]:
                    mask |= (1 << j)
            adj_masks[i] = mask
            
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]: continue
                common = adj_masks[i] & adj_masks[j]
                # Check if common contains an edge
                c = common
                while c:
                    bit = c & -c
                    u = bit.bit_length() - 1
                    c ^= bit
                    if (adj_masks[u] & common) > bit: # if u has another neighbor in common
                        # Wait, (adj_masks[u] & common) includes u? No, adj_masks[u] doesn't have u.
                        # So if there is any other bit, it's a K4.
                        if (adj_masks[u] & common) != 0:
                            return False
        return True

    def max_tf_size(adj):
        # Exhaustive search
        # Precompute triangle masks
        tri_masks = []
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]: continue
                for k in range(j+1, n):
                    if adj[i][k] and adj[j][k]:
                        tri_masks.append((1<<i)|(1<<j)|(1<<k))
                        
        if not tri_masks:
            return n
            
        best = 0
        # Iterate masks
        # Optimization: skip if popcount <= best
        for m in range(1 << n):
            if bin(m).count('1') <= best:
                continue
            valid = True
            for tm in tri_masks:
                if (m & tm) == tm:
                    valid = False
                    break
            if valid:
                best = bin(m).count('1')
                if best == n: break
        return best

    best_score = max_tf_size(adj)
    best_adj = [row[:] for row in adj]
    
    # Local Search
    cur_adj = [row[:] for row in adj]
    
    while time.time() - start_time < timeout:
        i, j = random.sample(range(n), 2)
        # Flip
        old = cur_adj[i][j]
        cur_adj[i][j] = cur_adj[j][i] = 1 - old
        
        if is_k4_free(cur_adj):
            score = max_tf_size(cur_adj)
            if score < best_score:
                best_score = score
                best_adj = [row[:] for row in cur_adj]
                # print(f"Improved to {score}")
            else:
                # Revert
                cur_adj[i][j] = cur_adj[j][i] = old
        else:
            # Revert
            cur_adj[i][j] = cur_adj[j][i] = old
            
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    return edges
