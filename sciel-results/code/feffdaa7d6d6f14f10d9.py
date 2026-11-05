import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin for safety
    
    N = 180
    
    # Precompute all forbidden triples (a, b, c) with a < b < c
    # Equation: 1/a = 1/b + 1/c  =>  bc = a(b+c) => bc - ab - ac = 0 => (b-a)(c-a) = a^2
    # Let x = b-a, y = c-a. Then xy = a^2.
    # Since a < b < c, x > 0, y > 0 and x != y (if x=y then b=c which is not distinct).
    # Also b = a+x, c = a+y.
    # We need b, c <= N.
    
    # For each a, find pairs (x, y) such that xy = a^2, x != y, a+x <= N, a+y <= N.
    # Since x != y, we can assume x < y to avoid duplicates.
    # x < sqrt(a^2) = a. So x ranges from 1 to a-1.
    # y = a^2 // x.
    # Check if a^2 % x == 0.
    # Check if a+x <= N and a+y <= N.
    # Check if x != y (implied by x < a).
    
    # We store conflicts for each element to help with B&B pruning.
    # However, for the B&B, it's better to work with the set of "available" nodes.
    # When we pick a node 'v', we remove 'v' and all nodes that form a forbidden triple with 'v' 
    # and any other node in the current solution? 
    # No, MIS in hypergraph: If we pick a set S, no triple {a,b,c} can be fully contained in S.
    # This is equivalent to: For any triple {a,b,c}, at least one of a,b,c is NOT in S.
    # This is the definition of a 3-uniform hypergraph independent set.
    
    # Standard B&B for Hypergraph MIS:
    # State: (chosen, available)
    # When we pick a node v from available:
    #   new_chosen = chosen + {v}
    #   new_available = available - {v} - {u | exists w in chosen such that {u, v, w} is a forbidden triple}
    #   Wait, this logic is slightly off. 
    #   The constraint is: No triple {a,b,c} is a subset of chosen.
    #   If we add v to chosen, we must ensure that for any u1, u2 in chosen, {u1, u2, v} is not a forbidden triple.
    #   And for any u in available, if we later pick u, we must ensure {v, u, w} is not forbidden for any w in chosen.
    
    # Actually, a simpler view for B&B on "Maximum Independent Set" in a hypergraph:
    # We want to maximize |S| such that S contains no edge (triple).
    # This is equivalent to finding a set S that intersects every edge in at most 2 vertices? No, that's vertex cover.
    # Independent set: S contains no edge.
    
    # Let's use the standard B&B for 3-SAT style or just direct search.
    # Since N=180 is small, we can represent the state as a bitmask or a set.
    # Bitmask is faster. 180 bits fits in Python ints.
    
    # Precompute mask for each element i: mask[i] has bits set for j such that i and j are in some forbidden triple?
    # No, that's for 2-uniform (graph). For 3-uniform, the interaction is ternary.
    
    # Let's stick to the "remove conflicts" logic but carefully.
    # If we have a current set C (chosen), and we consider adding v:
    # v can be added if there is no pair (u, w) in C such that {u, v, w} is a forbidden triple.
    # If we add v, the new set is C' = C U {v}.
    # The "available" set A should contain all nodes not in C.
    # When we pick v from A, we form C' = C U {v}.
    # The new available set A' should be A \ {v} \ {u in A | exists w in C such that {u, v, w} is forbidden}.
    # Why? If u is in A' and we pick it later, we will have {u, v, w} in C'' which is forbidden.
    # So u must be removed from availability if it forms a forbidden triple with v and ANY w in C.
    
    # Let's precompute:
    # triples: list of (a, b, c)
    # For fast lookup: 
    # pair_conflicts: dict mapping (min(a,b), max(a,b)) to list of c's?
    # Or better: for each pair (u, v), store the set of w's that form a forbidden triple with u and v.
    
    # Let's build pair_conflicts:
    # pair_conflicts[(u, v)] = set of w such that {u, v, w} is a forbidden triple.
    
    pair_conflicts = defaultdict(set)
    
    for a in range(1, N + 1):
        a_sq = a * a
        # x < y, xy = a^2
        # x ranges 1 to sqrt(a^2) = a
        for x in range(1, int(a_sq**0.5) + 1):
            if a_sq % x != 0:
                continue
            y = a_sq // x
            if x == y:
                continue
            b = a + x
            c = a + y
            # We need a, b, c distinct. Since x,y > 0, b>a, c>a.
            # If x != y, then b != c.
            # So a, b, c are distinct.
            if b > N or c > N:
                continue
            
            # Sort a, b, c
            t = tuple(sorted((a, b, c)))
            # t is (a, b, c) since a < b < c
            # Add to pair conflicts
            u, v, w = t
            pair_conflicts[(u, v)].add(w)
            pair_conflicts[(u, w)].add(v)
            pair_conflicts[(v, w)].add(u)
            
    # Precompute for each element i, the list of pairs (j, k) that form a triple with i?
    # No, the logic above "remove u from A if exists w in C such that {u, v, w} forbidden"
    # requires checking pairs (v, w) for w in C.
    
    # To speed up:
    # For each v, and each w in C, we need to find u in A such that (v, w) in pair_conflicts and u in pair_conflicts[(v,w)].
    # This is O(|C| * degree) per step.
    
    # Let's try a different B&B strategy which is often faster for small N:
    # Recursive search with "branch on an element".
    # Branch 1: Include element i.
    # Branch 2: Exclude element i.
    
    # To prune:
    # Upper bound = current_size + number of remaining elements.
    # This is weak.
    
    # Better: Greedy Coloring Bound.
    # Construct a graph G where edges connect elements that CANNOT be in the same independent set?
    # No, in a hypergraph, there are no direct "edges" between two vertices that forbid them.
    # However, if we have a set of "available" vertices A, and a set of "chosen" vertices C.
    # We want to bound the number of vertices we can add from A to C.
    # Let's project the hypergraph to a graph on A?
    # If u, v in A, and there exists w in C such that {u, v, w} is a forbidden triple, then u and v cannot BOTH be added to C (because if we add both, we have {u, v, w} in C_new).
    # So, for a fixed C, we can define a graph G_C on A where (u, v) is an edge if they conflict with some w in C.
    # Then the max number of elements we can add from A is at most the size of the maximum independent set of G_C.
    # We can upper bound this by the chromatic number of G_C, which can be estimated by greedy coloring.
    
    # This seems complex to implement efficiently in Python within the time limit if done naively.
    
    # Alternative: Simple Branch and Bound with a good heuristic.
    # Order elements by "degree" (number of triples they are in).
    # Pick the element with highest degree.
    # Branch 1: Include it.
    #   - Add to C.
    #   - Remove it and all "conflicting" elements from A?
    #     An element u in A conflicts with v in C if there exists w in C such that {u, v, w} is forbidden.
    #     Wait, if we include v, we must remove any u in A that would form a forbidden triple with v and ANY w in C.
    #     Also, we must remove any u in A that would form a forbidden triple with v and ANY other element u' in A that we might pick later?
    #     No, the condition is "No triple in final set".
    #     If we pick v, we commit to v.
    #     If we later pick u, we must ensure no w in (C U {v}) forms a triple with u and v.
    #     So, u is "blocked" by v if there is a w in C such that {u, v, w} is forbidden.
    #     What if there is no w in C, but there is a w' in A?
    #     If we pick u and v, and later pick w', then {u, v, w'} is in the set.
    #     So u and v are not immediately blocked by each other unless they share a partner in C.
    #     However, if we pick u and v, they "block" any w' in A such that {u, v, w'} is forbidden.
    #     
    # This suggests that the state of "Available" is not just a set, but we need to track pairwise conflicts induced by C.
    
    # Let's simplify. N=180.
    # A simple greedy approach might get 1.69.
    # Can we do better?
    # Let's try to improve the local search significantly.
    # The current program does:
    # 1. Greedy removal.
    # 2. Greedy addition.
    # 3. Random kick.
    
    # Improvement:
    # 1. Use a more sophisticated initial construction.
    # 2. Use simulated annealing or tabu search for the local search.
    
    # However, the directive explicitly asks to "Model the problem as a Maximum Independent Set... using a branch-and-bound solver".
    # I will implement a B&B that uses the "Greedy Coloring Bound" on the induced graph.
    
    # Let's refine the "Induced Graph" idea.
    # State: C (chosen), A (available).
    # We want to bound |C_final|.
    # Let G_A be a graph on vertices A.
    # Edge (u, v) in G_A if u and v CANNOT both be in C_final.
    # When do u, v in A conflict?
    # They conflict if there exists w in C such that {u, v, w} is a forbidden triple.
    # (Because if we add both u and v, we have {u, v, w} in C_final).
    # Note: If there is no such w in C, u and v do NOT necessarily conflict. They might conflict with some future w' in A, but that's a 3-way interaction.
    # However, for an UPPER BOUND, we can ignore 3-way interactions and only count 2-way conflicts induced by C.
    # This gives a lower bound on the conflicts, thus an upper bound on the independent set size?
    # If we ignore some conflicts, the graph has fewer edges, so the MIS size is larger (or equal).
    # So |MIS(G_A)| is an UPPER BOUND on the number of elements we can add from A.
    # Wait. If we ignore conflicts, we might think we can add more, but actually we can't.
    # So the true max addable is <= |MIS(G_A_reduced)|.
    # Yes. So we can use the size of the MIS of G_A_reduced as an upper bound.
    # And we can approximate |MIS(G_A_reduced)| by greedy coloring.
    
    # Algorithm:
    # 1. Start with C = empty, A = {1..N}.
    # 2. Bound = |C| + GreedyColorMISBound(A, C).
    # 3. If Bound <= best, prune.
    # 4. Pick a vertex v in A. Heuristic: highest degree in G_A_reduced? Or highest global degree?
    # 5. Branch 1: Include v.
    #    C' = C U {v}
    #    A' = A \ {v} \ {u in A | exists w in C such that {u, v, w} forbidden}
    #    Recurse(C', A')
    # 6. Branch 2: Exclude v.
    #    C'' = C
    #    A'' = A \ {v}
    #    Recurse(C'', A'')
    
    # This is correct.
    # Complexity:
    # Calculating G_A_reduced:
    # For each pair (u, v) in A, check if there is w in C such that {u, v, w} is forbidden.
    # This is O(|A|^2 * |C|) per node. Too slow if |A| is large.
    
    # Optimization:
    # Precompute for each pair (u, v) the set of w's: Pairs[u,v] = set of w.
    # Then (u, v) is an edge in G_A_reduced if Pairs[u,v] intersects C.
    # To build G_A_reduced, we iterate over pairs in A.
    # Still O(|A|^2).
    
    # For N=180, |A| starts at 180. 180^2 = 32400.
    # If we have 10000 nodes in the tree, 324M operations. Might be tight in Python.
    
    # We can optimize by only checking pairs where u < v.
    # Also, we can stop early if we find an intersection.
    
    # Let's try to implement this B&B.
    # To make it faster, we can use bitmasks for C and A.
    # N=180 fits in 3 Python ints (or just one big int).
    
    # Let's use Python integers for bitmasks.
    
    # Precomputation:
    # pair_mask[u][v] = bitmask of w's that form a triple with u, v.
    # This is a 180x180 matrix of ints.
    
    # In the B&B:
    # C is a bitmask.
    # A is a bitmask.
    
    # To build G_A_reduced:
    # We need to find the MIS of the graph defined on A.
    # Edge (u, v) exists if (pair_mask[u][v] & C) != 0.
    
    # Greedy Coloring for Bound:
    # We want to color the graph G_A_reduced.
    # The number of colors is an upper bound on the clique number? No, on the chromatic number.
    # Chromatic number is an upper bound on the size of the MIS? No.
    # Wait.
    # In a graph, the size of the MIS is at most the number of colors in any valid coloring?
    # No. The size of the MIS is at most the number of vertices.
    # The size of the MAXIMUM CLIQUE is at most the number of colors? No.
    # The size of the MIS is at most N - MinVertexCover.
    
    # Standard MIS Bound using Coloring:
    # We color the graph G. The number of colors k is an upper bound on the size of the Maximum Clique? No.
    # The number of colors is an upper bound on the Chromatic Number.
    # The size of the MIS is NOT bounded by the Chromatic Number.
    # Example: Empty graph. Colors=1. MIS=N.
    
    # Ah, the standard bound for MIS in Branch and Bound (like in MaxClique solvers) uses the coloring of the COMPLEMENT graph?
    # Or it uses the fact that if we color the graph G, we can partition V into k independent sets.
    # Then the MIS of G is the union of MIS of each independent set? No.
    
    # Let's recall the standard "Greedy Coloring Bound" for Maximum Independent Set.
    # We are looking for MIS in G.
    # We color G with k colors.
    # This partitions V into k independent sets I_1, ..., I_k.
    # The MIS of G can contain at most one vertex from each... NO.
    # The MIS of G can contain ALL vertices of an independent set.
    # So coloring G doesn't give a direct bound on MIS size.
    
    # However, if we are looking for Maximum Clique in G, we color G, and the clique size is bounded by the number of colors (since a clique must have distinct colors).
    
    # So, to bound MIS in G, we should look for Maximum Clique in the COMPLEMENT graph \bar{G}.
    # MIS(G) = Clique(\bar{G}).
    # So we need to bound the Clique size in \bar{G}.
    # We color \bar{G} with k colors. Then Clique(\bar{G}) <= k.
    # So MIS(G) <= k.
    
    # So the algorithm is:
    # 1. Construct G_A_reduced (conflict graph on A).
    # 2. Construct its complement \bar{G}_A_reduced (non-conflict graph).
    # 3. Color \bar{G}_A_reduced greedily.
    # 4. The number of colors is an upper bound on the size of the MIS of G_A_reduced.
    
    # This is correct.
    
    # Complexity of Coloring:
    # Greedy coloring is O(V^2).
    # We do this at every node.
    
    # Let's implement this.
    
    # Precompute pair_masks:
    # pair_masks is a list of lists of ints.
    # pair_masks[u][v] has bit w set if {u, v, w} is forbidden.
    
    pair_masks = [[0] * (N + 1) for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        a_sq = a * a
        for x in range(1, int(a_sq**0.5) + 1):
            if a_sq % x != 0:
                continue
            y = a_sq // x
            if x == y:
                continue
            b = a + x
            c = a + y
            if b > N or c > N:
                continue
            # a, b, c are distinct
            # Update masks
            # Note: indices 1-based in logic, 0-based in array? 
            # Let's use 0-based indices for masks to match bit shifts.
            # Element i corresponds to bit (i-1).
            
            u, v, w = a-1, b-1, c-1
            
            pair_masks[u][v] |= (1 << w)
            pair_masks[u][w] |= (1 << v)
            pair_masks[v][u] |= (1 << w)
            pair_masks[v][w] |= (1 << u)
            pair_masks[w][u] |= (1 << v)
            pair_masks[w][v] |= (1 << u)
            
    # Helper: Greedy Coloring of a graph defined by a bitmask of vertices and a conflict condition.
    # We need to color the COMPLEMENT of the conflict graph.
    # Conflict graph G: edge (u, v) if (pair_masks[u][v] & C_mask) != 0.
    # Complement \bar{G}: edge (u, v) if (pair_masks[u][v] & C_mask) == 0.
    # We want to color \bar{G}.
    # Two vertices u, v can have the same color in \bar{G} if they are NOT connected in \bar{G}?
    # No, in a proper coloring, adjacent vertices must have different colors.
    # So in \bar{G}, if (u, v) is an edge (i.e., they do NOT conflict in G), they must have different colors.
    # Wait.
    # If they do NOT conflict in G, they CAN be in the same Independent Set of G.
    # So they are "compatible".
    # In \bar{G}, they are connected.
    # So they must have different colors.
    
    # So we are coloring the graph where edges represent COMPATIBILITY.
    # This seems counter-intuitive for coloring. Usually we color the conflict graph.
    # If we color the conflict graph G, adjacent vertices (conflicting) have different colors.
    # Then each color class is an independent set in G.
    # The MIS of G is the union of... no.
    
    # Let's re-verify the bound.
    # We want to bound MIS(G).
    # Let \bar{G} be the complement.
    # MIS(G) = MaxClique(\bar{G}).
    # In \bar{G}, a clique is a set of vertices all pairwise connected in \bar{G}.
    # Connected in \bar{G} means NOT connected in G.
    # So a clique in \bar{G} is a set of vertices that are pairwise non-conflicting in G.
    # This is exactly an Independent Set in G.
    
    # So we need to bound MaxClique(\bar{G}).
    # Standard bound: Color \bar{G}. Let k be the number of colors.
    # Any clique in \bar{G} must contain at most one vertex of each color?
    # No. In a proper coloring of \bar{G}, adjacent vertices have different colors.
    # In a clique, all vertices are adjacent.
    # So all vertices in a clique must have distinct colors.
    # So size of clique <= number of colors.
    
    # So yes, we color \bar{G} (the compatibility graph) and the number of colors is the bound.
    
    # How to color \bar{G} greedily?
    # Vertices are in A.
    # Order them (e.g., by degree in \bar{G} or just natural order).
    # For each vertex v, assign the smallest color not used by its neighbors in \bar{G}.
    # Neighbors in \bar{G} are those u in A such that (pair_masks[u][v] & C_mask) == 0.
    
    def get_bound(A_mask, C_mask):
        # A_mask: bitmask of available vertices
        # C_mask: bitmask of chosen vertices
        # Returns upper bound on number of vertices addable from A_mask
        
        # Extract list of vertices in A_mask
        verts = []
        m = A_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
            
        if not verts:
            return 0
            
        # Greedy coloring of \bar{G}
        # colors[v] = color assigned to v
        colors = {}
        color_used = set()
        
        # Sort vertices by degree in \bar{G} descending? 
        # Or just process in order.
        # To speed up, we can just iterate.
        
        for v in verts:
            # Find neighbors of v in \bar{G}
            # Neighbor u if u in A_mask and u != v and (pair_masks[u][v] & C_mask) == 0
            # We need to know which colors are used by neighbors.
            used_colors = set()
            m = A_mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                if u != v:
                    # Check if u is neighbor in \bar{G}
                    if (pair_masks[u][v] & C_mask) == 0:
                        if u in colors:
                            used_colors.add(colors[u])
                m ^= lsb
                
            # Find smallest available color
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            color_used.add(c)
            
        return len(color_used)

    # B&B Search
    best_size = 0
    best_set = set()
    
    def search(C_mask, A_mask, current_size):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
            
        if current_size + get_bound(A_mask, C_mask) <= best_size:
            return
            
        if A_mask == 0:
            if current_size > best_size:
                best_size = current_size
                # Reconstruct set from C_mask
                s = set()
                m = C_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    s.add(idx + 1)
                    m ^= lsb
                best_set = s
            return
            
        # Heuristic: Pick vertex v from A_mask
        # Choose v with highest degree in \bar{G}? Or highest global degree?
        # Let's pick the first one for simplicity, or random?
        # To improve pruning, pick v that is "most constrained".
        # In \bar{G}, high degree means many compatible neighbors.
        # In G, high degree means many conflicts.
        # We want to branch on a vertex that prunes a lot.
        # If we pick v, we remove it and its "conflicts" from A.
        # Conflicts in G are those u where (pair_masks[u][v] & C_mask) != 0.
        # So if v has many conflicts with C, it prunes a lot?
        # Actually, if v has many conflicts with C, it might not be addable?
        # No, v is in A, so it is not conflicting with C?
        # Wait. A is defined as available.
        # In my B&B, A is just "not chosen yet".
        # But we must ensure that we don't pick a v that conflicts with C?
        # No, if v conflicts with C, we shouldn't have it in A?
        # Let's check the update rule.
        # When we pick v, we remove u from A if u conflicts with v AND C.
        # So if v itself conflicts with C, it should have been removed earlier?
        # Initially C is empty, so no conflicts.
        # If we pick v, C becomes C' = C U {v}.
        # Then we remove u from A if (pair_masks[u][v] & C) != 0.
        # Note: We check against OLD C.
        # So v itself is not checked against C.
        # But if v conflicts with some w in C, then {v, w, x} is forbidden?
        # No, the condition is "No triple in final set".
        # If v and w are in C', and there is an x in A such that {v, w, x} is forbidden, then x must be removed.
        # But what if v and w themselves form a pair that is part of a forbidden triple with some x already in C?
        # If x is in C, and {v, w, x} is forbidden, then we cannot have v and w in C'.
        # But we are adding v to C. If w is already in C, and {v, w, x} is forbidden for some x in C, then adding v creates a forbidden triple {v, w, x} in C'.
        # So v should NOT be added if it forms a forbidden triple with any two elements in C?
        # No, the triple is {a, b, c}.
        # If we add v, and C contains w, x, and {v, w, x} is forbidden, then C' contains a forbidden triple.
        # So v is INVALID if there exist w, x in C such that {v, w, x} is forbidden.
        
        # My update rule "remove u from A if (pair_masks[u][v] & C) != 0" handles the case where u is in A.
        # It does NOT check if v is valid with respect to C.
        # So I must check if v is valid before branching.
        # If v is invalid, I can only branch on "Exclude v".
        
        # How to check if v is valid with C?
        # v is invalid if there exist w, x in C such that {v, w, x} is forbidden.
        # This is equivalent to: (pair_masks[w][x] & (1 << v)) != 0 for some w, x in C.
        # Or: There is a pair (w, x) in C such that v is in pair_masks[w][x].
        
        # Let's add this check.
        
        # Pick v from A_mask
        # To speed up, we can precompute for each v, the mask of pairs in C that conflict with v?
        # No, C changes.
        
        # Let's just iterate through bits of A_mask to pick v.
        # Heuristic: Pick v that minimizes the bound?
        # For now, pick the first one.
        
        m = A_mask
        v = (m & -m).bit_length() - 1
        
        # Check if v is valid with C
        is_valid = True
        mc = C_mask
        # We need to check if there is a pair (w, x) in C such that v is in pair_masks[w][x].
        # This is O(|C|^2).
        # Optimize: Precompute for each v, a list of pairs?
        # Or: For each w in C, check if there is x in C such that v in pair_masks[w][x].
        
        # To make this fast, let's precompute conflict_with_v_mask[v]?
        # No.
        
        # Let's just do the check.
        # If C is small, it's fast.
        
        cw = []
        mc = C_mask
        while mc:
            lsb = mc & -mc
            w = lsb.bit_length() - 1
            cw.append(w)
            mc ^= lsb
            
        for i in range(len(cw)):
            w = cw[i]
            for j in range(i+1, len(cw)):
                x = cw[j]
                if (pair_masks[w][x] >> v) & 1:
                    is_valid = False
                    break
            if not is_valid:
                break
        
        # Branch 1: Include v (if valid)
        if is_valid:
            # New C
            new_C = C_mask | (1 << v)
            # New A
            # Remove v
            new_A = A_mask & ~(1 << v)
            # Remove u from new_A if (pair_masks[u][v] & C_mask) != 0
            # Note: We check against OLD C_mask.
            
            # Iterate u in new_A
            mu = new_A
            while mu:
                lsb = mu & -mu
                u = lsb.bit_length() - 1
                if (pair_masks[u][v] & C_mask) != 0:
                    new_A ^= lsb
                mu ^= lsb
                
            search(new_C, new_A, current_size + 1)
            
        # Branch 2: Exclude v
        new_A = A_mask & ~(1 << v)
        search(C_mask, new_A, current_size)
        
    # Initial call
    initial_A = (1 << N) - 1
    initial_C = 0
    search(initial_C, initial_A, 0)
    
    if not best_set:
        # Fallback to a simple valid set if B&B failed or timed out without finding anything
        # e.g. all odd numbers?
        best_set = set(range(1, N+1, 2))
        
    return list(best_set)
