"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 255
    # Numbers are 2 to 256
    # Index i corresponds to number i + 2
    
    # 1. Precompute Compatibility Graph (Complement of Conflict Graph)
    # We want to find a clique in this graph.
    # compat_mask[i] is a bitmask where bit j is 1 if i and j are compatible (lcm > 256).
    compat_mask = [0] * N
    weights = [0.0] * N
    
    for i in range(N):
        a = i + 2
        weights[i] = 1.0 / a
        mask = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 2
            g = gcd(a, b)
            # lcm(a, b) = a * b // g
            # Compatible if lcm > 256
            if a * b // g > 256:
                mask |= (1 << j)
        compat_mask[i] = mask

    # 2. Bron-Kerbosch with Pivot and Greedy Coloring for Maximum Weight Clique
    # We will work with the set of candidates 'P' as a bitmask.
    # To use coloring bounds effectively, we need an ordering of vertices.
    # Standard BK coloring works best when vertices are ordered by degree or weight.
    # Here, we want to maximize weight. We can order vertices by descending weight.
    # However, the coloring bound usually assumes that the color class gives an upper bound on the size.
    # For weighted clique, we can adapt the bound: 
    # The max weight of a clique in a subset of vertices is bounded by the sum of weights of the color classes?
    # Actually, a simpler and very effective bound for weighted independent set/clique is:
    # Sort vertices in P by some heuristic. Assign colors such that no two adjacent vertices (in the incompatibility graph, i.e., compatible in our graph? No, in the graph where we are finding cliques, adjacent means connected) share a color?
    # Wait, Bron-Kerbosch finds cliques in G.
    # The coloring bound for *unweighted* max clique: Color the subgraph induced by P. The max clique size <= max color.
    # For *weighted* max clique, a common relaxation is to sort vertices by weight and use a bound based on the sum of weights of vertices that could potentially be added.
    # A tighter specific bound for weighted clique:
    # Let P be the set of candidates.
    # We can compute an upper bound on the weight of any clique extendable from the current path.
    # One simple bound: Sum of weights of all vertices in P. (Too loose)
    # Better bound: Greedy coloring of the complement of the induced subgraph on P?
    # If we color the graph G[P] such that adjacent vertices have different colors, then any clique can contain at most one vertex from each color class.
    # Therefore, the maximum weight of a clique in P is <= the sum of the maximum weights in each color class? No, that's not right.
    # If C1, C2... Ck are color classes, a clique picks at most 1 from each. So Max Weight <= sum(max_{v in Ci} w(v)).
    # This is a valid upper bound!
    
    # To implement this efficiently within the recursive BK:
    # 1. Maintain P as a bitmask.
    # 2. At each step, we need to order the vertices in P and assign colors.
    # 3. If current_weight + bound <= best_weight, prune.
    
    # Optimization: Pre-sorting vertices by weight descending helps finding good solutions early.
    # Let's map original indices to a sorted order.
    # But bitmasks are tied to original indices for fast intersection.
    # We can just iterate through the set bits of P in an order determined by weight.
    
    best_weight = 0.0
    best_clique = [] # List of original indices
    
    # Helper to get list of indices from bitmask, sorted by weight descending
    # This is called frequently, so we should cache or optimize.
    # Given N=255, iterating bits is fast enough if we prune well.
    
    def get_sorted_vertices(P_mask):
        verts = []
        m = P_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        # Sort by weight descending
        verts.sort(key=lambda x: -weights[x])
        return verts

    def greedy_color_bound(P_mask):
        """
        Returns an upper bound on the maximum weight of a clique in the subgraph induced by P_mask.
        Uses greedy coloring on the graph induced by P_mask.
        Vertices are processed in order of descending weight.
        Color classes are sets of non-adjacent vertices (independent set in G[P]).
        Wait, for clique finding, we color G[P]. Adjacent vertices must have different colors.
        A clique can have at most one vertex from each color class.
        Bound = sum over color classes c of (max weight in c).
        """
        # Get vertices in P_mask sorted by weight descending
        verts = get_sorted_vertices(P_mask)
        
        # colors[v] = color index assigned to v
        colors = {}
        # color_classes[c] = list of vertices in color class c
        color_classes = []
        
        for v in verts:
            # Find the smallest color c such that v is not adjacent to any vertex in color_classes[c]
            # In our graph, 'adjacent' means compatible (edge exists).
            # So we need v to be NON-compatible (incompatible) with all vertices in color_classes[c]?
            # No. In graph G, an edge means compatible.
            # A clique is a set of mutually compatible vertices.
            # Coloring G: Adjacent vertices (compatible) must have different colors.
            # So if u and v are compatible, they cannot have the same color.
            # We want to assign colors such that if u, v are in the same color, they are NOT compatible (incompatible).
            # This forms an independent set in G.
            # Then a clique in G can pick at most 1 from each independent set (color class).
            
            # Check existing color classes
            assigned_color = -1
            for c, class_verts in enumerate(color_classes):
                # Check if v is incompatible with ALL vertices in this class
                # If v is compatible with ANY vertex in the class, v cannot join this class.
                can_join = True
                for u in class_verts:
                    # Check if v and u are compatible
                    if (compat_mask[v] >> u) & 1:
                        can_join = False
                        break
                if can_join:
                    assigned_color = c
                    break
            
            if assigned_color == -1:
                # Need a new color class
                assigned_color = len(color_classes)
                color_classes.append([])
            
            color_classes[assigned_color].append(v)
            colors[v] = assigned_color
            
        # Calculate bound
        bound = 0.0
        for c_verts in color_classes:
            if c_verts:
                # Max weight in this class
                max_w = max(weights[v] for v in c_verts)
                bound += max_w
        return bound

    # To speed up the coloring, we can use a simpler bound if the set is small,
    # or optimize the coloring. N=255 is small, but recursion depth can be high.
    # A simpler bound that is often effective:
    # Sum of weights of vertices in P that are compatible with the current clique?
    # No, that's the same as sum of weights in P.
    # The coloring bound is much tighter.
    
    # We will implement the recursive BK.
    # P: bitmask of candidates
    # X: bitmask of excluded candidates (already processed, not in P)
    # R: list of vertices in the current clique
    
    def bron_kerbosch(P, X, R, current_weight):
        nonlocal best_weight, best_clique, deadline
        
        if time.time() > deadline:
            return
            
        if P == 0 and X == 0:
            # Found a maximal clique
            if current_weight > best_weight:
                best_weight = current_weight
                best_clique = list(R)
            return
            
        # Bound check
        # If current_weight + bound(P) <= best_weight, prune
        # Note: bound(P) is an upper bound on the weight of any clique in P.
        # The total weight of the final clique will be current_weight + weight(clique_in_P).
        b = greedy_color_bound(P)
        if current_weight + b <= best_weight:
            return
            
        # Pivot selection
        # Choose a pivot u from P union X that maximizes |N(u) intersect P|
        # This reduces the number of recursive calls.
        # N(u) in our context is the set of neighbors of u in the graph (compatible vertices).
        # So we want u such that the number of vertices in P compatible with u is maximized.
        
        PX = P | X
        if PX == 0:
            return
            
        # Find pivot
        # Iterate through bits in PX
        pivot = -1
        max_count = -1
        m = PX
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            # Count neighbors of u in P
            # Neighbors of u are bits set in compat_mask[u]
            # Intersection with P
            neighbors_in_P = compat_mask[u] & P
            count = bin(neighbors_in_P).count('1')
            if count > max_count:
                max_count = count
                pivot = u
            m ^= lsb
            
        # The set of vertices to recurse on is P \ N(pivot)
        # If P \ N(pivot) is empty, we just recurse on P?
        # Standard BK: Iterate over vertices in P \ N(pivot).
        # If P \ N(pivot) is empty, we iterate over P.
        
        # Let's get the list of vertices to iterate
        # We want to process high weight vertices first to find good solutions early
        # But the pivot logic usually iterates P \ N(pivot).
        
        candidates_mask = P & ~compat_mask[pivot] if pivot != -1 else P
        
        # If candidates_mask is 0, we must iterate over P (standard BK behavior when pivot covers all P? 
        # Actually, if P \ N(pivot) is empty, it means P is a subset of N(pivot). 
        # In that case, we can just pick any vertex from P to start the recursion?
        # No, the theorem says we can restrict to P \ N(pivot). If it's empty, we don't need to recurse?
        # Wait, if P \ N(pivot) is empty, then every vertex in P is adjacent to pivot.
        # This implies that {pivot} U R is a superset of any maximal clique extension?
        # No.
        # Let's stick to the standard: Iterate over vertices in P \ N(pivot).
        # If that set is empty, we iterate over P. (This case is rare but possible).
        
        if candidates_mask == 0:
            candidates_mask = P
            
        # Get sorted list of candidates
        cand_list = get_sorted_vertices(candidates_mask)
        
        for v in cand_list:
            # Check if v is still in P (it might have been removed? No, we are iterating a static list)
            # But we need to ensure v is in the current P.
            if not (P & (1 << v)):
                continue
                
            # Recurse
            # New P = P intersect N(v)
            # New X = X intersect N(v)
            # New R = R U {v}
            
            new_P = P & compat_mask[v]
            new_X = X & compat_mask[v]
            
            R.append(v)
            bron_kerbosch(new_P, new_X, R, current_weight + weights[v])
            R.pop()
            
            # Move v from P to X
            P &= ~(1 << v)
            X |= (1 << v)
            
            # Check time again inside loop? 
            # The check at the top of BK handles the entry.
            # But if the loop is long, we might overshoot.
            if time.time() > deadline:
                return

    # Initial call
    # P = all vertices
    # X = empty
    # R = empty
    initial_P = (1 << N) - 1
    bron_kerbosch(initial_P, 0, [], 0.0)
    
    # Convert best_clique indices to numbers
    result_numbers = [idx + 2 for idx in best_clique]
    
    # If no clique found (should not happen, single nodes are cliques), return best single node
    if not result_numbers:
        # Find max weight single node
        max_w = 0
        max_idx = 0
        for i in range(N):
            if weights[i] > max_w:
                max_w = weights[i]
                max_idx = i
        result_numbers = [max_idx + 2]
        
    # Verification
    # Check validity
    for i in range(len(result_numbers)):
        for j in range(i+1, len(result_numbers)):
            a = result_numbers[i]
            b = result_numbers[j]
            g = gcd(a, b)
            if a * b // g <= 256:
                # Fallback: Return the greedy solution or just the best single node if invalid
                # This should not happen if BK is correct.
                # For safety, let's just return the first element if it's a single node, or re-run a simple greedy.
                greedy_set = []
                greedy_mask = 0
                for i in range(N):
                    if (compat_mask[i] & greedy_mask) != 0 or i not in [x-2 for x in greedy_set]:
                         # This logic is messy. Let's just return a valid single node.
                         pass
                
                # Simplest valid fallback: The number with the highest weight (2) is always a valid set of size 1.
                # But we might have a better valid set in best_clique if the check failed due to a bug.
                # Let's assume BK is correct. If it fails, return [2].
                return [2]

    return result_numbers

# EVOLVE-BLOCK-END
