import numpy as np
import itertools
import time
import random

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    d = 11
    
    # Strategy: 
    # We look for a set of integer vectors in Z^11.
    # The condition is: min_{i!=j} ||v_i - v_j||^2 >= max_i ||v_i||^2.
    # 
    # If we restrict to vectors with norm squared 4 (i.e., support size 4 with entries +/-1),
    # then max_norm = 4.
    # The condition becomes: min dist^2 >= 4.
    # For two such vectors v, w:
    # ||v - w||^2 = ||v||^2 + ||w||^2 - 2 v.w = 8 - 2 v.w.
    # We need 8 - 2 v.w >= 4  =>  4 >= 2 v.w  =>  v.w <= 2.
    # 
    # Let v and w have supports S_v and S_w.
    # v.w = sum_{k in S_v \cap S_w} v_k w_k.
    # Let I = S_v \cap S_w. Let m = |I|.
    # The maximum possible value of v.w is m (if signs match on all intersection).
    # So we need m <= 2.
    # If m <= 2, the condition v.w <= 2 is always satisfied (since max dot is 2).
    # If m = 3, we need v.w <= 2. The max dot is 3. We must avoid the case where signs match on all 3 intersection points.
    # If m = 4, v and w have the same support. 
    #   If v == w, dot = 4 (invalid, distinct points).
    #   If v != w, they differ in signs.
    #   Let k be the number of positions where signs differ.
    #   dot = (4-k)*1 + k*(-1) = 4 - 2k.
    #   We need 4 - 2k <= 2 => 2 <= 2k => k >= 1.
    #   So any two distinct vectors with the same support are valid (dist^2 = 8 - 2(4-2k) = 8 - 8 + 4k = 4k. Since k>=1, dist^2 >= 4).
    #
    # So the constraints are:
    # 1. No two vectors are identical.
    # 2. For any pair with intersection size 3, the signs must NOT match on all 3.
    # 3. For any pair with intersection size 4, they must be distinct (handled by 1).
    #
    # This allows us to pick multiple vectors with the same support, as long as they are distinct.
    # In fact, for a fixed support, we can pick any subset of the 16 sign patterns.
    # The only restriction is between supports with intersection 3.
    #
    # Let's model this as a graph coloring / independent set problem on the "supports"?
    # No, it's simpler.
    # Let's use a backtracking search.
    # We will iterate through all possible vectors (support, signs) in a specific order.
    # To make backtracking efficient, we should order them such that conflicts are detected early.
    #
    # Total candidates: C(11,4) * 16 = 330 * 16 = 5280.
    # 
    # We want to find the largest subset.
    # 
    # Key Insight for Pruning:
    # If we fix the support, we can decide which sign patterns to include.
    # However, the conflict with other supports depends on the specific signs.
    # 
    # Alternative Structural Approach:
    # Notice that if we pick a set of supports such that any two supports intersect in at most 2 points, 
    # then we can pick ALL 16 sign patterns for each support.
    # This is because the condition "intersection size <= 2" guarantees validity regardless of signs.
    # 
    # So, one class of solutions is:
    # Find a collection of 4-subsets of {1..11} such that any two intersect in at most 2.
    # For each such subset, add all 16 vectors.
    # Size = 16 * |Collection|.
    #
    # What is the maximum size of such a collection?
    # This is a packing problem.
    # Let's try to find a large packing of 4-subsets with intersection <= 2.
    # 
    # If we can find a packing of size P, we get 16P points.
    # Current best (275) is approx 17 supports * 16 = 272 + 3. 
    # So 17 supports is the current benchmark.
    # Can we find 18 or 19 supports?
    # 
    # Let's search for a packing of 4-subsets with intersection <= 2.
    # This is a Max Independent Set on a graph of 330 nodes (supports), where edges connect supports with intersection 3 or 4.
    # Intersection 4 means identical support (self-loop, ignore).
    # Intersection 3 is the conflict.
    # 
    # So we need a set of supports S such that for any S1, S2 in S, |S1 \cap S2| <= 2.
    # 
    # Let's solve this Max Independent Set on the 330-node graph.
    # This graph is much smaller than the 5280-node vector graph.
    # And once we have the supports, we can just take all 16 signs for each.
    # 
    # But wait, can we do better by mixing?
    # If we pick supports with intersection 3, we can still pick some signs.
    # Specifically, if S1 and S2 intersect in 3 points, we cannot pick signs that match on those 3 points for BOTH.
    # This creates a constraint between the sign choices of S1 and S2.
    # 
    # Let's first try to maximize the number of supports in a "pure" packing (intersection <= 2).
    # If we find a packing of size P, score is 16P.
    # If P=18, score=288.
    # If P=19, score=304.
    #
    # Let's run a greedy/local search on the 330 supports graph.
    
    supports = list(itertools.combinations(range(11), 4))
    n_sup = len(supports)
    
    # Build conflict matrix for supports (intersection > 2)
    # Conflict if intersection size is 3 or 4.
    # Intersection 4 is only self.
    # So conflict if intersection size == 3.
    
    conflict_sup = np.zeros((n_sup, n_sup), dtype=bool)
    for i in range(n_sup):
        si = set(supports[i])
        for j in range(i+1, n_sup):
            sj = set(supports[j])
            inter = si.intersection(sj)
            if len(inter) > 2:
                conflict_sup[i, j] = True
                conflict_sup[j, i] = True
                
    # Now find Max Independent Set on this 330-node graph.
    # We can use a simple greedy with local search, or even brute force with branch and bound if it's fast enough?
    # 330 nodes is still large for exact MIS, but greedy + hill climbing should work well.
    
    best_support_set = []
    best_size = 0
    
    # Greedy initialization
    deg = conflict_sup.sum(axis=1)
    order = np.argsort(deg)
    
    current_selected = []
    selected_set = set()
    for idx in order:
        if not np.any(conflict_sup[idx, current_selected]):
            current_selected.append(idx)
            selected_set.add(idx)
            
    best_size = len(current_selected)
    best_support_set = current_selected.copy()
    
    # Local Search: Try to add, swap, etc.
    # Since the graph is small (330), we can afford more iterations.
    
    for _ in range(1000):
        if time.time() > deadline - 100:
            break
            
        improved = False
        # Try to add any node
        for i in range(n_sup):
            if i not in selected_set:
                if not np.any(conflict_sup[i, current_selected]):
                    current_selected.append(i)
                    selected_set.add(i)
                    improved = True
                    break
        
        if improved:
            best_size = len(current_selected)
            best_support_set = current_selected.copy()
            continue
        
        # Try 1-for-2 swap: remove i, add j, k
        # This is O(N^3) which is 330^3 ~ 36M. Might be slow in Python.
        # Let's try a simpler 1-for-1 swap to improve "potential" or just random restarts.
        # Actually, let's try a random restart greedy.
        
        random.shuffle(order)
        current_selected = []
        selected_set = set()
        for idx in order:
            if not np.any(conflict_sup[idx, current_selected]):
                current_selected.append(idx)
                selected_set.add(idx)
        
        if len(current_selected) > best_size:
            best_size = len(current_selected)
            best_support_set = current_selected.copy()
            
    # Now we have best_support_set.
    # We can generate all 16 vectors for each support.
    points = []
    for s in best_support_set:
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(11, dtype=np.int64)
            for k, val in enumerate(signs):
                v[s[k]] = val
            points.append(v)
            
    # This gives 16 * best_size points.
    # Is this valid? Yes, by construction.
    
    # Can we do better by adding more vectors to supports that are already in the set?
    # No, we already added all 16.
    
    # Can we add vectors from supports NOT in the set?
    # If a support S_new intersects some S_old in 3 points, we can only add vectors from S_new
    # that are "compatible" with the vectors in S_old.
    # Compatibility: For v_new in S_new and v_old in S_old, if they intersect in 3 points,
    # signs must not match on all 3.
    # Since we have ALL 16 signs in S_old, for any v_new, there will be some v_old that matches
    # on the 3 intersection points (specifically, the one with matching signs on those 3 points).
    # Therefore, we CANNOT add ANY vector from a support that intersects an existing support in 3 points.
    # Because the existing support has "all" sign patterns, it "blocks" all patterns of the new support
    # that would be incompatible, but wait...
    # 
    # Let's re-verify.
    # Suppose S_old is in the set. We have all 16 vectors v_old.
    # Suppose S_new intersects S_old in 3 points I.
    # Pick any v_new. Let its signs on I be s_I.
    # There exists a v_old such that its signs on I are exactly s_I.
    # For this pair (v_new, v_old), the dot product on I is sum(s_I * s_I) = 3.
    # Since the intersection is exactly 3, the total dot product is 3.
    # We need dot <= 2.
    # So v_new conflicts with v_old.
    # Since v_old is in the set, v_new cannot be added.
    # 
    # Conclusion: If we take ALL 16 vectors for a support, it blocks ALL vectors from any support
    # that intersects it in 3 or more points.
    # 
    # So the "16 * |Packing|" strategy is optimal for the "all signs" approach.
    # 
    # Can we do better by NOT taking all 16 signs for some supports, allowing more supports?
    # This is the general MIS problem on the 5280 nodes.
    # However, the "all signs" strategy is a very strong heuristic.
    # The bottleneck is the number of supports.
    # 
    # Let's check if we can find a larger packing.
    # Current best_size is likely around 17 or 18.
    # 
    # Let's also try a mixed strategy:
    # 1. Find a large packing of supports (intersection <= 2). Let this be P.
    # 2. For supports in P, take all 16 signs.
    # 3. For supports NOT in P, check if we can add some signs.
    #    But as shown above, if a support S_new intersects ANY S_old in P in 3 points, 
    #    it is completely blocked because S_old has all signs.
    #    So we can only add vectors from supports that intersect ALL supports in P in <= 2 points.
    #    But if a support S_new intersects all S_old in P in <= 2, then S_new could have been added to P!
    #    So the set of supports we use is exactly a maximal independent set in the support graph.
    #    
    # Therefore, the problem reduces to finding the Maximum Independent Set in the graph of 330 supports.
    # The score is 16 * (Size of MIS).
    #
    # Let's refine the MIS search to be more robust.
    # We can use a branch and bound or a more sophisticated local search.
    # Given the small size (330), we can try to improve the greedy result with simulated annealing.
    
    # Let's implement a simple Simulated Annealing on the support set.
    # State: Binary vector of length 330.
    # Energy: Number of conflicts - lambda * (number of selected).
    # We want to minimize conflicts and maximize selected.
    # Actually, we just want a valid set with max size.
    
    # Let's stick to the greedy + random restarts, but increase restarts if time permits.
    
    # Also, consider that we might be able to use Norm 2 vectors?
    # If we mix Norm 2 and Norm 4:
    # Max norm is 4.
    # Norm 2 vectors must be dist >= 4 from all.
    # Norm 2 vectors have support 2.
    # If we include Norm 2 vectors, they might conflict with Norm 4 vectors.
    # However, the "16 * MIS" strategy is clean and likely near-optimal for integer lattices of this type.
    # The kissing number of D11 is 220 (Norm 2 vectors).
    # 16 * 14 = 224.
    # 16 * 17 = 272.
    # 16 * 18 = 288.
    # 16 * 19 = 304.
    # 
    # If we can find a packing of 19 supports, we get 304.
    # Is 19 possible?
    # Upper bound: Each support has 4 elements. Total elements used if disjoint would be 4*19=76.
    # But they must intersect.
    # 
    # Let's run the search.
    
    best_support_set_final = best_support_set
    best_size_final = best_size
    
    # If we have time, do more restarts
    for _ in range(5000):
        if time.time() > deadline - 50:
            break
        random.shuffle(order)
        current_selected = []
        selected_set = set()
        for idx in order:
            if not np.any(conflict_sup[idx, current_selected]):
                current_selected.append(idx)
                selected_set.add(idx)
        
        if len(current_selected) > best_size_final:
            best_size_final = len(current_selected)
            best_support_set_final = current_selected.copy()
            
    # Construct the points
    points = []
    for s in best_support_set_final:
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(11, dtype=np.int64)
            for k, val in enumerate(signs):
                v[s[k]] = val
            points.append(v)
            
    if not points:
        # Fallback to D11 minimal vectors
        for i in range(11):
            for j in range(i+1, 11):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        v = np.zeros(11, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        points.append(v)
                        
    P = np.array(points, dtype=np.int64)
    
    # Validation
    if len(P) < 2:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        
    norms_sq = np.sum(P**2, axis=1)
    max_norm = np.max(norms_sq)
    
    # Check min dist
    min_dist = float('inf')
    for i in range(len(P)):
        for j in range(i+1, len(P)):
            d_sq = np.sum((P[i] - P[j])**2)
            if d_sq < min_dist:
                min_dist = d_sq
                
    if min_dist >= max_norm:
        return P
    else:
        # Should not happen, but fallback
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0,0]], dtype=np.int64)

