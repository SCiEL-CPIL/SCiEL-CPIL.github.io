import numpy as np
import itertools
from scipy.spatial.distance import cdist

def construct():
    D = 11
    candidates = []
    
    # Type 1: two ±1's, rest 0 (norm^2 = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Type 2: one ±2, rest 0 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
    
    # Type 3: four ±1's, rest 0 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    # Type 4: three ±1's and one ±2? No, norm^2 = 3+4=7. Not minimal.
    # Let's also try: one ±3, rest 0 (norm^2 = 9) - probably not useful for dense packing.
    # Let's stick to norm^2 = 2 and 4 vectors for now, but improve the selection.
    
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    norms_sq = np.sum(candidates**2, axis=1)
    
    # We want to maximize the size of a subset S such that:
    # min_{u,v in S, u!=v} ||u-v|| >= max_{w in S} ||w||
    
    # Strategy:
    # 1. If we only pick norm^2=2 vectors, max_norm = sqrt(2). Min dist >= sqrt(2) => dist_sq >= 2.
    # 2. If we pick norm^2=4 vectors, max_norm = 2. Min dist >= 2 => dist_sq >= 4.
    # 3. Mixed: max_norm is determined by the largest norm in the set.
    
    # The current code uses dist_sq >= 4 for all pairs. This forces the condition for norm^2=4 vectors.
    # It is a valid sufficient condition for the whole set to be valid IF the max norm is 2.
    # If we select a set where all vectors have norm^2=2, we could potentially pack more if we allowed dist_sq >= 2.
    # But if we mix them, the max norm is 2, so we need dist_sq >= 4 for ALL pairs.
    # Therefore, the constraint dist_sq >= 4 is necessary and sufficient for any set containing at least one norm^2=4 vector.
    # For sets with ONLY norm^2=2 vectors, the constraint is dist_sq >= 2.
    
    # Let's try to find the best independent set under the constraint dist_sq >= 4.
    # This guarantees validity for any mix.
    # To potentially beat this, we could try:
    # A) Max independent set with dist_sq >= 4 (current approach, but better algorithm)
    # B) Max independent set with dist_sq >= 2, restricted to norm^2=2 vectors only.
    # C) Hybrid: Start with a set of norm^2=4 vectors (satisfying dist_sq>=4), then try to add norm^2=2 vectors 
    #    that satisfy dist_sq>=4 to existing ones. (This is covered by A).
    
    # Let's improve the greedy algorithm for A.
    # The number of candidates is:
    # Type 1: C(11,2)*4 = 55*4 = 220
    # Type 2: 11*2 = 22
    # Type 3: C(11,4)*16 = 330*16 = 5280
    # Total: 5522 vectors.
    
    # Building the conflict matrix is O(n^2 * D). 5522^2 * 11 ~ 335 million ops. This is fast in NumPy.
    
    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflict = (dist_sq < 4)
    np.fill_diagonal(conflict, False)
    
    # Also consider the case where we ONLY use norm^2=2 vectors and require dist_sq >= 2.
    # Let's solve both and take the best.
    
    # Case 1: Mixed set, constraint dist_sq >= 4
    # Case 2: Pure norm^2=2 set, constraint dist_sq >= 2
    
    # For Case 2:
    mask_t1 = norms_sq == 2
    candidates_t1 = candidates[mask_t1]
    n_t1 = len(candidates_t1)
    if n_t1 > 0:
        dots_t1 = candidates_t1 @ candidates_t1.T
        dist_sq_t1 = 2 * np.ones((n_t1, n_t1), dtype=np.int64) - 2 * dots_t1 # norms are all 2
        # Wait, norms_sq is 2 for all.
        # dist_sq = 2 + 2 - 2*dot = 4 - 2*dot.
        # We want dist_sq >= 2 => 4 - 2*dot >= 2 => 2*dot <= 2 => dot <= 1.
        # Since dot product of integer vectors with entries -1,0,1 is integer.
        # Conflict if dot > 1.
        # Max possible dot for two distinct vectors with two 1s is 2 (if they are identical).
        # If they are different, dot can be 1, 0, -1, -2.
        # Conflict if dot == 1. (dist_sq = 2) -- Wait, condition is dist_sq >= 2.
        # So dist_sq = 2 is ALLOWED.
        # Conflict if dist_sq < 2.
        # dist_sq = 4 - 2*dot.
        # dist_sq < 2 => 4 - 2*dot < 2 => 2*dot > 2 => dot > 1.
        # So conflict if dot >= 2.
        # Since max dot is 2 (identical), and we exclude identical (diagonal), max dot for distinct is 1?
        # No, two vectors can share one coordinate. E.g., (1,1,0..) and (1,0,1..). Dot = 1. Dist_sq = 2. Allowed.
        # Two vectors can share zero coordinates. Dot = 0. Dist_sq = 4. Allowed.
        # Two vectors can be opposite? (1,1..) and (-1,-1..). Dot = -2. Dist_sq = 8. Allowed.
        # So for distinct Type 1 vectors, is dist_sq ever < 2?
        # dist_sq = 4 - 2*dot.
        # Min dist_sq occurs when dot is max.
        # Max dot for distinct vectors with two 1s is 1.
        # So min dist_sq is 2.
        # Thus, ANY set of distinct Type 1 vectors satisfies dist_sq >= 2.
        # And max norm is sqrt(2).
        # So we can select ALL Type 1 vectors?
        # Let's check.
        # If we select all 220 Type 1 vectors, min dist is sqrt(2), max norm is sqrt(2).
        # Condition: min dist >= max norm. sqrt(2) >= sqrt(2). True.
        # So k=220 is a valid solution using only Type 1 vectors.
        
        # Can we do better by mixing?
        # If we add any Type 2 or Type 3 vector, max norm becomes 2.
        # Then we need min dist >= 2 => dist_sq >= 4.
        # This restricts the Type 1 vectors we can keep.
        # A Type 2 vector (2,0,...) has dot product 2 with (1,0,...) if the 1 is at the same position?
        # No, Type 1 has two 1s. (1,1,0..). Dot with (2,0,0..) is 2.
        # Dist_sq = 2 + 4 - 2*2 = 2.
        # We need dist_sq >= 4. So 2 < 4. Conflict.
        # So a Type 2 vector conflicts with any Type 1 vector that has a 1 at the same position.
        # There are 2*10 = 20 such vectors for each Type 2 vector?
        # Type 2 at i. Type 1 vectors with 1 at i: C(10,1)*2 (signs of other) * 2 (sign of i) = 40?
        # Total Type 1 vectors: 220.
        # So each Type 2 vector kills ~40 Type 1 vectors.
        # But Type 2 vectors also conflict with each other?
        # (2,0..) and (-2,0..). Dist_sq = 4+4-2*(-4) = 16. OK.
        # (2,0..) and (2,0..). Same.
        # (2,0..) and (0,2,..). Dot=0. Dist_sq=8. OK.
        # So Type 2 vectors don't conflict with each other (except identicals which we don't have).
        # But they conflict with Type 3 vectors?
        # Type 3 has four 1s. Dot with Type 2 (2 at i) is 2 if i is in the 4 positions.
        # Dist_sq = 4 + 4 - 2*2 = 4. Allowed (>=4).
        # If i is NOT in the 4 positions, dot=0. Dist_sq=8. Allowed.
        # So Type 2 and Type 3 don't conflict?
        # Wait, Type 3 norm^2=4. Type 2 norm^2=4.
        # Dist_sq = 4+4-2*dot = 8-2*dot.
        # Max dot is 2. Min dist_sq is 4.
        # So Type 2 and Type 3 are compatible!
        
        # What about Type 3 and Type 3?
        # Dot product of two Type 3 vectors.
        # They have 4 ones each.
        # Intersection size k. Dot = sum of products.
        # If they share k positions, dot = sum of signs.
        # Min dist_sq = 8 - 2*max_dot.
        # Max dot is 4 (if identical). Dist_sq = 0. Conflict.
        # If distinct, max dot is 3? (3 same signs, 1 diff). Dist_sq = 8-6=2. Conflict (<4).
        # If 2 same, 2 diff (one +1 one -1 in intersection? No, intersection is positions).
        # Let's say positions A and B. |A|=4, |B|=4.
        # Intersection size m.
        # Dot = sum_{i in A∩B} s_i t_i.
        # Max dot for distinct is 3 (if 3 match, 1 mismatch in intersection? No, if m=4, 3 match 1 mismatch => dot=2. If m=3, all match => dot=3).
        # So max dot for distinct Type 3 is 3.
        # Dist_sq = 8 - 6 = 2. Conflict.
        # So Type 3 vectors conflict with each other if they share 3 or 4 positions with matching signs.
        
        # So the structure is:
        # Pool 1: Type 1 (220 vectors). Compatible with each other (min dist 2).
        # Pool 2: Type 2 (22 vectors). Compatible with each other.
        # Pool 3: Type 3 (5280 vectors). Compatible with Type 2. Conflict with Type 1 (if dist_sq < 4).
        # Conflict T1-T2: Dist_sq = 2+4-2*dot = 6-2*dot.
        # Dot is 2 if T1 has 1 at T2's position.
        # Dist_sq = 2. Conflict.
        # Dot is 0 otherwise. Dist_sq = 6. OK.
        # So T2 conflicts with T1s that have a 1 at its position.
        
        # Conflict T1-T3: Dist_sq = 2+4-2*dot = 6-2*dot.
        # Dot is sum of products at intersection.
        # T1 has 2 positions. T3 has 4.
        # If T1's positions are in T3's positions:
        #   Dot = s1*t1 + s2*t2.
        #   Max dot = 2. Dist_sq = 2. Conflict.
        #   Dot = 0. Dist_sq = 6. OK.
        #   Dot = -2. Dist_sq = 10. OK.
        # If T1 has 1 position in T3:
        #   Dot = ±1. Dist_sq = 4 or 8. OK (>=4).
        # If T1 has 0 positions in T3:
        #   Dot = 0. Dist_sq = 6. OK.
        
        # So T1 conflicts with T3 only if T1's two positions are in T3's four positions AND signs match (dot=2).
        
        # Strategy:
        # 1. Select all Type 1 vectors? k=220.
        # 2. Select all Type 2 and Type 3?
        #    T2 and T3 are compatible.
        #    T3s conflict with each other.
        #    We need an independent set in the T3 graph.
        #    Plus all T2s.
        #    If we pick a large independent set of T3s, can we add T1s?
        #    T1s conflict with T2s and T3s.
        #    If we pick T2s, we kill many T1s.
        #    If we pick T3s, we kill some T1s.
        
        # Let's estimate.
        # Option A: All T1s. k=220.
        # Option B: All T2s (22) + Max Independent Set of T3s.
        #    T3 graph: 5280 nodes.
        #    Degree of a T3 node:
        #    Conflicts with other T3s if dot >= 3? No, dist_sq < 4 => dot > 2.
        #    Dot > 2 means dot >= 3.
        #    How many T3s have dot >= 3 with a given T3?
        #    This requires careful counting.
        
        # Let's just run the greedy on the full conflict matrix (dist_sq < 4) which handles all cases.
        # The current code does this.
        # Why is it only getting ~2.57?
        # Maybe the greedy is bad.
        # Let's improve the greedy.
        # Use a better heuristic: pick node with min degree, break ties randomly.
        # The current code does this.
        
        # Maybe we can use a local search / hill climbing on the independent set?
        # Or use a solver? No, too slow.
        
        # Let's try to increase the number of attempts in the greedy.
        # And maybe add a "repair" step.
        
        # Actually, let's look at the T3 independent set size.
        # If we can get a large T3 independent set, plus T2s, that might be > 220.
        
        # Let's run the greedy specifically on T3s to see the size.
        # And on the full set.
        
        # I will modify the code to:
        # 1. Compute conflict matrix for dist_sq < 4.
        # 2. Run greedy multiple times.
        # 3. Also try a greedy that prioritizes T3s? Or T1s?
        # 4. Try to add T1s to a T2+T3 set? No, the conflict matrix already handles it.
        
        # The issue might be that the greedy gets stuck in local optima.
        # Let's try a "randomized restart" with more iterations.
        # Current: 200 attempts.
        # Let's try 1000 attempts? 5522 nodes. Greedy is O(n^2) per attempt? No, O(n * degree).
        # With vectorization, it's fast.
        
        # Let's also try a different tie-breaking: max degree first? No, min degree is standard for MIS.
        
        # Let's try to explicitly construct a T3 independent set.
        # T3 vectors are (±1, ±1, ±1, ±1, 0, 0, 0, 0, 0, 0, 0) with 4 ones.
        # This is related to the Leech lattice?
        
        # Let's just boost the greedy performance.
        
        best_selected = []
        best_size = 0
        np.random.seed(42)
        
        n_attempts = 500
        for attempt in range(n_attempts):
            remaining = np.ones(n, dtype=bool)
            degrees = conflict.sum(axis=1).copy()
            selected = []
            
            while remaining.any():
                rem_indices = np.where(remaining)[0]
                if len(rem_indices) == 0:
                    break
                
                rem_degrees = degrees[rem_indices]
                min_deg = np.min(rem_degrees)
                min_mask = rem_degrees == min_deg
                min_indices_local = np.where(min_mask)[0]
                
                chosen_local = np.random.choice(min_indices_local)
                chosen_global = rem_indices[chosen_local]
                
                selected.append(chosen_global)
                
                neigh_mask = conflict[chosen_global]
                
                to_remove_mask = np.zeros(n, dtype=bool)
                to_remove_mask[chosen_global] = True
                to_remove_mask[neigh_mask] = True
                
                actual_remove = to_remove_mask & remaining
                
                if np.any(actual_remove):
                    deg_decrease = conflict[actual_remove][:, remaining].sum(axis=0)
                    degrees[remaining] -= deg_decrease
                    
                remaining[actual_remove] = False
            
            if len(selected) > best_size:
                best_size = len(selected)
                best_selected = selected
        
        result = candidates[best_selected]
        return result
