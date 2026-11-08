"""Construction of a kissing configuration in R^11 using integer vectors."""
import time
import numpy as np
from itertools import combinations, product

D = 11

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110  # safety margin from 150s limit

    # Generate candidate vectors: all vectors with entries in {-1, 0, 1}
    # with L1 norm (number of nonzeros) = 2 (weight-2 vectors)
    # This gives C(11,2) * 4 = 220 vectors, all with norm^2 = 2.
    # For any two distinct weight-2 vectors, ||a-b||^2 >= 2 = max_norm^2. ✓
    
    # Strategy: Use greedy approach to maximize count.
    # Start with all weight-2 vectors (220), then try to add more vectors
    # from a larger candidate pool while maintaining the constraint.
    
    # Generate all weight-2 vectors
    candidates_w2 = []
    for i, j in combinations(range(D), 2):
        for si, sj in product((1, -1), repeat=2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = si
            v[j] = sj
            candidates_w2.append(v)
    
    # Start with all weight-2 vectors
    selected = candidates_w2[:]
    max_norm_sq = 2  # all have norm^2 = 2
    
    # Try to add weight-1 vectors (norm^2 = 1)
    # For weight-1 vector e_i * s, and weight-2 vector v:
    # ||e_i*s - v||^2 = 1 + 2 - 2*s*v[i] = 3 - 2*s*v[i]
    # s*v[i] can be -1, 0, 1
    # distance^2 = 5, 3, or 1
    # We need distance^2 >= max_norm_sq = 2
    # So s*v[i] cannot be 1, meaning v[i] cannot be s
    # For a weight-1 vector s*e_i, we need: for ALL selected weight-2 vectors v, v[i] != s
    # i.e., no weight-2 vector has entry s at position i
    # But we have ALL weight-2 vectors, so for each position i, there exist vectors with v[i]=1 and v[i]=-1.
    # So NO weight-1 vector can be added. 
    
    # Try weight-3 vectors (norm^2 = 3)? Then max_norm_sq becomes 3.
    # For two weight-2 vectors: ||a-b||^2 >= 2, but we need >= 3. 
    # ||a-b||^2 = 4 - 2*(a·b). Need >= 3 => a·b <= 0.5 => a·b <= 0.
    # So weight-2 vectors can only share 0 positions, or 1 position with opposite signs.
    # This would force us to REMOVE many weight-2 vectors. Not worth it.
    
    # Alternative: Use a different base. What if we use only a subset of weight-2 vectors
    # that are mutually compatible with weight-1 vectors, plus the weight-1 vectors?
    
    # Let's try: include all 22 weight-1 vectors (±e_i for i=0..10).
    # Then max_norm_sq = 1 initially.
    # For two weight-1 vectors: ||s1*e_i - s2*e_j||^2
    # If i != j: 1 + 1 = 2 >= 1. OK.
    # If i == j, s1 != s2: ||s1*e_i - s2*e_i||^2 = (s1-s2)^2 = 4 >= 1. OK.
    # So all 22 weight-1 vectors work together. max_norm_sq = 1, min_dist^2 = 2.
    
    # Now add weight-2 vectors. For weight-2 v and weight-1 s*e_i:
    # ||v - s*e_i||^2 = 2 + 1 - 2*s*v[i] = 3 - 2*s*v[i]
    # Need >= 1 (current max_norm_sq). Always true since min is 1.
    # But if we add weight-2 vectors, max_norm_sq becomes 2.
    # Then we need ALL pairwise distances >= 2 (in squared terms, >= 2).
    # For weight-1 s*e_i and weight-2 v: need 3 - 2*s*v[i] >= 2 => s*v[i] <= 0.5 => s*v[i] <= 0.
    # So v[i] must be 0 or -s.
    
    # This means: for each weight-1 vector s*e_i, every weight-2 vector v we add must have v[i] != s.
    # Since we have both +e_i and -e_i in our weight-1 set, we need v[i] != 1 AND v[i] != -1, so v[i] = 0.
    # This means NO weight-2 vector can be added if we have all 22 weight-1 vectors.
    
    # So the trade-off: 22 weight-1 + 0 weight-2 = 22 vs 220 weight-2 + 0 weight-1 = 220.
    # Clearly 220 is better.
    
    # What about a mix? Suppose we keep weight-1 vectors for a subset of positions.
    # If we keep weight-1 vectors for positions in set S, then weight-2 vectors can only
    # have nonzeros outside S.
    # Number of weight-2 vectors with support outside S: C(11-|S|, 2) * 4
    # Total: |S| * 2 + C(11-|S|, 2) * 4
    
    # Let f(s) = 2s + 4*C(11-s, 2) = 2s + 4*(11-s)*(10-s)/2 = 2s + 2*(11-s)*(10-s)
    # f(s) = 2s + 2*(110 - 21s + s^2) = 2s + 220 - 42s + 2s^2 = 2s^2 - 40s + 220
    # f'(s) = 4s - 40 < 0 for s < 10. So f is decreasing for s in [0, 10].
    # f(0) = 220, f(1) = 2 - 40 + 220 = 182, f(2) = 8 - 80 + 220 = 148, ...
    # So s=0 gives the maximum: 220.
    
    # Therefore, the optimal among these simple constructions is 220.
    
    # Let me try a greedy approach with a larger candidate pool to see if we can do better.
    # Generate candidates: all vectors with entries in {-1, 0, 1} and L1 norm 1 or 2.
    # That's 22 (weight-1) + 220 (weight-2) = 242 candidates.
    
    # Greedy: sort by norm^2 ascending, add if constraint holds.
    # Constraint: for new vector v, min_{u in selected} ||v-u||^2 >= max(max_norm_sq, ||v||^2)
    
    all_candidates = []
    # Weight-1 vectors
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            all_candidates.append(v)
    # Weight-2 vectors
    for i, j in combinations(range(D), 2):
        for si, sj in product((1, -1), repeat=2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = si
            v[j] = sj
            all_candidates.append(v)
    
    # Sort by norm^2 ascending
    norms_sq_list = np.array([np.sum(v**2) for v in all_candidates])
    order = np.argsort(norms_sq_list)
    sorted_candidates = [all_candidates[i] for i in order]
    
    selected = []
    max_norm_sq = 0
    min_dist_sq = np.inf
    
    for v in sorted_candidates:
        if time.time() > deadline - 5:
            break
        v_norm_sq = int(np.sum(v**2))
        new_max_norm_sq = max(max_norm_sq, v_norm_sq)
        
        if not selected:
            selected.append(v)
            max_norm_sq = new_max_norm_sq
            min_dist_sq = np.inf
            continue
        
        # Check if v can be added
        # Need: min_{u in selected} ||v-u||^2 >= new_max_norm_sq
        # And: new_max_norm_sq <= min_dist_sq (existing min distance must still be >= new max norm)
        
        if new_max_norm_sq > min_dist_sq:
            continue  # Adding v would increase max norm beyond current min distance
        
        # Check distances to all selected vectors
        # For efficiency, compute distances
        sel_arr = np.array(selected)
        diffs = sel_arr - v
        dists_sq = np.sum(diffs**2, axis=1)
        new_min_dist_sq = int(np.min(dists_sq))
        
        if new_min_dist_sq >= new_max_norm_sq:
            selected.append(v)
            min_dist_sq = min(min_dist_sq, new_min_dist_sq)
            max_norm_sq = new_max_norm_sq
    
    # The greedy approach should find at least 220 (all weight-2 vectors).
    # Let's check how many we got.
    result = np.array(selected, dtype=np.int64)
    
    # If greedy didn't do better than 220, fall back to the known-good 220
    if len(result) < len(candidates_w2):
        result = np.array(candidates_w2, dtype=np.int64)
    
    return result
# EVOLVE-BLOCK-END
