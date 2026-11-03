import numpy as np
import random
import time

D = 11
DEADLINE_SECONDS = 115.0  # safety margin before 120s kill


def construct():
    start_time = time.time()
    
    # Generate candidate vectors: all non-zero vectors with entries in {-1, 0, 1}
    # but that's 3^11 - 1 = 177146 candidates. We need a smarter approach.
    
    # Strategy: Build vectors greedily. A vector v is compatible with a set S 
    # if for all u in S: |v - u|^2 >= max(|v|^2, max_{w in S} |w|^2)
    # Actually the condition is: minimum pairwise distance >= maximum norm.
    # Pairwise distance between v and u: |v-u|^2
    # Maximum norm: max(|w|^2) for w in the set including the new vector.
    
    # Let's think more carefully. We have a set of vectors. The constraint is:
    # For all pairs (v_i, v_j): |v_i - v_j|^2 >= max_{k} |v_k|^2
    # This means: min_{i!=j} |v_i - v_j|^2 >= max_k |v_k|^2
    
    # So if we add a new vector v to an existing set S with max_norm^2 = M:
    # We need: |v|^2 <= M (so the new max doesn't increase) OR we recompute
    # Actually if |v|^2 > M, then the new max is |v|^2, and we need all pairwise distances >= |v|^2.
    # Simpler approach: keep all vectors with the same norm. If all vectors have norm^2 = R,
    # then we need |v_i - v_j|^2 >= R for all pairs.
    # |v_i - v_j|^2 = |v_i|^2 + |v_j|^2 - 2*v_i.v_j = 2R - 2*v_i.v_j >= R
    # => v_i.v_j <= R/2
    
    # So: find a large set of integer vectors all with the same squared norm R,
    # such that pairwise dot products <= R/2.
    
    # Let's try vectors with exactly k non-zero entries, each being +/-1.
    # Then R = k.
    # Dot product of two such vectors: sum of products of overlapping entries.
    # If they share m positions with same sign and n positions with opposite sign,
    # dot = m - n. And m + n <= k (since each has k non-zero entries).
    # We need m - n <= k/2.
    
    # For k=2: vectors with exactly 2 non-zero entries. There are C(11,2)*4 = 55*4 = 220.
    # This is what the seed program does. R=2, need dot <= 1.
    # Two vectors sharing one position: dot = +/-1. If same sign: 1 <= 1 OK. If opposite: -1 OK.
    # Two vectors sharing two positions (same pair): dot = 2 or 0 or -2. 
    #   Same pair, same signs: dot=2 > 1 BAD.
    #   Same pair, one sign flipped: dot=0 OK.
    #   Same pair, both flipped: dot=2 BAD.
    #   Same pair, opposite pattern: dot=-2 OK.
    # So for each pair of positions, we can have at most 2 vectors (e.g., (+1,+1) and (-1,+1), 
    # or (+1,+1) and (+1,-1), etc.) -- actually we need to check all pairs.
    # The seed gets 220/2 = 110? No, seed gets all 220 but many violate.
    # Actually the seed just returns all 220 and the evaluator checks. Score 1.0 means it's valid.
    # Wait, let me re-check: the seed returns ALL vectors with exactly 2 non-zero entries.
    # That's 220 vectors. But some pairs have dot product 2, violating the constraint.
    # Unless the evaluator normalizes... "Normalizing each vector to norm 2 certifies..."
    # Hmm, maybe the evaluator checks differently. Let me re-read.
    # "minimum pairwise distance is at least the maximum vector norm"
    # Distance = Euclidean distance = sqrt(|v_i - v_j|^2)
    # Maximum vector norm = max |v_k|
    # So: min_{i!=j} sqrt(|v_i - v_j|^2) >= max_k |v_k|
    # Squared: min_{i!=j} |v_i - v_j|^2 >= (max_k |v_k|)^2 = max_k |v_k|^2
    
    # For the seed: all vectors have norm^2 = 2. So max norm^2 = 2.
    # Need |v_i - v_j|^2 >= 2 for all pairs.
    # For two vectors on the same pair of positions with same signs: 
    # v = (1,1,0,...), u = (1,1,0,...): |v-u|^2 = 0 < 2. VIOLATION!
    # But the seed has (1,1) and also (1,1) again? No, itertools.combinations gives each pair once,
    # and si,sj in (1,-1)^2 gives 4 vectors per pair: (1,1),(1,-1),(-1,1),(-1,-1).
    # (1,1) and (-1,-1): |v-u|^2 = (2)^2+(2)^2 = 8 >= 2. OK.
    # (1,1) and (1,-1): |v-u|^2 = 0 + 4 = 4 >= 2. OK.
    # (1,1) and (-1,1): |v-u|^2 = 4 + 0 = 4 >= 2. OK.
    # So within the same pair, all 4 are fine! The distance is at least 2 in squared terms.
    # What about vectors on different pairs? E.g., (1,1,0,...) and (1,0,1,...):
    # |v-u|^2 = 0 + 1 + 1 = 2 >= 2. OK!
    # (1,1,0,...) and (0,1,1,...): |v-u|^2 = 1 + 0 + 1 = 2 >= 2. OK!
    # (1,1,0,...) and (1,0,0,...) -- but (1,0,0,...) isn't in our set (needs 2 non-zeros).
    # So it seems all 220 vectors are valid! Score = 220/220 = 1.0. Makes sense.
    
    # Now to improve: we need MORE than 220 vectors.
    # Let's try adding vectors with 3 non-zero entries (R=3).
    # But then max norm^2 becomes 3, and we need all pairwise distances squared >= 3.
    # The existing 2-vec vectors have norm^2=2 < 3, so they're fine norm-wise.
    # But a 2-vec and a 3-vec: distance squared = |v|^2 + |u|^2 - 2 v.u = 2+3-2*v.u = 5-2*v.u >= 3
    # => v.u <= 1.
    # A 3-vec and another 3-vec: distance squared = 3+3-2*v.u = 6-2*v.u >= 3 => v.u <= 1.5, so v.u <= 1 (integer).
    # Two 2-vecs: need distance squared >= 3. Currently they have distance squared >= 2.
    # Some pairs of 2-vecs have distance squared exactly 2 (e.g., sharing one position, different other).
    # Those would violate if we add 3-vecs!
    
    # So we can't simply add 3-vecs. We need to either:
    # 1. Remove some 2-vecs to make room for 3-vecs
    # 2. Find a different family
    
    # Alternative: use vectors with entries in {-1, 0, 1} but allow more non-zero entries.
    # Or use a different approach: greedy construction.
    
    # Let me try a greedy approach with a pool of candidate vectors.
    # Pool: all vectors in {-1,0,1}^11 with exactly 2 or 3 non-zero entries.
    # That's C(11,2)*4 + C(11,3)*8 = 55*4 + 165*8 = 220 + 1320 = 1540 candidates.
    
    # Greedy: start with empty set. Add vectors one by one if compatible.
    # Compatible means: after adding, min pairwise dist squared >= max norm squared.
    
    # To be efficient, let's precompute norms and use a heuristic ordering.
    
    # Actually, let me try a simpler improvement first:
    # Use vectors with exactly 2 non-zero entries BUT also include vectors with exactly 1 non-zero entry?
    # A 1-vec has norm^2=1. If we add it, max norm might still be 2 (from 2-vecs).
    # Distance between 1-vec (e1) and 2-vec (e1+e2): |e1-(e1+e2)|^2 = |e2|^2 = 1 < 2. VIOLATION.
    # So 1-vecs don't work with 2-vecs.
    
    # Let me try the greedy approach with 2-vecs and 3-vecs.
    # Key insight: if we use ONLY 2-vecs, we get 220.
    # If we use ONLY 3-vecs: R=3, need pairwise dist^2 >= 3.
    # Two 3-vecs: dist^2 = 6 - 2*v.u >= 3 => v.u <= 1.
    # Number of 3-vecs: C(11,3)*8 = 1320.
    # Can we pack a large subset?
    
    # Let's do a greedy MIS on the conflict graph.
    # Two vectors conflict if their distance squared < max(norm^2 of both).
    
    # For a fixed weight class (all same R), conflict means dist^2 < R, i.e., v.u > R/2.
    # For R=2: v.u > 1, i.e., v.u >= 2. Only same vector or negation... 
    # Actually for 2-vecs: v.u can be -2,-1,0,1,2. Conflict if v.u > 1, i.e., v.u = 2.
    # v.u=2 means v and u have same support and same signs. So only the identical vector conflicts.
    # That's why all 220 work!
    
    # For R=3: v.u can be -3,-2,-1,0,1,2,3. Conflict if v.u > 1.5, i.e., v.u >= 2.
    # So two 3-vecs conflict if their dot product is 2 or 3.
    
    # Let's build the conflict graph for 3-vecs and find a large independent set.
    
    # Generate all 3-vecs
    candidates = []
    cand_indices = []
    for positions in [(a, b, c) for a in range(D) for b in range(a+1, D) for c in range(b+1, D)]:
        for s in [(1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1), (-1,1,1), (-1,1,-1), (-1,-1,1), (-1,-1,-1)]:
            v = [0]*D
            for pos, sign in zip(positions, s):
                v[pos] = sign
            candidates.append(v)
    
    n_cand = len(candidates)
    print(f"Number of 3-vec candidates: {n_cand}")
    
    # Compute dot products and find conflicts
    # Conflict: dot > R/2 = 1.5, so dot >= 2
    # Convert to numpy for speed
    cand_arr = np.array(candidates, dtype=np.int8)
    # Dot product matrix
    dots = cand_arr @ cand_arr.T
    # Conflict matrix: True if dot >= 2 (and i != j)
    conflict = (dots >= 2)
    np.fill_diagonal(conflict, False)
    
    print(f"Conflict matrix computed, time: {time.time()-start_time:.2f}s")
    
    # Greedy MIS with degree-based ordering
    # Try multiple orderings and take the best
    best_set = set()
    
    def greedy_mis(order, conflict_mat, selected_set):
        """Greedy: process in order, add if no conflict with already selected."""
        result = set(selected_set)
        for i in order:
            # Check if i conflicts with any in result
            if any(conflict_mat[i, j] for j in result):
                continue
            result.add(i)
        return result
    
    # Ordering 1: by degree (descending) - high degree first
    degrees = conflict.sum(axis=1)
    order1 = list(np.argsort(-degrees))
    
    # Ordering 2: by degree (ascending) - low degree first  
    order2 = list(np.argsort(degrees))
    
    # Ordering 3: random
    order3 = list(range(n_cand))
    random.shuffle(order3)
    
    # Try all three
    for order in [order1, order2, order3]:
        s = greedy_mis(order, conflict, set())
        if len(s) > len(best_set):
            best_set = s
    
    print(f"Best 3-vec set size: {len(best_set)}, time: {time.time()-start_time:.2f}s")
    
    # Now let's also try combining with 2-vecs
    # Generate 2-vecs
    candidates2 = []
    for a in range(D):
        for b in range(a+1, D):
            for sa in [1, -1]:
                for sb in [1, -1]:
                    v = [0]*D
                    v[a] = sa
                    v[b] = sb
                    candidates2.append(v)
    
    cand2_arr = np.array(candidates2, dtype=np.int8)
    n_cand2 = len(candidates2)
    
    # Check which 2-vecs are compatible with the selected 3-vecs
    # A 2-vec v and 3-vec u: dist^2 = 2+3-2*v.u = 5-2*v.u >= 3 (since max norm^2 = 3)
    # => v.u <= 1
    # Also 2-vec vs 2-vec: dist^2 = 4-2*v.u >= 3 => v.u <= 0.5 => v.u <= 0
    # Wait, that changes things! With 3-vecs present, max norm^2 = 3, so 2-vecs need
    # pairwise dist^2 >= 3 among themselves too.
    # Originally with only 2-vecs, max norm^2 = 2, so dist^2 >= 2 was enough.
    # Now with 3-vecs, we need dist^2 >= 3 for ALL pairs including 2-vec pairs.
    
    # So we need to recheck 2-vec pairs. Some 2-vec pairs that were OK before (dist^2=2) 
    # are now not OK (need dist^2 >= 3).
    
    # Let's find which 2-vec pairs have dist^2 < 3.
    dots2 = cand2_arr @ cand2_arr.T
    # dist^2 = 4 - 2*dots2. Need >= 3 => dots2 <= 0.5 => dots2 <= 0.
    # So 2-vecs conflict with each other if dot > 0, i.e., dot >= 1.
    conflict22 = (dots2 >= 1)
    np.fill_diagonal(conflict22, False)
    
    # 2-vec vs 3-vec: dot <= 1 for compatibility
    cross_dots = cand2_arr @ cand_arr.T  # shape (n_cand2, n_cand)
    # Conflict if cross_dot > 1
    conflict23 = (cross_dots > 1)  # shape (n_cand2, n_cand)
    
    # Now greedy: start with best 3-vec set, add compatible 2-vecs
    selected_3 = set(best_set)
    # Check which 2-vecs are compatible with all selected 3-vecs
    compatible_2 = []
    for i in range(n_cand2):
        ok = True
        for j in selected_3:
            if conflict23[i, j]:
                ok = False
                break
        if ok:
            compatible_2.append(i)
    
    print(f"Compatible 2-vecs with 3-set: {len(compatible_2)}")
    
    # Now find max independent set among compatible 2-vecs (conflict if dot22 >= 1)
    best_2_set = set()
    if compatible_2:
        # Build sub-conflict matrix
        comp_arr = np.array(compatible_2)
        sub_conflict = conflict22[np.ix_(comp_arr, comp_arr)]
        n_sub = len(comp_arr)
        
        # Greedy with degree ordering
        sub_degrees = sub_conflict.sum(axis=1)
        order_sub = list(np.argsort(-sub_degrees))
        
        for order in [order_sub, list(range(n_sub)), list(reversed(range(n_sub)))]:
            s = set()
            for idx in order:
                i = comp_arr[idx]
                if any(sub_conflict[idx, j_idx] for j_idx in s):
                    continue
                s.add(idx)
            if len(s) > len(best_2_set):
                best_2_set = s
    
    # Convert back to original indices
    final_2_set = set(comp_arr[i] for i in best_2_set)
    
    print(f"Final 2-vec set size: {len(final_2_set)}")
    print(f"Total vectors: {len(selected_3) + len(final_2_set)}")
    print(f"Time: {time.time()-start_time:.2f}s")
    
    # Build final array
    all_indices = []
    all_indices.extend(selected_3)  # these are indices into candidates (3-vecs)
    # Need to combine: 3-vecs from candidates, 2-vecs from candidates2
    
    final_vectors = []
    for idx in selected_3:
        final_vectors.append(candidates[idx])
    for idx in final_2_set:
        final_vectors.append(candidates2[idx])
    
    if not final_vectors:
        # Fallback to seed
        import itertools
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    final_vectors.append(v)
    
    return np.array(final_vectors, dtype=np.int64)


if __name__ == "__main__":
    result = construct()
    print(f"Result shape: {result.shape}")
    # Verify
    if result.shape[0] > 1:
        norms_sq = np.sum(result**2, axis=1)
        max_norm_sq = norms_sq.max()
        # Compute pairwise distances
        diff = result[:, np.newaxis, :] - result[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = dist_sq.min()
        print(f"Max norm^2: {max_norm_sq}, Min dist^2: {min_dist_sq}")
        print(f"Valid: {min_dist_sq >= max_norm_sq}")
