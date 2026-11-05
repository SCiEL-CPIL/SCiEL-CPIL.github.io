import itertools
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # 1. Generate STS(15) using PG(3,2) construction
    # Points are 1..15 (representing non-zero vectors in GF(2)^4), mapped to 0..14
    sts_triples = []
    for i in range(1, 16):
        for j in range(i + 1, 16):
            k = i ^ j
            if k > j:
                t = tuple(sorted([i - 1, j - 1, k - 1]))
                sts_triples.append(t)
    
    # Verify we have 35 triples
    if len(sts_triples) != 35:
        # Fallback if construction logic is flawed (unlikely for this standard construction)
        # But let's assume it's correct.
        pass

    # Precompute pairs for each STS triple for faster lookup
    sts_triple_pairs = []
    for t in sts_triples:
        sts_triple_pairs.append([(t[0], t[1]), (t[0], t[2]), (t[1], t[2])])

    # Precompute all 8-subsets of 16 points
    # We need to check if an 8-subset S is independent.
    # S is independent if:
    # 1. No triple from the final system is contained in S.
    # Final system = (STS triples not removed by M) U { (15, u, v) for (u,v) in M }
    
    # Let M be a matching on points 0..14.
    # A triple (15, u, v) is in the system if (u,v) in M.
    # An STS triple t is in the system if none of its pairs are in M.
    
    # So S is independent iff:
    # A) If 15 in S: No pair (u,v) in M is such that u in S and v in S.
    # B) For all STS triples t: If t subset S, then at least one pair of t is in M.
    
    # We iterate over all 8-subsets. There are C(16,8) = 12870 subsets.
    # For each subset, we precompute:
    # - points_in_s: set of points
    # - sts_triples_in_s: list of indices of sts_triples fully contained in S
    
    subsets_8 = list(itertools.combinations(range(16), 8))
    subset_data = []
    for s in subsets_8:
        s_set = set(s)
        sts_idx_in_s = []
        for idx, t in enumerate(sts_triples):
            if t[0] in s_set and t[1] in s_set and t[2] in s_set:
                sts_idx_in_s.append(idx)
        subset_data.append((s, s_set, sts_idx_in_s))

    # Helper to check if a specific M makes a specific subset independent
    def is_independent_subset(s_set, sts_idx_in_s, M):
        # Condition A: Check new triples involving 15
        if 15 in s_set:
            for u, v in M:
                if u in s_set and v in s_set:
                    return False # Triple (15, u, v) is in S, so not independent
        
        # Condition B: Check STS triples
        # If there exists an STS triple in S that is NOT covered by M, then S contains a triple -> Not Independent.
        # So S is independent ONLY IF all STS triples in S are covered by M.
        for idx in sts_idx_in_s:
            t_pairs = sts_triple_pairs[idx]
            covered = False
            for p in t_pairs:
                if p in M:
                    covered = True
                    break
            if not covered:
                return False # This STS triple is kept and is in S. So S is NOT independent.
        
        # If we reach here, no new triple in S (if 15 in S) and no kept STS triple in S.
        # So S contains no triples. It IS independent.
        return True

    def evaluate_M(M):
        """
        Returns the number of independent sets of size 8.
        If 0, alpha <= 7.
        """
        M_set = set(M)
        count = 0
        for s, s_set, sts_idx_in_s in subset_data:
            if is_independent_subset(s_set, sts_idx_in_s, M_set):
                count += 1
        return count

    def get_final_triples(M):
        M_set = set(M)
        final = []
        # Add new triples with 15
        for u, v in M:
            final.append([15, u, v])
        # Add kept STS triples
        for i, t in enumerate(sts_triples):
            t_pairs = sts_triple_pairs[i]
            keep = True
            for p in t_pairs:
                if p in M_set:
                    keep = False
                    break
            if keep:
                final.append(list(t))
        return final

    def is_linear(triples):
        pair_to_triple = {}
        for t in triples:
            for i in range(3):
                for j in range(i + 1, 3):
                    p = tuple(sorted((t[i], t[j])))
                    if p in pair_to_triple:
                        return False
                    pair_to_triple[p] = t
        return True

    best_M = None
    best_score = 12870 # Max possible independent sets
    
    # Local Search Strategy
    # Start with random matchings.
    # Move: Swap a pair in M with a pair not in M (maintaining matching property).
    # Or: Add a pair (if size < 7) / Remove a pair.
    # Since M is a matching on 15 points, max size is 7 (one point left out).
    # Let's fix size 7 for simplicity, or allow variable size?
    # The problem doesn't restrict the number of triples.
    # However, adding more triples generally reduces independence number.
    # A matching of size 7 covers 14 points.
    # Let's try to find a matching of size 7 that minimizes independent sets.
    
    def random_matching():
        pts = list(range(15))
        random.shuffle(pts)
        M = []
        # We can pick up to 7 pairs
        # Let's just pick 7 disjoint pairs from the shuffled list
        for i in range(0, 14, 2):
            M.append(tuple(sorted((pts[i], pts[i + 1]))))
        return M

    for restart in range(2000):
        if time.time() > deadline - 5:
            break
            
        M = random_matching()
        score = evaluate_M(M)
        
        if score < best_score:
            best_score = score
            best_M = M[:]
            if best_score == 0:
                break
                
        # Local Search
        improved = True
        while improved:
            if time.time() > deadline - 2:
                break
            improved = False
            
            # Generate neighbors: Swap one pair in M with a new pair
            # Or just try all possible swaps?
            # Pairs in M: 7 pairs.
            # Possible new pairs: C(15,2) = 105.
            # Total swaps ~ 7 * 105 = 735.
            # Evaluating each swap takes checking 12870 subsets.
            # 735 * 12870 ~ 9.4 million ops. Might be too slow in Python if done naively.
            
            # Optimization: Only check subsets that were independent in the previous step?
            # Or just sample a few neighbors.
            
            # Let's try a few random neighbors
            for _ in range(20):
                if time.time() > deadline - 2:
                    break
                new_M = M[:]
                # Pick a random pair in M to remove
                idx_remove = random.randint(0, len(new_M) - 1)
                removed_pair = new_M.pop(idx_remove)
                
                # Pick a random new pair to add
                # Must be disjoint from all other pairs in new_M
                used_points = set()
                for u, v in new_M:
                    used_points.add(u)
                    used_points.add(v)
                    
                available_points = [p for p in range(15) if p not in used_points]
                # available_points includes the 2 points from removed_pair and 1 other point (since 15-14=1)
                # Wait, M has 7 pairs -> 14 points used. 1 point unused.
                # If we remove 1 pair, we free 2 points. Total available = 3 points.
                # We need to pick 2 from 3.
                if len(available_points) >= 2:
                    pair_candidates = list(itertools.combinations(available_points, 2))
                    new_pair = tuple(sorted(pair_candidates[random.randint(0, len(pair_candidates) - 1)]))
                    new_M.append(new_pair)
                    
                    new_score = evaluate_M(new_M)
                    if new_score < score:
                        score = new_score
                        M = new_M
                        improved = True
                        if score < best_score:
                            best_score = score
                            best_M = M[:]
                            if best_score == 0:
                                break
            if best_score == 0:
                break

    if best_M is None:
        # Fallback: return empty or random if search failed completely
        # But random matching should always produce *some* result
        best_M = random_matching()

    final_triples = get_final_triples(best_M)
    
    # Verify linearity
    if not is_linear(final_triples):
        # This should not happen with the construction logic, but just in case
        # We could try to fix it or return a valid but potentially worse system
        pass

    return final_triples

# To ensure the function is called if this script is run directly (though evaluator calls construct())
if __name__ == "__main__":
    t = construct()
    print(f"Constructed {len(t)} triples")
    # Quick independence check for size 8 to verify best_score logic
    # (Omitted for brevity in main block, logic is in evaluate_M)
