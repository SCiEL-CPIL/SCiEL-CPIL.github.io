import time

def construct():
    start = time.time()
    
    # Strategy: Deterministic interval-growth
    # 
    # Key insight: If we maintain the invariant that for each cardinality c,
    # all subset sums of size c lie in a single interval [L_c, R_c],
    # and these intervals are pairwise disjoint for different c,
    # then the set is valid.
    #
    # When we add a new element x, the new sums for cardinality c+1 are
    # {s + x : s in sums of size c}. If sums of size c are in [L_c, R_c],
    # then new sums are in [L_c + x, R_c + x].
    #
    # We need [L_c + x, R_c + x] to be disjoint from all existing intervals [L_j, R_j] for j != c+1.
    #
    # A simple sufficient condition: if we ensure that each new interval is strictly
    # above all existing intervals, or strictly below, we maintain disjointness.
    #
    # Better approach: Use a "superincreasing" style where each new element is large enough
    # that its contribution dominates. But we want to maximize count, so we want small elements.
    #
    # Let's think differently. The constraint is:
    # For c != d, sum_c != sum_d for any subset of size c and any subset of size d.
    #
    # This is equivalent to: the set of all subset sums, indexed by cardinality, has
    # no collisions across different cardinalities.
    #
    # A known construction: powers of 2 give size 8. Can we do better?
    #
    # Let's try a different approach: use a set where the elements grow slowly but
    # the gaps between consecutive elements are large enough to separate the sum ranges.
    #
    # Actually, let's use a more principled approach. 
    # 
    # For a set S = {a_1, a_2, ..., a_k} with a_1 < a_2 < ... < a_k,
    # the minimum sum of size c is a_1 + ... + a_c.
    # The maximum sum of size c is a_{k-c+1} + ... + a_k.
    #
    # If we ensure that for all c < d, max_sum(c) < min_sum(d), then we're valid.
    # This is a sufficient condition (not necessary, but easy to enforce).
    #
    # max_sum(c) = sum of largest c elements
    # min_sum(d) = sum of smallest d elements
    #
    # We need: for all c < d, sum(largest c) < sum(smallest d)
    #
    # The most restrictive case is c = d-1.
    # So we need: sum(largest d-1) < sum(smallest d) for all d = 2, ..., k.
    #
    # Let's try to construct such a set greedily.
    # Start with a_1. Then choose a_2 such that a_1 < a_2 (trivially true if distinct and sorted).
    # For d=2: we need max_sum(1) < min_sum(2), i.e., a_1 < a_1 + a_2, which is always true.
    # For d=3: we need max_sum(2) < min_sum(3), i.e., a_1 + a_2 < a_1 + a_2 + a_3, always true.
    # Wait, this is always true! So this condition is trivially satisfied?
    #
    # No, I made an error. max_sum(c) is the sum of the LARGEST c elements, not the first c.
    # If S = {a_1 < a_2 < ... < a_k}, then:
    # min_sum(c) = a_1 + a_2 + ... + a_c
    # max_sum(c) = a_{k-c+1} + ... + a_k
    #
    # We need max_sum(c) < min_sum(d) for all c < d.
    # The tightest constraint is when c = d-1.
    # So we need: max_sum(d-1) < min_sum(d) for all d.
    # i.e., sum of largest d-1 elements < sum of smallest d elements.
    #
    # For d=2: a_k < a_1 + a_2. Since a_k is the largest, this is very restrictive.
    # For k=8, a_8 < a_1 + a_2. If a_1=1, a_2=2, then a_8 < 3, so a_8 can be at most 2,
    # but a_8 > a_7 > ... > a_2 = 2, contradiction. So this condition is too strong.
    #
    # The actual condition is weaker: we only need that no specific subset sum of size c
    # equals any specific subset sum of size d. It's not about the min/max ranges.
    #
    # Let's go back to a different approach. Use the fact that if we use a "base" representation
    # where each element is a power of a base B, then subset sums are unique if B is large enough.
    # But we want to maximize count, so we want small elements.
    #
    # Alternative: use a set where each element is a multiple of a large power of 2,
    # but with different bit patterns. This is getting complicated.
    #
    # Let's try a simpler heuristic: use a set of the form {1, 2, 4, ..., 2^{k-1}} plus
    # some additional elements that are "safe" to add.
    #
    # Actually, let's just use a well-known construction: the set of all numbers of the form
    # 2^i * 3^j for small i, j. But this might not work.
    #
    # Let's try a different strategy: use a set where the elements are spaced out enough
    # that the subset sums for different cardinalities don't overlap.
    #
    # A simple sufficient condition: if a_{i+1} > sum of all previous elements, then
    # the subset sums are all distinct (not just across cardinalities, but globally).
    # This is the superincreasing sequence. But it grows too fast.
    #
    # For our problem, we only need distinctness across cardinalities, not globally.
    # So we can be more lenient.
    #
    # Let's try: a_i = i * M for some large M. Then subset sums of size c are
    # M * (sum of c distinct indices from 1..k). The sum of c distinct indices from 1..k
    # ranges from c(c+1)/2 to c(2k-c+1)/2.
    # For different c, these ranges might overlap.
    #
    # I think the best approach is to use a randomized search but with a much faster
    # validity check. Let's use a bitmask-based DP that is very fast.
    
    # Fast validity check using sets of sums by cardinality
    def is_valid_fast(s):
        n = len(s)
        # dp[c] = set of sums using exactly c elements
        dp = [set() for _ in range(n + 1)]
        dp[0].add(0)
        for x in s:
            # Iterate backwards
            for c in range(n - 1, -1, -1):
                if dp[c]:
                    new_sums = {v + x for v in dp[c]}
                    dp[c + 1].update(new_sums)
        # Check disjointness
        seen = set()
        for c in range(n + 1):
            if not seen.isdisjoint(dp[c]):
                return False
            seen.update(dp[c])
        return True
    
    # Greedy construction with fast validity check
    # Try to build the largest possible set
    best_set = []
    current = []
    
    # Try to add elements greedily
    for x in range(1, 257):
        if x in current:
            continue
        new_set = current + [x]
        if is_valid_fast(new_set):
            current = new_set
            best_set = current
    
    # Try some random restarts to see if we can do better
    import random
    rng = random.Random(42)
    
    for _ in range(500):
        if time.time() - start > 70:
            break
        # Random restart: pick a random subset of candidates
        candidates = list(range(1, 257))
        rng.shuffle(candidates)
        current = []
        for x in candidates:
            if x in current:
                continue
            new_set = current + [x]
            if is_valid_fast(new_set):
                current = new_set
        if len(current) > len(best_set):
            best_set = current
    
    return [int(x) for x in best_set]
