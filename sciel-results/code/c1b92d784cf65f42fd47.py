import time
import random
from itertools import combinations

def construct():
    n = 12
    m = 40
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s
    
    # Represent sets as bitmasks for fast comparability check
    # A subset B is a strict superset of A iff (A | B) == B and A != B
    # i.e., A & B == A and A != B
    
    def is_strict_subset(a, b):
        """Check if a is a strict subset of b (using bitmasks)"""
        return (a & b) == a and a != b
    
    def count_comparable(fam):
        """Count unordered pairs where one is strict subset of other"""
        c = 0
        for i in range(len(fam)):
            for j in range(i+1, len(fam)):
                if is_strict_subset(fam[i], fam[j]) or is_strict_subset(fam[j], fam[i]):
                    c += 1
        return c
    
    def count_comparable_except(fam, idx):
        """Count comparable pairs involving fam[idx] with all others"""
        c = 0
        x = fam[idx]
        for j in range(len(fam)):
            if j == idx:
                continue
            y = fam[j]
            if is_strict_subset(x, y) or is_strict_subset(y, x):
                c += 1
        return c
    
    # Generate all possible masks
    all_masks = list(range(1 << n))
    
    # Initial family: build a long chain + extend
    # Chain: empty set, then add one element at a time
    # For a good chain, use a fixed ordering
    order = list(range(n))
    chain = []
    current = 0
    chain.append(current)  # empty set
    for i in range(n):
        current |= (1 << order[i])
        chain.append(current)
    # chain has 13 sets (sizes 0 through 12)
    
    # Now add 27 more sets. Strategy: for each size k, add sets that are comparable with many chain members.
    # A set S of size k is comparable with chain member of size i if:
    #   i < k: S must contain the chain member of size i (i.e., chain[i] ⊆ S)
    #   i > k: chain member of size i must contain S (i.e., S ⊆ chain[i])
    #   i == k: only if equal (but we need distinct, so no)
    
    # For a set S of size k, it's comparable with chain[i] for i < k iff chain[i] ⊆ S
    # and for i > k iff S ⊆ chain[i]
    
    # To maximize comparability with the chain, for size k, we want S to contain as many small chain members as possible
    # and be contained in as many large chain members as possible.
    
    # Best sets for size k: those that are "aligned" with the chain.
    # The chain members are: C_0 = {}, C_1 = {o0}, C_2 = {o0,o1}, ..., C_12 = {0,1,...,11}
    # A set S of size k is contained in C_i for all i >= k if S ⊆ C_k (the k-th chain member)
    # Actually S ⊆ C_i iff all elements of S are in {o0,...,o_{i-1}}
    
    # Let's just do a greedy + local search approach
    
    # Build initial family
    family = list(chain)  # 13 sets
    used = set(family)
    
    # For each size k from 0 to 12, we want to add sets
    # Count how many sets we have per size
    from collections import Counter
    size_count = Counter()
    for s in family:
        size_count[s.bit_count()] += 1
    
    # Add sets greedily: pick the set that maximizes comparable pairs with existing family
    # Candidate pool: all masks not yet used
    candidates = [mask for mask in all_masks if mask not in used]
    
    # Score a candidate: number of existing family members it's comparable with
    def candidate_score(mask, fam):
        c = 0
        for y in fam:
            if is_strict_subset(mask, y) or is_strict_subset(y, mask):
                c += 1
        return c
    
    # Greedily add 27 more sets
    for _ in range(m - len(family)):
        if time.time() > deadline - 5.0:
            break
        best_mask = -1
        best_score = -1
        # Sample from candidates for speed
        sample_size = min(500, len(candidates))
        sample = random.sample(candidates, sample_size)
        for mask in sample:
            sc = candidate_score(mask, family)
            if sc > best_score:
                best_score = sc
                best_mask = mask
        if best_mask >= 0:
            family.append(best_mask)
            used.add(best_mask)
            candidates.remove(best_mask)
    
    # Ensure we have exactly m sets
    while len(family) < m:
        for mask in all_masks:
            if mask not in used:
                family.append(mask)
                used.add(mask)
                break
    
    # Local search: iteratively try to improve
    best_family = list(family)
    best_score = count_comparable(best_family)
    
    # Precompute for speed: for each mask, which other masks is it comparable with
    # But that's 4096*4096 which is too big. Instead, use the family-level count.
    
    iter_count = 0
    while time.time() < deadline - 2.0:
        improved = False
        # For each position, try replacing with a random candidate
        for i in range(m):
            if time.time() > deadline - 2.0:
                break
            # Try a few random replacements
            for _ in range(20):
                new_mask = random.choice(all_masks)
                if new_mask in used:
                    continue
                # Remove old, add new
                old_mask = family[i]
                old_comp = count_comparable_except(family, i)
                # Temporarily swap
                family[i] = new_mask
                new_comp = count_comparable_except(family, i)
                # Delta in total score: (new_comp - old_comp) + (pairs not involving i stay same)
                # But wait, count_comparable_except only counts pairs involving i.
                # The total score change = new_comp - old_comp
                delta = new_comp - old_comp
                if delta > 0:
                    used.discard(old_mask)
                    used.add(new_mask)
                    best_score += delta
                    best_family = list(family)
                    improved = True
                else:
                    family[i] = old_mask
        # If no improvement, try a different strategy: swap two sets
        if not improved:
            # Try swapping two random sets with two random candidates
            for _ in range(50):
                if time.time() > deadline - 2.0:
                    break
                i, j = random.sample(range(m), 2)
                new_i = random.choice(all_masks)
                new_j = random.choice(all_masks)
                if new_i in used or new_j in used or new_i == new_j:
                    continue
                old_i = family[i]
                old_j = family[j]
                # Count change
                old_score_part = count_comparable_except(family, i) + count_comparable_except(family, j) - (1 if (is_strict_subset(old_i, old_j) or is_strict_subset(old_j, old_i)) else 0)
                family[i] = new_i
                family[j] = new_j
                new_score_part = count_comparable_except(family, i) + count_comparable_except(family, j) - (1 if (is_strict_subset(new_i, new_j) or is_strict_subset(new_j, new_i)) else 0)
                delta = new_score_part - old_score_part
                if delta > 0:
                    used.discard(old_i)
                    used.discard(old_j)
                    used.add(new_i)
                    used.add(new_j)
                    best_score += delta
                    best_family = list(family)
                    improved = True
                else:
                    family[i] = old_i
                    family[j] = old_j
            if not improved:
                # Reset to best and try with different random seed
                family = list(best_family)
                used = set(family)
                # Add some random perturbation
                for _ in range(5):
                    idx = random.randint(0, m-1)
                    new_mask = random.choice(all_masks)
                    if new_mask not in used:
                        old = family[idx]
                        family[idx] = new_mask
                        used.discard(old)
                        used.add(new_mask)
    
    # Final: convert to lists of integers
    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    
    return result
