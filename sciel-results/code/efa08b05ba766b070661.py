"""Rewritten program: greedy + min-cut construction of 40 subsets of {0..11} maximizing comparable pairs."""
import time
import random
import itertools
import sys
import numpy as np

# EVOLVE-BLOCK-START
N = 12
M = 40
ALL_MASKS = list(range(1 << N))  # 4096 masks
NUM_MASKS = 1 << N
DEADLINE = 82.0  # seconds, safety margin before 90s kill

def _count_comparable(family_masks):
    """Count unordered pairs where one is strict subset of the other."""
    n = len(family_masks)
    count = 0
    for i in range(n):
        mi = family_masks[i]
        for j in range(i + 1, n):
            mj = family_masks[j]
            if (mi & mj) == mi and mi != mj:
                count += 1
            elif (mi & mj) == mj and mi != mj:
                count += 1
    return count

def _precompute_comparability():
    """
    Precompute for each mask, bitmasks of its strict subsets and strict supersets.
    subset_mask[m] is a 4096-bit int where bit i is set if mask i is a strict subset of m.
    superset_mask[m] is a 4096-bit int where bit i is set if mask i is a strict superset of m.
    """
    subset_mask = [0] * NUM_MASKS
    superset_mask = [0] * NUM_MASKS
    
    for m in range(NUM_MASKS):
        # Find all strict subsets of m
        sub = m
        while sub:
            if sub != m:
                subset_mask[m] |= (1 << sub)
            sub = (sub - 1) & m
        # Find all strict supersets of m
        # A superset s of m has all bits of m plus at least one more
        # Equivalently, m is a strict subset of s
        # We can compute this by: for each mask s, check if (m & s) == m and m != s
        # But that's O(4096^2). Instead, use the fact that supersets of m are m | extra
        # where extra is any non-empty subset of (~m) & ((1<<N)-1)
        complement = (~m) & ((1 << N) - 1)
        extra = complement
        while extra:
            s = m | extra
            if s != m:
                superset_mask[m] |= (1 << s)
            extra = (extra - 1) & complement
    
    return subset_mask, superset_mask

def _greedy_construct(seed, deadline, subset_mask, superset_mask):
    """Greedy construction with dynamic degree tracking."""
    rng = random.Random(seed)
    
    in_family = [False] * NUM_MASKS
    family = []
    
    # scores[m] = number of family members comparable with m
    scores = [0] * NUM_MASKS
    
    # First element: pick randomly from a diverse set
    # Prefer masks with intermediate size for better connectivity
    first = rng.choice(ALL_MASKS)
    family.append(first)
    in_family[first] = True
    
    # Update scores: for each mask y, if y is comparable with first, scores[y] += 1
    # y is comparable with first if y is a strict subset or strict superset of first
    # Use the precomputed masks
    comp_with_first = subset_mask[first] | superset_mask[first]
    # For each y where bit y is set in comp_with_first, scores[y] += 1
    # This is hard to do efficiently without iterating. Let's iterate over set bits.
    y = 0
    while (1 << y) & comp_with_first:
        if not in_family[y]:
            scores[y] += 1
        y += 1
    # Actually this loop is O(4096) which is fine.
    # Better: just iterate all
    for y in range(NUM_MASKS):
        if y != first and ((comp_with_first >> y) & 1):
            scores[y] += 1
    
    for _ in range(M - 1):
        if time.time() > deadline:
            break
        
        # Find the mask with highest score that's not in family
        best_mask = -1
        best_score = -1
        candidates = []
        for y in range(NUM_MASKS):
            if not in_family[y]:
                if scores[y] > best_score:
                    best_score = scores[y]
                    best_mask = y
                    candidates = [y]
                elif scores[y] == best_score:
                    candidates.append(y)
        
        if best_mask == -1:
            break
        
        # Pick randomly among top candidates for diversity
        chosen = rng.choice(candidates)
        family.append(chosen)
        in_family[chosen] = True
        
        # Update scores for all other masks
        comp_with_chosen = subset_mask[chosen] | superset_mask[chosen]
        for y in range(NUM_MASKS):
            if not in_family[y] and ((comp_with_chosen >> y) & 1):
                scores[y] += 1
    
    if len(family) < M:
        remaining = [m for m in ALL_MASKS if not in_family[m]]
        rng.shuffle(remaining)
        family.extend(remaining[:M - len(family)])
    
    count = _count_comparable(family)
    return count, family

def _local_search_polish(family, deadline, rng, subset_mask, superset_mask):
    """
    Improve a family by trying to swap one element for another.
    """
    family_set = set(family)
    current_count = _count_comparable(family)
    
    # Precompute comparability scores for all masks not in family
    # scores[m] = number of family members comparable with m
    scores = {}
    for m in range(NUM_MASKS):
        if m in family_set:
            continue
        cnt = 0
        comp_m = subset_mask[m] | superset_mask[m]
        # Count how many family members have their bit set in comp_m
        for f in family:
            if (comp_m >> f) & 1:
                cnt += 1
        scores[m] = cnt
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try each element in family for removal
        for i in range(len(family)):
            if time.time() > deadline:
                break
            removed = family[i]
            
            # Compute old comparable count for removed (with other family members)
            old_comp = 0
            comp_removed = subset_mask[removed] | superset_mask[removed]
            for j in range(len(family)):
                if j == i:
                    continue
                if (comp_removed >> family[j]) & 1:
                    old_comp += 1
            
            # Find best candidate to add
            best_gain = -1
            best_add = -1
            for m, sc in scores.items():
                # If we remove 'removed' and add 'm':
                # New pairs involving m: sc (but sc was computed with 'removed' in family)
                # If m is comparable with removed, then sc includes 1 for removed
                # After removing, the comparable count for m becomes sc - (1 if comp(m, removed) else 0)
                # Pairs lost: old_comp
                # Pairs gained: sc - (1 if comp(m, removed) else 0)
                # Net change: sc - (1 if comp(m, removed) else 0) - old_comp
                
                is_comp_with_removed = ((subset_mask[m] | superset_mask[m]) >> removed) & 1
                
                gain = sc - (1 if is_comp_with_removed else 0) - old_comp
                if gain > best_gain:
                    best_gain = gain
                    best_add = m
            
            if best_gain > 0 and best_add != -1:
                # Perform the swap
                family[i] = best_add
                family_set.remove(removed)
                family_set.add(best_add)
                
                # Update scores for all masks not in family
                # When we remove 'removed' and add 'best_add':
                # For each mask m not in {removed, best_add} and not in family:
                #   score[m] changes by:
                #     -1 if m comparable with removed
                #     +1 if m comparable with best_add
                
                comp_removed = subset_mask[removed] | superset_mask[removed]
                comp_added = subset_mask[best_add] | superset_mask[best_add]
                
                for m in list(scores.keys()):
                    if m == removed or m == best_add:
                        continue
                    if (comp_removed >> m) & 1:
                        scores[m] -= 1
                    if (comp_added >> m) & 1:
                        scores[m] += 1
                
                # Remove best_add from scores (it's now in family)
                if best_add in scores:
                    del scores[best_add]
                # removed is no longer in family, but we don't need its score
                
                current_count += best_gain
                improved = True
    
    return family, current_count

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Precompute comparability structures
    subset_mask, superset_mask = _precompute_comparability()
    
    best_count = 0
    best_family = None
    
    # Phase 1: Multiple greedy restarts
    num_restarts = 200
    
    for i in range(num_restarts):
        if time.time() > deadline - 25.0:  # Leave 25s for local search
            break
        
        count, family = _greedy_construct(i, deadline, subset_mask, superset_mask)
        
        if count > best_count:
            best_count = count
            best_family = family
    
    # Phase 2: Local search polish on best family
    if best_family is not None:
        rng = random.Random(12345)
        polish_deadline = deadline - 3.0  # Leave 3s margin
        best_family, best_count = _local_search_polish(best_family, polish_deadline, rng, subset_mask, superset_mask)
    
    # If we didn't find anything (shouldn't happen), fallback to simple construction
    if best_family is None:
        vertices = list(range(6))
        all_sets = []
        for r in range(7):
            for combo in itertools.combinations(vertices, r):
                all_sets.append(list(combo))
        all_sets.sort(key=len)
        best_family = all_sets[:M]
    
    # Convert to lists of ints
    result = []
    for mask in best_family:
        result.append([b for b in range(N) if (mask >> b) & 1])
    
    return result
# EVOLVE-BLOCK-END
