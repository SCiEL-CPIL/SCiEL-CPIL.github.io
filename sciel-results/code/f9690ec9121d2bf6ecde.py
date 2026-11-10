"""
Maximize comparable pairs among 40 distinct subsets of {0..11}.
Strategy: fast precompute + greedy + strong local search (1-swap, 2-swap) + simulated annealing.
"""
import time
import random
import itertools
from collections import deque

# EVOLVE-BLOCK-START
N = 12
M = 40
NUM_MASKS = 1 << N  # 4096
ALL_MASKS = list(range(NUM_MASKS))
DEADLINE_MARGIN = 5.0  # seconds safety margin before 90s kill

def _precompute_comparability():
    """
    Precompute for each mask m:
    - comp_mask[m]: a 4096-bit integer where bit j is set iff m and j are strictly comparable
    - comp_count[m]: number of masks strictly comparable with m
    """
    comp_mask = [0] * NUM_MASKS
    comp_count = [0] * NUM_MASKS
    
    # For each pair (m1, m2) with m1 < m2, check comparability
    for m1 in range(NUM_MASKS):
        for m2 in range(m1 + 1, NUM_MASKS):
            if (m1 & m2) == m1:
                # m1 is subset of m2
                if m1 != m2:
                    comp_mask[m1] |= (1 << m2)
                    comp_mask[m2] |= (1 << m1)
                    comp_count[m1] += 1
                    comp_count[m2] += 1
            elif (m1 & m2) == m2:
                # m2 is subset of m1
                if m1 != m2:
                    comp_mask[m1] |= (1 << m2)
                    comp_mask[m2] |= (1 << m1)
                    comp_count[m1] += 1
                    comp_count[m2] += 1
    
    return comp_mask, comp_count

def _count_comparable(family_masks, comp_mask):
    """Count unordered pairs where one is strict subset of the other."""
    n = len(family_masks)
    count = 0
    for i in range(n):
        mi = family_masks[i]
        for j in range(i + 1, n):
            mj = family_masks[j]
            if (comp_mask[mi] >> mj) & 1:
                count += 1
    return count

def _greedy_construct(seed, comp_mask, comp_count):
    """
    Greedy construction: start with a random mask, then repeatedly add the mask
    that maximizes comparable pairs with the current family.
    """
    rng = random.Random(seed)
    
    in_family = [False] * NUM_MASKS
    family = []
    
    # Pick a random starting mask
    first = rng.choice(ALL_MASKS)
    family.append(first)
    in_family[first] = True
    
    # Score for each mask: number of family members it's comparable with
    scores = [0] * NUM_MASKS
    for y in ALL_MASKS:
        if y == first:
            continue
        if (comp_mask[first] >> y) & 1:
            scores[y] = 1
    
    for _ in range(M - 1):
        # Find best candidate
        best_score = -1
        best_candidates = []
        for y in ALL_MASKS:
            if not in_family[y]:
                if scores[y] > best_score:
                    best_score = scores[y]
                    best_candidates = [y]
                elif scores[y] == best_score:
                    best_candidates.append(y)
        
        if not best_candidates:
            break
        
        # Pick randomly among best candidates
        chosen = rng.choice(best_candidates)
        family.append(chosen)
        in_family[chosen] = True
        
        # Update scores
        for y in ALL_MASKS:
            if not in_family[y]:
                if (comp_mask[chosen] >> y) & 1:
                    scores[y] += 1
    
    # Fill if needed
    if len(family) < M:
        remaining = [m for m in ALL_MASKS if not in_family[m]]
        rng.shuffle(remaining)
        family.extend(remaining[:M - len(family)])
    
    return family

def _local_search_1swap(family, comp_mask, deadline):
    """
    Local search: try all 1-swap moves (remove one, add one).
    Repeat until no improvement or deadline.
    """
    family_set = set(family)
    current_count = _count_comparable(family, comp_mask)
    
    # Precompute scores: for each mask not in family, how many family members is it comparable with
    scores = [0] * NUM_MASKS
    for m in range(NUM_MASKS):
        if m in family_set:
            continue
        for f in family:
            if (comp_mask[m] >> f) & 1:
                scores[m] += 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # For each position i, try removing family[i] and adding the best candidate
        for i in range(len(family)):
            if time.time() > deadline:
                break
            
            removed = family[i]
            
            # Count how many family members removed is comparable with
            old_comp = 0
            for j in range(len(family)):
                if j == i:
                    continue
                if (comp_mask[removed] >> family[j]) & 1:
                    old_comp += 1
            
            # Find best candidate to add
            best_gain = 0
            best_add = -1
            for m in range(NUM_MASKS):
                if m in family_set:
                    continue
                # Gain = (comparable count of m with family minus 1 if m comparable with removed) - old_comp
                is_comp_with_removed = (comp_mask[m] >> removed) & 1
                gain = scores[m] - is_comp_with_removed - old_comp
                if gain > best_gain:
                    best_gain = gain
                    best_add = m
            
            if best_gain > 0 and best_add != -1:
                # Perform swap
                family[i] = best_add
                family_set.remove(removed)
                family_set.add(best_add)
                
                # Update scores
                for m in range(NUM_MASKS):
                    if m in family_set:
                        continue
                    if (comp_mask[m] >> removed) & 1:
                        scores[m] -= 1
                    if (comp_mask[m] >> best_add) & 1:
                        scores[m] += 1
                
                current_count += best_gain
                improved = True
    
    return family, current_count

def _local_search_2swap(family, comp_mask, deadline):
    """
    Local search: try all 2-swap moves (remove two, add two).
    This is more expensive but can escape deeper local optima.
    """
    family_set = set(family)
    current_count = _count_comparable(family, comp_mask)
    
    # Precompute scores
    scores = [0] * NUM_MASKS
    for m in range(NUM_MASKS):
        if m in family_set:
            continue
        for f in family:
            if (comp_mask[m] >> f) & 1:
                scores[m] += 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try pairs of positions to remove
        for i in range(len(family)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(family)):
                if time.time() > deadline:
                    break
                
                rem1 = family[i]
                rem2 = family[j]
                
                # Count comparable pairs lost
                # rem1's comparable count with other family members (excluding rem2)
                comp1 = 0
                comp2 = 0
                for k in range(len(family)):
                    if k == i or k == j:
                        continue
                    if (comp_mask[rem1] >> family[k]) & 1:
                        comp1 += 1
                    if (comp_mask[rem2] >> family[k]) & 1:
                        comp2 += 1
                # Check if rem1 and rem2 are comparable
                if (comp_mask[rem1] >> rem2) & 1:
                    comp1 += 1
                    comp2 += 1
                total_lost = comp1 + comp2  # This double-counts the pair (rem1,rem2) if comparable, but we want to count each lost pair once
                
                # Actually, let me recount properly:
                # Total comparable pairs involving rem1 or rem2 with other family members
                lost_pairs = 0
                for k in range(len(family)):
                    if k == i or k == j:
                        continue
                    if (comp_mask[rem1] >> family[k]) & 1:
                        lost_pairs += 1
                    if (comp_mask[rem2] >> family[k]) & 1:
                        lost_pairs += 1
                if (comp_mask[rem1] >> rem2) & 1:
                    lost_pairs += 1  # The pair (rem1, rem2) itself
                
                # Find best pair to add
                best_gain = 0
                best_add1 = -1
                best_add2 = -1
                
                # We need to find two masks a, b not in family such that
                # gain = (comparable pairs of a with family minus those with rem1,rem2) +
                #        (comparable pairs of b with family minus those with rem1,rem2) +
                #        (1 if a and b comparable else 0) - lost_pairs > 0
                
                # This is O(n^2) which is 4096^2 = 16M, too slow.
                # Instead, let's try a simpler heuristic:
                # For each candidate a, compute its "adjusted score" assuming rem1,rem2 are removed
                # Then pick the top 2 adjusted scores and check if they're comparable.
                
                # Adjusted score for mask m: scores[m] - (1 if m comparable with rem1) - (1 if m comparable with rem2)
                # But we also need to add 1 if the two chosen masks are comparable with each other.
                
                # Let's just try the top few candidates
                candidates = []
                for m in range(NUM_MASKS):
                    if m in family_set:
                        continue
                    adj_score = scores[m]
                    if (comp_mask[m] >> rem1) & 1:
                        adj_score -= 1
                    if (comp_mask[m] >> rem2) & 1:
                        adj_score -= 1
                    candidates.append((adj_score, m))
                
                candidates.sort(reverse=True)
                
                # Try top 20 pairs
                for idx1 in range(min(20, len(candidates))):
                    if time.time() > deadline:
                        break
                    for idx2 in range(idx1 + 1, min(20, len(candidates))):
                        a = candidates[idx1][1]
                        b = candidates[idx2][1]
                        
                        # Gain = adj_score[a] + adj_score[b] + (1 if a and b comparable else 0) - lost_pairs
                        # But adj_score already accounts for removing rem1,rem2
                        # Wait, adj_score[a] = scores[a] - (comparable with rem1) - (comparable with rem2)
                        # scores[a] counts comparable with ALL family members including rem1,rem2
                        # So adj_score[a] = comparable with (family - {rem1,rem2})
                        # Similarly for b
                        # Total gain = adj_score[a] + adj_score[b] + (1 if a,b comparable) - lost_pairs
                        
                        ab_comp = 1 if (comp_mask[a] >> b) & 1 else 0
                        gain = candidates[idx1][0] + candidates[idx2][0] + ab_comp - lost_pairs
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_add1 = a
                            best_add2 = b
                
                if best_gain > 0 and best_add1 != -1 and best_add2 != -1:
                    # Perform 2-swap
                    family[i] = best_add1
                    family[j] = best_add2
                    family_set.remove(rem1)
                    family_set.remove(rem2)
                    family_set.add(best_add1)
                    family_set.add(best_add2)
                    
                    # Update scores
                    for m in range(NUM_MASKS):
                        if m in family_set:
                            continue
                        if (comp_mask[m] >> rem1) & 1:
                            scores[m] -= 1
                        if (comp_mask[m] >> rem2) & 1:
                            scores[m] -= 1
                        if (comp_mask[m] >> best_add1) & 1:
                            scores[m] += 1
                        if (comp_mask[m] >> best_add2) & 1:
                            scores[m] += 1
                    
                    current_count += best_gain
                    improved = True
    
    return family, current_count

def _simulated_annealing(family, comp_mask, deadline, seed):
    """
    Simulated annealing to escape local optima.
    """
    rng = random.Random(seed)
    family = list(family)
    family_set = set(family)
    current_count = _count_comparable(family, comp_mask)
    best_count = current_count
    best_family = list(family)
    
    # Precompute scores
    scores = [0] * NUM_MASKS
    for m in range(NUM_MASKS):
        if m in family_set:
            continue
        for f in family:
            if (comp_mask[m] >> f) & 1:
                scores[m] += 1
    
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    
    while time.time() < deadline and T > T_min:
        # Random 1-swap move
        i = rng.randint(0, len(family) - 1)
        removed = family[i]
        
        # Pick a random candidate not in family
        candidates = [m for m in range(NUM_MASKS) if m not in family_set]
        if not candidates:
            break
        add = rng.choice(candidates)
        
        # Compute gain
        old_comp = 0
        for j in range(len(family)):
            if j == i:
                continue
            if (comp_mask[removed] >> family[j]) & 1:
                old_comp += 1
        
        is_comp_with_removed = (comp_mask[add] >> removed) & 1
        gain = scores[add] - is_comp_with_removed - old_comp
        
        # Accept or reject
        if gain >= 0 or rng.random() < pow(2.718281828, gain / T):
            # Accept
            family[i] = add
            family_set.remove(removed)
            family_set.add(add)
            
            # Update scores
            for m in range(NUM_MASKS):
                if m in family_set:
                    continue
                if (comp_mask[m] >> removed) & 1:
                    scores[m] -= 1
                if (comp_mask[m] >> add) & 1:
                    scores[m] += 1
            
            current_count += gain
            
            if current_count > best_count:
                best_count = current_count
                best_family = list(family)
        
        T *= cooling
    
    return best_family, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 90.0 - DEADLINE_MARGIN
    
    # Precompute comparability
    comp_mask, comp_count = _precompute_comparability()
    
    best_count = 0
    best_family = None
    
    # Phase 1: Multiple greedy restarts
    num_restarts = 200
    restart_deadline = deadline - 30.0  # Leave 30s for local search and annealing
    
    for i in range(num_restarts):
        if time.time() > restart_deadline:
            break
        
        family = _greedy_construct(i, comp_mask, comp_count)
        count = _count_comparable(family, comp_mask)
        
        if count > best_count:
            best_count = count
            best_family = list(family)
    
    # Phase 2: Local search 1-swap on best family
    if best_family is not None:
        ls1_deadline = deadline - 15.0
        best_family, best_count = _local_search_1swap(best_family, comp_mask, ls1_deadline)
    
    # Phase 3: Local search 2-swap on best family
    if best_family is not None:
        ls2_deadline = deadline - 8.0
        best_family, best_count = _local_search_2swap(best_family, comp_mask, ls2_deadline)
    
    # Phase 4: Simulated annealing
    if best_family is not None:
        sa_deadline = deadline - 2.0
        best_family, best_count = _simulated_annealing(best_family, comp_mask, sa_deadline, 42)
    
    # Final local search
    if best_family is not None:
        final_deadline = deadline - 0.5
        best_family, best_count = _local_search_1swap(best_family, comp_mask, final_deadline)
    
    # Fallback
    if best_family is None:
        vertices = list(range(6))
        all_sets = []
        for r in range(7):
            for combo in itertools.combinations(vertices, r):
                all_sets.append(list(combo))
        all_sets.sort(key=len)
        best_family = all_sets[:M]
    
    # Convert masks to lists of ints
    result = []
    for mask in best_family:
        result.append([b for b in range(N) if (mask >> b) & 1])
    
    return result
# EVOLVE-BLOCK-END
