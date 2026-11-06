import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 87  # Safety margin before 90s limit

    # Generate all 4-element subsets of [16] as bitmasks (0-indexed bits for 0..15)
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    
    n_total = len(all_masks)  # 1820
    rng = random.Random(42)

    # Precompute all pairwise intersections
    # inter[i][j] = all_masks[i] & all_masks[j] for i <= j
    inter = [[0] * n_total for _ in range(n_total)]
    for i in range(n_total):
        mi = all_masks[i]
        inter[i][i] = mi
        for j in range(i + 1, n_total):
            inter[i][j] = mi & all_masks[j]
            inter[j][i] = inter[i][j]

    # For each mask, precompute list of masks that intersect it in a specific way
    # Actually, let's build a faster validity check using precomputed data
    
    # For each pair (i, j), we know their intersection. 
    # A sunflower of size 3 (a, b, c) requires:
    # inter[a][b] == inter[a][c] == inter[b][c]
    
    # Let's precompute for each mask m, and for each possible intersection value v,
    # the set of masks that have intersection v with m.
    # But that's a lot of memory. Let's use a different approach.
    
    # Alternative: For each mask m, precompute a dictionary:
    #   inter_dict[m][v] = list of indices i such that all_masks[i] & m == v
    # Then to check if m can be added to family F:
    #   For each v, if there are at least 2 indices in inter_dict[m][v] that are in F,
    #   AND their pairwise intersection equals v, then adding m creates a sunflower.
    
    # Let's precompute inter_dict for all masks.
    # This is O(n_total^2 * 16) which is about 1820^2 * 16 ≈ 53 million operations.
    # That might be too slow. Let's be smarter.
    
    # Actually, let's just precompute the intersection matrix and use it directly.
    # The key optimization is to avoid recomputing intersections.
    
    # Let's use a different strategy: maintain for each pair of masks in the family,
    # their intersection value. When adding a new mask, we only need to check pairs
    # involving the new mask.
    
    # Actually, the original is_valid_addition is O(|F|^2) per addition.
    # With |F| up to 250, that's 250^2 = 62,500 operations per addition.
    # For 1820 masks, that's about 113 million operations for one full pass.
    # That's too slow for multiple passes.
    
    # Let's optimize by using bit operations and early termination.
    
    def is_valid_addition_fast(fam, fam_set, new_idx):
        """Check if all_masks[new_idx] can be added to fam without creating a sunflower."""
        n = len(fam)
        if n < 2:
            return True
        
        new_mask = all_masks[new_idx]
        
        # Group existing masks by their intersection with new_mask
        groups = defaultdict(list)
        for i in fam:
            inter_val = inter[i][new_idx]
            groups[inter_val].append(i)
        
        # For each group, check if any pair has intersection equal to the group key
        for inter_val, indices in groups.items():
            if len(indices) < 2:
                continue
            for idx_a in range(len(indices)):
                a = indices[idx_a]
                for idx_b in range(idx_a + 1, len(indices)):
                    b = indices[idx_b]
                    if inter[a][b] == inter_val:
                        return False
        return True

    best_family = []
    best_size = 0

    # Phase 1: Multiple greedy trials with different orderings
    # Use more trials since the check is faster now
    n_trials = 20
    for trial in range(n_trials):
        if time.time() > deadline - 30:
            break
        order = list(range(n_total))
        if trial > 0:
            rng.shuffle(order)
        family = []
        family_set = set()
        for idx in order:
            if is_valid_addition_fast(family, family_set, idx):
                family.append(idx)
                family_set.add(idx)
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)
    
    family = list(best_family)
    family_set = set(family)

    # Phase 2: Try to add any remaining sets (fill-in)
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for m_idx in range(n_total):
            if m_idx in family_set:
                continue
            if is_valid_addition_fast(family, family_set, m_idx):
                family.append(m_idx)
                family_set.add(m_idx)
                improved = True
                if len(family) > best_size:
                    best_size = len(family)
                    best_family = list(family)
                if len(family) >= 250:
                    break
        if len(family) >= 250:
            break

    # Phase 3: Iterated Local Search with multiple swap attempts
    current_family = list(family)
    current_set = set(family)

    max_iterations = 50000
    for it in range(max_iterations):
        if time.time() > deadline - 3:
            break
        if not current_family:
            break

        # Pick a random set to remove
        remove_idx_pos = rng.randrange(len(current_family))
        removed_mask_idx = current_family[remove_idx_pos]
        current_family.pop(remove_idx_pos)
        current_set.discard(removed_mask_idx)

        # Try to add a better set
        added = False
        # Try a larger sample of candidates
        n_candidates = min(2000, n_total)
        candidates = rng.sample(range(n_total), n_candidates)
        for new_idx in candidates:
            if new_idx in current_set:
                continue
            if is_valid_addition_fast(current_family, current_set, new_idx):
                current_family.append(new_idx)
                current_set.add(new_idx)
                added = True
                break

        if added:
            if len(current_family) > best_size:
                best_size = len(current_family)
                best_family = list(current_family)
        else:
            # Restore original
            current_family.insert(remove_idx_pos, removed_mask_idx)
            current_set.add(removed_mask_idx)

        if best_size >= 250:
            break

    # Phase 4: Final polish - try to add any remaining sets
    if time.time() < deadline - 2:
        family = list(best_family)
        family_set = set(family)
        for m_idx in range(n_total):
            if m_idx in family_set:
                continue
            if is_valid_addition_fast(family, family_set, m_idx):
                family.append(m_idx)
                family_set.add(m_idx)
                if len(family) > best_size:
                    best_size = len(family)
                    best_family = list(family)
                if len(family) >= 250:
                    break

    result = []
    for idx in best_family:
        mask = all_masks[idx]
        elements = [i + 1 for i in range(16) if (mask >> i) & 1]
        result.append(sorted(elements))

    return result
