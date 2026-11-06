import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # Precompute all 4-subsets of [1,16] as bitmasks
    all_masks = []
    for comb in combinations(range(16), 4):
        mask = 0
        for i in comb:
            mask |= (1 << i)
        all_masks.append(mask)
    n_total = len(all_masks)  # 1820

    rng = random.Random(42)

    def is_valid_addition(fam, new_mask):
        """Check if new_mask can be added to fam without creating a sunflower."""
        n = len(fam)
        if n < 2:
            return True
        # Group family members by their intersection with new_mask
        inter_groups = defaultdict(list)
        for i in range(n):
            inter = fam[i] & new_mask
            inter_groups[inter].append(i)
        # For each group, check if any pair in the group has pairwise intersection equal to the group's intersection
        for inter_val, indices in inter_groups.items():
            if len(indices) < 2:
                continue
            # Check all pairs in this group
            for a_idx in range(len(indices)):
                for b_idx in range(a_idx + 1, len(indices)):
                    a = indices[a_idx]
                    b = indices[b_idx]
                    if (fam[a] & fam[b]) == inter_val:
                        return False
        return True

    def greedy_build(order):
        """Build a family greedily following the given order of mask indices."""
        family = []
        for idx in order:
            m = all_masks[idx]
            if is_valid_addition(family, m):
                family.append(m)
        return family

    best_family = []
    best_size = 0

    # Phase 1: Multiple greedy trials with different orderings
    for trial in range(20):
        if time.time() > deadline - 20.0:
            break
        order = list(range(n_total))
        if trial == 0:
            pass  # Natural order
        else:
            rng.shuffle(order)
        family = greedy_build(order)
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)

    # Phase 2: Try to add any remaining sets to the best family
    family = list(best_family)
    family_set = set(family)
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        for m in all_masks:
            if m in family_set:
                continue
            if is_valid_addition(family, m):
                family.append(m)
                family_set.add(m)
                improved = True
                if len(family) > best_size:
                    best_size = len(family)
                    best_family = list(family)
                if len(family) >= 250:
                    break
        if len(family) >= 250:
            break

    # Phase 3: Iterated Local Search with swaps
    current_family = list(family)
    current_set = set(family)
    current_size = len(current_family)

    # Precompute a pool of candidate masks not in current family for faster swap attempts
    def get_candidate_pool(fam_set, pool_size=500):
        available = [m for m in all_masks if m not in fam_set]
        if len(available) <= pool_size:
            return available
        return rng.sample(available, pool_size)

    max_ils_iters = 20000
    for it in range(max_ils_iters):
        if time.time() > deadline - 2.0:
            break
        if not current_family:
            break

        # Remove a random set from current family
        remove_idx = rng.randrange(len(current_family))
        removed_mask = current_family.pop(remove_idx)
        current_set.discard(removed_mask)
        current_size -= 1

        # Try to add a better set
        added = False
        # Try a random sample of candidates
        n_try = min(200, n_total - current_size)
        candidates = rng.sample(all_masks, n_try)
        for new_mask in candidates:
            if new_mask in current_set:
                continue
            if is_valid_addition(current_family, new_mask):
                current_family.append(new_mask)
                current_set.add(new_mask)
                current_size += 1
                added = True
                break

        if added:
            if current_size > best_size:
                best_size = current_size
                best_family = list(current_family)
        else:
            # Restore original
            current_family.insert(remove_idx, removed_mask)
            current_set.add(removed_mask)
            current_size += 1

        if best_size >= 250:
            break

    # Convert bitmasks back to sorted lists of 1-indexed elements
    result = []
    for mask in best_family:
        elements = [i + 1 for i in range(16) if (mask >> i) & 1]
        result.append(sorted(elements))

    return result
