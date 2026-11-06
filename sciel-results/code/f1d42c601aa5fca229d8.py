import time
import random
import numpy as np
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55

    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)

    n_total = len(all_masks)
    rng = random.Random(42)

    # Precompute intersection matrix using numpy
    masks_array = np.array(all_masks, dtype=np.uint16)
    inter_matrix = (masks_array[:, None] & masks_array[None, :]).astype(np.uint16)

    def is_valid_addition(fam_indices, new_idx):
        n = len(fam_indices)
        if n < 2:
            return True
        # Get intersections of new with all family members
        inter = inter_matrix[new_idx, fam_indices]
        # Group by intersection value
        groups = {}
        for i in range(n):
            v = int(inter[i])
            if v in groups:
                groups[v].append(i)
            else:
                groups[v] = [i]
        # Check each group
        for v, indices in groups.items():
            if len(indices) < 2:
                continue
            for a_pos in range(len(indices)):
                a = indices[a_pos]
                fam_a = fam_indices[a]
                for b_pos in range(a_pos + 1, len(indices)):
                    b = indices[b_pos]
                    fam_b = fam_indices[b]
                    if inter_matrix[fam_a][fam_b] == v:
                        return False
        return True

    best_family = []
    best_size = 0

    # Multiple greedy trials with different orderings
    for trial in range(10):
        if time.time() > deadline - 25:
            break
        order = list(range(n_total))
        if trial > 0:
            rng.shuffle(order)
        family = []
        for idx in order:
            if is_valid_addition(family, idx):
                family.append(idx)
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)

    family = list(best_family)
    family_set = set(family)

    # Try to add any remaining sets
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for idx in range(n_total):
            if idx in family_set:
                continue
            if is_valid_addition(family, idx):
                family.append(idx)
                family_set.add(idx)
                improved = True
                if len(family) > best_size:
                    best_size = len(family)
                    best_family = list(family)
                if len(family) >= 250:
                    break
        if len(family) >= 250:
            break

    # Iterated Local Search
    current_family = list(family)
    current_set = set(family)

    for it in range(100000):
        if time.time() > deadline - 2:
            break
        if not current_family:
            break

        remove_idx = rng.randrange(len(current_family))
        removed = current_family[remove_idx]
        current_family.pop(remove_idx)
        current_set.discard(removed)

        added = False
        n_candidates = min(2000, n_total)
        candidates = rng.sample(range(n_total), n_candidates)
        for new_idx in candidates:
            if new_idx in current_set:
                continue
            if is_valid_addition(current_family, new_idx):
                current_family.append(new_idx)
                current_set.add(new_idx)
                added = True
                break

        if added:
            if len(current_family) > best_size:
                best_size = len(current_family)
                best_family = list(current_family)
        else:
            current_family.insert(remove_idx, removed)
            current_set.add(removed)

        if best_size >= 250:
            break

    result = []
    for idx in best_family:
        mask = all_masks[idx]
        elements = [i + 1 for i in range(16) if (mask >> i) & 1]
        result.append(sorted(elements))

    return result
