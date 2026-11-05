import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill

    # All 4-subsets of [1,16] as bitmasks (16 bits)
    all_bitmasks = []
    all_sets_list = []
    for c in combinations(range(1, 17), 4):
        mask = 0
        for x in c:
            mask |= (1 << (x - 1))
        all_bitmasks.append(mask)
        all_sets_list.append(c)
    n_sets = len(all_bitmasks)  # 1820

    # Precompute popcount for 16-bit masks (intersection sizes)
    # Actually we just need equality of intersection masks, not sizes.

    def is_valid_addition(cand_mask, family_masks):
        """Check if cand_mask can be added to family_masks without creating a sunflower.
        A sunflower: three distinct sets A,B,C with A∩B = A∩C = B∩C (as sets).
        For the new set C=cand, we need: no pair (A,B) in family with A∩B = A∩C = B∩C.
        Strategy: group family members by their intersection with cand.
        For each group with common intersection S, check if any pair in the group
        has mutual intersection == S.
        """
        groups = {}
        for fm in family_masks:
            inter = cand_mask & fm
            if inter in groups:
                groups[inter].append(fm)
            else:
                groups[inter] = [fm]

        for S, members in groups.items():
            n = len(members)
            if n < 2:
                continue
            # Check all pairs in this group
            for i in range(n):
                mi = members[i]
                for j in range(i + 1, n):
                    if (mi & members[j]) == S:
                        return False
        return True

    def greedy_conflict_aware(family_masks, family_indices, rng, max_time=None):
        """Greedy: repeatedly pick the candidate that is valid and has the highest
        'blocking score' (number of pairs in family it would complete a sunflower with).
        This is a surrogate: sets that block more sunflowers are 'more compatible'.
        """
        in_family = set(family_indices)
        all_candidates = [i for i in range(n_sets) if i not in in_family]

        while all_candidates:
            if max_time and time.time() > max_time:
                break

            # Compute blocking score for each candidate:
            # For candidate c, blocking score = number of pairs (a,b) in family
            # such that a∩b = a∩c = b∩c (i.e., c would COMPLETE a sunflower).
            # We want to AVOID completing sunflowers, so we want LOW blocking score.
            # Actually, we want to ADD c if it's valid (blocking score = 0).
            # Among valid candidates, prefer ones with... hmm.
            #
            # Alternative: prefer candidates that have MANY distinct intersection
            # patterns with the family (more "diverse"), as they're less likely to
            # create sunflowers later.
            #
            # Simpler: just pick a random valid candidate. But let's try:
            # prefer candidates whose intersection with the family is "spread out"
            # (many distinct intersection values), as this means they interact
            # differently with many family members.

            best_idx = None
            best_score = -1
            # Sample a subset of candidates for speed
            sample_size = min(len(all_candidates), 200)
            sample = rng.sample(all_candidates, sample_size)

            for idx in sample:
                cm = all_bitmasks[idx]
                if not is_valid_addition(cm, family_masks):
                    continue
                # Score: number of distinct intersection values with family
                # Higher diversity = better (less likely to form sunflowers)
                inter_vals = set()
                for fm in family_masks:
                    inter_vals.add(cm & fm)
                score = len(inter_vals)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                # No valid candidate in sample; check all remaining
                for idx in all_candidates:
                    cm = all_bitmasks[idx]
                    if is_valid_addition(cm, family_masks):
                        best_idx = idx
                        break

            if best_idx is None:
                break

            # Add best_idx
            family_masks.append(all_bitmasks[best_idx])
            family_indices.append(best_idx)
            in_family.add(best_idx)
            all_candidates.remove(best_idx)

        return len(family_indices)

    def local_search_add(family_masks, family_indices, rng, max_iters=500):
        """Try to add more sets via random sampling."""
        in_family = set(family_indices)
        added = 0
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            idx = rng.randint(0, n_sets - 1)
            if idx in in_family:
                continue
            if is_valid_addition(all_bitmasks[idx], family_masks):
                family_masks.append(all_bitmasks[idx])
                family_indices.append(idx)
                in_family.add(idx)
                added += 1
        return added

    def local_search_swap(family_masks, family_indices, rng, max_iters=200):
        """Try swap: remove one set, try to add two others. Keep if net gain > 0."""
        improved = 0
        in_family = set(family_indices)
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            rem_pos = rng.randint(0, len(family_masks) - 1)
            removed_mask = family_masks[rem_pos]
            removed_idx = family_indices[rem_pos]

            # Temporarily remove
            family_masks[rem_pos] = None
            family_indices[rem_pos] = -1
            in_family.discard(removed_idx)

            active_masks = [m for m in family_masks if m is not None]

            # Try to add up to 2 new sets
            added_new = []
            for _ in range(3):
                cand_idx = rng.randint(0, n_sets - 1)
                if cand_idx == removed_idx or cand_idx in in_family:
                    continue
                if is_valid_addition(all_bitmasks[cand_idx], active_masks):
                    family_masks.append(all_bitmasks[cand_idx])
                    family_indices.append(cand_idx)
                    in_family.add(cand_idx)
                    active_masks.append(all_bitmasks[cand_idx])
                    added_new.append(cand_idx)

            new_size = len([m for m in family_masks if m is not None])
            old_size = len(family_masks) - 1  # size before removal

            if new_size > old_size:
                # Keep the improvement
                improved += 1
                # Clean up: remove the None entry
                family_masks.pop(rem_pos)
                family_indices.pop(rem_pos)
            else:
                # Revert
                for _ in added_new:
                    family_masks.pop()
                    family_indices.pop()
                in_family.discard(added_new[0]) if added_new else None
                if len(added_new) > 1:
                    in_family.discard(added_new[1])
                family_masks[rem_pos] = removed_mask
                family_indices[rem_pos] = removed_idx
                in_family.add(removed_idx)

        return improved

    rng = random.Random(42)
    best_family_indices = []
    best_size = 0

    # Phase 1: Multiple greedy constructions
    num_trials = 40
    for trial in range(num_trials):
        if time.time() > deadline - 15:
            break

        family_masks = []
        family_indices = []

        if trial == 0:
            # Natural order
            for idx in range(n_sets):
                if time.time() > deadline - 15:
                    break
                if is_valid_addition(all_bitmasks[idx], family_masks):
                    family_masks.append(all_bitmasks[idx])
                    family_indices.append(idx)
        elif trial == 1:
            # Reverse order
            for idx in range(n_sets - 1, -1, -1):
                if time.time() > deadline - 15:
                    break
                if is_valid_addition(all_bitmasks[idx], family_masks):
                    family_masks.append(all_bitmasks[idx])
                    family_indices.append(idx)
        else:
            # Conflict-aware greedy
            greedy_conflict_aware(family_masks, family_indices, rng, max_time=deadline - 15)

        # Local search to add more
        local_search_add(family_masks, family_indices, rng, max_iters=200)

        if len(family_indices) > best_size:
            best_size = len(family_indices)
            best_family_indices = list(family_indices)

        if best_size >= 250:
            break

    # Phase 2: Local search swaps on the best family
    if time.time() < deadline - 8 and best_family_indices:
        family_masks = [all_bitmasks[i] for i in best_family_indices]
        family_indices = list(best_family_indices)

        for round_num in range(15):
            if time.time() > deadline - 8:
                break
            local_search_swap(family_masks, family_indices, rng, max_iters=100)
            local_search_add(family_masks, family_indices, rng, max_iters=200)

            if len(family_indices) > best_size:
                best_size = len(family_indices)
                best_family_indices = list(family_indices)

            if best_size >= 250:
                break

    # Phase 3: More aggressive random add if time permits
    if time.time() < deadline - 3 and best_family_indices:
        family_masks = [all_bitmasks[i] for i in best_family_indices]
        family_indices = list(best_family_indices)
        in_family = set(family_indices)

        for _ in range(500):
            if time.time() > deadline - 3:
                break
            idx = rng.randint(0, n_sets - 1)
            if idx in in_family:
                continue
            if is_valid_addition(all_bitmasks[idx], family_masks):
                family_masks.append(all_bitmasks[idx])
                family_indices.append(idx)
                in_family.add(idx)
                if len(family_indices) > best_size:
                    best_size = len(family_indices)
                    best_family_indices = list(family_indices)
                if best_size >= 250:
                    break

    # Return as sorted lists
    return [sorted(all_sets_list[i]) for i in best_family_indices]
