import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit

    # Precompute all 4-subsets of [1,16] as bitmasks
    all_masks = []
    for comb in combinations(range(1, 17), 4):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        all_masks.append(mask)

    N = len(all_masks)  # 1820
    mask_to_idx = {m: i for i, m in enumerate(all_masks)}

    def is_sunflower(a, b, c):
        """Check if three 4-element sets (as bitmasks) form a sunflower."""
        ab = a & b
        ac = a & c
        bc = b & c
        return ab == ac == bc

    def can_add(candidate, family):
        """Check if candidate can be added to family without forming a sunflower."""
        for i in range(len(family)):
            a = family[i]
            for j in range(i + 1, len(family)):
                b = family[j]
                if is_sunflower(a, b, candidate):
                    return False
        return True

    def can_add_with_blockers(candidate, family):
        """Check if candidate can be added. Returns (valid, blocker_pair or None)."""
        for i in range(len(family)):
            a = family[i]
            for j in range(i + 1, len(family)):
                b = family[j]
                if is_sunflower(a, b, candidate):
                    return False, (i, j)
        return True, None

    def greedy_build(order, deadline):
        """Build family greedily in the given order."""
        family = []
        family_set = set()
        for idx in order:
            if time.time() > deadline:
                break
            mask = all_masks[idx]
            if can_add(mask, family):
                family.append(mask)
                family_set.add(mask)
        return family

    def local_search(family, deadline, max_steps=2000):
        """Local search: try swaps and additions to improve family size."""
        family = family[:]
        family_set = set(family)
        steps_no_improve = 0
        best_size = len(family)

        for step in range(max_steps):
            if time.time() > deadline:
                break
            if steps_no_improve > 300:
                break

            # Try to add a random candidate
            idx = random.randint(0, N - 1)
            c = all_masks[idx]

            if c in family_set:
                steps_no_improve += 1
                continue

            valid, blockers = can_add_with_blockers(c, family)

            if valid:
                family.append(c)
                family_set.add(c)
                if len(family) > best_size:
                    best_size = len(family)
                    steps_no_improve = 0
                else:
                    steps_no_improve += 1
            else:
                # Try swap: remove one blocker, add c
                if blockers:
                    i, j = blockers
                    # Randomly pick one to remove
                    victim_idx = random.choice([i, j])
                    victim = family[victim_idx]

                    # Remove victim
                    family.pop(victim_idx)
                    family_set.remove(victim)

                    # Try to add c
                    if can_add(c, family):
                        family.append(c)
                        family_set.add(c)
                        # Size unchanged, but structure improved
                        steps_no_improve += 1
                    else:
                        # Restore
                        family.append(victim)
                        family_set.add(victim)
                        steps_no_improve += 1
                else:
                    steps_no_improve += 1

        return family

    best_family = []
    best_size = 0

    # Phase 1: Multiple greedy builds with different orderings
    # Use lexicographic order first
    lex_order = list(range(N))
    family = greedy_build(lex_order, deadline - 10)
    if len(family) > best_size:
        best_size = len(family)
        best_family = family[:]

    # Random restarts
    rng = random.Random(12345)
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break

        order = list(range(N))
        rng.shuffle(order)
        family = greedy_build(order, deadline - 15)

        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]

    # Phase 2: Local search on best family
    if time.time() < deadline - 5:
        improved = local_search(best_family, deadline - 5, max_steps=3000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_family = improved

    # Phase 3: Try to add any remaining sets
    if time.time() < deadline - 2:
        family_set = set(best_family)
        added_any = True
        while added_any and time.time() < deadline - 2:
            added_any = False
            for idx in range(N):
                if time.time() > deadline - 2:
                    break
                mask = all_masks[idx]
                if mask not in family_set and can_add(mask, best_family):
                    best_family.append(mask)
                    family_set.add(mask)
                    added_any = True
                    if len(best_family) > best_size:
                        best_size = len(best_family)

    # Convert bitmasks back to sorted lists
    result = []
    for m in best_family:
        s = [i for i in range(1, 17) if (m >> i) & 1]
        result.append(sorted(s))

    return result
