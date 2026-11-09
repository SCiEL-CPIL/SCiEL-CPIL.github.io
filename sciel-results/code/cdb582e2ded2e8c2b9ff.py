import time
import random

L = 5354228880  # lcm(1..24)
N = 24
ALL = list(range(1, 25))

# Precompute reciprocals as L//i
RECIP = [0] + [L // i for i in range(1, 25)]  # RECIP[i] = L//i


def has_subsum(recips_list, target, exclude_idx=None):
    """
    Check if target can be formed by a subset of recips_list.
    recips_list: list of (L//i) values (positive ints).
    exclude_idx: index in recips_list to exclude (or None).
    Returns True if some non-empty subset sums to exactly target.
    """
    # Collect eligible weights
    if exclude_idx is None:
        weights = [w for w in recips_list if 0 < w <= target]
    else:
        weights = [w for i, w in enumerate(recips_list) if i != exclude_idx and 0 < w <= target]

    if not weights:
        return False
    if target == 0:
        return False

    # Deduplicate and sort
    weights = sorted(set(weights))
    n = len(weights)
    if n == 0:
        return False

    # If sum of all < target, impossible
    if sum(weights) < target:
        return False

    # Quick check: if target is in weights, done
    if target in weights:
        return True

    # Meet-in-the-middle
    mid = n // 2
    left = weights[:mid]
    right = weights[mid:]

    def gen_sums(arr, max_val):
        sums = {0}
        for w in arr:
            new = set()
            for s in sums:
                ns = s + w
                if ns <= max_val:
                    new.add(ns)
            sums |= new
        return sums

    left_sums = gen_sums(left, target)
    right_sums = gen_sums(right, target)

    # Check
    for s in left_sums:
        if (target - s) in right_sums:
            return True
    return False


def find_violation(S):
    """
    S: list of ints.
    Returns (violator_idx, contributing_subset) if violation, else (None, None).
    """
    if not S:
        return None, None
    m = len(S)
    recip_list = [L // x for x in S]
    for i, x in enumerate(S):
        target = L // x
        if has_subsum(recip_list, target, exclude_idx=i):
            return i, None
    return None, None


def is_valid(S):
    """
    S: list of ints. Return True if S is a valid reciprocal-subsum-free set.
    """
    if not S:
        return True
    recip_list = [L // x for x in S]
    m = len(S)
    for i in range(m):
        target = L // S[i]
        # Quick check: if target > sum of all other reciprocals, skip
        others_sum = sum(recip_list) - recip_list[i]
        if target > others_sum:
            continue
        if has_subsum(recip_list, target, exclude_idx=i):
            return False
    return True


def construct():
    start = time.time()
    deadline = start + 85.0

    def now():
        return time.time()

    best_set = None
    best_size = 0

    # Start with a known-valid seed: the upper half [13..24] is 12 elements
    seed = list(range(13, 25))
    if is_valid(seed):
        best_set = seed[:]
        best_size = len(seed)
    else:
        best_set = [24]
        best_size = 1

    rng = random.Random(42)

    def greedy_add(order, start_set=None):
        S = set(start_set) if start_set is not None else set()
        S_list = list(S)
        for x in order:
            if now() > deadline:
                break
            if x in S:
                continue
            trial = S_list + [x]
            if is_valid(trial):
                S.add(x)
                S_list = list(S)
        return S_list

    # Try greedy from empty with various orderings
    orderings = []
    orderings.append(ALL[:])
    orderings.append(ALL[::-1])
    orderings.append(sorted(ALL, key=lambda x: L // x))
    orderings.append(sorted(ALL, key=lambda x: L // x, reverse=True))
    orderings.append(sorted(ALL, key=lambda x: x % 7))

    for _ in range(15):
        o = ALL[:]
        rng.shuffle(o)
        orderings.append(o)

    for o in orderings:
        if now() > deadline:
            break
        res = greedy_add(o, start_set=None)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res[:]

    def repair(S):
        S = set(S)
        while now() < deadline and S:
            S_list = list(S)
            viol_idx, _ = find_violation(S_list)
            if viol_idx is None:
                return S_list
            S.remove(S_list[viol_idx])
        return list(S)

    # Repair full set
    S_full = repair(ALL[:])
    if len(S_full) > best_size:
        best_size = len(S_full)
        best_set = S_full[:]

    def add_back(S):
        S = set(S)
        missing = [x for x in ALL if x not in S]
        for x in missing:
            if now() > deadline:
                break
            trial = list(S) + [x]
            if is_valid(trial):
                S.add(x)
        return list(S)

    if best_set is not None:
        improved = add_back(best_set)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved[:]

    def local_search(S, max_iters=200):
        S = set(S)
        S_list = list(S)
        iters = 0
        improved = False
        while iters < max_iters and now() < deadline:
            iters += 1
            improved = False

            # Try additions first
            missing = [x for x in ALL if x not in S]
            if missing:
                rng2 = random.Random(iters + len(S) * 1000)
                rng2.shuffle(missing)
                for x in missing:
                    if now() > deadline:
                        break
                    trial = list(S) + [x]
                    if is_valid(trial):
                        S.add(x)
                        S_list = list(S)
                        improved = True
                        break
            if improved:
                continue

            # Try swaps: remove a, add b
            if len(S) > 0:
                a = rng.choice(S_list)
                missing = [x for x in ALL if x not in S]
                if missing:
                    rng.shuffle(missing)
                    for b in missing:
                        if now() > deadline:
                            break
                        new_set = (S - {a}) | {b}
                        if is_valid(list(new_set)):
                            S = new_set
                            S_list = list(S)
                            improved = True
                            break

            if not improved:
                break

        return S_list

    # Apply local search to best
    if best_set is not None:
        ls = local_search(best_set, max_iters=100)
        if len(ls) > best_size:
            best_size = len(ls)
            best_set = ls[:]

    # Random restarts: pick random subsets, repair, add back, local search
    for _ in range(30):
        if now() > deadline:
            break
        target_size = random.randint(max(2, best_size), 24)
        rand_set = random.sample(ALL, min(target_size, 24))
        repaired = repair(rand_set)
        added = add_back(repaired)
        if len(added) > best_size:
            best_size = len(added)
            best_set = added[:]
            ls = local_search(added, max_iters=50)
            if len(ls) > best_size:
                best_size = len(ls)
                best_set = ls[:]

    # Final: ensure validity
    if best_set is None or not is_valid(best_set):
        best_set = [24]
        best_size = 1
    else:
        best_set = list(best_set)
        best_size = len(best_set)

    return best_set


if __name__ == "__main__":
    print(construct())
