import time
from itertools import combinations
from functools import lru_cache

L = 5354228880  # lcm(1..24)
N = 24
ALL = list(range(1, 25))

# Precompute reciprocals as L//i
RECIP = [0] + [L // i for i in range(1, 25)]  # RECIP[i] = L//i


def has_subsum(weights, target):
    """
    Check if target can be formed by a non-empty subset of weights (positive ints).
    Uses DP with a set for the subset-sum problem.
    """
    if not weights or target <= 0:
        return False
    # Quick check
    if target in weights:
        return True
    if sum(weights) < target:
        return False
    
    # DP: reachable sums
    reachable = {0}
    for w in weights:
        if w > target:
            continue
        new_sums = set()
        for s in reachable:
            ns = s + w
            if ns <= target:
                new_sums.add(ns)
        reachable |= new_sums
        if target in reachable:
            return True
    return False


def is_valid(S):
    """
    S: list of ints. Return True if S is a valid reciprocal-subsum-free set.
    """
    if not S:
        return True
    recip_list = [L // x for x in S]
    m = len(S)
    for i in range(m):
        target = recip_list[i]
        others = [recip_list[j] for j in range(m) if j != i]
        if target > sum(others):
            continue
        if has_subsum(others, target):
            return False
    return True


def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin

    def now():
        return time.time()

    best_set = None
    best_size = 0

    # Try seed: upper half
    seed = list(range(13, 25))
    if is_valid(seed):
        best_set = seed[:]
        best_size = len(seed)
    else:
        best_set = [24]
        best_size = 1

    import random

    # Greedy addition with multiple orderings
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

    orderings = []
    orderings.append(ALL[:])
    orderings.append(ALL[::-1])
    orderings.append(sorted(ALL, key=lambda x: L // x))
    orderings.append(sorted(ALL, key=lambda x: L // x, reverse=True))
    orderings.append(sorted(ALL, key=lambda x: x % 7))
    orderings.append(sorted(ALL, key=lambda x: x % 13))

    rng = random.Random(42)
    for _ in range(30):
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

    # Repair: start from full set, remove violators
    def find_violation(S):
        if not S:
            return None
        recip_list = [L // x for x in S]
        m = len(S)
        for i in range(m):
            target = recip_list[i]
            others = [recip_list[j] for j in range(m) if j != i]
            if target <= sum(others) and has_subsum(others, target):
                return i
        return None

    def repair(S):
        S = set(S)
        while now() < deadline and S:
            S_list = list(S)
            viol_idx = find_violation(S_list)
            if viol_idx is None:
                return S_list
            S.remove(S_list[viol_idx])
        return list(S)

    S_full = repair(ALL[:])
    if len(S_full) > best_size:
        best_size = len(S_full)
        best_set = S_full[:]

    # Add back missing elements
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

    # Local search with swaps
    def local_search(S, max_iters=200):
        S = set(S)
        S_list = list(S)
        iters = 0
        improved = False
        while iters < max_iters and now() < deadline:
            iters += 1
            improved = False

            # Try additions
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

            # Try swaps
            if len(S) > 0:
                a = rng.choice(list(S))
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

    if best_set is not None:
        ls = local_search(best_set, max_iters=200)
        if len(ls) > best_size:
            best_size = len(ls)
            best_set = ls[:]

    # Random restarts
    for _ in range(50):
        if now() > deadline:
            break
        target_size = random.randint(max(2, best_size), 24)
        rand_set = random.sample(ALL, min(target_size, 24))
        repaired = repair(rand_set)
        added = add_back(repaired)
        if len(added) > best_size:
            best_size = len(added)
            best_set = added[:]
            ls = local_search(added, max_iters=100)
            if len(ls) > best_size:
                best_size = len(ls)
                best_set = ls[:]

    # Final validation
    if best_set is None or not is_valid(best_set):
        best_set = [24]
        best_size = 1
    else:
        best_set = list(best_set)
        best_size = len(best_set)

    return best_set


if __name__ == "__main__":
    print(construct())
