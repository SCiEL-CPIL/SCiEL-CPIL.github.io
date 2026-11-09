import time
from itertools import combinations

L = 5354228880  # lcm(1..24)
N = 24
ALL = list(range(1, 25))
RECIP = [0] + [L // i for i in range(1, 25)]  # RECIP[i] = L//i


def has_subsum(recips_list, target, exclude_idx=None):
    """
    Check if target can be formed by a subset of recips_list (non-empty).
    recips_list: list of (L//i) values.
    exclude_idx: index to exclude, or None.
    Returns True if some non-empty subset sums to exactly target.
    """
    if target == 0:
        return False

    if exclude_idx is None:
        weights = [w for w in recips_list if 0 < w <= target]
    else:
        weights = [w for i, w in enumerate(recips_list) if i != exclude_idx and 0 < w <= target]

    if not weights:
        return False

    weights = sorted(set(weights))
    n = len(weights)
    if n == 0:
        return False

    if sum(weights) < target:
        return False

    if target in weights:
        return True

    # Meet-in-the-middle
    mid = n // 2
    left = weights[:mid]
    right = weights[mid:]

    def gen_sums(arr):
        sums = {0}
        for w in arr:
            new_sums = set()
            for s in sums:
                ns = s + w
                if ns <= target:
                    new_sums.add(ns)
            sums |= new_sums
        return sums

    left_sums = gen_sums(left)
    right_sums = gen_sums(right)

    for s in left_sums:
        if (target - s) in right_sums:
            return True
    return False


def is_valid(S):
    """
    S: list of ints. Return True if S is a valid reciprocal-subsum-free set.
    """
    if not S:
        return True
    recip_list = [L // x for x in S]
    for i in range(len(S)):
        target = L // S[i]
        if has_subsum(recip_list, target, exclude_idx=i):
            return False
    return True


def build_conflict_graph():
    """
    Build conflict graph: edge (x, y) exists if y is part of some subset of OTHER elements
    that sums to 1/x. Actually, the condition is: x conflicts with y if there exists a subset
    T of S\{x} such that sum(1/t for t in T) = 1/x AND y is in T... 
    
    Wait, let me re-read the problem. The condition is:
    "no reciprocal of a selected element may equal the sum of reciprocals of a subset of the OTHER selected elements"
    
    So for a set S, it's invalid if there exists x in S and a non-empty subset T of S\{x}
    such that 1/x = sum(1/t for t in T).
    
    For MIS formulation: we want a set S where NO such x exists.
    This is NOT simply a pairwise conflict graph. It's a higher-order constraint.
    
    However, we can still use branch-and-bound on subsets directly.
    """
    pass


def construct():
    start = time.time()
    deadline = start + 85.0

    def now():
        return time.time()

    # Precompute: for each element x, which subsets of {1..24}\{x} sum to 1/x?
    # This is expensive, so we'll use it in the validity check.

    best_set = None
    best_size = 0

    # Start with a known-valid seed
    seed = list(range(13, 25))
    if is_valid(seed):
        best_set = seed[:]
        best_size = len(seed)
    else:
        best_set = [24]
        best_size = 1

    import random

    # Local search with additions and swaps
    def local_search(S, max_iters=200, seed_offset=0):
        """
        S: valid list. Try additions and swaps to improve.
        Returns improved list.
        """
        S = set(S)
        S_list = list(S)
        rng = random.Random(seed_offset + len(S) * 1000)
        iters = 0
        while iters < max_iters and now() < deadline:
            iters += 1
            improved = False

            # Try additions first
            missing = [x for x in ALL if x not in S]
            if missing:
                shuffled_missing = missing[:]
                rng.shuffle(shuffled_missing)
                for x in shuffled_missing:
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
                missing = [x for x in ALL if x not in S]
                if missing:
                    a = rng.choice(list(S))
                    shuffled_missing = missing[:]
                    rng.shuffle(shuffled_missing)
                    for b in shuffled_missing:
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

    # Apply local search to initial best
    if best_set is not None:
        ls = local_search(best_set, max_iters=150, seed_offset=0)
        if len(ls) > best_size:
            best_size = len(ls)
            best_set = ls[:]

    # Random restarts
    for restart in range(60):
        if now() > deadline:
            break
        # Random subset
        target_size = random.randint(max(2, best_size), 24)
        rand_set = random.sample(ALL, min(target_size, 24))
        
        # Repair: remove violators
        S = set(rand_set)
        while now() < deadline and S:
            S_list = list(S)
            # Find first violation
            recip_list = [L // x for x in S_list]
            viol_idx = None
            for i, x in enumerate(S_list):
                target = L // x
                if has_subsum(recip_list, target, exclude_idx=i):
                    viol_idx = i
                    break
            if viol_idx is None:
                break
            S.remove(S_list[viol_idx])
        
        # Add back
        S_list = list(S)
        missing = [x for x in ALL if x not in S]
        for x in missing:
            if now() > deadline:
                break
            trial = S_list + [x]
            if is_valid(trial):
                S.add(x)
                S_list = list(S)
        
        # Local search
        ls = local_search(list(S), max_iters=80, seed_offset=restart)
        if len(ls) > best_size:
            best_size = len(ls)
            best_set = ls[:]

    # Final polish
    if best_set is not None and now() < deadline - 5.0:
        ls_final = local_search(best_set, max_iters=200, seed_offset=9999)
        if len(ls_final) > best_size:
            best_size = len(ls_final)
            best_set = ls_final[:]

    # Ensure validity
    if best_set is None or not is_valid(best_set):
        best_set = [24]
        best_size = 1
    else:
        best_set = list(best_set)
        best_size = len(best_set)

    return best_set


if __name__ == "__main__":
    print(construct())
