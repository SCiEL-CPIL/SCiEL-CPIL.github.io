import time
import random
import math

_DEADLINE = 85.0  # seconds wall-clock budget (safety margin below 90s)
_N = 256


def _is_good_pair(a, b):
    return (a * b) % (a + b) != 0


def _validate(S):
    """Return True if S is a valid subset of [1,256] (pairwise distinct-sum non-divisibility)."""
    s = set(S)
    if len(s) != len(S):
        return False
    if any(x < 1 or x > _N for x in s):
        return False
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a, b = S[i], S[j]
            if (a * b) % (a + b) == 0:
                return False
    return True


def _build_conflict_matrix():
    """Precompute which pairs (i,j) conflict, i.e., (i*j)%(i+j)==0."""
    # conflicts[i] is a set of j > i that conflict with i
    conflicts = [set() for _ in range(_N + 1)]
    for a in range(1, _N + 1):
        for b in range(a + 1, _N + 1):
            if (a * b) % (a + b) == 0:
                conflicts[a].add(b)
                conflicts[b].add(a)
    return conflicts


def _can_add(x, S_set, conflicts):
    """Check if x can be added to S_set without conflicts."""
    for y in S_set:
        if y in conflicts[x]:
            return False
    return True


def _greedy_build(order, conflicts):
    """Greedy add elements in given order if they don't conflict with current set."""
    S = []
    S_set = set()
    for x in order:
        if _can_add(x, S_set, conflicts):
            S.append(x)
            S_set.add(x)
    return S, S_set


def _local_search_add_only(S, S_set, remaining, conflicts, time_limit):
    """Try to add elements that fit. Return (S, S_set, remaining)."""
    S = list(S)
    S_set = set(S)
    remaining = set(remaining)
    start = time.time()
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        for x in list(remaining):
            if time.time() >= time_limit:
                break
            if _can_add(x, S_set, conflicts):
                S.append(x)
                S_set.add(x)
                remaining.discard(x)
                improved = True
    return S, S_set, remaining


def _two_swap_search(S, S_set, remaining, conflicts, time_limit):
    """
    Try 2-for-1 swaps: remove two elements from S, add up to three from remaining
    such that the net size increases. Also try 1-for-2 (remove 1, add 2).
    This is more powerful than single addition.
    """
    S = list(S)
    S_set = set(S)
    remaining = set(remaining)
    start = time.time()
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        # Try 1-for-2: remove one element r, try to add two elements x,y from remaining
        for r in list(S):
            if time.time() >= time_limit:
                break
            S_set.discard(r)
            # Find candidates that can be added
            candidates = []
            for x in remaining:
                if _can_add(x, S_set, conflicts):
                    candidates.append(x)
            # Try to add two from candidates
            added = 0
            for x in candidates:
                if time.time() >= time_limit:
                    break
                if added >= 2:
                    break
                if _can_add(x, S_set, conflicts):
                    S.append(x)
                    S_set.add(x)
                    remaining.discard(x)
                    added += 1
            if added > 1:
                # We removed r and added 'added' elements; net gain = added - 1
                # But we need to make sure r is not re-added (it's not in remaining)
                S.remove(r)
                improved = True
                break
            else:
                # Re-add r if we didn't improve
                S_set.add(r)
        if time.time() >= time_limit:
            break
        # Try 2-for-3: remove two elements, add three
        S_list = list(S)
        for i in range(len(S_list)):
            if time.time() >= time_limit:
                break
            for j in range(i + 1, len(S_list)):
                if time.time() >= time_limit:
                    break
                r1, r2 = S_list[i], S_list[j]
                S_set.discard(r1)
                S_set.discard(r2)
                candidates = []
                for x in remaining:
                    if _can_add(x, S_set, conflicts):
                        candidates.append(x)
                added = 0
                for x in candidates:
                    if time.time() >= time_limit:
                        break
                    if added >= 3:
                        break
                    if _can_add(x, S_set, conflicts):
                        S.append(x)
                        S_set.add(x)
                        remaining.discard(x)
                        added += 1
                if added > 2:
                    S.remove(r1)
                    S.remove(r2)
                    improved = True
                    break
                else:
                    S_set.add(r1)
                    S_set.add(r2)
            if improved:
                break
    return S, S_set, remaining


def _simulated_annealing(S, S_set, remaining, conflicts, time_limit):
    """
    Simulated annealing: randomly remove an element, then try to greedily add
    elements to maximize size. Accept if size improves or with some probability.
    """
    S = list(S)
    S_set = set(S)
    remaining = set(remaining)
    start = time.time()
    best_S = list(S)
    best_size = len(S)
    rng = random.Random(12345)
    temperature = 1.0
    cooling = 0.995

    while time.time() < time_limit:
        # Randomly remove one element
        idx = rng.randint(0, len(S) - 1)
        removed = S.pop(idx)
        S_set.discard(removed)
        remaining.add(removed)

        # Greedily try to add elements
        new_S, new_S_set, new_remaining = _local_search_add_only(
            S, S_set, remaining, conflicts, time.time() + 0.05
        )

        new_size = len(new_S)
        if new_size > best_size:
            best_S = list(new_S)
            best_size = new_size
            S = new_S
            S_set = new_S_set
            remaining = new_remaining
        else:
            # Revert with some probability
            if rng.random() < temperature:
                # Keep the change
                S = new_S
                S_set = new_S_set
                remaining = new_remaining
            else:
                # Revert
                S = list(best_S)
                S_set = set(S)
                remaining = set(range(1, _N + 1)) - S_set

        temperature *= cooling
        if temperature < 0.01:
            break

    return best_S, set(best_S), set(range(1, _N + 1)) - set(best_S)


def construct():
    t0 = time.time()
    rng = random.Random(42)

    conflicts = _build_conflict_matrix()

    def _deadline_ok():
        return time.time() - t0 < _DEADLINE

    # Start with all odds as a strong baseline
    best = list(range(1, _N + 1, 2))
    best_size = len(best)

    # 1) Greedy least-conflicting
    if _deadline_ok():
        S, S_set = [], set()
        remaining = set(range(1, _N + 1))
        while remaining:
            best_x = None
            best_c = None
            for x in remaining:
                c = len(conflicts[x] & S_set)
                if best_c is None or c < best_c or (c == best_c and x < best_x):
                    best_x = x
                    best_c = c
                if c == 0:
                    break
            if best_c == 0:
                S.append(best_x)
                S_set.add(best_x)
                remaining.discard(best_x)
            else:
                break
        if _validate(S) and len(S) > best_size:
            best = S
            best_size = len(S)

    # 2) Multiple random greedy orderings
    if _deadline_ok():
        for _ in range(500):
            if time.time() - t0 > _DEADLINE - 5:
                break
            order = list(range(1, _N + 1))
            rng.shuffle(order)
            S, S_set = _greedy_build(order, conflicts)
            if _validate(S) and len(S) > best_size:
                best = S
                best_size = len(S)

    # 3) Local search: add-only on best
    if _deadline_ok():
        S_set = set(best)
        remaining = set(range(1, _N + 1)) - S_set
        S_new, _, _ = _local_search_add_only(best, S_set, remaining, conflicts, _DEADLINE - 2)
        if _validate(S_new) and len(S_new) > best_size:
            best = S_new
            best_size = len(S_new)

    # 4) Two-swap search
    if _deadline_ok():
        S_set = set(best)
        remaining = set(range(1, _N + 1)) - S_set
        S_new, _, _ = _two_swap_search(best, S_set, remaining, conflicts, _DEADLINE - 2)
        if _validate(S_new) and len(S_new) > best_size:
            best = S_new
            best_size = len(S_new)

    # 5) Simulated annealing
    if _deadline_ok():
        S_set = set(best)
        remaining = set(range(1, _N + 1)) - S_set
        S_new, _, _ = _simulated_annealing(best, S_set, remaining, conflicts, _DEADLINE - 1)
        if _validate(S_new) and len(S_new) > best_size:
            best = S_new
            best_size = len(S_new)

    # Final safety: ensure validity
    if not _validate(best):
        best = list(range(1, _N + 1, 2))

    return best
