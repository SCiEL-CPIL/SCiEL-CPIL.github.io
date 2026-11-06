"""Greedy seed (no early stop) + systematic 1-swap local search with restarts.

Strategy:
  * Build a maximal 4-AP-free set by scanning 1..256 and adding x whenever it
    does not create a nonconstant 4-AP. This gives a good, reproducible seed.
  * Use incremental 3-AP delta computations so each candidate move is O(N)
    rather than O(N^2).
  * Run an exhaustive 1-swap local search (remove one, add one) until no
    improving swap exists, plus single additions.
  * On local optimum, perturb by removing a few random elements and restart.
  * Track the best set seen across all restarts and return it.
"""
import time
import random

N = 256
DEADLINE = 55.0  # safety margin under 90 s kill


def _has_4ap_with(selected, x):
    """Return True if selected ∪ {x} contains a nonconstant 4-AP involving x."""
    s = selected
    # x as 1st term: x, x+d, x+2d, x+3d
    for d in range(1, (N - x) // 3 + 1):
        if (x + d) in s and (x + 2 * d) in s and (x + 3 * d) in s:
            return True
    # x as 2nd term: x-d, x, x+d, x+2d
    for d in range(1, x):
        if (x - d) in s and (x + d) in s and (x + 2 * d) in s:
            return True
    # x as 3rd term: x-2d, x-d, x, x+d
    for d in range(1, x // 2 + 1):
        if (x - 2 * d) in s and (x - d) in s and (x + d) in s:
            return True
    # x as 4th term: x-3d, x-2d, x-d, x
    for d in range(1, x // 3 + 1):
        if (x - 3 * d) in s and (x - 2 * d) in s and (x - d) in s:
            return True
    return False


def _delta_add(selected, x):
    """Number of new 3-APs created by adding x to selected."""
    s = selected
    cnt = 0
    # x as first term: x, x+d, x+2d
    for d in range(1, (N - x) // 2 + 1):
        if (x + d) in s and (x + 2 * d) in s:
            cnt += 1
    # x as middle term: x-d, x, x+d
    for d in range(1, min(x, N - x) + 1):
        if (x - d) in s and (x + d) in s:
            cnt += 1
    # x as last term: x-2d, x-d, x
    for d in range(1, x // 2 + 1):
        if (x - 2 * d) in s and (x - d) in s:
            cnt += 1
    return cnt


def _delta_remove(selected, x):
    """Number of 3-APs destroyed by removing x from selected."""
    s = selected
    cnt = 0
    for d in range(1, (N - x) // 2 + 1):
        if (x + d) in s and (x + 2 * d) in s:
            cnt += 1
    for d in range(1, min(x, N - x) + 1):
        if (x - d) in s and (x + d) in s:
            cnt += 1
    for d in range(1, x // 2 + 1):
        if (x - 2 * d) in s and (x - d) in s:
            cnt += 1
    return cnt


def _count_3aps(selected):
    """Total number of distinct 3-term APs in selected."""
    cnt = 0
    for x in selected:
        cnt += _delta_add(selected - {x}, x)
    return cnt


def _greedy_seed():
    """Build a maximal 4-AP-free set by scanning 1..N and adding x if safe."""
    selected = set()
    for x in range(1, N + 1):
        if not _has_4ap_with(selected, x):
            selected.add(x)
    return selected


def _local_search(selected, current_3ap, start_time):
    """Exhaustive 1-swap + single-add local search until no improvement."""
    while time.time() - start_time < DEADLINE:
        improved = False
        elems = list(selected)
        missing = [x for x in range(1, N + 1) if x not in selected]
        if not elems or not missing:
            break

        # Exhaustive 1-swap: try every (rem, add) pair
        for rem in elems:
            if time.time() - start_time > DEADLINE:
                return selected, current_3ap, True
            rem_delta = _delta_remove(selected, rem)
            for add in missing:
                if time.time() - start_time > DEADLINE:
                    return selected, current_3ap, True
                # Check 4-AP for the swap: remove rem, add add
                trial_without_add = selected - {rem}
                if _has_4ap_with(trial_without_add, add):
                    continue
                add_delta = _delta_add(trial_without_add, add)
                delta = add_delta - rem_delta
                if delta > 0:
                    selected = trial_without_add | {add}
                    current_3ap += delta
                    improved = True
                    break
            if improved:
                break

        if not improved:
            # Try single additions
            found = False
            for add in missing:
                if time.time() - start_time > DEADLINE:
                    return selected, current_3ap, True
                if _has_4ap_with(selected, add):
                    continue
                delta = _delta_add(selected, add)
                if delta > 0:
                    selected.add(add)
                    current_3ap += delta
                    found = True
                    break
            if not found:
                return selected, current_3ap, False  # local optimum

    return selected, current_3ap, True


def construct():
    start = time.time()
    random.seed(12345)

    # Greedy seed: add x if it doesn't create a 4-AP
    selected = _greedy_seed()
    current_3ap = _count_3aps(selected)
    best_set = set(selected)
    best_3ap = current_3ap

    # Iterated local search with random restarts
    restart_count = 0
    while time.time() - start < DEADLINE:
        selected, current_3ap, is_local_opt = _local_search(selected, current_3ap, start)
        if current_3ap > best_3ap:
            best_3ap = current_3ap
            best_set = set(selected)
        if is_local_opt:
            # Perturbation: remove k random elements to escape the local optimum
            restart_count += 1
            elems = list(selected)
            if not elems:
                break
            k = min(3, len(elems) // 10)  # remove a small number
            if k < 1:
                k = 1
            to_remove = random.sample(elems, k)
            for rem in to_remove:
                current_3ap -= _delta_remove(selected, rem)
                selected.discard(rem)
            if len(selected) == 0:
                # Fallback: rebuild from scratch
                selected = _greedy_seed()
                current_3ap = _count_3aps(selected)
            # Occasionally do a full restart from a fresh greedy seed with
            # a different random perturbation
            if restart_count % 5 == 0:
                fresh = _greedy_seed()
                fresh_3ap = _count_3aps(fresh)
                if fresh_3ap >= current_3ap:
                    selected = fresh
                    current_3ap = fresh_3ap
                    if current_3ap > best_3ap:
                        best_3ap = current_3ap
                        best_set = set(selected)

    return sorted(best_set)
