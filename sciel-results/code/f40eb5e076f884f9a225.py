import time
import random

L = 5354228880  # lcm(1..24)

def _has_conflict(selected):
    n = len(selected)
    if n < 2:
        return False
    w = [L // x for x in selected]
    for i in range(n):
        target = w[i]
        others = w[:i] + w[i+1:]
        m = len(others)
        if m == 0:
            continue
        mid = m // 2
        left = others[:mid]
        right = others[mid:]
        left_sums = {0}
        for v in left:
            left_sums = left_sums | {s + v for s in left_sums}
        right_sums = {0}
        for v in right:
            right_sums = right_sums | {s + v for s in right_sums}
        if target in right_sums:
            return True
        for rs in right_sums:
            if target - rs in left_sums:
                return True
    return False

def _is_valid(selected):
    return not _has_conflict(selected)

def _greedy_remove(order):
    current = list(order)
    changed = True
    while changed:
        changed = False
        for i in range(len(current)):
            trial = current[:i] + current[i+1:]
            if _is_valid(trial):
                current = trial
                changed = True
                break
    return current

def construct():
    deadline = time.time() + 50.0
    rng = random.Random(12345)
    best = [1]
    best_size = 1

    # deterministic full
    cur = _greedy_remove(list(range(1, 25)))
    if len(cur) > best_size:
        best, best_size = cur, len(cur)

    # random restarts
    while time.time() < deadline:
        order = list(range(1, 25))
        rng.shuffle(order)
        cur = _greedy_remove(order)
        if len(cur) > best_size:
            best, best_size = cur, len(cur)
            if best_size == 24:
                break

    # try to add to best
    for cand in range(1, 25):
        if time.time() > deadline:
            break
        if cand in best:
            continue
        trial = list(best) + [cand]
        if _is_valid(trial):
            best = trial
            best_size = len(best)

    if not best:
        best = [1]
    return list(best)
