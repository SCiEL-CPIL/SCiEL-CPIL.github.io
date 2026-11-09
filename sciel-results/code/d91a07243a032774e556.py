"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time
import random

_DEADLINE = time.time() + 280

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _precompute_conflicts(all_points):
    """For each point p, compute set of pairs (i,j) indices such that (p, all_points[i], all_points[j]) is isosceles."""
    n = len(all_points)
    conflicts = [set() for _ in range(n)]
    for p_idx in range(n):
        p = all_points[p_idx]
        for i in range(n):
            if i == p_idx:
                continue
            a = all_points[i]
            for j in range(i + 1, n):
                if j == p_idx:
                    continue
                b = all_points[j]
                d1 = _d2(p, a)
                d2 = _d2(p, b)
                d3 = _d2(a, b)
                if d1 == d2 or d1 == d3 or d2 == d3:
                    conflicts[p_idx].add((min(i, j), max(i, j)))
    return conflicts

def _can_add_idx(p_idx, chosen_indices, conflicts):
    conf = conflicts[p_idx]
    for i in range(len(chosen_indices)):
        for j in range(i + 1, len(chosen_indices)):
            a, b = chosen_indices[i], chosen_indices[j]
            pair = (min(a, b), max(a, b))
            if pair in conf:
                return False
    return True

def _is_maximal_idx(chosen_indices, all_indices, conflicts):
    chosen_set = set(chosen_indices)
    for p_idx in all_indices:
        if p_idx not in chosen_set:
            conf = conflicts[p_idx]
            blocked = False
            for i in range(len(chosen_indices)):
                for j in range(i + 1, len(chosen_indices)):
                    a, b = chosen_indices[i], chosen_indices[j]
                    pair = (min(a, b), max(a, b))
                    if pair in conf:
                        blocked = True
                        break
                if blocked:
                    break
            if not blocked:
                return False
    return True

def _greedy_random_idx(all_indices, conflicts, rng):
    order = list(all_indices)
    rng.shuffle(order)
    chosen = []
    for p_idx in order:
        if _can_add_idx(p_idx, chosen, conflicts):
            chosen.append(p_idx)
    return chosen

def _prune_idx(chosen_indices, all_indices, conflicts):
    chosen_list = list(chosen_indices)
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        for i in range(len(chosen_list)):
            candidate = chosen_list[:i] + chosen_list[i + 1:]
            if _is_maximal_idx(candidate, all_indices, conflicts):
                chosen_list = candidate
                improved = True
                break
    return chosen_list

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    all_indices = list(range(len(all_points)))
    conflicts = _precompute_conflicts(all_points)

    best = None
    rng = random.Random(42)

    for attempt in range(20):
        if time.time() > _DEADLINE:
            break
        chosen = _greedy_random_idx(all_indices, conflicts, rng)
        pruned = _prune_idx(chosen, all_indices, conflicts)
        if best is None or len(pruned) < len(best):
            best = pruned

    if best is None:
        best = _greedy_random_idx(all_indices, conflicts, rng)

    return [all_points[i] for i in best]
# EVOLVE-BLOCK-END
