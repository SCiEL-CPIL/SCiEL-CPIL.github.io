"""No-isosceles set: maximize number of grid points in [0,N)^2 with no isosceles triangle."""
import time
import random

N = 64

# EVOLVE-BLOCK-START
def _sqdist(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy


def _can_add(p, pts, dist_from):
    # pts: list of points; dist_from: dict point -> set of sq-dist to others
    seen = set()
    for a in pts:
        d = _sqdist(p, a)
        # apex at a: d equals some existing distance from a
        if d in dist_from[a]:
            return False
        # apex at p: d equals distance p->b for some b
        if d in seen:
            return False
        seen.add(d)
    return True


def _add(p, pts, dist_from, dist_list):
    # dist_list[a] = list of (other, d) to update dist_from
    dist_from[p] = set()
    for a in pts:
        d = _sqdist(p, a)
        dist_from[p].add(d)
        dist_from[a].add(d)
        dist_list[a].append((p, d))
    pts.append(p)
    dist_list.append([])


def _remove(p, pts, dist_from, dist_list):
    idx = pts.index(p)
    # remove p from each other's structures
    for a in pts:
        if a is p:
            continue
        d = _sqdist(p, a)
        dist_from[a].discard(d)
    # remove entries referencing p from dist_list
    for i in range(len(pts)):
        lst = dist_list[i]
        dist_list[i] = [(o, d) for (o, d) in lst if o is not p]
    pts.pop(idx)
    dist_list.pop(idx)
    dist_from.pop(p)


def _greedy_order(points, rng):
    return points


def _initial_greedy(points, rng, seed):
    rng = random.Random(seed)
    order = list(points)
    rng.shuffle(order)
    pts, dist_from, dist_list = [], {}, []
    for p in order:
        if _can_add(p, pts, dist_from):
            _add(p, pts, dist_from, dist_list)
    return pts, dist_from, dist_list


def _local_search(pts, dist_from, dist_list, all_points, rng, deadline):
    # Try to add random points; if stuck, try swap: remove one, add another.
    best = len(pts)
    cand_pool = [p for p in all_points if p not in set(pts)]
    rng.shuffle(cand_pool)
    ci = 0
    while time.time() < deadline:
        # try adding next candidate
        added = False
        while ci < len(cand_pool):
            p = cand_pool[ci]
            ci += 1
            if _can_add(p, pts, dist_from):
                _add(p, pts, dist_from, dist_list)
                added = True
                if len(pts) > best:
                    best = len(pts)
                break
        if added:
            # refresh pool of not-in-set candidates occasionally
            if ci >= len(cand_pool):
                cand_pool = [p for p in all_points if p not in set(pts)]
                rng.shuffle(cand_pool)
                ci = 0
            continue
        # pool exhausted: try a swap to unblock
        if not pts:
            break
        a = rng.choice(pts)
        # remove a
        _remove(a, pts, dist_from, dist_list)
        # try to add a random other point
        added2 = False
        for _ in range(20):
            p = rng.choice(all_points)
            if p == a:
                continue
            if _can_add(p, pts, dist_from):
                _add(p, pts, dist_from, dist_list)
                added2 = True
                if len(pts) > best:
                    best = len(pts)
                break
        if not added2:
            # re-add a to keep size
            _add(a, pts, dist_from, dist_list)
        # refresh pool
        cand_pool = [p for p in all_points if p not in set(pts)]
        rng.shuffle(cand_pool)
        ci = 0
    return pts, dist_from, dist_list, best


def construct():
    start = time.time()
    deadline = start + 280  # safety margin vs 330s limit

    all_points = [(x, y) for x in range(N) for y in range(N)]
    rng = random.Random(12345)

    # Multiple restarts with different seeds, keep best
    best_pts, best_from, best_list, best_size = [], {}, [], 0

    for seed in range(1, 200):
        if time.time() > deadline - 20:
            break
        pts, dist_from, dist_list = _initial_greedy(all_points, rng, seed)
        pts, dist_from, dist_list, sz = _local_search(
            pts, dist_from, dist_list, all_points, rng, deadline - 5)
        if sz > best_size:
            best_size = sz
            best_pts, best_from, best_list = pts, dist_from, dist_list
            if time.time() > deadline - 30:
                break

    return list(best_pts)
# EVOLVE-BLOCK-END
