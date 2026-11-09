import time
import random
import math
from math import gcd

_START = time.time()
_DEADLINE = _START + 145.0
_BOUND = 1000


def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def _line_key(p, q):
    x1, y1 = p
    x2, y2 = q
    dx, dy = x2 - x1, y2 - y1
    g = _gcd(dx, dy)
    if g:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    c = x1 * dy - y1 * dx
    return (dx, dy, c)


def _build_line_map(points):
    """Returns dict: line_key -> set of point indices on that line."""
    n = len(points)
    lm = {}
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            if k not in lm:
                lm[k] = set()
            lm[k].add(i)
            lm[k].add(j)
    return lm


def _score_from_lmap(lm):
    n4 = sum(1 for s in lm.values() if len(s) == 4)
    n5 = sum(1 for s in lm.values() if len(s) >= 5)
    n3 = sum(1 for s in lm.values() if len(s) == 3)
    return n4, n5, n3


def _valid(pts):
    lm = _build_line_map(pts)
    for s in lm.values():
        if len(s) >= 5:
            return False
    return True


def _build_ag24():
    pts = [[x, y] for x in range(4) for y in range(4)]
    return pts


def _build_staggered():
    pts = []
    for row in range(4):
        for col in range(4):
            pts.append([col + row, row])
    return pts


def _build_hexagonal():
    pts = [[0, 0]]
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1)]:
        pts.append([dx, dy])
    for p in [(2,0),(-2,0),(0,2),(0,-2),(2,1),(-2,-1),(1,2),(-1,-2)]:
        pts.append(list(p))
    return pts[:16]


def _build_random_seed(rng, scale=12):
    pts = []
    attempts = 0
    while len(pts) < 16 and attempts < 5000:
        x = rng.randint(-scale, scale)
        y = rng.randint(-scale, scale)
        if [x, y] in pts:
            continue
        cand = pts + [[x, y]]
        if len(cand) >= 5:
            lm = _build_line_map(cand)
            bad = False
            for s in lm.values():
                if len(s) >= 5:
                    bad = True
                    break
            if bad:
                continue
        pts.append([x, y])
        attempts += 1
    while len(pts) < 16:
        x = rng.randint(-scale, scale)
        y = rng.randint(-scale, scale)
        if [x, y] not in pts:
            pts.append([x, y])
    return pts[:16]


def _build_perturbed_ag24(rng, n_perturb=3):
    pts = [list(p) for p in _build_ag24()]
    for _ in range(n_perturb):
        i = rng.randrange(16)
        pts[i] = [pts[i][0] + rng.randint(-2, 2), pts[i][1] + rng.randint(-2, 2)]
    if len(set(map(tuple, pts))) == 16:
        return pts
    return _build_ag24()


def _find_pairs_on_line(lm, pts):
    """Find pairs of points that are already on the same 3-point or 4-point line.
    For each such line, the pair can be swapped or one moved to try to create a new 4-line."""
    candidates = []
    for k, s in lm.items():
        if len(s) == 3:
            # Try to find a 4th point that could be moved to this line
            idx_list = list(s)
            for i in idx_list:
                for j in idx_list:
                    if i < j:
                        candidates.append((i, j, k, s))
        elif len(s) == 4:
            idx_list = list(s)
            for i in idx_list:
                for j in idx_list:
                    if i < j:
                        candidates.append((i, j, k, s))
    return candidates


def _try_pair_swap(pts, i, j, rng, budget):
    """Try swapping points i and j to new positions that might create 4-lines."""
    n = len(pts)
    cur = [list(p) for p in pts]
    lm = _build_line_map(cur)
    n4, n5, n3 = _score_from_lmap(lm)
    cur_score = n4 - n5  # primary metric

    # Try moving point i to various positions near point j
    best_pos_i = None
    best_pos_j = None
    best_new_score = cur_score
    best_new_pts = None

    x0, y0 = cur[i]
    x1, y1 = cur[j]

    # Try positions in a window around the midpoint and around each point
    search_positions = set()
    # Around point i
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            nx, ny = x0 + dx, y0 + dy
            if abs(nx) <= _BOUND and abs(ny) <= _BOUND:
                search_positions.add((nx, ny))
    # Around point j
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            nx, ny = x1 + dx, y1 + dy
            if abs(nx) <= _BOUND and abs(ny) <= _BOUND:
                search_positions.add((nx, ny))
    # Around midpoint
    mx, my = (x0 + x1) // 2, (y0 + y1) // 2
    for dx in range(-4, 5):
        for dy in range(-4, 5):
            nx, ny = mx + dx, my + dy
            if abs(nx) <= _BOUND and abs(ny) <= _BOUND:
                search_positions.add((nx, ny))

    # Try swapping i and j positions
    if [x1, y1] != [x0, y0]:
        test = [list(p) for p in cur]
        test[i] = [x1, y1]
        test[j] = [x0, y0]
        if len(set(map(tuple, test))) == n:
            lm2 = _build_line_map(test)
            if all(len(s) < 5 for s in lm2.values()):
                n4b, n5b, n3b = _score_from_lmap(lm2)
                sc = n4b - n5b
                if sc > best_new_score:
                    best_new_score = sc
                    best_new_pts = test

    # Try moving i to a search position and j stays, or vice versa
    for (nx, ny) in search_positions:
        if [nx, ny] in cur:
            continue
        # Move i to (nx, ny)
        test = [list(p) for p in cur]
        test[i] = [nx, ny]
        if len(set(map(tuple, test))) == n:
            lm2 = _build_line_map(test)
            if all(len(s) < 5 for s in lm2.values()):
                n4b, n5b, n3b = _score_from_lmap(lm2)
                sc = n4b - n5b
                if sc > best_new_score:
                    best_new_score = sc
                    best_new_pts = test
        # Move j to (nx, ny)
        test = [list(p) for p in cur]
        test[j] = [nx, ny]
        if len(set(map(tuple, test))) == n:
            lm2 = _build_line_map(test)
            if all(len(s) < 5 for s in lm2.values()):
                n4b, n5b, n3b = _score_from_lmap(lm2)
                sc = n4b - n5b
                if sc > best_new_score:
                    best_new_score = sc
                    best_new_pts = test

    if best_new_pts is not None:
        return best_new_pts
    return None


def _directed_pair_move(pts, i, j, rng, budget):
    """Move point i to a position that, together with j and 2 others, forms a 4-line.
    Strategy: for each pair (i,j), find if there exist 2 other points such that
    all 4 could be collinear, then try to position i or j accordingly."""
    n = len(pts)
    cur = [list(p) for p in pts]
    lm = _build_line_map(cur)
    n4, n5, n3 = _score_from_lmap(lm)
    cur_score = n4 - n5

    best_new_pts = None
    best_new_score = cur_score

    # For each pair of other points (k, l), check if line through j and k
    # could include i at some position, or line through i and k could include j
    # More practically: try to move i so that i, j, k, l are collinear
    # for some k, l

    for k in range(n):
        if k == i or k == j:
            continue
        for l in range(n):
            if l == i or l == j or l == k:
                continue
            # Check if j, k, l are collinear (or close)
            kj = _line_key(cur[j], cur[k])
            kl = _line_key(cur[j], cur[l])
            if kj == kl:
                # j, k, l are on the same line. Try to place i on this line.
                # Find a point on this line that's not already occupied
                # Line through j and k: parametric form
                xj, yj = cur[j]
                xk, yk = cur[k]
                dx, dy = xk - xj, yk - yj
                g = _gcd(dx, dy)
                if g:
                    dx //= g
                    dy //= g
                # Points on this line: (xj + t*dx, yj + t*dy) for integer t
                # Find t values that give valid positions not in cur
                used = set(map(tuple, cur))
                for t in range(-8, 9):
                    if t == 0:  # j's position
                        continue
                    nx, ny = xj + t * dx, yj + t * dy
                    if abs(nx) > _BOUND or abs(ny) > _BOUND:
                        continue
                    if (nx, ny) in used:
                        continue
                    test = [list(p) for p in cur]
                    test[i] = [nx, ny]
                    if len(set(map(tuple, test))) == n:
                        lm2 = _build_line_map(test)
                        if all(len(s) < 5 for s in lm2.values()):
                            n4b, n5b, n3b = _score_from_lmap(lm2)
                            sc = n4b - n5b
                            if sc > best_new_score:
                                best_new_score = sc
                                best_new_pts = test
                                break
    return best_new_pts


def _swap_local_search(pts, rng, time_budget):
    """Local search using pair swaps and directed pair moves."""
    n = len(pts)
    cur = [list(p) for p in pts]
    best_pts = [list(p) for p in cur]

    lm = _build_line_map(cur)
    n4, n5, n3 = _score_from_lmap(lm)
    cur_score = n4 - n5
    best_score = cur_score
    step = 0
    no_improve = 0

    while time.time() < time_budget:
        improved = False

        # Phase 1: Directed pair moves (targeted)
        # Pick pairs that are on 3-point lines or close to forming 4-lines
        lm = _build_line_map(cur)
        # Find 3-point lines
        three_lines = [s for s in lm.values() if len(s) == 3]
        four_lines = [s for s in lm.values() if len(s) == 4]

        # For each 3-point line, try to find a 4th point to move onto it
        for s in three_lines:
            if time.time() >= time_budget:
                break
            idx_list = list(s)
            # Try moving a point not on this line to join it
            line_key = None
            for a in idx_list:
                for b in idx_list:
                    if a < b:
                        line_key = _line_key(cur[a], cur[b])
                        break
                if line_key:
                    break
            if line_key is None:
                continue
            dx, dy, c = line_key
            # Find candidate positions on this line
            # Line: dy*x - dx*y = c (or equivalent)
            # Try a few points on this line
            x0, y0 = cur[idx_list[0]]
            # Parametric: (x0 + t*dx, y0 + t*dy)
            used = set(map(tuple, cur))
            found_pos = False
            for t in range(-10, 11):
                if t == 0:
                    continue
                nx, ny = x0 + t * dx, y0 + t * dy
                if abs(nx) > _BOUND or abs(ny) > _BOUND:
                    continue
                if (nx, ny) in used:
                    continue
                # Find a point to move here
                for i in range(n):
                    if i in s:
                        continue
                    test = [list(p) for p in cur]
                    test[i] = [nx, ny]
                    if len(set(map(tuple, test))) == n:
                        lm2 = _build_line_map(test)
                        if all(len(ss) < 5 for ss in lm2.values()):
                            n4b, n5b, n3b = _score_from_lmap(lm2)
                            sc = n4b - n5b
                            if sc > cur_score:
                                cur = test
                                cur_score = sc
                                improved = True
                                if sc > best_score:
                                    best_score = sc
                                    best_pts = [list(p) for p in cur]
                                found_pos = True
                                break
                if found_pos:
                    break
            if improved:
                break

        # Phase 2: Random pair moves
        if not improved and step > 2:
            # Try random pair of points
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                j = (i + 1) % n

            # Try swapping
            if [cur[j][0], cur[j][1]] != [cur[i][0], cur[i][1]]:
                test = [list(p) for p in cur]
                test[i] = [cur[j][0], cur[j][1]]
                test[j] = [cur[i][0], cur[i][1]]
                if len(set(map(tuple, test))) == n:
                    lm2 = _build_line_map(test)
                    if all(len(s) < 5 for s in lm2.values()):
                        n4b, n5b, n3b = _score_from_lmap(lm2)
                        sc = n4b - n5b
                        if sc >= cur_score:
                            cur = test
                            cur_score = sc
                            improved = True
                            if sc > best_score:
                                best_score = sc
                                best_pts = [list(p) for p in cur]

            # Try moving one of the pair randomly
            if not improved:
                move_i = i if rng.random() < 0.5 else j
                for _ in range(50):
                    if time.time() >= time_budget:
                        break
                    dx = rng.randint(-6, 6)
                    dy = rng.randint(-6, 6)
                    nx = cur[move_i][0] + dx
                    ny = cur[move_i][1] + dy
                    if abs(nx) > _BOUND or abs(ny) > _BOUND:
                        continue
                    if [nx, ny] in cur:
                        continue
                    test = [list(p) for p in cur]
                    test[move_i] = [nx, ny]
                    if len(set(map(tuple, test))) == n:
                        lm2 = _build_line_map(test)
                        if all(len(s) < 5 for s in lm2.values()):
                            n4b, n5b, n3b = _score_from_lmap(lm2)
                            sc = n4b - n5b
                            if sc >= cur_score:
                                cur = test
                                cur_score = sc
                                improved = True
                                if sc > best_score:
                                    best_score = sc
                                    best_pts = [list(p) for p in cur]
                                break

        step += 1
        no_improve = 0 if improved else no_improve + 1

        # Random kick if stuck
        if no_improve > 5 and time.time() < time_budget:
            i = rng.randrange(n)
            for _ in range(300):
                nx = rng.randint(-_BOUND, _BOUND)
                ny = rng.randint(-_BOUND, _BOUND)
                if [nx, ny] in cur:
                    continue
                test = [list(p) for p in cur]
                test[i] = [nx, ny]
                if len(set(map(tuple, test))) != n:
                    continue
                lm2 = _build_line_map(test)
                if any(len(s) >= 5 for s in lm2.values()):
                    continue
                n4b, n5b, n3b = _score_from_lmap(lm2)
                sc = n4b - n5b
                if sc >= cur_score - 1:  # allow small degradation
                    cur = test
                    cur_score = sc
                    break

    return best_pts, best_score


def _final_check(pts):
    if len(pts) != 16:
        return False
    if len(set(map(tuple, pts))) != 16:
        return False
    for p in pts:
        if abs(p[0]) > _BOUND or abs(p[1]) > _BOUND:
            return False
    lm = _build_line_map(pts)
    for s in lm.values():
        if len(s) >= 5:
            return False
    return True


def construct():
    global _START, _DEADLINE
    rng = random.Random(123)

    # Build diverse seeds
    seeds = []
    seeds.append(_build_ag24())
    seeds.append(_build_staggered())
    seeds.append(_build_hexagonal())

    # Random seeds
    for _ in range(3):
        seeds.append(_build_random_seed(rng, scale=15))

    # Perturbed AG(2,4)
    for _ in range(3):
        seeds.append(_build_perturbed_ag24(rng, rng.randint(2, 5)))

    # Additional structured seeds
    # 4x4 grid with one row shifted
    pts = []
    for row in range(4):
        shift = 1 if row == 2 else 0
        for col in range(4):
            pts.append([col + shift, row])
    if _final_check(pts):
        seeds.append(pts)

    # Diagonal-shifted grid
    pts = []
    for row in range(4):
        for col in range(4):
            pts.append([col + (row * 2) % 5, row])
    if len(set(map(tuple, pts))) == 16 and _final_check(pts):
        seeds.append(pts)

    # Dedupe
    seen = set()
    uniq = []
    for s in seeds:
        if not _final_check(s):
            continue
        key = tuple(sorted(map(tuple, s)))
        if key not in seen:
            seen.add(key)
            uniq.append(s)

    if not uniq:
        pts = _build_ag24()
        if _final_check(pts):
            return pts
        return [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],
                [0,2],[1,2],[2,2],[3,2],[0,3],[1,3],[2,3],[3,3]]

    best_overall_pts = None
    best_overall_score = -999

    # Allocate time per seed
    t_start = time.time()
    n_seeds = len(uniq)
    per_seed_time = max(2.0, min(15.0, (_DEADLINE - 15.0 - t_start) / max(1, n_seeds)))

    for idx, s in enumerate(uniq):
        if time.time() >= _DEADLINE - 10.0:
            break
        t_end = min(_DEADLINE - 8.0, time.time() + per_seed_time)
        pts, score = _swap_local_search(s, rng, t_end)
        if _final_check(pts):
            if score > best_overall_score:
                best_overall_score = score
                best_overall_pts = pts

    # If we have time left, do more search on the best
    if best_overall_pts is not None and time.time() < _DEADLINE - 5.0:
        t_end = _DEADLINE - 3.0
        pts, score = _swap_local_search(best_overall_pts, rng, t_end)
        if _final_check(pts):
            if score > best_overall_score:
                best_overall_score = score
                best_overall_pts = pts

    if best_overall_pts is None:
        pts = _build_ag24()
        if _final_check(pts):
            return pts
        return [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],
                [0,2],[1,2],[2,2],[3,2],[0,3],[1,3],[2,3],[3,3]]

    return [list(p) for p in best_overall_pts]
