import time
import math
import random
import itertools
from collections import defaultdict

_START = time.time()
_DEADLINE = _START + 115.0  # safety margin before the 150 s kill
_BOUND = 1000


def _gcd(a, b):
    a = abs(a)
    b = abs(b)
    while b:
        a, b = b, a % b
    return a


def _line_key(p, q):
    x1, y1 = p
    x2, y2 = q
    dx = x2 - x1
    dy = y2 - y1
    g = _gcd(dx, dy)
    if g:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    return (dx, dy, x1 * dy - y1 * dx)


def _score(points):
    """Returns (num_exact4, num_5plus, num_3, num_2)."""
    counts = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            counts[k] = counts.get(k, 0) + 1
    n4 = n5 = n3 = n2 = 0
    for c in counts.values():
        if c == 6:
            n4 += 1
        elif c >= 10:
            n5 += 1
        elif c == 3:
            n3 += 1
        elif c == 2:
            n2 += 1
    return n4, n5, n3, n2


def _valid(points):
    """True iff no five are collinear (no line with >=5 points)."""
    counts = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            counts[k] = counts.get(k, 0) + 1
            if counts[k] >= 10:
                return False
    return True


# --- Seeds ---------------------------------------------------------------

def _seed_grid():
    return [[x, y] for x in range(4) for y in range(4)]


def _seed_perturbed_grid():
    pts = [[x, y] for x in range(4) for y in range(4)]
    pts[0] = [0, 0]
    pts[3] = [3, 3]
    pts[4] = [0, 5]
    pts[15] = [4, 3]
    return pts


def _seed_3x3_plus():
    pts = [[x, y] for x in range(3) for y in range(3)]
    extra = [(3, 1), (-1, 1), (1, 3), (1, -1), (3, 3), (-1, -1), (4, 0)]
    for e in extra:
        pts.append(list(e))
    return pts


def _seed_lattice():
    base = [[x, y] for x in range(5) for y in range(5)]
    random.shuffle(base)
    return base[:16]


def _seed_diagonal_grid():
    pts = []
    for i in range(4):
        for j in range(4):
            pts.append([i, j])
    pts[5] = [1, 5]
    pts[10] = [2, -1]
    pts[15] = [5, 3]
    return pts


def _seed_staggered():
    pts = []
    for row in range(4):
        offset = row
        for col in range(4):
            pts.append([col + offset, row])
    return pts


def _seed_hexagonal():
    pts = []
    pts.append([0, 0])
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]:
        pts.append([dx, dy])
    ring2 = [(2, 0), (-2, 0), (0, 2), (0, -2), (2, 1), (-2, -1), (1, 2), (-1, -2)]
    for p in ring2:
        pts.append(list(p))
    return pts[:16]


def _seed_affine_grid():
    pts = []
    for x in range(4):
        for y in range(4):
            pts.append([x + y, y])
    return pts


def _seed_mixed():
    pts = []
    for i in range(4):
        for j in range(4):
            if i % 2 == 0:
                pts.append([i, j])
            else:
                pts.append([i + 1, j])
    return pts


def _all_seeds():
    return [_seed_grid(), _seed_perturbed_grid(), _seed_3x3_plus(),
            _seed_lattice(), _seed_diagonal_grid(), _seed_staggered(),
            _seed_hexagonal(), _seed_affine_grid(), _seed_mixed()]


# --- Spectral / incidence-matrix move family -----------------------------

def _line_membership(points):
    """Map line_key -> list of point indices on that line."""
    n = len(points)
    members = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            members[k].append(i)
            members[k].append(j)
    # Deduplicate (each pair adds both endpoints; a point may appear via multiple pairs)
    dedup = {}
    for k, lst in members.items():
        dedup[k] = list(set(lst))
    return dedup


def _spectral_move(points, rng, budget):
    """
    Targeted move: pick a line with exactly 3 points (a 'near-4' line) and try to
    move one of the 3 points (or another point) to a position on that line to
    create a 4-point line, without creating a 5-collinear.
    Also target lines with 5+ points to break them.
    """
    n = len(points)
    cur = [list(p) for p in points]
    best_key = None
    best_pts = None
    best_cand = None

    members = _line_membership(cur)

    # Collect candidate lines to complete (size 3) or break (size >=5)
    target_lines = []
    for k, pts_on_line in members.items():
        if len(pts_on_line) == 3:
            target_lines.append((k, pts_on_line, 'complete'))
        elif len(pts_on_line) >= 5:
            target_lines.append((k, pts_on_line, 'break'))

    if not target_lines:
        return None, None

    # Sort: prefer completing, then breaking
    target_lines.sort(key=lambda t: (0 if t[2] == 'complete' else 1, -len(t[1])))

    for k, pts_on_line, action in target_lines[:12]:
        if time.time() >= budget:
            break

        # Determine the direction vector of this line
        if len(pts_on_line) >= 2:
            p1 = cur[pts_on_line[0]]
            p2 = cur[pts_on_line[1]]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            g = _gcd(dx, dy)
            if g:
                dx //= g
                dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
        else:
            continue

        if action == 'complete':
            # Try to find a new position on this line (not occupied) that,
            # when we move some point to it, creates a 4-line.
            # We'll move one of the 3 points to a new spot on the same line,
            # or move a 4th point onto this line.
            # Strategy: for each of the 3 points, try moving it slightly along
            # the line to a new integer position, OR try moving a non-member
            # point onto this line at an integer position.
            existing = set(tuple(p) for p in cur)
            # Try moving a non-member point onto this line
            non_members = [i for i in range(n) if i not in pts_on_line]
            for idx in non_members[:6]:
                if time.time() >= budget:
                    break
                # Find integer positions on the line near the existing 3 points
                base_x = sum(cur[i][0] for i in pts_on_line) // 3
                base_y = sum(cur[i][1] for i in pts_on_line) // 3
                for t in range(-4, 5):
                    nx = base_x + t * dx
                    ny = base_y + t * dy
                    if abs(nx) > _BOUND or abs(ny) > _BOUND:
                        continue
                    np_ = [nx, ny]
                    if tuple(np_) in existing:
                        continue
                    cand = [list(p) for p in cur]
                    cand[idx] = np_
                    if not _valid(cand):
                        continue
                    n4, n5, n3, n2 = _score(cand)
                    key = (n4, -n5, n3, n2)
                    if best_key is None or key > best_key:
                        best_key = key
                        best_pts = [list(p) for p in cand]
                        best_cand = (k, idx, np_)

        elif action == 'break':
            # Move one of the 5+ points off this line to break the 5-collinear
            for idx in pts_on_line[:5]:
                if time.time() >= budget:
                    break
                x0, y0 = cur[idx]
                for _ in range(30):
                    nx = x0 + rng.randint(-6, 6)
                    ny = y0 + rng.randint(-6, 6)
                    if abs(nx) > _BOUND or abs(ny) > _BOUND:
                        continue
                    np_ = [nx, ny]
                    if np_ == cur[idx]:
                        continue
                    if any(np_ == p for p in cur):
                        continue
                    cand = [list(p) for p in cur]
                    cand[idx] = np_
                    if not _valid(cand):
                        continue
                    n4, n5, n3, n2 = _score(cand)
                    key = (n4, -n5, n3, n2)
                    if best_key is None or key > best_key:
                        best_key = key
                        best_pts = [list(p) for p in cand]
                        best_cand = (k, idx, np_)

    return best_pts, best_key


# --- Local search ---------------------------------------------------------

def _local_search(points, rng, time_budget):
    """Single-point local search with incremental scoring + spectral moves."""
    n = len(points)
    cur = [list(p) for p in points]
    cur_n4, cur_n5, cur_n3, cur_n2 = _score(cur)
    best_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
    best_pts = [list(p) for p in cur]
    step = 0

    while time.time() < time_budget:
        improved = False

        # --- Spectral / incidence-matrix targeted move ---
        if step % 3 == 0:  # every 3rd step
            spec_pts, spec_key = _spectral_move(cur, rng, time_budget)
            if spec_pts is not None and spec_key is not None:
                if spec_key > best_key:
                    best_key = spec_key
                    best_pts = [list(p) for p in spec_pts]
                if spec_key > (cur_n4, -cur_n5, cur_n3, cur_n2):
                    cur = spec_pts
                    cur_n4, cur_n5, cur_n3, cur_n2 = spec_key[0], -spec_key[1], spec_key[2], spec_key[3]
                    improved = True

        # --- Single-point moves ---
        for i in range(n):
            if time.time() >= time_budget:
                break
            x0, y0 = cur[i]
            w = 4 + (step // 15)
            w = min(w, 15)

            best_local = None
            best_local_key = None

            for dx in range(-w, w + 1):
                if time.time() >= time_budget:
                    break
                for dy in range(-w, w + 1):
                    nx, ny = x0 + dx, y0 + dy
                    if abs(nx) > _BOUND or abs(ny) > _BOUND:
                        continue
                    np_ = [nx, ny]
                    if any(np_ == p for p in cur):
                        continue

                    cand = [list(p) for p in cur]
                    cand[i] = np_

                    # Quick validity: check lines through moved point
                    valid = True
                    for j in range(n):
                        if j == i:
                            continue
                        k = _line_key(np_, cand[j])
                        count = 1
                        for m in range(n):
                            if m == i or m == j:
                                continue
                            if _line_key(cand[m], cand[j]) == k:
                                count += 1
                        if count >= 5:
                            valid = False
                            break
                    if not valid:
                        continue

                    n4, n5, n3, n2 = _score(cand)
                    key = (n4, -n5, n3, n2)
                    if best_local_key is None or key > best_local_key:
                        best_local_key = key
                        best_local = np_

            if best_local is not None:
                cur_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
                if best_local_key > cur_key:
                    cur[i] = best_local
                    cur_n4, cur_n5, cur_n3, cur_n2 = best_local_key[0], -best_local_key[1], best_local_key[2], best_local_key[3]
                    improved = True
                    if (cur_n4, -cur_n5, cur_n3, cur_n2) > best_key:
                        best_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
                        best_pts = [list(p) for p in cur]

        step += 1

        if not improved and step > 2:
            # Random kick to escape local optimum
            if time.time() >= time_budget:
                break
            i = rng.randrange(n)
            x0, y0 = cur[i]
            for _ in range(300):
                nx = x0 + rng.randint(-8, 8)
                ny = y0 + rng.randint(-8, 8)
                if abs(nx) > _BOUND or abs(ny) > _BOUND:
                    continue
                np_ = [nx, ny]
                if any(np_ == p for p in cur):
                    continue
                cand = [list(p) for p in cur]
                cand[i] = np_
                if not _valid(cand):
                    continue
                n4, n5, n3, n2 = _score(cand)
                key = (n4, -n5, n3, n2)
                cur_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
                if key >= cur_key:
                    cur[i] = np_
                    cur_n4, cur_n5, cur_n3, cur_n2 = n4, n5, n3, n2
                    if (cur_n4, -cur_n5, cur_n3, cur_n2) > best_key:
                        best_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
                        best_pts = [list(p) for p in cur]
                    break

    return best_pts, best_key


def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    rng = random.Random(42)

    seeds = _all_seeds()

    # Evaluate seeds
    best_score = -1
    best_pts = None
    for seed in seeds:
        n4, n5, n3, n2 = _score(seed)
        if n5 == 0:
            score = n4
        else:
            score = n4 - 100 * n5
        if score > best_score:
            best_score = score
            best_pts = [p[:] for p in seed]

    global_best_pts = [p[:] for p in best_pts]
    global_best_score = best_score

    # Run local search with multiple restarts
    restart = 0
    max_restarts = 8
    while time.time() < deadline:
        # Pick a starting point
        if restart == 0:
            start_pts = [p[:] for p in best_pts]
        elif restart % 2 == 0:
            start_pts = [p[:] for p in global_best_pts]
        else:
            start_pts = [p[:] for p in rng.choice(seeds)]
            # Perturb
            for _ in range(5):
                idx = rng.randint(0, 15)
                start_pts[idx][0] += rng.randint(-5, 5)
                start_pts[idx][1] += rng.randint(-5, 5)

        # Time budget per restart
        remaining = deadline - time.time()
        per_restart = max(5.0, remaining / (max_restarts - restart + 1))
        time_budget = time.time() + per_restart

        pts, key = _local_search(start_pts, rng, time_budget)
        n4, n5, n3, n2 = _score(pts)
        score = n4 - 100 * n5
        if score > global_best_score:
            global_best_score = score
            global_best_pts = [p[:] for p in pts]

        restart += 1

    # Final validation
    n4, n5, n3, n2 = _score(global_best_pts)
    if n5 > 0:
        # Fallback to a known-valid seed
        for seed in seeds:
            s4, s5, _, _ = _score(seed)
            if s5 == 0:
                return seed
        return _seed_grid()

    return global_best_pts


if __name__ == "__main__":
    pts = construct()
    n4, n5, n3, n2 = _score(pts)
    print(f"Points: {pts}")
    print(f"Exact-4 lines: {n4}, 5+ collinear: {n5}, 3-collinear: {n3}, 2-collinear: {n2}")
    print(f"Time: {time.time() - _START:.1f}s")
