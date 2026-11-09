"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd
from collections import defaultdict

def _line_key(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (0, 0, x1)
    if dy == 0:
        return (0, 1, y1)
    if dy < 0:
        dx, dy = -dx, -dy
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    c = dy * x1 - dx * y1
    return (1, dx, dy, c)

def _count_ordinary_lines(pts):
    n = len(pts)
    if n < 3:
        return 0
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(pts[i][0], pts[i][1], pts[j][0], pts[j][1])
            lines[key] = lines.get(key, 0) + 1
    return sum(1 for c in lines.values() if c == 1)

def _evaluate(points):
    return _count_ordinary_lines(points)

def _not_all_collinear(points):
    n = len(points)
    if n < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for k in range(2, n):
        x2, y2 = points[k]
        if (x1 - x0) * (y2 - y0) != (y1 - y0) * (x2 - x0):
            return True
    return False

def _generate_hexagonal_candidates(scale=2):
    s = scale
    pts = [[0, 0]]
    for i in range(6):
        a = i * math.pi / 3
        pts.append([round(s * math.cos(a)), round(s * math.sin(a))])
    for i in range(6):
        a = i * math.pi / 3 + math.pi / 6
        pts.append([round(s * 0.5 * math.cos(a)), round(s * 0.5 * math.sin(a))])
    seen = set()
    out = []
    for p in pts:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            out.append(p)
    while len(out) < 12:
        out.append([len(out), 0])
    return out[:12]

def _generate_structured_configs():
    configs = []
    configs.append([[x, y] for x in range(4) for y in range(3)])
    configs.append([[x, y] for x in range(3) for y in range(4)])
    configs.append([[x, y] for x in range(6) for y in range(2)])
    configs.append([[x, y] for x in range(2) for y in range(6)])
    configs.append(_generate_hexagonal_candidates(2))
    configs.append(_generate_hexagonal_candidates(3))
    configs.append(_generate_hexagonal_candidates(4))
    # 4x4 minus 4 corners
    c44 = [[x, y] for x in range(4) for y in range(4)
           if not (x in (0, 3) and y in (0, 3))]
    configs.append(c44)
    # Two parallel lines of 6
    configs.append([[x, 0] for x in range(6)] + [[x, 3] for x in range(6)])
    # Three parallel lines of 4
    configs.append([[x, y] for x in range(4) for y in range(3)])
    return configs

def construct():
    start = time.time()
    deadline = start + 138  # safety margin before 150s kill
    n = 12
    rng = random.Random(42)

    candidates = _generate_structured_configs()
    for _ in range(15):
        base = [[x, y] for x in range(4) for y in range(3)]
        pert = []
        for (x, y) in base:
            if rng.random() < 0.25:
                pert.append([x + rng.randint(-3, 3), y + rng.randint(-3, 3)])
            else:
                pert.append([x, y])
        seen = set()
        uniq = []
        for p in pert:
            t = tuple(p)
            if t not in seen:
                seen.add(t)
                uniq.append(p)
        if len(uniq) >= 12:
            candidates.append(uniq[:12])

    best_points = None
    best_score = float('inf')
    for cand in candidates:
        if len(cand) != 12:
            continue
        seen = set()
        valid = True
        for p in cand:
            t = tuple(p)
            if t in seen:
                valid = False
                break
            seen.add(t)
        if not valid:
            continue
        if not _not_all_collinear(cand):
            continue
        s = _evaluate(cand)
        if s < best_score:
            best_score = s
            best_points = [p[:] for p in cand]

    if best_points is None:
        best_points = [[i, i * i] for i in range(12)]
        best_score = _evaluate(best_points)

    # ---- Fast incremental ordinary-line counter ----
    # lines[k] = list of point indices on line k
    def _build_lines(pts):
        lm = {}
        for i in range(n):
            for j in range(i + 1, n):
                k = _line_key(pts[i][0], pts[i][1], pts[j][0], pts[j][1])
                if k not in lm:
                    lm[k] = []
                lm[k].append(i)
                lm[k].append(j)
        # dedup (each pair added once, so no dupes)
        return lm

    def _ordinary_count(lm):
        return sum(1 for v in lm.values() if len(v) == 2)

    def _remove_point(lm, idx):
        """Remove point idx from all lines; drop empty lines."""
        to_remove = []
        for k, v in lm.items():
            if idx in v:
                v.remove(idx)
                if len(v) == 0:
                    to_remove.append(k)
        for k in to_remove:
            del lm[k]

    def _add_point(lm, idx, x, y):
        """Add point idx at (x,y) to all lines through it and existing points."""
        for j in range(n):
            if j == idx:
                continue
            k = _line_key(x, y, pts[j][0], pts[j][1])
            if k not in lm:
                lm[k] = []
            if idx not in lm[k]:
                lm[k].append(idx)
            if j not in lm[k]:
                lm[k].append(j)

    # ---- Local search with incremental updates ----
    def _local_search(pts, score, time_limit):
        pts = [p[:] for p in pts]
        lm = _build_lines(pts)
        cur = score
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            order = list(range(n))
            rng.shuffle(order)
            for idx in order:
                if time.time() >= time_limit:
                    break
                old_x, old_y = pts[idx]

                # Build candidate positions
                cand_pos = set()
                others = [(pts[j][0], pts[j][1]) for j in range(n) if j != idx]
                occ = set(others)

                # Positions on lines through pairs of others (to create 3-pt lines)
                for i in range(len(others)):
                    for j in range(i + 1, len(others)):
                        x1, y1 = others[i]
                        x2, y2 = others[j]
                        dx = x2 - x1
                        dy = y2 - y1
                        if dx == 0 and dy == 0:
                            continue
                        if dx < 0 or (dx == 0 and dy < 0):
                            dx, dy = -dx, -dy
                        g = gcd(abs(dx), abs(dy))
                        if g > 0:
                            dxn, dyn = dx // g, dy // g
                        else:
                            dxn, dyn = dx, dy
                        for t in range(-3, 4):
                            nx = x1 + t * dxn
                            ny = y1 + t * dyn
                            if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                                if (nx, ny) not in occ:
                                    cand_pos.add((nx, ny))

                # Directed: move point idx to a position that joins two existing
                # 2-point lines (creating a 3-point line at the intersection)
                two_pt_lines = [k for k, v in lm.items() if len(v) == 2]
                for a in range(len(two_pt_lines)):
                    for b in range(a + 1, len(two_pt_lines)):
                        ka = two_pt_lines[a]
                        kb = two_pt_lines[b]
                        if ka == kb:
                            continue
                        # Solve intersection of the two lines
                        # Line form: (a,b,c) where a*x + b*y = c  (we use dy*x - dx*y = c)
                        # Actually our key is (1, dx, dy, c) meaning dy*x - dx*y = c
                        # and (0,0,x0) meaning x = x0, (0,1,y0) meaning y = y0
                        def _line_eq(k):
                            if k[0] == 0:
                                if k[1] == 0:
                                    return (1, 0, -k[2])  # x = k[2]
                                else:
                                    return (0, 1, -k[2])  # y = k[2]
                            else:
                                # dy*x - dx*y = c  =>  a=dy, b=-dx, c=c
                                return (k[2], -k[1], -k[3])

                        a1, b1, c1 = _line_eq(ka)
                        a2, b2, c2 = _line_eq(kb)
                        det = a1 * b2 - a2 * b1
                        if det == 0:
                            continue  # parallel
                        ix = (c1 * b2 - c2 * b1) / det
                        iy = (a1 * c2 - a2 * c1) / det
                        # Check if integer
                        if abs(ix - round(ix)) < 1e-9 and abs(iy - round(iy)) < 1e-9:
                            ix_i, iy_i = round(ix), round(iy)
                            if abs(ix_i) <= 10**9 and abs(iy_i) <= 10**9:
                                if (ix_i, iy_i) not in occ:
                                    cand_pos.add((ix_i, iy_i))

                # Random nearby
                for _ in range(20):
                    bx, by = rng.choice(others)
                    nx = bx + rng.randint(-4, 4)
                    ny = by + rng.randint(-4, 4)
                    if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                        if (nx, ny) not in occ:
                            cand_pos.add((nx, ny))

                # Include original
                cand_pos.add((old_x, old_y))
                pos_list = list(cand_pos)
                rng.shuffle(pos_list)

                for nx, ny in pos_list:
                    if nx == old_x and ny == old_y:
                        continue
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    # Check duplicate
                    dup = False
                    for j, (px, py) in enumerate(pts):
                        if j != idx and px == nx and py == ny:
                            dup = True
                            break
                    if dup:
                        continue

                    # Apply move
                    _remove_point(lm, idx)
                    pts[idx] = [nx, ny]
                    _add_point(lm, idx, nx, ny)

                    # Check not all collinear
                    if not _not_all_collinear(pts):
                        _remove_point(lm, idx)
                        pts[idx] = [old_x, old_y]
                        _add_point(lm, idx, old_x, old_y)
                        continue

                    new_score = _ordinary_count(lm)
                    if new_score < cur:
                        cur = new_score
                        improved = True
                        break
                    else:
                        _remove_point(lm, idx)
                        pts[idx] = [old_x, old_y]
                        _add_point(lm, idx, old_x, old_y)

                    if improved:
                        break

                if improved:
                    break

        return pts, cur

    # Multi-restart local search
    for restart in range(40):
        if time.time() > deadline:
            break

        if restart == 0:
            cur_pts = [p[:] for p in best_points]
            cur_score = best_score
        else:
            cur_pts = [p[:] for p in best_points]
            for _ in range(rng.randint(2, 4)):
                idx = rng.randrange(n)
                cur_pts[idx] = [rng.randint(-20, 20), rng.randint(-20, 20)]
            seen = set()
            valid = True
            for p in cur_pts:
                t = tuple(p)
                if t in seen:
                    valid = False
                    break
                seen.add(t)
            if not valid or not _not_all_collinear(cur_pts):
                continue
            cur_score = _evaluate(cur_pts)
            if cur_score > best_score + 5:
                continue

        remaining = deadline - time.time()
        if remaining < 1:
            break
        # Allocate time per restart
        alloc = min(remaining / max(1, 40 - restart), 3.0)
        cur_pts, cur_score = _local_search(cur_pts, cur_score, time.time() + alloc)

        if cur_score < best_score:
            best_score = cur_score
            best_points = [p[:] for p in cur_pts]
            if best_score <= 6:
                break

    # Final validation
    assert len(best_points) == 12
    assert all(isinstance(p, list) and len(p) == 2 for p in best_points)
    assert all(abs(p[0]) <= 10**9 and abs(p[1]) <= 10**9 for p in best_points)
    assert _not_all_collinear(best_points)
    seen = set()
    for p in best_points:
        assert tuple(p) not in seen
        seen.add(tuple(p))

    return best_points
# EVOLVE-BLOCK-END
