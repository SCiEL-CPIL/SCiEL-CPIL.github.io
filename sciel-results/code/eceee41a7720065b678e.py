"""Optimized design: maximize collinear triples (no four collinear), then exact score."""
import time
import random
from math import gcd

DEADLINE = 110.0  # safety margin vs 120s wall clock
N = 20
MAXC = 100


def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = gcd(abs(dx), abs(dy))
    if g == 0:
        return (0, 0, 0)
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    # a = -dy, b = dx, c = dy*x1 - dx*y1
    a = -dy
    b = dx
    c = dy * x1 - dx * y1
    g2 = gcd(abs(a), gcd(abs(b), abs(c)))
    if g2 != 0:
        a //= g2
        b //= g2
        c //= g2
    return (a, b, c)


def _count_lines(pts):
    """Return (num_lines_with_3plus, max_line_count)."""
    lines = {}
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(pts[i], pts[j])
            lines[k] = lines.get(k, 0) + 1
    cnt3 = 0
    mx = 0
    for v in lines.values():
        if v >= 3:
            cnt3 += 1
        if v > mx:
            mx = v
    return cnt3, mx


def _valid(pts):
    _, mx = _count_lines(pts)
    return mx <= 3


def _triple_edges(pts):
    """Set of frozensets of 3 indices that are collinear."""
    lines = {}
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(pts[i], pts[j])
            lines.setdefault(k, set()).add(i)
            lines[k].add(j)
    edges = set()
    for s in lines.values():
        if len(s) >= 3:
            idx = sorted(s)
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    for c in range(b + 1, len(idx)):
                        edges.add(frozenset((idx[a], idx[b], idx[c])))
    return edges


def _max_independent(edges):
    """Exact largest subset with no collinear triple via branch and bound."""
    m = len(edges)
    if m == 0:
        return 20
    best = [0]

    def bnb(remaining, size):
        if size + len(remaining) <= best[0]:
            return
        if not remaining:
            if size > best[0]:
                best[0] = size
            return
        # pick a vertex in the smallest edge to branch
        e = min(remaining, key=len)
        for v in e:
            new_rem = set()
            for ed in remaining:
                if v in ed:
                    continue
                new_rem.add(ed)
            bnb(new_rem, size + 1)
            if size + len(new_rem) <= best[0]:
                return
        # also option: exclude nothing (all vertices) is already covered by picking one
        # We must also consider NOT including any vertex of e? Standard: include one, or skip.
        # Simpler exact: try include each v, and the bound handles the rest.

    bnb(set(edges), 0)
    return best[0]


def _score(pts):
    edges = _triple_edges(pts)
    return _max_independent(edges)


def _neighbor_swap(pts):
    i = random.randrange(N)
    j = random.randrange(N)
    if i == j:
        return pts
    np_ = pts[:]
    np_[i], np_[j] = np_[j], np_[i]
    return np_


def _random_point():
    x = random.randint(-MAXC, MAXC)
    y = random.randint(-MAXC, MAXC)
    return [x, y]


def construct():
    t0 = time.time()
    rng = random.Random(12345)

    # Start from a structured seed: a 4x5 grid (many triples, but has 4-collinear).
    # We'll fix 4-collinear by perturbing, but first build a valid base via parabola
    # then improve by adding collinearity.
    base = [[x, x * x] for x in range(-10, 10)]
    base = [p[:] for p in base]
    base_score = _score(base)
    best = base[:]
    best_score = base_score

    # Try to improve by local search on the objective:
    #   maximize number of 3-collinear lines (proxy for smaller independent set)
    #   subject to no four collinear.
    cur = base[:]
    cur_cnt, _ = _count_lines(cur)
    cur_score = base_score

    improved = True
    while time.time() - t0 < DEADLINE:
        # propose a swap
        cand = _neighbor_swap(cur)
        if not _valid(cand):
            continue
        c_cnt, c_mx = _count_lines(cand)
        # objective: more 3+ lines is better (lower independent set)
        if c_cnt > cur_cnt:
            cur = cand
            cur_cnt = c_cnt
            # recompute exact score periodically
            if cur_cnt != cur_score:
                pass
            s = _score(cur)
            if s < best_score:
                best_score = s
                best = cur[:]
            cur_score = s
        elif c_cnt == cur_cnt:
            # tie: maybe still reduce score; check occasionally
            if rng.random() < 0.1:
                s = _score(cand)
                if s < best_score:
                    best_score = s
                    best = cand[:]
                cur = cand
                cur_score = s

        # occasionally do a random restart with a different structured base
        if rng.random() < 0.02 and time.time() - t0 < DEADLINE - 20:
            # grid-like base with perturbation to break 4-collinear
            cand2 = []
            for i in range(N):
                gx = (i % 5) - 2
                gy = (i // 5) - 2
                px = gx * 20 + rng.randint(-3, 3)
                py = gy * 20 + rng.randint(-3, 3)
                if abs(px) > MAXC:
                    px = MAXC if px > 0 else -MAXC
                if abs(py) > MAXC:
                    py = MAXC if py > 0 else -MAXC
                cand2.append([px, py])
            # dedupe
            seen = set()
            ded = []
            for p in cand2:
                if (p[0], p[1]) not in seen:
                    seen.add((p[0], p[1]))
                    ded.append(p)
            while len(ded) < N:
                p = _random_point()
                if (p[0], p[1]) not in seen:
                    seen.add((p[0], p[1]))
                    ded.append(p)
            if _valid(ded):
                c_cnt2, _ = _count_lines(ded)
                if c_cnt2 > cur_cnt:
                    cur = ded
                    cur_cnt = c_cnt2
                    s = _score(cur)
                    if s < best_score:
                        best_score = s
                        best = cur[:]
                    cur_score = s

    # final exact score of best
    final_score = _score(best)
    if final_score < best_score:
        best_score = final_score
    return best
