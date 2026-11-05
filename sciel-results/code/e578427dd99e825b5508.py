# EVOLVE-BLOCK-START
import random, time, math, sys
from itertools import combinations

BOX = 100
N = 20

def _now():
    return time.time()

def _collinear3(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]) == 0

def _triple_count(pts):
    c = 0
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                if _collinear3(pts[i], pts[j], pts[k]):
                    c += 1
    return c

def _has_four_collinear(pts):
    for i in range(N):
        for j in range(i+1, N):
            cnt = 0
            pi, pj = pts[i], pts[j]
            for k in range(N):
                if k == i or k == j: continue
                if _collinear3(pi, pj, pts[k]):
                    cnt += 1
                    if cnt >= 2:  # total 4
                        return True
    return False

def _valid(pts):
    if len(set(map(tuple, pts))) != N:
        return False
    if any(abs(x) > BOX or abs(y) > BOX for x, y in pts):
        return False
    return not _has_four_collinear(pts)

def _max_3free(pts):
    """Exact branch-and-bound via 2-coloring: minimize max(color_class_size).
    Branch on smallest undecided hyperedge. Returns size of largest 3-free subset."""
    n = N
    # Build triples as bitmasks
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear3(pts[i], pts[j], pts[k]):
                    triples.append((1<<i)|(1<<j)|(1<<k))
    if not triples:
        return n
    
    # For each vertex, store which triples contain it
    vertex_triples = [[] for _ in range(n)]
    for idx, t in enumerate(triples):
        for v in range(n):
            if t & (1<<v):
                vertex_triples[v].append(idx)
    
    best = [n]  # upper bound on max color class size
    nodes = [0]
    CAP = 500000
    
    # State: color[v] = -1 undecided, 0 or 1 assigned
    # We want to minimize max(count_0, count_1) subject to no triple all same color
    # Equivalent: find coloring where both color classes are 3-free
    # Score = max(size of color0, size of color1)
    
    color = [-1] * n
    count = [0, 0]
    
    def search(depth):
        nodes[0] += 1
        if nodes[0] > CAP:
            return
        # Prune: if current max count already >= best, no improvement possible
        # Actually we need both counts to stay below best
        if count[0] >= best[0] or count[1] >= best[0]:
            return
        # Check if all assigned
        if depth == n:
            if max(count[0], count[1]) < best[0]:
                best[0] = max(count[0], count[1])
            return
        # Upper bound: even if we assign optimally, can we beat best?
        remaining = n - depth
        # Best case: distribute remaining evenly
        min_possible_max = max(count[0], count[1])
        # If we add all remaining to the smaller class
        smaller = min(count[0], count[1])
        larger = max(count[0], count[1])
        # We can at best make them equal-ish
        total = count[0] + count[1] + remaining
        lower_bound_max = (total + 1) // 2  # ceil(total/2)
        if lower_bound_max >= best[0]:
            return
        
        # Find smallest undecided hyperedge containing vertex 'depth'
        # Actually, branch on vertex 'depth' but use hyperedge constraint for pruning
        v = depth
        
        # Try assigning color 0
        feasible_0 = True
        for t_idx in vertex_triples[v]:
            t = triples[t_idx]
            # Check if other two vertices in this triple are both color 0
            for u in range(n):
                if u == v or not (t & (1<<u)):
                    continue
                if color[u] == 0:
                    # Need to check the third
                    pass
            # More careful: check all pairs in triple with v
            others = [u for u in range(n) if u != v and (t & (1<<u))]
            if len(others) == 2:
                if color[others[0]] == 0 and color[others[1]] == 0:
                    feasible_0 = False
                    break
        if feasible_0:
            color[v] = 0
            count[0] += 1
            search(depth + 1)
            count[0] -= 1
            color[v] = -1
        
        # Try assigning color 1
        feasible_1 = True
        for t_idx in vertex_triples[v]:
            t = triples[t_idx]
            others = [u for u in range(n) if u != v and (t & (1<<u))]
            if len(others) == 2:
                if color[others[0]] == 1 and color[others[1]] == 1:
                    feasible_1 = False
                    break
        if feasible_1:
            color[v] = 1
            count[1] += 1
            search(depth + 1)
            count[1] -= 1
            color[v] = -1
    
    # Use iterative deepening or just direct search
    # First get a quick upper bound via greedy
    def greedy_ub():
        # Greedy: assign colors to minimize max class
        c = [-1] * n
        cnt = [0, 0]
        for v in range(n):
            # Try color with smaller count first
            for col in sorted([0,1], key=lambda x: cnt[x]):
                ok = True
                for t_idx in vertex_triples[v]:
                    t = triples[t_idx]
                    others = [u for u in range(n) if u != v and (t & (1<<u))]
                    if len(others) == 2:
                        if c[others[0]] == col and c[others[1]] == col:
                            ok = False
                            break
                if ok:
                    c[v] = col
                    cnt[col] += 1
                    break
            if c[v] == -1:
                c[v] = 0
                cnt[0] += 1
        return max(cnt[0], cnt[1])
    
    ub = greedy_ub()
    best[0] = ub
    
    # Now branch and bound trying to find better
    search(0)
    return best[0]

def _construct_line_packed():
    """Construct 20 points with exactly 3 on many distinct lines.
    Use a grid-like structure but perturb to avoid 4-collinear."""
    random.seed(42)
    
    # Strategy: place points in clusters of 3 on different lines
    # 6 lines x 3 points = 18, plus 2 free points
    # Choose lines with different slopes to minimize accidental collinearities
    
    # Use parametric lines: y = m*x + b, pick integer points
    lines_def = [
        # (slope, intercept, 3 x-values)
        (1, 0, [-10, 0, 10]),
        (-1, 5, [-10, 0, 10]),
        (0, 20, [-15, 0, 15]),
        (0, -20, [-15, 0, 15]),
        (2, -30, [-5, 0, 5]),
        (-2, 30, [-5, 0, 5]),
    ]
    
    pts = []
    for m, b, xs in lines_def:
        for x in xs:
            y = m * x + b
            pts.append([x, y])
    
    # Add 2 more points that don't create 4-collinear
    pts.append([3, 7])
    pts.append([-7, -3])
    
    # Check validity and fix if needed
    if _valid(pts):
        return pts
    
    # Fallback: random search for valid line-packed config
    for attempt in range(10000):
        random.seed(attempt)
        p = []
        used_lines = set()
        for i in range(6):
            m = random.randint(-3, 3)
            b = random.randint(-40, 40)
            xs = sorted(random.sample(range(-50, 51), 3))
            line_key = (m, b)
            if line_key in used_lines:
                continue
            used_lines.add(line_key)
            for x in xs:
                y = m * x + b
                if abs(y) <= BOX:
                    p.append([x, y])
        while len(p) < 18:
            x, y = random.randint(-50, 50), random.randint(-50, 50)
            p.append([x, y])
        # Add 2 more
        p.append([random.randint(-50,50), random.randint(-50,50)])
        p.append([random.randint(-50,50), random.randint(-50,50)])
        p = p[:20]
        if _valid(p):
            return p
    
    # Ultimate fallback: parabola
    return [[x, x*x] for x in range(-10, 10)]

def _local_search(pts, deadline):
    """Improve via local search: move points to increase triple count while staying valid."""
    cur_triples = _triple_count(pts)
    best_triples = cur_triples
    best_pts = [list(p) for p in pts]
    best_score = _max_3free(pts)
    best_score_pts = [list(p) for p in pts]
    
    T = 1.0
    it = 0
    while time.time() < deadline:
        it += 1
        i = random.randrange(N)
        r = random.random()
        np_ = [list(p) for p in pts]
        if r < 0.5:
            np_[i] = [random.randint(-BOX, BOX), random.randint(-BOX, BOX)]
        elif r < 0.75:
            dx = random.randint(-6, 6)
            dy = random.randint(-6, 6)
            np_[i][0] = max(-BOX, min(BOX, np_[i][0] + dx))
            np_[i][1] = max(-BOX, min(BOX, np_[i][1] + dy))
        else:
            j = random.randrange(N)
            np_[i], np_[j] = np_[j], np_[i]
        
        if not _valid(np_):
            continue
        
        tc = _triple_count(np_)
        if tc > cur_triples or random.random() < math.exp(-(cur_triples - tc) / T):
            pts = np_
            cur_triples = tc
            if tc > best_triples:
                best_triples = tc
                best_pts = [list(p) for p in np_]
                # Evaluate true score periodically
                if it % 50 == 0:
                    st = _max_3free(np_)
                    if st < best_score:
                        best_score = st
                        best_score_pts = [list(p) for p in np_]
        T = max(0.2, T * 0.999)
        
        if best_score <= 10:
            break
    
    # Final evaluation
    for cand in (best_pts, best_score_pts, pts):
        if _valid(cand):
            st = _max_3free(cand)
            if st < best_score:
                best_score = st
                best_score_pts = [list(p) for p in cand]
    
    return best_score_pts

def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin
    
    # Phase 1: Constructive line-packed initialization
    pts = _construct_line_packed()
    
    # Phase 2: Local search to improve
    result = _local_search(pts, deadline)
    
    if result is not None and _valid(result):
        return result
    
    # Fallback
    return [[x, x*x] for x in range(-10, 10)]
# EVOLVE-BLOCK-END
