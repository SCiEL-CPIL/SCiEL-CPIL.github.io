# EVOLVE-BLOCK-START
import time
import numpy as np


def _cross_fast(a, b, c):
    """Signed cross product (b-a) x (c-b)."""
    return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])


def _is_valid_fast(pts):
    """Check validity constraints: shape, range, distinctness, non-collinearity."""
    if pts.shape != (33, 2):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    if np.any(pts < -1e-6) or np.any(pts > 1 + 1e-6):
        return False
    # pairwise squared distance
    d2 = (pts[:, None, 0] - pts[None, :, 0]) ** 2 + (pts[:, None, 1] - pts[None, :, 1]) ** 2
    if np.any(np.triu(d2, 1) < 1e-12):
        return False
    # collinearity check
    n = 33
    for i in range(n):
        for j in range(i + 1, n):
            for l in range(j + 1, n):
                v = pts[i, 0] * (pts[j, 1] - pts[l, 1]) + pts[j, 0] * (pts[l, 1] - pts[i, 1]) + pts[l, 0] * (pts[i, 1] - pts[j, 1])
                if abs(v) < 1e-9:
                    return False
    return True


def _count_convex7_fast(pts):
    """Count convex 7-gons via dynamic program over convex chains."""
    n = len(pts)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    p = pts[order]
    C = 0
    for i in range(n):
        later = list(range(i + 1, n))
        if len(later) < 6:
            continue
        later.sort(key=lambda j: np.arctan2(p[j, 1] - p[i, 1], p[j, 0] - p[i, 0]))
        m = len(later)
        dp = [[0] * 8 for _ in range(m)]
        for j in range(m):
            dp[j][2] = 1
        for t in range(3, 7):
            for j in range(t - 1, m):
                s = 0
                for k in range(j - 1, -1, -1):
                    if dp[k][t - 1] == 0:
                        continue
                    a = p[i] if k == 0 else p[later[k - 1]]
                    b = p[later[k]]
                    c = p[later[j]]
                    if _cross_fast(a, b, c) > 1e-9:
                        s += dp[k][t - 1]
                dp[j][t] = s
        for j in range(5, m):
            if dp[j][6] == 0:
                continue
            a = p[later[j - 1]]
            b = p[later[j]]
            c = p[i]
            if _cross_fast(a, b, c) <= 1e-9:
                continue
            a = p[i]
            b = p[later[0]]
            c = p[later[1]]
            if _cross_fast(a, b, c) <= 1e-9:
                continue
            C += dp[j][6]
    return C


def _generate_zigzag(rng, n=33):
    """Generate a zigzag pattern: alternating high and low y for consecutive x."""
    xs = np.linspace(0.05, 0.95, n)
    ys = np.zeros(n)
    for i in range(n):
        if i % 2 == 0:
            ys[i] = 0.85 + rng.uniform(-0.02, 0.02)
        else:
            ys[i] = 0.15 + rng.uniform(-0.02, 0.02)
    # Add small x jitter
    xs += rng.uniform(-0.01, 0.01, n)
    return np.column_stack([xs, ys])


def _generate_clusters(rng, n=33):
    """Generate points in a few tight clusters."""
    n_clusters = 3
    pts = np.zeros((n, 2))
    centers = rng.uniform(0.2, 0.8, (n_clusters, 2))
    for i in range(n):
        c = i % n_clusters
        pts[i] = centers[c] + rng.normal(0, 0.05, 2)
    pts = np.clip(pts, 0.01, 0.99)
    return pts


def _generate_grid(rng, n=33):
    """Generate a perturbed grid pattern."""
    cols = 6
    rows = 6
    xs = np.linspace(0.1, 0.9, cols)
    ys = np.linspace(0.1, 0.9, rows)
    grid_pts = []
    for r in range(rows):
        for c in range(cols):
            if len(grid_pts) < n:
                grid_pts.append([xs[c] + rng.uniform(-0.03, 0.03),
                                 ys[r] + rng.uniform(-0.03, 0.03)])
    return np.array(grid_pts[:n])


def _hill_climb(pts, rng, deadline, step0=0.01, max_iter=50000):
    """Local search: perturb points to reduce convex 7-gons."""
    C_best = _count_convex7_fast(pts)
    best = pts.copy()
    step = step0
    no_improve = 0
    for it in range(max_iter):
        if time.time() > deadline:
            break
        idx = rng.randint(0, 33)
        delta = rng.normal(0, step, 2)
        new_pts = best.copy()
        new_pts[idx] = best[idx] + delta
        new_pts = np.clip(new_pts, 0.0, 1.0)
        if not _is_valid_fast(new_pts):
            continue
        C = _count_convex7_fast(new_pts)
        if C < C_best:
            best = new_pts
            C_best = C
            no_improve = 0
            if it % 500 == 0:
                step = max(step * 0.95, 1e-4)
        else:
            no_improve += 1
            if no_improve > 200:
                no_improve = 0
                step = max(step * 0.9, 1e-5)
                if step < 1e-5:
                    break
    return best, C_best


def _crossover(p1, p2, rng):
    """Uniform crossover between two point sets."""
    mask = rng.rand(33) > 0.5
    child = np.where(mask[:, None], p1, p2)
    return child


def _mutate(pts, rng, rate=0.3, strength=0.02):
    """Gaussian mutation on a fraction of points."""
    child = pts.copy()
    for i in range(33):
        if rng.rand() < rate:
            child[i] += rng.normal(0, strength, 2)
    child = np.clip(child, 0.0, 1.0)
    return child


def _tournament_select(pop, scores, rng, k=3):
    """Tournament selection."""
    idxs = rng.randint(0, len(pop), k)
    best_idx = idxs[np.argmin(scores[idxs])]  # lower C is better
    return pop[best_idx].copy()


def _ga(pts_init, rng, deadline, pop_size=20, max_gen=200):
    """Genetic algorithm to minimize convex 7-gons."""
    # Initialize population
    pop = [pts_init.copy()]
    for _ in range(pop_size - 1):
        p = _mutate(pts_init, rng, rate=0.5, strength=0.05)
        if _is_valid_fast(p):
            pop.append(p)
        else:
            pop.append(pts_init.copy())
    
    # Evaluate initial population
    scores = []
    for p in pop:
        if _is_valid_fast(p):
            scores.append(_count_convex7_fast(p))
        else:
            scores.append(10**9)
    
    best_idx = np.argmin(scores)
    best_pts = pop[best_idx].copy()
    best_C = scores[best_idx]
    
    for gen in range(max_gen):
        if time.time() > deadline - 60:
            break
        
        # Create next generation
        new_pop = []
        new_scores = []
        
        # Elitism: keep best
        new_pop.append(best_pts.copy())
        new_scores.append(best_C)
        
        while len(new_pop) < pop_size:
            if time.time() > deadline - 60:
                break
            p1 = _tournament_select(pop, scores, rng)
            p2 = _tournament_select(pop, scores, rng)
            child = _crossover(p1, p2, rng)
            child = _mutate(child, rng, rate=0.3, strength=0.015)
            if _is_valid_fast(child):
                C = _count_convex7_fast(child)
                new_pop.append(child)
                new_scores.append(C)
                if C < best_C:
                    best_C = C
                    best_pts = child.copy()
            else:
                new_pop.append(p1.copy())
                new_scores.append(scores[np.argmin(scores)])
        
        pop = new_pop
        scores = new_scores
    
    return best_pts, best_C


def erdos_szekeres_problem7() -> np.ndarray:
    deadline = time.time() + 580.0
    
    rng = np.random.RandomState(123)
    
    # Generate multiple diverse initial candidates
    candidates = []
    candidates.append(_generate_zigzag(rng))
    candidates.append(_generate_clusters(rng))
    candidates.append(_generate_grid(rng))
    
    # Also try a cup-cap style
    base_cup_xs = np.linspace(0.02, 0.98, 17)
    base_cup_ys = np.linspace(0.05, 0.95, 17) ** 1.6
    base_cap_xs = np.linspace(0.02, 0.98, 16)
    base_cap_ys = 1.0 - np.linspace(0.05, 0.95, 16) ** 1.6
    cup_xs = base_cup_xs + rng.uniform(-1e-4, 1e-4, 17)
    cup_ys = base_cup_ys + rng.uniform(-1e-4, 1e-4, 17)
    cap_xs = base_cap_xs + rng.uniform(-1e-4, 1e-4, 16)
    cap_ys = base_cap_ys + rng.uniform(-1e-4, 1e-4, 16)
    cup_cap = np.column_stack([np.concatenate([cup_xs, cap_xs]),
                                 np.concatenate([cup_ys, cap_ys])])
    candidates.append(cup_cap)
    
    # Evaluate all candidates
    best_pts = None
    best_C = float('inf')
    for cand in candidates:
        if _is_valid_fast(cand):
            C = _count_convex7_fast(cand)
            if C < best_C:
                best_C = C
                best_pts = cand.copy()
    
    if best_pts is None:
        best_pts = rng.rand(33, 2)
        best_C = _count_convex7_fast(best_pts)
    
    # Phase 1: Hill climb from best candidate
    best_pts, best_C = _hill_climb(best_pts, rng, deadline, step0=0.01, max_iter=30000)
    
    # Phase 2: Genetic algorithm for further improvement
    if time.time() < deadline - 120:
        ga_pts, ga_C = _ga(best_pts, rng, deadline, pop_size=15, max_gen=100)
        if ga_C < best_C:
            best_pts = ga_pts
            best_C = ga_C
    
    # Phase 3: Final hill climb
    if time.time() < deadline - 60:
        best_pts, best_C = _hill_climb(best_pts, rng, deadline, step0=0.005, max_iter=20000)
    
    # Final validity check
    if not _is_valid_fast(best_pts):
        rng2 = np.random.RandomState(0)
        best_pts = rng2.rand(33, 2)
    
    return best_pts


# EVOLVE-BLOCK-END
