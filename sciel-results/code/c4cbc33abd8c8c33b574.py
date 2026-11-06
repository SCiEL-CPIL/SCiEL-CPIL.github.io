"""Structural change: polar-coordinate DE + L-BFGS-B with symmetry-breaking perturbations."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

_DEADLINE = 150.0
_EPS = 1e-12


def _diameter(pts):
    n = len(pts)
    d = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            d = max(d, dx * dx + dy * dy)
    return math.sqrt(d)


def _diameter_sq(pts):
    n = len(pts)
    d = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            d = max(d, dx * dx + dy * dy)
    return d


def _log_product(pts):
    """Sum of log distances over ordered pairs (i != j)."""
    n = len(pts)
    s = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            d2 = dx * dx + dy * dy
            if d2 < _EPS:
                return -1e18
            s += 0.5 * math.log(d2)
    return s


def _score_from_log_product_and_diameter(log_prod, diam):
    """Score = Delta / n^n where Delta = product over ordered pairs after scaling to diam=2."""
    n = 8
    if diam < _EPS:
        return 0.0
    # log_prod is sum of log(d_ij) over ordered pairs (i≠j)
    # After scaling to diam=2: each distance *= 2/diam
    # log_scaled = log_prod + n*(n-1)*log(2/diam)
    log_scaled = log_prod + n * (n - 1) * math.log(2.0 / diam)
    log_score = log_scaled - n * n * math.log(n)
    return math.exp(log_score)


def _score(pts):
    diam = _diameter(pts)
    if diam < _EPS:
        return 0.0
    log_prod = _log_product(pts)
    return _score_from_log_product_and_diameter(log_prod, diam)


def _objective_polar(x):
    """Objective in polar coordinates: x = [r0, r1, ..., r7, a0, a1, ..., a7]
    Maximize log product of distances (equivalent to minimizing negative).
    We work in polar coords to naturally enforce distinctness and explore
    radial/angle configurations.
    """
    n = 8
    rs = x[:n]
    as_ = x[n:]
    # Build cartesian
    pts = np.empty((n, 2))
    for i in range(n):
        pts[i, 0] = rs[i] * math.cos(as_[i])
        pts[i, 1] = rs[i] * math.sin(as_[i])
    # Compute log product of all ordered pair distances
    s = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            d2 = dx * dx + dy * dy
            if d2 < 1e-20:
                return 1e18
            s += 0.5 * math.log(d2)
    # s = sum over ordered pairs of log(d_ij) = log(product)
    # We want to maximize this, so minimize -s
    return -s


def _objective_cart(x, penalty_weight=0.0):
    """Minimize negative log product with soft penalty for non-unit diameter."""
    n = 8
    pts = x.reshape(n, 2)
    # diameter squared
    d2 = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            d2 = max(d2, dx * dx + dy * dy)
    diam = math.sqrt(d2) if d2 > 0 else _EPS
    # log product over ordered pairs
    s = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dx = pts[i, 0] - pts[j][0]
            dy = pts[i, 1] - pts[j][1]
            dd = dx * dx + dy * dy
            if dd < 1e-20:
                return 1e18
            s += 0.5 * math.log(dd)
    # s = log(product over ordered pairs)
    # penalty to keep diameter near 2 (normalization)
    pen = penalty_weight * (diam - 2.0) ** 2
    return -s + pen


def _refine(pts0, time_budget):
    """Local refinement using L-BFGS-B in cartesian coords with increasing penalty."""
    n = 8
    x0 = np.array(pts0, dtype=float).reshape(n * 2)
    diam = _diameter(pts0)
    if diam > _EPS:
        x0 = x0 * (2.0 / diam)
    best_x = x0.copy()
    best_score = _score(best_x.reshape(n, 2).tolist())

    t_start = time.time()
    penalties = [0.001, 0.01, 0.1, 1.0, 10.0]
    for pw in penalties:
        if time.time() - t_start > time_budget:
            break
        try:
            res = minimize(
                _objective_cart, x0, args=(pw,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-10}
            )
            if res.fun < 1e17:
                cand = res.x.reshape(n, 2).tolist()
                sc = _score(cand)
                if sc > best_score:
                    best_score = sc
                    best_x = res.x.copy()
                x0 = res.x.copy()
        except Exception:
            break
    return best_x.reshape(n, 2).tolist(), best_score


def _make_polar_init(rng):
    """Generate a random polar-coordinate initialization.
    Returns array of shape (16,) = [r0..r7, a0..a7]
    """
    n = 8
    # Radii: mix of small and large to encourage diverse structures
    rs = np.zeros(n)
    for i in range(n):
        rs[i] = rng.uniform(0.3, 1.5)
    # Angles: start near regular octagon with perturbation
    base_ang = 2 * math.pi * np.arange(n) / n
    # Add random perturbation
    pert = rng.normal(0, 0.3, n)
    as_ = base_ang + pert
    # Also try some structured inits
    choice = rng.integers(0, 4)
    if choice == 0:
        # Two rings: 4 points at radius ~1, 4 at radius ~0.5
        for i in range(4):
            rs[i] = 1.0 + rng.normal(0, 0.1)
            rs[i + 4] = 0.5 + rng.normal(0, 0.1)
        as_ = 2 * math.pi * np.arange(n) / n + rng.normal(0, 0.2, n)
    elif choice == 1:
        # Alternating radii
        for i in range(n):
            rs[i] = 1.0 if i % 2 == 0 else 0.6
            rs[i] += rng.normal(0, 0.05)
        as_ = 2 * math.pi * np.arange(n) / n + rng.normal(0, 0.15, n)
    elif choice == 2:
        # Three groups: 3+3+2
        for i in range(3):
            rs[i] = 1.2 + rng.normal(0, 0.05)
        for i in range(3, 6):
            rs[i] = 0.7 + rng.normal(0, 0.05)
        for i in range(6, 8):
            rs[i] = 0.3 + rng.normal(0, 0.05)
        as_ = 2 * math.pi * np.arange(n) / n + rng.normal(0, 0.25, n)
    # Clamp radii to positive
    rs = np.maximum(rs, 0.05)
    x = np.concatenate([rs, as_])
    return x


def _polar_to_cart(polar_x):
    n = 8
    rs = polar_x[:n]
    as_ = polar_x[n:]
    pts = np.empty((n, 2))
    for i in range(n):
        pts[i, 0] = rs[i] * math.cos(as_[i])
        pts[i, 1] = rs[i] * math.sin(as_[i])
    return pts


def construct():
    t_start = time.time()
    deadline = _DEADLINE - 5.0  # safety margin
    rng = np.random.default_rng(42)

    # Regular octagon as baseline
    octagon = [[math.cos(2 * math.pi * k / 8), math.sin(2 * math.pi * k / 8)] for k in range(8)]
    best_pts = octagon
    best_score = _score(octagon)

    # Phase 1: Differential Evolution in polar coordinates
    # Bounds for polar coords: r in [0.05, 3.0], angle in [0, 2*pi]
    bounds = [(0.05, 3.0)] * 8 + [(0.0, 2 * math.pi)] * 8
    de_time_budget = 40.0  # seconds for DE phase

    def _de_objective(polar_x):
        if time.time() - t_start > deadline:
            return 1e18
        return _objective_polar(polar_x)

    try:
        # Use DE with moderate population
        de_res = differential_evolution(
            _de_objective,
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=True,
            updating='deferred'
        )
        if de_res.fun < 1e17:
            de_cart = _polar_to_cart(de_res.x)
            de_score = _score(de_cart.tolist())
            if de_score > best_score:
                best_score = de_score
                best_pts = de_cart.tolist()
    except Exception:
        pass

    # Phase 2: Multiple local refinements from diverse starts
    n_local_trials = 8
    remaining_time = deadline - (time.time() - t_start)
    if remaining_time > 10:
        per_trial = remaining_time / (n_local_trials + 1)
        for trial in range(n_local_trials):
            if time.time() - t_start > deadline:
                break
            # Generate diverse polar initializations
            polar_x = _make_polar_init(rng)
            cart = _polar_to_cart(polar_x)
            # Also try refining the current best with small perturbations
            if trial % 2 == 0:
                # Perturb best solution
                pert_amt = 0.05 * (1.0 - trial / n_local_trials)
                cart = np.array(best_pts) + rng.normal(0, pert_amt, (8, 2))
            # Ensure distinct
            ok = True
            for i in range(8):
                for j in range(i + 1, 8):
                    dx = cart[i, 0] - cart[j, 0]
                    dy = cart[i, 1] - cart[j, 1]
                    if dx * dx + dy * dy < 1e-8:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            try:
                refined, sc = _refine(cart.tolist(), per_trial)
                if sc > best_score:
                    best_score = sc
                    best_pts = refined
            except Exception:
                pass

    # Phase 3: Final polish
    if time.time() - t_start < deadline - 5.0:
        remaining = deadline - 5.0 - (time.time() - t_start)
        if remaining > 5.0:
            try:
                refined, sc = _refine(best_pts, remaining)
                if sc > best_score:
                    best_score = sc
                    best_pts = refined
            except Exception:
                pass

    # Ensure coordinates are within bounds
    result = []
    for p in best_pts:
        x = float(p[0])
        y = float(p[1])
        if abs(x) >= 1e6:
            x = math.copysign(1e6 - 1.0, x)
        if abs(y) >= 1e6:
            y = math.copysign(1e6 - 1.0, y)
        result.append([x, y])
    return result
