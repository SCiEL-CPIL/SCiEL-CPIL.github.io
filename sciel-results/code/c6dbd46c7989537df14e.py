"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as optimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin against 150s limit

    # Precompute all triples
    triples = list(combinations(range(N), 3))
    idx_i = np.array([t[0] for t in triples])
    idx_j = np.array([t[1] for t in triples])
    idx_k = np.array([t[2] for t in triples])
    n_triples = len(triples)

    def _compute_areas(x):
        """Vectorized triangle area computation."""
        x_i = x[idx_i]
        x_j = x[idx_j]
        x_k = x[idx_k]
        v1 = x_j - x_i
        v2 = x_k - x_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return 0.5 * np.abs(cross)

    def _hull_area(x):
        """Compute convex hull area."""
        try:
            hull = ConvexHull(x)
            return hull.volume
        except Exception:
            return 0.0

    def score(x_flat):
        """Returns the minimum triangle area divided by convex hull area."""
        x = x_flat.reshape(-1, 2)
        hull_area = _hull_area(x)
        if hull_area < 1e-10:
            return 0.0
        areas = _compute_areas(x)
        min_area = np.min(areas)
        return min_area / hull_area

    def soft_min_score(x_flat, beta):
        """Soft-min version of the score for smoother optimization."""
        x = x_flat.reshape(-1, 2)
        hull_area = _hull_area(x)
        if hull_area < 1e-10:
            return 0.0
        areas = _compute_areas(x)
        scaled = -areas / beta
        max_scaled = np.max(scaled)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min = -beta * log_sum_exp
        return soft_min / hull_area

    def annealed_objective(x_flat, beta):
        return -soft_min_score(x_flat, beta)

    def regular_polygon(n=N, cx=0.0, cy=0.0, r=1.0, phase=0.0):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + phase
        return np.column_stack([cx + r*np.cos(angles), cy + r*np.sin(angles)])

    def optimize_with_annealing(x_start, beta_start=0.5, beta_end=0.005,
                                  n_anneal_steps=12, max_iter_per_step=400):
        """Perform optimization with annealing of the soft-min parameter."""
        x_curr = x_start.copy()
        for step in range(n_anneal_steps):
            if time.time() > deadline - 5:
                break
            t = step / max(n_anneal_steps - 1, 1)
            beta = beta_start * (beta_end / beta_start) ** t

            def obj(x_flat):
                return annealed_objective(x_flat, beta)

            try:
                res = optimize.minimize(
                    obj, x_curr,
                    method='Nelder-Mead',
                    options={'maxiter': max_iter_per_step, 'xatol': 1e-8,
                             'fatol': 1e-8, 'adaptive': True}
                )
                x_curr = res.x
            except Exception:
                break
        return x_curr

    def polish(x_start, max_iter=3000):
        """Direct Nelder-Mead optimization on the hard objective."""
        if time.time() > deadline - 3:
            return x_start
        try:
            res = optimize.minimize(
                lambda x: -score(x), x_start,
                method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-10,
                         'fatol': 1e-10, 'adaptive': True}
            )
            return res.x
        except Exception:
            return x_start

    def polish_lgbfgs(x_start, max_iter=500):
        """L-BFGS-B polish for faster convergence near optimum."""
        if time.time() > deadline - 3:
            return x_start
        try:
            res = optimize.minimize(
                lambda x: -score(x), x_start,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            return res.x
        except Exception:
            return x_start

    best_x = None
    best_score = -np.inf

    def try_update(x_candidate):
        nonlocal best_x, best_score
        s = score(x_candidate)
        if s > best_score:
            best_score = s
            best_x = x_candidate.copy()
        return s

    # Phase 0: Regular polygon baseline
    x_reg = regular_polygon(N).flatten()
    try_update(x_reg)

    # Phase 1: Generate diverse starting configurations
    starts = []

    # Regular polygon with various phases
    for phase in [0.0, np.pi/N, 2*np.pi/N, np.pi/(2*N), np.pi/(3*N)]:
        starts.append(regular_polygon(N, phase=phase).flatten())

    # Perturbed regular polygons
    for scale in [0.01, 0.03, 0.05, 0.08, 0.12]:
        pert = regular_polygon(N) + np.random.randn(N, 2) * scale
        starts.append(pert.flatten())

    # Slightly irregular radial perturbations
    for k in [2, 3, 4, 5, 6]:
        base = regular_polygon(N)
        r = np.linalg.norm(base, axis=1)
        theta = np.arctan2(base[:, 1], base[:, 0])
        for amp in [0.03, 0.05, 0.08]:
            r_pert = r * (1 + amp * np.sin(k*theta + np.random.uniform(0, 2*np.pi)))
            starts.append(np.column_stack([r_pert*np.cos(theta), r_pert*np.sin(theta)]).flatten())

    # Two-ring structures (some points inside)
    for n_inner in [2, 3, 4, 5]:
        n_outer = N - n_inner
        angles_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + np.pi/n_inner
        angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
        for r_inner in [0.3, 0.4, 0.5, 0.6]:
            x_inner = np.column_stack([r_inner*np.cos(angles_inner), r_inner*np.sin(angles_inner)])
            x_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
            starts.append(np.vstack([x_inner, x_outer]).flatten())

    # Random points in disk (sorted by angle)
    for _ in range(5):
        angles = np.sort(np.random.uniform(0, 2*np.pi, N))
        radii = np.sqrt(np.random.uniform(0.4, 1.0, N))
        starts.append(np.column_stack([radii*np.cos(angles), radii*np.sin(angles)]).flatten())

    # Three-ring structure
    for r1, r2 in [(0.2, 0.5), (0.3, 0.6), (0.25, 0.55)]:
        angles1 = np.linspace(0, 2*np.pi, 3, endpoint=False)
        angles2 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
        angles3 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/10
        x1 = np.column_stack([r1*np.cos(angles1), r1*np.sin(angles1)])
        x2 = np.column_stack([r2*np.cos(angles2), r2*np.sin(angles2)])
        x3 = np.column_stack([np.cos(angles3), np.sin(angles3)])
        starts.append(np.vstack([x1, x2, x3]).flatten())

    # Optimize each start with annealing
    for i, x_start in enumerate(starts):
        if time.time() > deadline - 20:
            break
        try:
            x_annealed = optimize_with_annealing(
                x_start, beta_start=0.3, beta_end=0.003,
                n_anneal_steps=10, max_iter_per_step=250
            )
            try_update(x_annealed)
        except Exception:
            pass

    # Phase 2: Polish the best solution
    if best_x is not None and time.time() < deadline - 15:
        x_polished = polish(best_x, max_iter=4000)
        try_update(x_polished)

    # Phase 3: L-BFGS-B polish
    if best_x is not None and time.time() < deadline - 10:
        x_lgbfgs = polish_lgbfgs(best_x, max_iter=300)
        try_update(x_lgbfgs)

    # Phase 4: More restarts from perturbed best solution
    for i in range(8):
        if time.time() > deadline - 12:
            break
        if best_x is not None:
            x_start = (best_x.reshape(-1, 2) + np.random.randn(N, 2) * 0.015).flatten()
        else:
            x_start = regular_polygon(N).flatten()
        try:
            x_annealed = optimize_with_annealing(
                x_start, beta_start=0.15, beta_end=0.003,
                n_anneal_steps=8, max_iter_per_step=200
            )
            try_update(x_annealed)
            # Quick polish
            if time.time() < deadline - 5:
                x_polished = polish(x_annealed, max_iter=1500)
                try_update(x_polished)
        except Exception:
            pass

    # Phase 5: Final multi-method polish
    if best_x is not None and time.time() < deadline - 5:
        x_polished = polish(best_x, max_iter=5000)
        try_update(x_polished)
        if time.time() < deadline - 3:
            x_lgbfgs = polish_lgbfgs(best_x, max_iter=500)
            try_update(x_lgbfgs)
            if time.time() < deadline - 2:
                x_polished2 = polish(best_x, max_iter=3000)
                try_update(x_polished2)

    if best_x is None:
        best_x = regular_polygon(N).flatten()

    return best_x.reshape(-1, 2)

# EVOLVE-BLOCK-END
