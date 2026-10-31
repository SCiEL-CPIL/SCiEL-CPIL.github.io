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
                    options={'maxiter': max_iter_per_step, 'xatol': 1e-7,
                             'fatol': 1e-7, 'adaptive': True}
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

    # Phase 1: Annealed optimization from multiple diverse starts
    starts = []

    # Regular polygon with various phases
    for phase in [0.0, np.pi/N, 2*np.pi/N, np.pi/(2*N)]:
        starts.append(regular_polygon(N, phase=phase).flatten())

    # Perturbed regular polygons
    for scale in [0.02, 0.05, 0.1]:
        pert = regular_polygon(N) + np.random.randn(N, 2) * scale
        starts.append(pert.flatten())

    # Slightly irregular radial perturbations
    for k in [2, 3, 4, 5]:
        base = regular_polygon(N)
        r = np.linalg.norm(base, axis=1)
        theta = np.arctan2(base[:, 1], base[:, 0])
        r_pert = r * (1 + 0.05 * np.sin(k*theta + np.random.uniform(0, 2*np.pi)))
        starts.append(np.column_stack([r_pert*np.cos(theta), r_pert*np.sin(theta)]).flatten())

    # Two-ring structures (some points inside)
    for n_inner in [3, 4, 5]:
        n_outer = N - n_inner
        angles_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False)
        angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False) + np.pi/n_outer
        x_inner = np.column_stack([0.5*np.cos(angles_inner), 0.5*np.sin(angles_inner)])
        x_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        starts.append(np.vstack([x_inner, x_outer]).flatten())

    # Random points in disk (sorted by angle)
    for _ in range(3):
        angles = np.sort(np.random.uniform(0, 2*np.pi, N))
        radii = np.sqrt(np.random.uniform(0.5, 1.0, N))
        starts.append(np.column_stack([radii*np.cos(angles), radii*np.sin(angles)]).flatten())

    # Optimize each start with annealing
    for i, x_start in enumerate(starts):
        if time.time() > deadline - 15:
            break
        try:
            x_annealed = optimize_with_annealing(
                x_start, beta_start=0.4, beta_end=0.005,
                n_anneal_steps=10, max_iter_per_step=300
            )
            try_update(x_annealed)
        except Exception:
            pass

    # Phase 2: Polish the best solution
    if best_x is not None and time.time() < deadline - 10:
        x_polished = polish(best_x, max_iter=3000)
        try_update(x_polished)

    # Phase 3: More restarts from perturbed best solution
    for i in range(5):
        if time.time() > deadline - 10:
            break
        if best_x is not None:
            x_start = (best_x.reshape(-1, 2) + np.random.randn(N, 2) * 0.02).flatten()
        else:
            x_start = regular_polygon(N).flatten()
        try:
            x_annealed = optimize_with_annealing(
                x_start, beta_start=0.2, beta_end=0.005,
                n_anneal_steps=8, max_iter_per_step=200
            )
            try_update(x_annealed)
            # Quick polish
            if time.time() < deadline - 5:
                x_polished = polish(x_annealed, max_iter=1000)
                try_update(x_polished)
        except Exception:
            pass

    # Phase 4: Final polish
    if best_x is not None and time.time() < deadline - 3:
        x_polished = polish(best_x, max_iter=5000)
        try_update(x_polished)

    if best_x is None:
        best_x = regular_polygon(N).flatten()

    return best_x.reshape(-1, 2)

# EVOLVE-BLOCK-END
