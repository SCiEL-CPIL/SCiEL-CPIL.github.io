"""Circle packing in unit square: 26 disjoint circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
EPS = 1e-8
DEADLINE = 145.0  # safety margin before 150s kill


def _penalty_objective(params, penalty=1000.0):
    """Penalty objective: minimize -sum(r) + penalty*(overlaps + wall violations)."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = np.maximum(p[:, 2], 1e-6)

    score = -np.sum(r)

    # Wall boundary violations
    wx = np.maximum(0.0, r - x) + np.maximum(0.0, r - (1.0 - x))
    wy = np.maximum(0.0, r - y) + np.maximum(0.0, r - (1.0 - y))
    wall = np.sum(wx) + np.sum(wy)

    # Pairwise overlap violations
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    overlap = np.maximum(0.0, (r[iu[0]] + r[iu[1]]) - dist)
    pair = np.sum(overlap)

    return score + penalty * (pair + wall)


def _delaunay_edges(params):
    """Get Delaunay triangulation edges for circle centers."""
    p = params.reshape(N, 3)
    centers = p[:, :2]
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        return list(edges)
    except Exception:
        # Fallback to all pairs if Delaunay fails
        return [(i, j) for i in range(N) for j in range(i+1, N)]


def _contact_graph_objective(params, edges, penalty=10000.0):
    """Optimize over contact graph edges only (Delaunay neighbors)."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = np.maximum(p[:, 2], 1e-6)

    score = -np.sum(r)

    # Wall boundary violations
    wx = np.maximum(0.0, r - x) + np.maximum(0.0, r - (1.0 - x))
    wy = np.maximum(0.0, r - y) + np.maximum(0.0, r - (1.0 - y))
    wall = np.sum(wx) + np.sum(wy)

    # Contact graph overlap violations (sparse)
    pair = 0.0
    for i, j in edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx * dx + dy * dy)
        ov = (r[i] + r[j]) - dist
        if ov > 0:
            pair += ov

    return score + penalty * (pair + wall)


def _contact_graph_grad(params, edges, penalty=10000.0):
    """Gradient for contact graph objective."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = np.maximum(p[:, 2], 1e-6)
    g = np.zeros_like(params)

    # d(-sum r)/dr = -1
    g[2::3] = -1.0

    # Wall contributions
    for i in range(N):
        if r[i] - x[i] > 0:
            g[3*i] += penalty * (-1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - (1.0 - x[i]) > 0:
            g[3*i] += penalty * (+1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - y[i] > 0:
            g[3*i+1] += penalty * (-1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - (1.0 - y[i]) > 0:
            g[3*i+1] += penalty * (+1.0)
            g[3*i+2] += penalty * (+1.0)

    # Contact graph pairwise
    for i, j in edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx * dx + dy * dy)
        dist_safe = max(dist, 1e-10)
        ov = (r[i] + r[j]) - dist
        if ov > 0:
            adx = dx / dist_safe
            ady = dy / dist_safe
            g[3*i] += penalty * (-adx)
            g[3*j] += penalty * (+adx)
            g[3*i+1] += penalty * (-ady)
            g[3*j+1] += penalty * (+ady)
            g[3*i+2] += penalty * (+1.0)
            g[3*j+2] += penalty * (+1.0)

    return g


def _penalty_grad(params, penalty=1000.0):
    """Analytic gradient of penalty objective."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = np.maximum(p[:, 2], 1e-6)
    n = N
    g = np.zeros_like(params)

    # d(-sum r)/dr = -1
    g[2::3] = -1.0

    # Wall contributions
    for i in range(n):
        if r[i] - x[i] > 0:
            g[3*i] += penalty * (-1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - (1.0 - x[i]) > 0:
            g[3*i] += penalty * (+1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - y[i] > 0:
            g[3*i+1] += penalty * (-1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - (1.0 - y[i]) > 0:
            g[3*i+1] += penalty * (+1.0)
            g[3*i+2] += penalty * (+1.0)

    # Pairwise
    iu = np.triu_indices(n, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    dist_safe = np.maximum(dist, 1e-10)
    ov = (r[iu[0]] + r[iu[1]]) - dist
    active = ov > 0
    if np.any(active):
        ax = iu[0][active]
        ay = iu[1][active]
        adx = dx[active] / dist_safe[active]
        ady = dy[active] / dist_safe[active]
        for k in range(len(ax)):
            i = ax[k]
            j = ay[k]
            g[3*i] += penalty * (-adx[k])
            g[3*j] += penalty * (+adx[k])
            g[3*i+1] += penalty * (-ady[k])
            g[3*j+1] += penalty * (+ady[k])
            g[3*i+2] += penalty * (+1.0)
            g[3*j+2] += penalty * (+1.0)

    return g


def _grid_initial(seed):
    """Grid of equal circles with small jitter."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4))
    cells = cells[:N]
    arr = np.array(cells, dtype=float).reshape(-1)
    arr += rng.uniform(-1e-5, 1e-5, size=arr.size)
    return arr


def _random_initial(seed):
    """Random positions with modest radii."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.15, 0.85, size=N)
    y = rng.uniform(0.15, 0.85, size=N)
    r = np.full(N, 0.08)
    params = np.empty(3 * N)
    params[0::3] = x
    params[1::3] = y
    params[2::3] = r
    return params


def _bounds():
    lo = np.zeros(3 * N)
    hi = np.zeros(3 * N)
    for i in range(N):
        lo[3*i] = 0.0
        hi[3*i] = 1.0
        lo[3*i+1] = 0.0
        hi[3*i+1] = 1.0
        lo[3*i+2] = 1e-6
        hi[3*i+2] = 0.5
    return (lo, hi)


def _stochastic_refine(params, deadline, rng, n_iters=6000):
    """Simulated annealing style local search with adaptive step size."""
    val = _penalty_objective(params, penalty=5000.0)
    best = params.copy()
    best_val = val
    T = 0.03
    step = 0.015
    accept_count = 0
    total_count = 0

    for it in range(n_iters):
        if time.time() > deadline:
            break
        i = rng.randint(N)
        cand = params.copy()
        cand[3*i] = params[3*i] + rng.uniform(-step, step)
        cand[3*i+1] = params[3*i+1] + rng.uniform(-step, step)
        cand[3*i+2] = params[3*i+2] + rng.uniform(-step*0.4, step*0.4)
        cand[3*i] = np.clip(cand[3*i], 0.0, 1.0)
        cand[3*i+1] = np.clip(cand[3*i+1], 0.0, 1.0)
        cand[3*i+2] = np.clip(cand[3*i+2], 1e-6, 0.5)

        new_val = _penalty_objective(cand, penalty=5000.0)
        d = new_val - val
        total_count += 1
        if d < 0 or rng.rand() < np.exp(-d / max(T, 1e-8)):
            params = cand
            val = new_val
            accept_count += 1
            if val < best_val:
                best_val = val
                best = params.copy()
        T *= 0.9992
        step *= 0.9997

        # Adaptive step: if acceptance rate too low, increase step; if too high, decrease
        if total_count % 200 == 0 and total_count > 0:
            rate = accept_count / total_count
            if rate < 0.1:
                step *= 1.3
            elif rate > 0.7:
                step *= 0.7
            accept_count = 0
            total_count = 0

    return best, best_val


def _vectorized_local_search(params, deadline, rng, n_iters=10000):
    """Fast vectorized local search: try multiple perturbations per iteration."""
    p = params.reshape(N, 3).copy()
    best_score = np.sum(np.maximum(p[:, 2], 1e-6))
    best_p = p.copy()
    T = 0.02
    step = 0.01

    for it in range(n_iters):
        if time.time() > deadline:
            break

        # Try 3 random circles per iteration
        indices = rng.randint(N, size=3)
        for i in indices:
            cand = p.copy()
            cand[i, 0] = p[i, 0] + rng.uniform(-step, step)
            cand[i, 1] = p[i, 1] + rng.uniform(-step, step)
            cand[i, 2] = p[i, 2] + rng.uniform(-step*0.3, step*0.3)
            cand[i, 0] = np.clip(cand[i, 0], 0.0, 1.0)
            cand[i, 1] = np.clip(cand[i, 1], 0.0, 1.0)
            cand[i, 2] = np.clip(cand[i, 2], 1e-6, 0.5)

            # Quick validity check: only check this circle against others
            valid = True
            r_i = cand[i, 2]
            x_i, y_i = cand[i, 0], cand[i, 1]

            # Wall check
            if r_i > x_i + 1e-9 or r_i > 1.0 - x_i + 1e-9 or r_i > y_i + 1e-9 or r_i > 1.0 - y_i + 1e-9:
                continue

            # Check against other circles
            for j in range(N):
                if j == i:
                    continue
                dx = x_i - p[j, 0]
                dy = y_i - p[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if r_i + p[j, 2] > dist + 1e-9:
                    valid = False
                    break

            if not valid:
                continue

            score = np.sum(np.maximum(cand[:, 2], 1e-6))
            if score > best_score:
                best_score = score
                best_p = cand.copy()
                p = cand.copy()

        T *= 0.999
        step *= 0.9995

    return best_p.reshape(-1)


def _enforce_validity(params):
    """Enforce all constraints: walls and pairwise non-overlap by shrinking radii."""
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

    # Clamp to walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    # Iteratively resolve overlaps by shrinking the smaller circle
    for _ in range(500):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 0
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            # Shrink the smaller circle by ov (or split if equal)
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                # If still overlapping, shrink the other
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
        # Re-clamp to walls
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    return p


def _score_valid(p):
    """Compute sum of radii for a valid configuration."""
    return np.sum(p[:, 2])


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    lo, hi = _bounds()
    bounds_list = list(zip(lo, hi))

    best_params = None
    best_score = -np.inf

    # Multi-start with different initializations
    initializers = [
        lambda s: _grid_initial(s),
        lambda s: _random_initial(s),
    ]

    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828]
    rng_master = np.random.RandomState(0)

    for seed in seeds:
        if time.time() > deadline - 5:
            break
        for init_fn in initializers:
            if time.time() > deadline - 5:
                break
            params = init_fn(seed)
            rng = np.random.RandomState(seed + 1000)

            # Multi-penalty L-BFGS-B with full pairwise
            for pen in [100.0, 500.0, 2000.0]:
                if time.time() > deadline - 3:
                    break

                def obj(p, pen=pen):
                    return _penalty_objective(p, penalty=pen)

                def grad(p, pen=pen):
                    return _penalty_grad(p, penalty=pen)

                try:
                    res = minimize(obj, params, jac=grad, method='L-BFGS-B',
                                   bounds=bounds_list,
                                   options={'maxiter': 150, 'maxfun': 200,
                                            'ftol': 1e-12, 'gtol': 1e-8})
                    params = res.x
                except Exception:
                    pass

            # Delaunay-based contact graph optimization
            if time.time() < deadline - 2:
                edges = _delaunay_edges(params)
                for pen in [2000.0, 10000.0]:
                    if time.time() > deadline - 2:
                        break

                    def cg_obj(p, pen=pen, edges=edges):
                        return _contact_graph_objective(p, edges, penalty=pen)

                    def cg_grad(p, pen=pen, edges=edges):
                        return _contact_graph_grad(p, edges, penalty=pen)

                    try:
                        res = minimize(cg_obj, params, jac=cg_grad, method='L-BFGS-B',
                                       bounds=bounds_list,
                                       options={'maxiter': 200, 'maxfun': 300,
                                                'ftol': 1e-12, 'gtol': 1e-8})
                        params = res.x
                    except Exception:
                        pass

            # Stochastic refinement
            if time.time() < deadline - 2:
                params, _ = _stochastic_refine(params, deadline - 1, rng, n_iters=3000)

            # Vectorized local search
            if time.time() < deadline - 1:
                params = _vectorized_local_search(params, deadline - 0.5, rng, n_iters=5000)

            # Final polish with high penalty
            if time.time() < deadline - 1:
                try:
                    res = minimize(lambda p: _penalty_objective(p, penalty=5000.0),
                                   params,
                                   jac=lambda p: _penalty_grad(p, penalty=5000.0),
                                   method='L-BFGS-B', bounds=bounds_list,
                                   options={'maxiter': 100, 'maxfun': 150,
                                            'ftol': 1e-12, 'gtol': 1e-8})
                    params = res.x
                except Exception:
                    pass

            # Enforce validity and score
            valid_p = _enforce_validity(params)
            score = _score_valid(valid_p)

            if score > best_score:
                best_score = score
                best_params = valid_p.copy()

    # If we still have time, do one more round of refinement on the best
    if best_params is not None and time.time() < deadline - 2:
        rng = np.random.RandomState(7777)
        refined, _ = _stochastic_refine(best_params.reshape(-1), deadline - 1, rng, n_iters=4000)
        valid_refined = _enforce_validity(refined)
        score_refined = _score_valid(valid_refined)
        if score_refined > best_score:
            best_params = valid_refined
            best_score = score_refined

        # Delaunay polish on best
        if time.time() < deadline - 1:
            edges = _delaunay_edges(best_params.reshape(-1))
            try:
                res = minimize(lambda p: _contact_graph_objective(p, edges, penalty=20000.0),
                               best_params.reshape(-1),
                               jac=lambda p: _contact_graph_grad(p, edges, penalty=20000.0),
                               method='L-BFGS-B', bounds=bounds_list,
                               options={'maxiter': 150, 'maxfun': 250,
                                        'ftol': 1e-12, 'gtol': 1e-8})
                polished = _enforce_validity(res.x)
                polished_score = _score_valid(polished)
                if polished_score > best_score:
                    best_params = polished
                    best_score = polished_score
            except Exception:
                pass

        # Final L-BFGS polish
        if time.time() < deadline - 0.5:
            try:
                res = minimize(lambda p: _penalty_objective(p, penalty=10000.0),
                               best_params.reshape(-1),
                               jac=lambda p: _penalty_grad(p, penalty=10000.0),
                               method='L-BFGS-B', bounds=bounds_list,
                               options={'maxiter': 50, 'maxfun': 80,
                                        'ftol': 1e-12, 'gtol': 1e-8})
                final_p = _enforce_validity(res.x)
                final_score = _score_valid(final_p)
                if final_score > best_score:
                    best_params = final_p
            except Exception:
                pass

    if best_params is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_params
