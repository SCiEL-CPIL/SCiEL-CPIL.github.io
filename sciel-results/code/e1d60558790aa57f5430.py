"""Fifty equal circles in a square – model-and-optimize packing optimizer."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist

N = 50
HALF = 0.5
DEADLINE = 145.0  # Safety margin before 150s limit


def _score(c):
    """Compute the exact radius score for given centres (N,2)."""
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    if np.any(c < -HALF) or np.any(c > HALF):
        return 0.0
    d = pdist(c)
    min_pair = d.min() / 2.0
    dist_walls = np.min(np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ]), axis=0)
    min_wall = dist_walls.min()
    return min(min_pair, min_wall)


def _hexagonal_init(rng, jitter=0.005):
    """Generate a hexagonal close packing initial layout."""
    points = []
    r = 0.072
    dx = 2 * r
    dy = r * np.sqrt(3)
    
    max_x = HALF - r
    max_y = HALF - r
    
    row = 0
    y = -max_y
    while y <= max_y + 1e-9:
        x_start = -max_x
        if row % 2 == 1:
            x_start += dx / 2
        x = x_start
        while x <= max_x + 1e-9:
            points.append([x + rng.uniform(-jitter, jitter), y + rng.uniform(-jitter, jitter)])
            x += dx
        y += dy
        row += 1
        if len(points) > N * 3:
            break
            
    pts = np.array(points)
    if len(pts) < N:
        extra = rng.uniform(-HALF + r, HALF - r, size=(N - len(pts), 2))
        pts = np.vstack([pts, extra])
    
    dists_to_center = np.sum(pts**2, axis=1)
    sorted_idx = np.argsort(dists_to_center)
    pts = pts[sorted_idx[:N]]
    return pts


def _grid_init(rng):
    """Grid initial layout."""
    x = np.linspace(-HALF, HALF, 8)
    y = np.linspace(-HALF, HALF, 7)
    grid = np.array([(xi, yi) for yi in y for xi in x])
    grid = grid + rng.uniform(-0.01, 0.01, size=grid.shape)
    dists_to_center = np.sum(grid**2, axis=1)
    sorted_idx = np.argsort(dists_to_center)
    pts = grid[sorted_idx[:N]]
    return pts


def _random_init(rng):
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))


def _optimize_smooth(c0, beta_start=200.0, beta_end=2000.0, n_stages=5, t_limit=None):
    """Multi-stage smooth optimization using L-BFGS-B."""
    c = c0.copy()
    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (2 * N)
    betas = np.linspace(beta_start, beta_end, n_stages)
    
    for beta in betas:
        if t_limit is not None and time.time() > t_limit:
            break
        try:
            def obj(x, b=beta):
                c_local = x.reshape(N, 2)
                c_local = np.clip(c_local, -HALF + 1e-10, HALF - 1e-10)
                d_pairs = pdist(c_local) / 2.0
                wall_dists = np.stack([
                    c_local[:, 0] + HALF,
                    HALF - c_local[:, 0],
                    c_local[:, 1] + HALF,
                    HALF - c_local[:, 1]
                ]).flatten()
                all_dists = np.concatenate([d_pairs, wall_dists])
                min_d = all_dists.min()
                exp_terms = np.exp(-b * (all_dists - min_d))
                soft_min = min_d - np.log(exp_terms.sum()) / b
                return -soft_min
            
            res = minimize(
                obj,
                c.ravel(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-15, 'iprint': 0}
            )
            c_new = np.clip(res.x.reshape(N, 2), -HALF + 1e-10, HALF - 1e-10)
            if _score(c_new) >= _score(c) - 1e-12:
                c = c_new
        except Exception:
            pass
    
    return c


def _epigraph_objective(x):
    """Epigraph objective: minimize -t + penalty for constraint violations."""
    c = x[:2*N].reshape(N, 2)
    t = x[2*N]
    c = np.clip(c, -HALF + 1e-10, HALF - 1e-10)
    
    d_pairs = pdist(c) / 2.0
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ]).flatten()
    all_dists = np.concatenate([d_pairs, wall_dists])
    
    diff = t - all_dists
    pos_diff = np.maximum(0, diff)
    penalty = np.sum(pos_diff ** 2)
    return -t + 10.0 * penalty


def _epigraph_grad(x):
    """Gradient of epigraph objective."""
    c = x[:2*N].reshape(N, 2)
    t = x[2*N]
    c = np.clip(c, -HALF + 1e-10, HALF - 1e-10)
    
    d_pairs = pdist(c) / 2.0
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ]).flatten()
    all_dists = np.concatenate([d_pairs, wall_dists])
    
    diff = t - all_dists
    pos_diff = np.maximum(0, diff)
    
    grad = np.zeros(2*N + 1)
    grad[2*N] = -1.0 + 20.0 * np.sum(pos_diff)
    
    # Gradient with respect to centers
    # d/dc of sum(max(0, t - d_i)^2) = -2 * sum(max(0, t - d_i) * d(d_i)/dc)
    
    # Pairwise distances
    D = cdist(c, c)
    np.fill_diagonal(D, 1.0)
    D_half = D / 2.0
    
    iu = np.triu_indices(N, k=1)
    n_pairs = len(iu[0])
    diffs = c[iu[0]] - c[iu[1]]
    dists = D_half[iu[0], iu[1]]
    safe_dists = np.where(dists < 1e-15, 1.0, dists)
    grads = diffs / (2.0 * safe_dists[:, np.newaxis])
    
    # For each pair, the gradient contribution to circle i is -2 * pos_diff * grad
    # and to circle j is +2 * pos_diff * grad
    for k in range(n_pairs):
        i, j = iu[0][k], iu[1][k]
        pd = pos_diff[k]
        if pd > 0:
            grad[2*i] -= 20.0 * pd * grads[k, 0]
            grad[2*i+1] -= 20.0 * pd * grads[k, 1]
            grad[2*j] += 20.0 * pd * grads[k, 0]
            grad[2*j+1] += 20.0 * pd * grads[k, 1]
    
    # Wall distances
    wall_idx = n_pairs
    for i in range(N):
        # c[i,0] + HALF
        wd = wall_dists[4*i]
        if wd < t:
            grad[2*i] -= 20.0 * (t - wd)
        # HALF - c[i,0]
        wd = wall_dists[4*i+1]
        if wd < t:
            grad[2*i] += 20.0 * (t - wd)
        # c[i,1] + HALF
        wd = wall_dists[4*i+2]
        if wd < t:
            grad[2*i+1] -= 20.0 * (t - wd)
        # HALF - c[i,1]
        wd = wall_dists[4*i+3]
        if wd < t:
            grad[2*i+1] += 20.0 * (t - wd)
    
    return grad


def _optimize_epigraph(c0, t_limit=None):
    """Epigraph optimization using L-BFGS-B."""
    c = c0.copy()
    r0 = _score(c)
    t0 = r0 * 0.999
    
    x = np.concatenate([c.ravel(), [t0]])
    bounds_c = [(-HALF + 1e-9, HALF - 1e-9)] * (2 * N)
    bounds_t = [(0.0, HALF)]
    bounds = bounds_c + bounds_t
    
    try:
        res = minimize(
            _epigraph_objective,
            x,
            jac=_epigraph_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15, 'iprint': 0}
        )
        c_new = np.clip(res.x[:2*N].reshape(N, 2), -HALF + 1e-10, HALF - 1e-10)
        if _score(c_new) > _score(c) - 1e-12:
            c = c_new
    except Exception:
        pass
    
    return c


def _slsqp_constraint_fun(x_):
    c_ = x_[:2*N].reshape(N, 2)
    t_ = x_[2*N]
    d = pdist(c_) / 2.0 - t_
    wall_dists = np.stack([
        c_[:, 0] + HALF - t_,
        HALF - c_[:, 0] - t_,
        c_[:, 1] + HALF - t_,
        HALF - c_[:, 1] - t_
    ]).flatten()
    return np.concatenate([d, wall_dists])


def _slsqp_constraint_jac(x_):
    c_ = x_[:2*N].reshape(N, 2)
    t_ = x_[2*N]
    n_cons = N*(N-1)//2 + 4*N
    n_vars = 2*N + 1
    jac = np.zeros((n_cons, n_vars))
    
    D = cdist(c_, c_)
    np.fill_diagonal(D, 1.0)
    D_half = D / 2.0
    
    iu = np.triu_indices(N, k=1)
    n_pairs = len(iu[0])
    
    diffs = c_[iu[0]] - c_[iu[1]]
    dists = D_half[iu[0], iu[1]]
    safe_dists = np.where(dists < 1e-15, 1.0, dists)
    grads = diffs / (2.0 * safe_dists[:, np.newaxis])
    
    jac[:n_pairs, 2*iu[0]] = grads[:, 0]
    jac[:n_pairs, 2*iu[0]+1] = grads[:, 1]
    jac[:n_pairs, 2*iu[1]] = -grads[:, 0]
    jac[:n_pairs, 2*iu[1]+1] = -grads[:, 1]
    jac[:n_pairs, 2*N] = -1.0
    
    wall_idx = n_pairs
    for i in range(N):
        jac[wall_idx, 2*i] = 1.0
        jac[wall_idx, 2*N] = -1.0
        wall_idx += 1
        jac[wall_idx, 2*i] = -1.0
        jac[wall_idx, 2*N] = -1.0
        wall_idx += 1
        jac[wall_idx, 2*i+1] = 1.0
        jac[wall_idx, 2*N] = -1.0
        wall_idx += 1
        jac[wall_idx, 2*i+1] = -1.0
        jac[wall_idx, 2*N] = -1.0
        wall_idx += 1
    
    return jac


def _polish_with_slsqp(c0, t_limit=None, max_iter=200):
    """SLSQP polish with hard constraints."""
    c = c0.copy()
    r0 = _score(c)
    t0 = r0 * 0.999
    
    x = np.concatenate([c.ravel(), [t0]])
    bounds_c = [(-HALF + 1e-12, HALF - 1e-12)] * (2 * N)
    bounds_t = [(0.0, HALF)]
    bounds = bounds_c + bounds_t
    
    def obj(x_):
        return -x_[2*N]
    
    def obj_grad(x_):
        grad = np.zeros(2*N + 1)
        grad[2*N] = -1.0
        return grad
    
    try:
        res = minimize(
            obj,
            x,
            method='SLSQP',
            bounds=bounds,
            jac=obj_grad,
            constraints=[{
                'type': 'ineq',
                'fun': _slsqp_constraint_fun,
                'jac': _slsqp_constraint_jac
            }],
            options={'maxiter': max_iter, 'ftol': 1e-14, 'disp': False}
        )
        
        c_new = np.clip(res.x[:2*N].reshape(N, 2), -HALF + 1e-12, HALF - 1e-12)
        if _score(c_new) > _score(c) - 1e-12:
            c = c_new
    except Exception:
        pass
    
    return c


def _coordinate_descent(c0, t_limit=None, max_sweeps=200):
    """Coordinate descent polish."""
    c = c0.copy()
    c = np.clip(c, -HALF + 1e-12, HALF - 1e-12)
    
    for sweep in range(max_sweeps):
        if t_limit is not None and time.time() > t_limit:
            break
        
        improved = False
        for i in range(N):
            min_dist = np.inf
            min_constraint = None
            
            wall_dists = np.array([
                c[i, 0] + HALF,
                HALF - c[i, 0],
                c[i, 1] + HALF,
                HALF - c[i, 1]
            ])
            min_wall = wall_dists.min()
            if min_wall < min_dist:
                min_dist = min_wall
                min_constraint = ('wall', np.argmin(wall_dists))
            
            for j in range(N):
                if j == i:
                    continue
                d = np.linalg.norm(c[i] - c[j]) / 2.0
                if d < min_dist:
                    min_dist = d
                    min_constraint = ('pair', j)
            
            if min_constraint[0] == 'wall':
                wall_idx = min_constraint[1]
                if wall_idx == 0:
                    direction = np.array([1, 0])
                elif wall_idx == 1:
                    direction = np.array([-1, 0])
                elif wall_idx == 2:
                    direction = np.array([0, 1])
                else:
                    direction = np.array([0, -1])
            else:
                j = min_constraint[1]
                direction = c[i] - c[j]
                norm = np.linalg.norm(direction)
                if norm > 1e-12:
                    direction /= norm
                else:
                    direction = np.array([1, 0])
            
            best_step = 0
            best_r = _score(c)
            
            for step in [0.001, 0.002, 0.005, 0.01, 0.02]:
                c_trial = c.copy()
                c_trial[i] += direction * step
                c_trial[i] = np.clip(c_trial[i], -HALF + 1e-12, HALF - 1e-12)
                r_trial = _score(c_trial)
                if r_trial > best_r + 1e-15:
                    best_r = r_trial
                    best_step = step
            
            if best_step > 0:
                c[i] += direction * best_step
                c[i] = np.clip(c[i], -HALF + 1e-12, HALF - 1e-12)
                improved = True
        
        if not improved:
            break
    
    return c


def _powell_polish(c0, t_limit=None, maxiter=3000):
    """Powell's method polish for final refinement."""
    def objective(x):
        c_local = x.reshape(N, 2)
        c_local = np.clip(c_local, -HALF + 1e-12, HALF - 1e-12)
        r = _score(c_local)
        return -r
    
    try:
        res = minimize(
            objective,
            c0.ravel(),
            method='Powell',
            options={'maxiter': maxiter, 'ftol': 1e-14, 'xtol': 1e-14, 'disp': False}
        )
        c_new = np.clip(res.x.reshape(N, 2), -HALF + 1e-12, HALF - 1e-12)
        if _score(c_new) > _score(c0) - 1e-12:
            return c_new
    except Exception:
        pass
    
    return c0


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    inits = [
        _hexagonal_init(rng, jitter=0.005),
        _grid_init(rng),
        _random_init(rng)
    ]
    
    best_centres = inits[0]
    best_score = _score(best_centres)
    
    n_starts = 25
    for i in range(n_starts):
        if time.time() - t0 > DEADLINE - 20:
            break
        
        if i < len(inits):
            x0 = inits[i]
        else:
            sigma = 0.01 + 0.02 * (i % 5)
            x0 = best_centres + rng.normal(0, sigma, size=(N, 2))
            x0 = np.clip(x0, -HALF + 1e-9, HALF - 1e-9)
        
        try:
            c_opt = _optimize_smooth(x0, beta_start=100.0, beta_end=1000.0, n_stages=3, t_limit=t0 + DEADLINE - 15)
            s1 = _score(c_opt)
            
            if time.time() - t0 < DEADLINE - 15:
                c_opt = _optimize_epigraph(c_opt, t_limit=t0 + DEADLINE - 10)
                s2 = _score(c_opt)
                if s2 > s1:
                    s1 = s2
            
            if time.time() - t0 < DEADLINE - 10:
                c_opt = _polish_with_slsqp(c_opt, t_limit=t0 + DEADLINE - 5, max_iter=150)
                s3 = _score(c_opt)
                if s3 > s1:
                    s1 = s3
            
            if s1 > best_score:
                best_score = s1
                best_centres = c_opt.copy()
                
        except Exception:
            pass

    if time.time() - t0 < DEADLINE - 10:
        try:
            best_centres = _polish_with_slsqp(best_centres, t_limit=t0 + DEADLINE - 5, max_iter=300)
            best_score = _score(best_centres)
        except Exception:
            pass

    if time.time() - t0 < DEADLINE - 5:
        try:
            best_centres = _coordinate_descent(best_centres, t_limit=t0 + DEADLINE - 3)
            s_cd = _score(best_centres)
            if s_cd > best_score:
                best_score = s_cd
        except Exception:
            pass

    if time.time() - t0 < DEADLINE - 3:
        try:
            best_centres = _powell_polish(best_centres, t_limit=t0 + DEADLINE - 1, maxiter=2000)
            s_pow = _score(best_centres)
            if s_pow > best_score:
                best_score = s_pow
        except Exception:
            pass

    best_centres = np.clip(best_centres, -HALF + 1e-12, HALF - 1e-12)
    return best_centres
