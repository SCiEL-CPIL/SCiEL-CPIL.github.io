"""Optimized program: 14 points in 3D minimizing (max/min pairwise dist)^2."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 14
D = 3
_DEADLINE = time.time() + 140.0  # safety margin before 150s kill

# Precompute pairwise indices
_IDX_I, _IDX_J = np.triu_indices(N, k=1)
_NUM_PAIRS = len(_IDX_I)


def _pairwise_dists2(pts):
    """Compute squared pairwise distances for N points in D dims."""
    pts = np.asarray(pts, dtype=float)
    diffs = pts[_IDX_I] - pts[_IDX_J]
    return np.sum(diffs * diffs, axis=1)


def _score(pts):
    """Return (max_d/min_d)^2."""
    d2 = _pairwise_dists2(pts)
    d2 = np.maximum(d2, 1e-24)
    return d2.max() / d2.min()


def _logsumexp_max(x, tau):
    """Smooth max: (1/tau) * logsumexp(tau * x)."""
    m = x.max()
    return m + np.log(np.sum(np.exp(tau * (x - m)))) / tau


def _logsumexp_min(x, tau):
    """Smooth min: -(1/tau) * logsumexp(-tau * x)."""
    m = x.min()
    return m - np.log(np.sum(np.exp(-tau * (x - m)))) / tau


def _objective_and_grad(x, tau):
    """
    Objective: log(max_d) - log(min_d)  (minimizing this minimizes max/min)
    Using smooth surrogates in squared-distance space.
    x: flat array of shape (N*D,)
    """
    pts = x.reshape(N, D)
    d2 = _pairwise_dists2(pts)
    d2 = np.maximum(d2, 1e-24)

    # Smooth max of d2, smooth min of d2
    max_d2 = _logsumexp_max(d2, tau)
    min_d2 = _logsumexp_min(d2, tau)

    # Objective: 0.5*(log(max_d2) - log(min_d2)) = log(max_d/min_d)
    # Minimizing this is equivalent to minimizing max/min
    obj = 0.5 * (np.log(max_d2) - np.log(min_d2))

    # Gradient w.r.t. x
    # d(max_d2)/d(d2_i) = exp(tau*(d2_i - max)) / sum(exp(tau*(d2_j - max)))
    m_max = d2.max()
    exp_max = np.exp(tau * (d2 - m_max))
    w_max = exp_max / exp_max.sum()  # weights for max

    m_min = d2.min()
    exp_min = np.exp(-tau * (d2 - m_min))
    w_min = exp_min / exp_min.sum()  # weights for min

    # d(obj)/d(d2_i) = 0.5 * (w_max_i / max_d2 - w_min_i / min_d2)
    grad_d2 = 0.5 * (w_max / max_d2 - w_min / min_d2)

    # d(d2_i)/d(x) where d2_i = sum_k (pts[i,k] - pts[j,k])^2
    # grad_x = sum_i grad_d2_i * d(d2_i)/dx
    # d(d2_i)/d(pts[i,k]) = 2*(pts[i,k] - pts[j,k])
    # d(d2_i)/d(pts[j,k]) = -2*(pts[i,k] - pts[j,k])

    diffs = pts[_IDX_I] - pts[_IDX_J]  # (_NUM_PAIRS, D)
    # For each pair, contribution to gradient
    # grad_pts[i] += 2 * grad_d2[pair] * (pts[i] - pts[j])
    # grad_pts[j] -= 2 * grad_d2[pair] * (pts[i] - pts[j])

    grad_pts = np.zeros_like(pts)
    np.add.at(grad_pts, _IDX_I, 2.0 * grad_d2[:, None] * diffs)
    np.add.at(grad_pts, _IDX_J, -2.0 * grad_d2[:, None] * diffs)

    return obj, grad_pts.flatten()


def _make_constraints(pts_ref):
    """
    Fix point 0 at origin, point 1 at (0,0,1) to remove symmetries.
    Returns constraint list for SLSQP.
    """
    def c_origin(x):
        return x[:D]  # point 0 should be at origin

    def c_zaxis(x):
        # point 1 should be at (0, 0, 1)
        return [x[D] - 0.0, x[D+1] - 0.0, x[D+2] - 1.0]

    constraints = [
        {'type': 'eq', 'fun': c_origin},
        {'type': 'eq', 'fun': c_zaxis},
    ]
    return constraints


def _generate_initial_points(rng, n_starts=20):
    """Generate diverse initial configurations."""
    inits = []
    for i in range(n_starts):
        if i == 0:
            # Regular-ish starting point: sphere distribution
            pts = np.zeros((N, D))
            # Use spherical Fibonacci-like distribution
            golden = (1 + np.sqrt(5)) / 2
            for k in range(N):
                z = 1 - 2 * (k + 0.5) / N
                r = np.sqrt(1 - z*z)
                theta = 2 * np.pi * k / golden
                pts[k] = [r * np.cos(theta), r * np.sin(theta), z]
            # Apply symmetry: point 0 at origin, point 1 on z-axis
            # Translate so point 0 is at origin
            pts = pts - pts[0]
            # Rotate so point 1 is on z-axis
            v = pts[1]
            if np.linalg.norm(v) > 1e-10:
                v = v / np.linalg.norm(v)
                # Rotation matrix to map v to z-axis
                z = np.array([0, 0, 1.0])
                axis = np.cross(v, z)
                s = np.linalg.norm(axis)
                c = np.dot(v, z)
                if s < 1e-10:
                    R = np.eye(3) if c > 0 else np.diag([1, -1, -1])
                else:
                    axis = axis / s
                    K = np.array([[0, -axis[2], axis[1]],
                                  [axis[2], 0, -axis[0]],
                                  [-axis[1], axis[0], 0]])
                    R = np.eye(3) + np.sin(np.arccos(c)) * K + (1 - c) * (K @ K)
                    # Actually use Rodrigues
                    theta = np.arccos(np.clip(c, -1, 1))
                    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
                pts = pts @ R.T
            inits.append(pts.flatten())
        else:
            # Random perturbations
            pts = np.random.randn(N, D) * 0.5
            pts[0] = 0
            pts[1] = [0, 0, 1]
            inits.append(pts.flatten())
    return inits


def construct():
    global _DEADLINE
    start_time = time.time()
    rng = np.random.RandomState(42)

    best_score = np.inf
    best_pts = None

    # Phase 1: Multi-start SLSQP with moderate tau
    n_starts = 15
    tau_initial = 500.0

    inits = _generate_initial_points(rng, n_starts)

    for i, x0 in enumerate(inits):
        if time.time() > _DEADLINE - 30:
            break

        try:
            res = minimize(
                _objective_and_grad,
                x0,
                args=(tau_initial,),
                method='SLSQP',
                jac=True,
                constraints=_make_constraints(None),
                options={'maxiter': 300, 'ftol': 1e-12},
            )
            if res.success and res.x is not None:
                pts = res.x.reshape(N, D)
                s = _score(pts)
                if s < best_score:
                    best_score = s
                    best_pts = pts.copy()
        except Exception:
            pass

    # Phase 2: Polish best candidate with high tau and more iterations
    if best_pts is not None:
        for tau_polish in [2000.0, 5000.0]:
            if time.time() > _DEADLINE - 10:
                break
            try:
                x0 = best_pts.flatten()
                res = minimize(
                    _objective_and_grad,
                    x0,
                    args=(tau_polish,),
                    method='SLSQP',
                    jac=True,
                    constraints=_make_constraints(None),
                    options={'maxiter': 500, 'ftol': 1e-14},
                )
                if res.success and res.x is not None:
                    pts = res.x.reshape(N, D)
                    s = _score(pts)
                    if s < best_score:
                        best_score = s
                        best_pts = pts.copy()
            except Exception:
                pass

    # Phase 3: If time remains, do a few more random restarts from best
    if best_pts is not None and time.time() < _DEADLINE - 15:
        for _ in range(5):
            if time.time() > _DEADLINE - 5:
                break
            try:
                noise = rng.randn(N, D) * 0.05
                x0 = (best_pts + noise).flatten()
                # Re-apply symmetry constraints approximately
                x0[:D] = 0.0
                x0[D:D+3] = [0, 0, 1]
                res = minimize(
                    _objective_and_grad,
                    x0,
                    args=(5000.0,),
                    method='SLSQP',
                    jac=True,
                    constraints=_make_constraints(None),
                    options={'maxiter': 300, 'ftol': 1e-13},
                )
                if res.success and res.x is not None:
                    pts = res.x.reshape(N, D)
                    s = _score(pts)
                    if s < best_score:
                        best_score = s
                        best_pts = pts.copy()
            except Exception:
                pass

    if best_pts is None:
        # Fallback: simple grid
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)

    return best_pts
