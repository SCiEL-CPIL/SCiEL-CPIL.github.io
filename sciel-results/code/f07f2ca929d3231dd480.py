"""Fifty equal spheres in a unit cube: optimise common radius via soft-min + L-BFGS-B."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50


def _score(centres: np.ndarray) -> float:
    """Return the feasible common radius for a placement (0.0 if invalid)."""
    if centres.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # distance to walls
    wall = 0.5 - np.abs(centres)
    r_wall = float(np.min(wall.min(axis=0)))
    if r_wall <= 0.0:
        return 0.0
    # pairwise distances
    d = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    r_pair = float(np.min(d) / 2.0)
    return min(r_wall, r_pair)


def _soft_min_loss(centres: np.ndarray, beta: float) -> float:
    """
    Surrogate objective: we want to maximize min(min_pair_dist/2, min_wall_dist).
    Equivalent to minimizing -min(...).
    Use log-sum-exp to smooth the min.
    Loss = -(1/beta) * log( sum_i exp(-beta * d_i) )  where d_i are the candidate radii.
    But we need a single radius. Let's build a vector of all constraint margins:
    For each pair i<j: margin = |c_i - c_j| / 2
    For each sphere i, each wall k: margin = 1/2 - |c_ik|
    Then soft_min = -(1/beta)*log(sum exp(-beta * m))  approximates min(m).
    We maximize that, so loss = -soft_min = (1/beta)*log(sum exp(-beta * m)).
    """
    c = centres
    # pairwise
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    iu = np.triu_indices(N, 1)
    pair_margins = d[iu] / 2.0
    # wall margins
    wall_margins = (0.5 - np.abs(c)).reshape(-1)
    all_margins = np.concatenate([pair_margins, wall_margins])
    # clamp to avoid overflow in exp
    # soft_min = -1/beta * log(sum(exp(-beta*m)))
    # For numerical stability, shift by min
    m_min = np.min(all_margins)
    log_sum_exp = m_min + np.log(np.sum(np.exp(-beta * (all_margins - m_min))))
    soft_min = -log_sum_exp
    return -soft_min  # minimize this


def _grad(centres: np.ndarray, beta: float) -> np.ndarray:
    """Gradient of the soft-min loss w.r.t. centres."""
    c = centres
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    iu = np.triu_indices(N, 1)
    pair_margins = d[iu] / 2.0
    wall_margins = (0.5 - np.abs(c)).reshape(-1)
    all_margins = np.concatenate([pair_margins, wall_margins])
    m_min = np.min(all_margins)
    shifted = -beta * (all_margins - m_min)
    # clip for stability
    shifted = np.clip(shifted, -500, 500)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    # d(soft_min)/d(marg) = -weights (since soft_min = -logsumexp)
    # d(loss)/d(marg) = weights
    # Now back-propagate to centres.
    grad = np.zeros_like(c)
    # pairwise part
    # pair_margins[k] = |c_i - c_j|/2 for (i,j) = iu[k]
    i_idx = iu[0]
    j_idx = iu[1]
    diff = c[i_idx] - c[j_idx]  # shape (num_pairs, 3)
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    unit = diff / norms  # (num_pairs, 3)
    # d(pair_margin)/d(c_i) = unit/2, d(pair_margin)/d(c_j) = -unit/2
    w = weights[:len(i_idx)]  # (num_pairs,)
    grad[i_idx] -= w[:, None] * unit / 2.0
    grad[j_idx] += w[:, None] * unit / 2.0
    # wall part
    # wall_margins = (0.5 - |c_ik|) flattened
    # d(margin)/d(c_ik) = -sign(c_ik)
    wall_grad = -np.sign(c).reshape(-1)
    grad_flat = grad.reshape(-1)
    grad_flat[0:0]  # just to keep reference
    # add wall contributions
    grad_flat2 = grad_flat.copy()
    grad_flat2[len(iu[0]):] += weights[len(iu[0]):] * wall_grad
    # Actually grad_flat is already modified; let's redo cleanly
    grad = np.zeros_like(c)
    w_pair = weights[:len(i_idx)]
    grad[i_idx] -= w_pair[:, None] * unit / 2.0
    grad[j_idx] += w_pair[:, None] * unit / 2.0
    w_wall = weights[len(i_idx):]
    wall_contrib = -w_wall.reshape(N, 3) * np.sign(c)
    grad += wall_contrib
    return grad


def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before 150s kill

    # Initial guess: cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    best = grid[:N].copy()
    best_r = _score(best)

    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    for trial in range(8):
        if time.time() > deadline:
            break
        init = best + rng.normal(0, 0.02, size=(N, 3))
        # Clip into cube
        init = np.clip(init, -0.48, 0.48)
        
        # Anneal beta from 20 to 800
        betas = np.linspace(20, 800, 12)
        cur = init.copy()
        for beta in betas:
            if time.time() > deadline:
                break
            res = minimize(
                _soft_min_loss,
                cur,
                args=(beta,),
                jac=_grad,
                method='L-BFGS-B',
                options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-6},
            )
            cur = res.x
            cur = np.clip(cur, -0.49, 0.49)
        
        r = _score(cur)
        if r > best_r:
            best_r = r
            best = cur.copy()

    # Final polish with high beta
    if time.time() < deadline - 10:
        for beta in [1000, 2000]:
            if time.time() > deadline:
                break
            res = minimize(
                _soft_min_loss,
                best,
                args=(beta,),
                jac=_grad,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8},
            )
            candidate = np.clip(res.x, -0.499, 0.499)
            r = _score(candidate)
            if r > best_r:
                best_r = r
                best = candidate

    return best
