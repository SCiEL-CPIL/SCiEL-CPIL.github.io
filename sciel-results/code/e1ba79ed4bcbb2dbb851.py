"""
Contact-graph push-apart packer for 100 equal spheres in a unit cube.

Strategy:
  - Represent the packing by its contact graph (nearest-neighbor pairs).
  - Use a smooth repulsion energy E = sum_{i<j} 1/(d_ij^2 + eps) to push
    overlapping or too-close spheres apart (no sqrt, no min, differentiable).
  - Alternate between:
      (a) fast Newton / BFGS local relaxation of the repulsion energy,
      (b) global "cluster kick" moves that reposition random subsets of
          spheres to escape local minima,
      (c) periodic re-evaluation of the contact graph.
  - Score is the true min-radius; we only use the energy as a surrogate
    for local optimisation.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 100
HALF = 0.5
EPS = 1e-6  # regularisation for repulsion

# ---------------------------------------------------------------------------
# Scoring (exact, as defined by the task)
# ---------------------------------------------------------------------------
def _score(points):
    """Return the maximal common radius for the given set of centres."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Inside cube [-0.5, 0.5]^3
    margin = HALF - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = dists.min() / 2.0
    return min(r_pair, r_margin)

# ---------------------------------------------------------------------------
# Repulsion energy (smooth, no min, differentiable)
# ---------------------------------------------------------------------------
def _repulsion_energy(pts):
    """
    E = sum_{i<j} 1/(d_ij^2 + EPS)  +  wall penalty
    Wall penalty pushes points back into the cube.
    """
    pts = np.asarray(pts, dtype=np.float64)
    # Pairwise squared distances (vectorised)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = (diff ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    # Only sum upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    E = np.sum(1.0 / (d2[iu] + EPS))
    # Wall penalty: quadratic for points outside [-0.5, 0.5]
    outside = np.maximum(0.0, np.abs(pts) - HALF)
    E += 100.0 * np.sum(outside ** 2)
    return E

def _repulsion_grad(pts):
    """
    Gradient of the repulsion energy w.r.t. positions.
    dE/dx_i = -2 * sum_{j != i} (x_i - x_j) / (d_ij^2 + EPS)^2
    """
    pts = np.asarray(pts, dtype=np.float64)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    d2 = (diff ** 2).sum(axis=2)  # (N, N)
    np.fill_diagonal(d2, 1.0)  # avoid div by zero on diagonal
    # weight_ij = 2 / (d_ij^2 + EPS)^2
    w = 2.0 / (d2 + EPS) ** 2
    np.fill_diagonal(w, 0.0)
    # grad_i = -sum_j w_ij * (x_i - x_j)
    grad = -np.sum(w[:, :, np.newaxis] * diff, axis=1)
    # Wall gradient
    outside = np.maximum(0.0, np.abs(pts) - HALF)
    sign = np.sign(pts)
    grad += 200.0 * outside * sign
    return grad

# ---------------------------------------------------------------------------
# Local relaxation using L-BFGS-B on the repulsion energy
# ---------------------------------------------------------------------------
def _local_relax(pts, maxiter=200, time_budget=5.0):
    """
    Minimise repulsion energy locally. Returns improved points.
    """
    x0 = pts.flatten()
    bounds = [(-HALF, HALF)] * (N * 3)
    t0 = time.time()

    def energy(x):
        if time.time() - t0 > time_budget:
            return _repulsion_energy(x0.reshape(N, 3))  # freeze
        p = x.reshape(N, 3)
        return _repulsion_energy(p)

    def grad(x):
        if time.time() - t0 > time_budget:
            return _repulsion_grad(x0.reshape(N, 3)).flatten()
        p = x.reshape(N, 3)
        return _repulsion_grad(p).flatten()

    try:
        result = minimize(energy, x0, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': maxiter, 'ftol': 1e-12,
                                   'gtol': 1e-8, 'maxls': 20})
        if result.fun < _repulsion_energy(pts):
            return result.x.reshape(N, 3)
    except Exception:
        pass
    return pts

# ---------------------------------------------------------------------------
# Contact-graph aware cluster kick
# ---------------------------------------------------------------------------
def _build_contact_graph(pts, r_target, k=6):
    """
    For each sphere, find its k nearest neighbours.
    Returns a list of edges (i, j) with i < j.
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = (diff ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    # For each i, get k nearest
    edges = set()
    for i in range(N):
        nn = np.argsort(d2[i])[:k]
        for j in nn:
            if i < j:
                edges.add((i, j))
            else:
                edges.add((j, i))
    return list(edges)

def _cluster_kick(pts, r_target, n_clusters=5, kick_scale=0.08):
    """
    Partition spheres into random clusters, reposition each cluster's
    centroid by a small random kick, then relax locally.
    This helps escape local minima by moving groups of spheres together.
    """
    new_pts = pts.copy()
    # Random partition into clusters
    labels = np.random.randint(0, n_clusters, N)
    for c in range(n_clusters):
        mask = labels == c
        if mask.sum() < 2:
            continue
        centroid = new_pts[mask].mean(axis=0)
        kick = np.random.normal(0, kick_scale, 3)
        new_pts[mask] += kick
        # Keep inside cube
        new_pts[mask] = np.clip(new_pts[mask], -HALF, HALF)
    return new_pts

# ---------------------------------------------------------------------------
# Multi-scale local search with contact-graph updates
# ---------------------------------------------------------------------------
def _multi_scale_search(pts, time_budget=60.0):
    """
    Alternating local relaxation and cluster kicks, with periodic
    contact-graph re-evaluation to guide kick scale.
    """
    best_pts = pts.copy()
    best_score = _score(best_pts)
    t0 = time.time()
    iteration = 0

    while time.time() - t0 < time_budget:
        iteration += 1

        # Local relaxation
        relaxed = _local_relax(best_pts, maxiter=100, time_budget=3.0)
        rel_score = _score(relaxed)
        if rel_score > best_score:
            best_pts = relaxed
            best_score = rel_score

        # Check time
        if time.time() - t0 > time_budget - 10:
            break

        # Cluster kick (adapt scale based on current radius)
        r_est = best_score
        kick_scale = max(0.02, 0.1 * r_est * 3)
        kicked = _cluster_kick(best_pts, r_est,
                               n_clusters=random.randint(3, 8),
                               kick_scale=kick_scale)
        # Relax after kick
        kicked_relaxed = _local_relax(kicked, maxiter=80, time_budget=3.0)
        kick_score = _score(kicked_relaxed)
        if kick_score > best_score:
            best_pts = kicked_relaxed
            best_score = kick_score

        # Occasionally do a single-sphere nudge for fine-tuning
        if iteration % 5 == 0 and time.time() - t0 < time_budget - 5:
            idx = random.randint(0, N - 1)
            nudge = best_pts.copy()
            nudge[idx] += np.random.normal(0, 0.03, 3)
            nudge[idx] = np.clip(nudge[idx], -HALF, HALF)
            nudge_relaxed = _local_relax(nudge, maxiter=60, time_budget=2.0)
            nudge_score = _score(nudge_relaxed)
            if nudge_score > best_score:
                best_pts = nudge_relaxed
                best_score = nudge_score

    return best_pts

# ---------------------------------------------------------------------------
# Initialisation: slightly perturbed grid + a few random starts
# ---------------------------------------------------------------------------
def _grid_init(nx=5, ny=5, nz=4, jitter=0.02):
    xs = np.linspace(-HALF + 0.08, HALF - 0.08, nx)
    ys = np.linspace(-HALF + 0.08, HALF - 0.08, ny)
    zs = np.linspace(-HALF + 0.1, HALF - 0.1, nz)
    points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    points += np.random.normal(0, jitter, points.shape)
    points = np.clip(points, -HALF, HALF)
    return points

def _random_init():
    return np.random.uniform(-HALF + 0.05, HALF - 0.05, (N, 3))

# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    random.seed(12345)
    np.random.seed(12345)

    best_pts = None
    best_score = 0.0

    # ---- Phase 1: Try multiple initialisations ----
    inits = []
    inits.append(_grid_init())
    inits.append(_grid_init(nx=4, ny=5, nz=5, jitter=0.03))
    inits.append(_random_init())

    for init in inits:
        if time.time() > deadline - 40:
            break
        # Quick local relaxation
        relaxed = _local_relax(init, maxiter=150, time_budget=8.0)
        s = _score(relaxed)
        if s > best_score:
            best_score = s
            best_pts = relaxed.copy()

    if best_pts is None:
        best_pts = _grid_init()
        best_score = _score(best_pts)

    # ---- Phase 2: Multi-scale search from the best initialisation ----
    remaining = deadline - time.time()
    if remaining > 20:
        best_pts = _multi_scale_search(best_pts, time_budget=min(remaining - 10, 80.0))
        best_score = _score(best_pts)

    # ---- Phase 3: Final fine polish ----
    remaining = deadline - time.time()
    if remaining > 5:
        polished = _local_relax(best_pts, maxiter=300, time_budget=min(remaining - 2, 15.0))
        p_score = _score(polished)
        if p_score > best_score:
            best_pts = polished
            best_score = p_score

    # Ensure valid output
    best_pts = np.clip(best_pts, -HALF, HALF)
    return best_pts

# EVOLVE-BLOCK-END
