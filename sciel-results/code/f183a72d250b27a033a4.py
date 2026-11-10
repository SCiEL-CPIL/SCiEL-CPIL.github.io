"""Optimized logarithmic energy minimization for 150 points on the sphere.

Strategy:
- High-quality structured starts (Fibonacci lattice, Gauss-Legendre spherical grid).
- Riemannian L-BFGS: work in tangent space with projected gradient and
  retraction (normalize after a Euclidean step), which is far more efficient
  than Cartesian L-BFGS with renormalization.
- Multiple diverse starts with a time budget, plus a final tight polish.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 150.0
SAFETY_MARGIN = 6.0


def _energy_and_euclid_grad(pts):
    """Compute logarithmic energy and Euclidean gradient.

    Energy: sum_{i<j} log(1/|x_i - x_j|)
    Euclidean gradient for point i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]          # (n,n,3)
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)                # (n,n)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)

    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))

    inv_dist_sq = np.zeros_like(dist_sq)
    mask = dist_sq > 1e-12
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    np.fill_diagonal(inv_dist_sq, 0.0)

    grad = np.einsum('ij,ijk->ik', inv_dist_sq, diff)
    return energy, grad


def _riemannian_grad(pts):
    """Project Euclidean gradient onto tangent space at each point."""
    _, eg = _energy_and_euclid_grad(pts)
    # Tangent projection: g_R = g_E - (g_E . x) x
    dot = np.einsum('ij,ij->i', eg, pts)
    rg = eg - dot[:, np.newaxis] * pts
    return rg


def _retract(pts, steps):
    """First-order retraction: move by step then renormalize."""
    new = pts + steps
    norms = np.linalg.norm(new, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return new / norms


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _fibonacci_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _gauss_legendre_grid():
    """Gauss-Legendre based spherical grid: good moment properties."""
    # Use Gauss-Legendre nodes for z, and uniform angle offsets per ring
    z = np.polynomial.legendre.leggauss(N)[::-1]  # ascending
    # Slightly perturb to avoid exact symmetry degeneracy
    pts = np.zeros((N, 3))
    for k, zk in enumerate(z):
        zk = np.clip(zk, -1.0 + 1e-12, 1.0 - 1e-12)
        r = np.sqrt(max(0.0, 1.0 - zk * zk))
        ang = 2.0 * np.pi * k / N + 0.3  # stagger
        pts[k] = [r * np.cos(ang), r * np.sin(ang), zk]
    return _normalize(pts)


def _random_points(seed):
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, 3))
    return _normalize(pts)


def _riemannian_lbfgs(pts0, max_iter=4000, ftol=1e-13, gtol=1e-10,
                      time_limit=None, start_time=None, m=20):
    """Riemannian L-BFGS with retraction and backtracking line search."""
    if start_time is None:
        start_time = time.time()
    pts = _normalize(pts0.copy())
    energy, _ = _energy_and_euclid_grad(pts)
    g = _riemannian_grad(pts)
    gnorm = np.linalg.norm(g)

    # L-BFGS memory
    S = []  # steps
    Y = []  # gradient differences
    W = 0.0  # gamma

    def two_loop_recursion(q, m_):
        alpha = np.zeros(m_)
        beta = np.zeros(m_)
        rho = np.zeros(m_)
        for i in range(m_):
            S_i = S[i]
            Y_i = Y[i]
            ynorm_sq = np.einsum('ij,ij->', Y_i, Y_i)
            if ynorm_sq < 1e-16:
                rho[i] = 0.0
                continue
            rho[i] = 1.0 / ynorm_sq
            alpha[i] = rho[i] * np.einsum('ij,ij->', S_i, q)
            q = q - alpha[i] * Y_i
        r = W * q
        for i in range(m_ - 1, -1, -1):
            beta[i] = rho[i] * np.einsum('ij,ij->', Y_i, r)
            r = r + (alpha[i] - beta[i]) * S[i]
        return r

    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        if gnorm < gtol:
            break

        # Search direction
        if it == 0 or len(S) == 0:
            d = -g
        else:
            q = -g
            d = two_loop_recursion(q, len(S))

        dnorm = np.linalg.norm(d)
        if dnorm < 1e-15:
            break

        # Backtracking line search with Armijo
        alpha = 1.0
        c1 = 1e-4
        energy0 = energy
        gdotd = np.einsum('ij,ij->', g, d)
        # Ensure descent
        if gdotd > 0:
            d = -g
            gdotd = -np.einsum('ij,ij->', g, g)
            dnorm = np.linalg.norm(d)

        for _ in range(30):
            new_pts = _retract(pts, alpha * d)
            new_energy, _ = _energy_and_euclid_grad(new_pts)
            if new_energy <= energy0 + c1 * alpha * gdotd:
                break
            alpha *= 0.5

        new_g = _riemannian_grad(new_pts)
        new_gnorm = np.linalg.norm(new_g)

        # Update L-BFGS memory
        if new_energy < energy0 - 1e-16:
            s = new_pts - pts
            y = new_g - g
            sy = np.einsum('ij,ij->', s, y)
            if sy > 1e-14:
                S.append(s)
                Y.append(y)
                if len(S) > m:
                    S.pop(0)
                    Y.pop(0)
                W = np.einsum('ij,ij->', s, s) / sy

        if abs(new_energy - energy0) < ftol * (abs(energy0) + 1e-10):
            energy = new_energy
            pts = new_pts
            g = new_g
            gnorm = new_gnorm
            break

        energy = new_energy
        pts = new_pts
        g = new_g
        gnorm = new_gnorm

    return _normalize(pts), energy


def _polish(pts0, time_limit):
    """Final tight Riemannian L-BFGS polish."""
    start = time.time()
    tl = max(1.0, time_limit - (time.time() - start))
    pts, e = _riemannian_lbfgs(
        pts0, max_iter=6000, ftol=1e-15, gtol=1e-12,
        time_limit=tl, start_time=start, m=25)
    return pts, e


def _repulsion_relax(pts0, n_iter=300, time_limit=None, start_time=None):
    """Simple molecular-dynamics-like repulsion relaxation to escape bad minima."""
    if start_time is None:
        start_time = time.time()
    pts = _normalize(pts0.copy())
    # Estimate typical step size from nearest-neighbor distance
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1e10)
    nn = np.sqrt(np.min(dist_sq, axis=1))
    step = 0.05 * np.median(nn)

    for it in range(n_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        _, eg = _energy_and_euclid_grad(pts)
        g = eg - np.einsum('ij,ij->i', eg, pts)[:, np.newaxis] * pts
        # Cooling
        cool = 1.0 - 0.5 * (it / max(1, n_iter))
        new_pts = _retract(pts, step * cool * g)
        pts = new_pts
    return _normalize(pts)


def construct():
    start_time = time.time()

    def remaining():
        return max(0.0, DEADLINE - SAFETY_MARGIN - (time.time() - start_time))

    # Candidate starts
    starts = [
        ("fib", _fibonacci_spiral()),
        ("gl", _gauss_legendre_grid()),
        ("rand42", _random_points(42)),
        ("rand123", _random_points(123)),
        ("rand7", _random_points(7)),
    ]

    best_pts = starts[0][1]
    best_energy = _energy_and_euclid_grad(best_pts)[0]

    # Phase 1: repulsion relax each start, then Riemannian L-BFGS
    for name, s in starts:
        tl = remaining()
        if tl < 3.0:
            break
        try:
            # Short relax to escape obvious bad minima
            tl_relax = min(8.0, tl * 0.25)
            relaxed = _repulsion_relax(s, n_iter=120, time_limit=tl_relax,
                                       start_time=time.time())
            tl_opt = remaining()
            if tl_opt < 2.0:
                continue
            pts, e = _riemannian_lbfgs(
                relaxed, max_iter=3000, ftol=1e-13, gtol=1e-10,
                time_limit=tl_opt, start_time=time.time(), m=20)
            e = _energy_and_euclid_grad(pts)[0]
            if e < best_energy:
                best_energy = e
                best_pts = pts
        except Exception:
            pass

    # Phase 2: polish the best
    tl = remaining()
    if tl > 2.0:
        try:
            pts, e = _polish(best_pts, time_limit=tl)
            e = _energy_and_euclid_grad(pts)[0]
            if e < best_energy:
                best_energy = e
                best_pts = pts
        except Exception:
            pass

    # Final safety check
    best_pts = _normalize(best_pts)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral()

    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    if min_dist < 1e-9:
        best_pts = _fibonacci_spiral()

    return best_pts


if __name__ == "__main__":
    pts = construct()
    e, _ = _energy_and_euclid_grad(pts)
    print(f"Energy: {e}")
    print(f"Shape: {pts.shape}")
