"""Improved packer: hex-lattice init + vectorized force refinement + L-BFGS-B polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 135.0  # seconds; leaves 15s margin under 150s kill


def _score(pts):
    """Compute the packing radius r for centres pts (N,2)."""
    n = len(pts)
    if n == 0:
        return 0.0
    # distance to boundary
    d = 1.0 - np.linalg.norm(pts, axis=1)
    r_bound = np.min(d)
    # pairwise distances (vectorized)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pair = np.min(dist) / 2.0
    return min(r_bound, r_pair)


def _force_refine(pts, max_iter=20000, tol=1e-12):
    """
    Vectorized force-based local refinement.
    Points repel each other when closer than 2r and are pulled inward when too close to boundary.
    """
    n = len(pts)
    pts = pts.copy().astype(np.float64)
    r = _score(pts)
    k = 10.0

    for it in range(max_iter):
        if time.time() > DEADLINE - 5.0:
            break

        forces = np.zeros_like(pts)

        # Vectorized pairwise repulsion
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, 1.0)  # avoid div by zero

        # Only apply force where dist < 2r
        mask = dist < 2.0 * r
        # Force magnitude: k * (2r - dist) / dist, applied along direction
        # For pairs where dist < 2r, force on i from j is k*(2r-dist)/dist * (pts[i]-pts[j])
        force_mag = np.zeros_like(dist)
        force_mag[mask] = k * (2.0 * r - dist[mask]) / dist[mask]
        # Sum forces: for each i, sum over j of force_mag[i,j] * diff[i,j]
        # But we need to be careful: diff[i,j] = pts[i] - pts[j]
        # Force on i from j = force_mag[i,j] * diff[i,j] (pushes i away from j)
        forces = np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)

        # Boundary constraint: if |p| > 1 - r, push INWARD (toward origin)
        norms = np.linalg.norm(pts, axis=1)
        boundary_mask = norms > (1.0 - r)
        if np.any(boundary_mask):
            # Push toward origin: force = -k * (norm - (1-r)) / norm * pts
            excess = norms[boundary_mask] - (1.0 - r)
            forces[boundary_mask] -= k * excess / norms[boundary_mask] * pts[boundary_mask]

        # Adaptive step size
        step = 0.01
        pts_new = pts + step * forces

        # Clip to ensure inside container
        norms_new = np.linalg.norm(pts_new, axis=1)
        max_norm = 1.0 - r * 0.5
        mask_out = norms_new > max_norm
        if np.any(mask_out):
            pts_new[mask_out] *= max_norm / norms_new[mask_out]

        new_r = _score(pts_new)
        if new_r < r - 1e-10:
            # Revert if worse
            break
        if abs(new_r - r) < tol:
            r = new_r
            pts = pts_new
            break
        pts = pts_new
        r = new_r

    return pts


def _lbfgs_polish(pts):
    """Use L-BFGS-B to maximize the minimum distance (packing radius)."""
    def neg_score(x):
        p = x.reshape(-1, 2)
        return -_score(p)

    # Constraints: each point must be inside unit circle (|p| <= 1)
    # We use bounds: each coordinate in [-1, 1]
    bounds = [(-1.0, 1.0)] * (2 * N)

    result = minimize(neg_score, pts.flatten(), method='L-BFGS-B',
                      bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
    if result.success or result.fun < neg_score(pts.flatten()):
        return result.x.reshape(-1, 2)
    return pts


def _hex_lattice_init():
    """Generate a hexagonal lattice of points inside the unit disk, pick best N."""
    best_pts = None
    best_r = 0.0
    for spacing in np.linspace(0.1, 0.25, 30):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            idx = np.argsort(np.hypot(points[:, 0], points[:, 1]), kind="stable")[:N]
            pts = points[idx]
            r = _score(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
    return best_pts, best_r


def construct():
    t0 = time.time()
    # Start with hex lattice
    pts, r = _hex_lattice_init()
    if pts is None:
        pts = np.zeros((N, 2))
        for i in range(N):
            angle = 2 * np.pi * i / N
            pts[i] = [0.9 * np.cos(angle), 0.9 * np.sin(angle)]
        r = _score(pts)

    # Force-based refinement (vectorized, faster)
    if time.time() < DEADLINE - 15.0:
        pts = _force_refine(pts, max_iter=30000, tol=1e-13)

    # L-BFGS-B polish for finer convergence
    if time.time() < DEADLINE - 5.0:
        pts = _lbfgs_polish(pts)

    # Final force refinement pass
    if time.time() < DEADLINE - 3.0:
        pts = _force_refine(pts, max_iter=10000, tol=1e-14)

    # Ensure validity: clip any point outside
    norms = np.linalg.norm(pts, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        pts[mask] /= norms[mask][:, np.newaxis] * 1.0001

    return pts
