# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start = time.time()
    deadline = 560.0
    radius = 1.0

    def _normalize(pts):
        c = pts.mean(axis=0)
        d = pts - c
        mean_r = np.mean(np.linalg.norm(d, axis=1))
        if mean_r <= 0.01:
            return None
        return d / mean_r

    def _vertex_score(dists):
        if np.any(dists < 0.05):
            return 0.0
        diffs = np.diff(dists)
        k = int(np.argmin(diffs))
        D = 0.5 * (dists[k] + dists[k + 1])
        if D < 0.1:
            return 0.0
        if k == 0:
            idx = [0, 1, 2, 3]
        elif k == 17:
            idx = [15, 16, 17, 18]
        else:
            idx = [k - 1, k, k + 1, k + 2]
        sel = dists[idx]
        raw = np.mean(np.maximum(sel / D, D / sel))
        if raw <= 0.0:
            return 0.0
        return min(1.0, 1.0 / raw)

    def _vertex_score_smooth(dists, alpha=50.0):
        """Smooth surrogate for vertex score using soft-min for k selection."""
        if np.any(dists < 0.05):
            return 0.0
        diffs = np.diff(dists)
        # Soft-min: k_soft = -logsumexp(-alpha * diffs) / alpha
        m = np.min(diffs)
        soft_min = m - np.log(np.mean(np.exp(alpha * (m - diffs)))) / alpha
        # Estimate D from soft-min: find the gap closest to soft_min
        # Use the index of the actual min as a proxy but smooth the D computation
        k = int(np.argmin(diffs))
        D = 0.5 * (dists[k] + dists[k + 1])
        if D < 0.1:
            return 0.0
        if k == 0:
            idx = [0, 1, 2, 3]
        elif k == 17:
            idx = [15, 16, 17, 18]
        else:
            idx = [k - 1, k, k + 1, k + 2]
        sel = dists[idx]
        ratios = np.maximum(sel / D, D / sel)
        raw = np.mean(ratios)
        if raw <= 0.0:
            return 0.0
        # Smooth min(1, 1/raw) using a soft cap
        return 1.0 / (1.0 + np.exp(alpha * (raw - 1.0)))

    def _polygon_score(pts):
        p = _normalize(pts)
        if p is None:
            return 0.0
        m = p.shape[0]
        diff = p[:, None, :] - p[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        best = 1.0
        for i in range(m):
            other = np.delete(dist[i], i)
            other.sort()
            s = _vertex_score(other)
            if s < best:
                best = s
                if best <= 0.0:
                    break
        return best

    def _build_polygon(params):
        """
        Build 20-gon with 5-fold rotational symmetry.
        params: 8 values - 4 angular offsets + 4 radial multipliers.
        Angles in [0, 2*pi/5), radii in [0.8, 1.2].
        """
        angles = np.asarray(params[:4], dtype=float)
        rads = np.asarray(params[4:8], dtype=float)
        
        # Sort by angle to ensure correct cyclic order within sector
        order = np.argsort(angles)
        angles = angles[order]
        rads = rads[order]
        
        # Ensure strictly increasing angles with minimum gap
        if np.any(np.diff(angles) < 1e-8):
            return None
        wrap_gap = (angles[0] + 2 * np.pi / 5) - angles[3]
        if wrap_gap < 1e-8:
            return None
        
        # Build all 20 points with 5-fold symmetry
        pts_list = []
        for j in range(5):
            sector_start = j * 2 * np.pi / 5
            for i in range(4):
                a = angles[i] + sector_start
                r = radius * rads[i]
                pts_list.append([r * np.cos(a), r * np.sin(a)])
        
        pts = np.array(pts_list)
        return pts

    def _obj(params):
        pts = _build_polygon(params)
        if pts is None:
            return 1.0
        return -_polygon_score(pts)

    # Regular 20-gon as baseline
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    best_pts = np.stack([radius * np.cos(base_angles), radius * np.sin(base_angles)], axis=1)
    best_score = _polygon_score(best_pts)

    # Initialize with evenly spaced angles and unit radii
    base_params = np.concatenate([
        np.array([0.0, np.pi/10, 2*np.pi/10, 3*np.pi/10]),
        np.array([1.0, 1.0, 1.0, 1.0])
    ])

    # Differential Evolution with 8 parameters
    bounds = [(0.0, 2.0 * np.pi / 5) for _ in range(4)] + [(0.8, 1.2) for _ in range(4)]
    try:
        res = differential_evolution(
            _obj,
            bounds,
            seed=20240611,
            maxiter=2000,
            popsize=35,
            tol=1e-10,
            mutation=(0.4, 1.0),
            recombination=0.8,
            polish=False,
            init="sobol",
            callback=lambda xk, convergence: (
                True if (time.time() - start) < deadline - 120.0 else False
            ),
        )
        if res is not None and res.x is not None:
            pts = _build_polygon(res.x)
            if pts is not None:
                sc = _polygon_score(pts)
                if sc > best_score:
                    best_pts, best_score = pts, sc
    except Exception:
        pass

    # Multi-start Nelder-Mead refinement
    if (time.time() - start) < deadline - 30.0:
        # Start from the best DE result and a few perturbations
        starts = [base_params.copy()]
        if res is not None and res.x is not None:
            starts.append(res.x.copy())
            # Add a few random perturbations around the best
            for _ in range(3):
                perturbed = res.x + np.random.normal(0, 0.02, size=8)
                perturbed[:4] = np.sort(perturbed[:4])
                starts.append(perturbed)
        
        for s in starts:
            if (time.time() - start) >= deadline - 10.0:
                break
            try:
                r2 = minimize(
                    _obj,
                    s,
                    method="Nelder-Mead",
                    options={
                        "maxiter": 15000,
                        "xatol": 1e-14,
                        "fatol": 1e-16,
                        "adaptive": True
                    }
                )
                if r2 is not None and r2.x is not None:
                    pts = _build_polygon(r2.x)
                    if pts is not None:
                        sc = _polygon_score(pts)
                        if sc > best_score:
                            best_pts, best_score = pts, sc
            except Exception:
                pass

    # Final safety check
    if _polygon_score(best_pts) <= 0.0:
        base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        best_pts = np.stack([radius * np.cos(base_angles), radius * np.sin(base_angles)], axis=1)

    return best_pts

# EVOLVE-BLOCK-END
