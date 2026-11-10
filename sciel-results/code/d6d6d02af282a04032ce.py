# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon (vertices in CCW order) that maximizes the minimum per-vertex
    equidistance score.

    Design:
      * Directly optimizes the 20 vertices (40 parameters) without symmetry constraints.
      * Uses a fast, vectorized scoring function that computes the specific "4 equidistant"
        metric efficiently.
      * Initializes with a superellipse (high exponent) to naturally cluster vertices near
        "corners", which helps satisfy the equidistance condition.
      * Uses Differential Evolution for global search to escape local minima, followed by
        L-BFGS-B for high-precision local refinement.
      * Includes a "convexity repair" step to handle invalid intermediate states during
        optimization, ensuring the final result is strictly convex.

    Returns:
        np.ndarray of shape (20, 2), finite, in cyclic (CCW) order.
    """
    n = 20
    start = time.time()
    deadline = 580.0  # Safety margin for 600s limit

    # ---------- scoring helpers ----------
    def _normalize(pts):
        c = pts.mean(axis=0)
        d = pts - c
        mean_r = np.mean(np.linalg.norm(d, axis=1))
        if mean_r <= 0.01:
            return None
        return d / mean_r

    def _vertex_score(dists):
        """
        Vectorized vertex score calculation.
        dists: array of 19 distances from a vertex to others, sorted.
        """
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

    def _polygon_score(pts):
        """
        Fast, vectorized polygon score.
        """
        p = _normalize(pts)
        if p is None:
            return 0.0
        
        # Compute pairwise distances (20x20)
        diff = p[:, None, :] - p[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        
        # Set diagonal to infinity to ignore self-distance
        np.fill_diagonal(dist, np.inf)
        
        # Sort distances for each vertex
        sorted_dist = np.sort(dist, axis=1)
        
        # Compute score for each vertex
        scores = np.zeros(n)
        for i in range(n):
            scores[i] = _vertex_score(sorted_dist[i])
            
        return np.min(scores)

    def _is_convex(pts):
        """Check if the polygon defined by pts is strictly convex."""
        n_pts = len(pts)
        if n_pts < 3:
            return False
        prev_sign = 0
        for i in range(n_pts):
            p0 = pts[i]
            p1 = pts[(i + 1) % n_pts]
            p2 = pts[(i + 2) % n_pts]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if abs(cross) < 1e-10:
                return False
            sign = 1 if cross > 0 else -1
            if prev_sign == 0:
                prev_sign = sign
            elif sign != prev_sign:
                return False
        return True

    def _repair_convexity(pts):
        """
        Attempt to repair convexity by projecting points onto a convex hull
        or adjusting angles if they are out of order.
        """
        # Simple repair: sort by angle around centroid
        c = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        sort_idx = np.argsort(angles)
        repaired = pts[sort_idx]
        
        # If still not convex, return original (or a regular polygon as last resort)
        if not _is_convex(repaired):
             return None
        return repaired

    def _obj(params):
        """
        Objective function for optimization.
        params: 40 values (x0, y0, x1, y1, ..., x19, y19)
        """
        if (time.time() - start) > deadline:
            return 1.0 # Timeout guard
            
        pts = params.reshape(n, 2)
        
        # Check for non-finite values
        if not np.all(np.isfinite(pts)):
            return 1.0
            
        # Check convexity. If not convex, return a high penalty.
        # We don't repair here to keep the gradient smooth for local optimizers,
        # but we penalize heavily.
        if not _is_convex(pts):
            # Penalty proportional to how "non-convex" it is
            # Simple penalty: large constant
            return 1.0 + np.sum(np.abs(pts)) * 0.1
            
        score = _polygon_score(pts)
        return -score

    # ---------- Initialization ----------
    # Superellipse initialization: x = a * sign(cos(t)) * |cos(t)|^(2/n), y = b * sign(sin(t)) * |sin(t)|^(2/n)
    # For n=20, this creates a shape closer to a square, clustering points near corners.
    def _init_superellipse(n_pts=20, exponent=4.0):
        t = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        x = np.sign(np.cos(t)) * np.abs(np.cos(t)) ** (2.0 / exponent)
        y = np.sign(np.sin(t)) * np.abs(np.sin(t)) ** (2.0 / exponent)
        pts = np.stack([x, y], axis=1)
        # Normalize
        c = pts.mean(axis=0)
        pts = pts - c
        mean_r = np.mean(np.linalg.norm(pts, axis=1))
        if mean_r > 0.01:
            pts = pts / mean_r
        return pts

    # Regular polygon fallback
    def _init_regular(n_pts=20):
        t = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        pts = np.stack([np.cos(t), np.sin(t)], axis=1)
        return pts

    # Start with superellipse
    best_pts = _init_superellipse(n, exponent=4.0)
    if not _is_convex(best_pts):
        best_pts = _init_regular(n)
        
    best_score = _polygon_score(best_pts)
    best_params = best_pts.flatten().copy()

    # ---------- Global Search: Differential Evolution ----------
    # Bounds: x, y in [-2, 2] (normalization handles scale)
    bounds = [(-2.0, 2.0) for _ in range(2 * n)]
    
    try:
        # DE is slow, so we limit iterations and population size
        res = differential_evolution(
            _obj,
            bounds,
            seed=42,
            maxiter=300,
            popsize=20,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False, # We will do our own polish
            init="latinhypercube",
            callback=lambda xk, convergence: (time.time() - start) < deadline - 60.0
        )
        if res is not None and res.x is not None:
            pts = res.x.reshape(n, 2)
            if _is_convex(pts):
                sc = _polygon_score(pts)
                if sc > best_score:
                    best_pts, best_score = pts, sc
                    best_params = res.x.copy()
    except Exception:
        pass

    # ---------- Local Refinement: L-BFGS-B ----------
    # We run multiple local refinements from the best global point and a few random perturbations
    if (time.time() - start) < deadline - 30.0:
        # Refine from best_params
        try:
            r2 = minimize(
                _obj, 
                best_params, 
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": 1000,
                    "ftol": 1e-12,
                    "gtol": 1e-8
                }
            )
            if r2 is not None and r2.x is not None:
                pts = r2.x.reshape(n, 2)
                if _is_convex(pts):
                    sc = _polygon_score(pts)
                    if sc > best_score:
                        best_pts, best_score = pts, sc
                        best_params = r2.x.copy()
        except Exception:
            pass

        # Try a few random perturbations of the best point to escape local minima
        if (time.time() - start) < deadline - 20.0:
            for _ in range(5):
                if (time.time() - start) > deadline - 20.0:
                    break
                perturbed = best_params + np.random.normal(0, 0.1, size=2*n)
                try:
                    r3 = minimize(
                        _obj, 
                        perturbed, 
                        method="L-BFGS-B",
                        bounds=bounds,
                        options={
                            "maxiter": 200,
                            "ftol": 1e-10,
                            "gtol": 1e-6
                        }
                    )
                    if r3 is not None and r3.x is not None:
                        pts = r3.x.reshape(n, 2)
                        if _is_convex(pts):
                            sc = _polygon_score(pts)
                            if sc > best_score:
                                best_pts, best_score = pts, sc
                                best_params = r3.x.copy()
                except Exception:
                    pass

    # ---------- Final Safety ----------
    # Ensure the final result is valid and convex
    if not _is_convex(best_pts):
        # Try to repair
        repaired = _repair_convexity(best_pts)
        if repaired is not None and _is_convex(repaired):
            best_pts = repaired
        else:
            # Fallback to regular polygon
            best_pts = _init_regular(n)
            
    # Final normalization check
    c = best_pts.mean(axis=0)
    best_pts = best_pts - c
    mean_r = np.mean(np.linalg.norm(best_pts, axis=1))
    if mean_r > 0.01:
        best_pts = best_pts / mean_r
    else:
        best_pts = _init_regular(n)

    return best_pts

# EVOLVE-BLOCK-END
