# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon (vertices in CCW order) that maximizes the minimum per-vertex
    equidistance score.

    Design:
      * Uses 4-fold rotational symmetry: the 20 vertices are defined by 5 distinct
        angular offsets in the first quadrant (0 to pi/2), repeated 4 times with 90-degree
        rotations. This reduces the search space from 20 dimensions to 5, allowing for
        much higher precision optimization.
      * Guarantees convexity and distinct vertices by ensuring the 5 offsets are strictly
        increasing within [0, pi/2) and separated by a minimum gap.
      * Uses Differential Evolution for global search and Nelder-Mead for local refinement.

    Returns:
        np.ndarray of shape (20, 2), finite, in cyclic (CCW) order.
    """
    n = 20
    start = time.time()
    deadline = 540.0
    radius = 1.0

    # ---------- scoring helpers ----------
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

    def _build_polygon_from_offsets(offsets):
        """
        Takes 5 offsets in [0, pi/2) and constructs the 20-point polygon.
        Ensures strict convexity by sorting and checking gaps.
        """
        # Sort offsets
        offs = np.sort(np.asarray(offsets, dtype=float))
        
        # Ensure they are within [0, pi/2) and distinct
        # If any are out of bounds or too close, penalize heavily or clamp
        # For DE, we can just let it handle the bounds, but we need to ensure valid geometry
        # We map the search space [0, pi/2] to valid distinct points.
        
        # A simple robust way: 
        # Let the search variables be raw values. 
        # We will interpret them as cumulative gaps or just sort them.
        # If we just sort them, we might have duplicates.
        # To avoid duplicates, we can add a small epsilon based on index.
        
        # Actually, a better parameterization for DE:
        # Search in 5D space.
        # Sort the 5 values.
        # If min gap < 1e-5, return a very bad score.
        
        if np.any(offs < 0) or np.any(offs >= np.pi/2):
             # If DE pushes out of bounds (though bounds are set), handle gracefully
             return None, -1.0

        gaps = np.diff(offs)
        # Also check gap from last to first (wrapping within the quadrant? No, they are distinct in one quadrant)
        # The vertices in the first quadrant are offs[0]...offs[4].
        # The next quadrant starts at offs[0] + pi/2.
        # The gap between the last vertex in Q1 (offs[4]) and the first in Q2 (offs[0]+pi/2) is (offs[0]+pi/2) - offs[4].
        # We need this to be > 0.
        wrap_gap = (offs[0] + np.pi/2) - offs[4]
        
        if np.any(gaps < 1e-8) or wrap_gap < 1e-8:
            return None, -1.0

        # Construct full angles
        # Q1: offs
        # Q2: offs + pi/2
        # Q3: offs + pi
        # Q4: offs + 3*pi/2
        q1 = offs
        q2 = offs + np.pi/2
        q3 = offs + np.pi
        q4 = offs + 3*np.pi/2
        
        # Concatenate and sort to get cyclic order
        angles = np.sort(np.concatenate([q1, q2, q3, q4]))
        
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        pts = np.stack([x, y], axis=1)
        return pts, 1.0

    def _obj_symmetric(offsets):
        pts, valid = _build_polygon_from_offsets(offsets)
        if not valid:
            return 1.0 # Maximize negative score, so return large positive (bad)
        return -_polygon_score(pts)

    # ---------- baseline: regular polygon ----------
    # Regular polygon has 5-fold symmetry? No, 20-fold.
    # In our 4-fold symmetric model, the regular 20-gon corresponds to:
    # Angles: 0, pi/10, 2pi/10, 3pi/10, 4pi/10
    base_offsets = np.linspace(0, np.pi/2, 5, endpoint=False)
    # Note: linspace(0, pi/2, 5) gives 0, pi/8, pi/4, 3pi/8, pi/2. 
    # Regular 20-gon has step 2pi/20 = pi/10.
    # pi/10, 2pi/10, 3pi/10, 4pi/10.
    # So base_offsets should be [0, pi/10, 2pi/10, 3pi/10, 4pi/10]?
    # Wait, if we use 4-fold symmetry, the vertices in Q1 are 0, pi/10, 2pi/10, 3pi/10, 4pi/10.
    # Yes.
    base_offsets = np.array([0, np.pi/10, 2*np.pi/10, 3*np.pi/10, 4*np.pi/10])
    
    best_pts, _ = _build_polygon_from_offsets(base_offsets)
    if best_pts is None:
        # Fallback to regular
        base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        best_pts = np.stack([radius * np.cos(base_angles), radius * np.sin(base_angles)], axis=1)
    
    best_score = _polygon_score(best_pts)
    best_offsets = base_offsets.copy()

    # ---------- Global Search: Differential Evolution ----------
    # Bounds for offsets: [0, pi/2]
    bounds = [(0.0, np.pi/2) for _ in range(5)]
    
    try:
        res = differential_evolution(
            _obj_symmetric,
            bounds,
            seed=20240611,
            maxiter=2000,
            popsize=30,
            tol=1e-10,
            mutation=(0.5, 1.0),
            recombination=0.8,
            polish=False,
            init="sobol",
            callback=lambda xk, convergence: (
                True if (time.time() - start) < deadline - 120.0 else False
            ),
        )
        if res is not None and res.x is not None:
            pts, valid = _build_polygon_from_offsets(res.x)
            if valid:
                sc = _polygon_score(pts)
                if sc > best_score:
                    best_pts, best_score = pts, sc
                    best_offsets = res.x.copy()
    except Exception:
        pass

    # ---------- Local Refinement: Nelder-Mead ----------
    if (time.time() - start) < deadline - 30.0:
        try:
            # Start from best offsets found
            x0 = best_offsets.copy()
            
            # Ensure x0 is strictly inside bounds for NM stability
            # NM doesn't handle bounds well, but our penalty function handles invalid states.
            
            r2 = minimize(
                _obj_symmetric, 
                x0, 
                method="Nelder-Mead",
                options={
                    "maxiter": 10000,
                    "xatol": 1e-10,
                    "fatol": 1e-12,
                    "adaptive": True
                }
            )
            
            if r2 is not None and r2.x is not None:
                pts, valid = _build_polygon_from_offsets(r2.x)
                if valid:
                    sc = _polygon_score(pts)
                    if sc > best_score:
                        best_pts, best_score = pts, sc
        except Exception:
            pass

    # ---------- Final Safety ----------
    if _polygon_score(best_pts) <= 0.0:
        # Fallback to regular polygon
        base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        best_pts = np.stack([radius * np.cos(base_angles), radius * np.sin(base_angles)], axis=1)
        
    return best_pts

# EVOLVE-BLOCK-END
