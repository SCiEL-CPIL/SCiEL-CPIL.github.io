"""Rewritten program: cover the unit sphere with 50 equal caps.

Strategy:
  1. Build several diverse initial configurations (Fibonacci spiral, icosahedral
     symmetry, random, and a perturbed known-good pattern).
  2. For each, run a gradient-descent / Lloyd-style optimisation that minimises
     the covering radius (max angular distance to nearest point).
  3. Use a deadline guard so we always return within the wall-clock limit.
"""

import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
import math

N = 50
DEADLINE_SECONDS = 115  # safety margin below 130s hard limit

def _covering_radius_deg(pts):
    """Compute the covering radius in degrees for a set of unit vectors.
    
    Uses exact circumcenter computation for each triangular facet of the convex hull.
    Returns (covering_radius_deg, worst_facet_indices, worst_circumcenter).
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: use a dense sampling
        n_samples = 20000
        i_s = np.arange(n_samples) + 0.5
        z_s = 1.0 - 2.0 * i_s / n_samples
        phi_s = np.pi * (1.0 + np.sqrt(5.0)) * i_s
        r_s = np.sqrt(np.maximum(0, 1.0 - z_s * z_s))
        samples = np.column_stack([r_s * np.cos(phi_s), r_s * np.sin(phi_s), z_s])
        max_dist = 0.0
        chunk = 5000
        for start in range(0, n_samples, chunk):
            end = min(start + chunk, n_samples)
            s = samples[start:end]
            dots = s @ pts.T
            dots = np.clip(dots, -1.0, 1.0)
            min_dist = np.arccos(dots).min(axis=1)
            max_dist = max(max_dist, min_dist.max())
        return math.degrees(max_dist), None, None
    
    max_circ_r = 0.0
    worst_simplex = None
    worst_cc = None
    
    for simplex in hull.simplices:
        tri = pts[simplex]  # (3, 3)
        a, b, c = tri[0], tri[1], tri[2]
        
        # Compute circumcenter of spherical triangle using exact formula
        # For unit vectors a, b, c, the circumcenter is proportional to a×b + b×c + c×a
        # (this is the direction of the circumcenter on the sphere)
        ab = np.cross(a, b)
        bc = np.cross(b, c)
        ca = np.cross(c, a)
        
        cc_vec = ab + bc + ca
        cc_norm = np.linalg.norm(cc_vec)
        
        if cc_norm < 1e-12:
            # Degenerate case: use the normal to the plane
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            cc_vec = n / n_norm
            cc_norm = 1.0
        
        cc = cc_vec / cc_norm
        
        # Ensure cc is on the "outside" (same side as the outward normal)
        centroid = (a + b + c) / 3.0
        if np.dot(cc, centroid) < 0:
            cc = -cc
            cc_vec = -cc_vec
            cc_norm = -cc_norm
        
        # Circumradius: angular distance from cc to any vertex
        dot_a = np.clip(np.dot(cc, a), -1.0, 1.0)
        circ_r = np.arccos(dot_a)
        
        if circ_r > max_circ_r:
            max_circ_r = circ_r
            worst_simplex = simplex
            worst_cc = cc
    
    return math.degrees(max_circ_r), worst_simplex, worst_cc


def _optimize(pts, max_iters=500, lr_init=0.01, deadline=None):
    """Hybrid optimization: Lloyd iterations + targeted worst-facet refinement.
    
    Returns (best_pts, best_cr).
    """
    pts = pts.copy()
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    best_pts = pts.copy()
    best_cr, _, _ = _covering_radius_deg(pts)
    
    lr = lr_init
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Phase 1: Lloyd iteration with Voronoi centroids
        n_samples = 5000
        i_s = np.arange(n_samples) + 0.5
        z_s = 1.0 - 2.0 * i_s / n_samples
        phi_s = np.pi * (1.0 + np.sqrt(5.0)) * i_s
        r_s = np.sqrt(np.maximum(0, 1.0 - z_s * z_s))
        samples = np.column_stack([r_s * np.cos(phi_s), r_s * np.sin(phi_s), z_s])
        
        dots = samples @ pts.T
        nearest = np.argmax(dots, axis=1)
        
        new_pts = np.zeros_like(pts)
        for k in range(N):
            mask = (nearest == k)
            if mask.sum() > 0:
                new_pts[k] = samples[mask].mean(axis=0)
            else:
                new_pts[k] = pts[k]
        
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        new_pts = new_pts / norms
        
        alpha = 0.5
        candidate = (1 - alpha) * pts + alpha * new_pts
        candidate = candidate / np.linalg.norm(candidate, axis=1, keepdims=True)
        
        cr, worst_simplex, worst_cc = _covering_radius_deg(candidate)
        if cr < best_cr:
            best_cr = cr
            best_pts = candidate.copy()
            pts = candidate
            lr *= 0.99
        else:
            alpha = 0.1
            candidate = (1 - alpha) * pts + alpha * new_pts
            candidate = candidate / np.linalg.norm(candidate, axis=1, keepdims=True)
            cr, worst_simplex, worst_cc = _covering_radius_deg(candidate)
            if cr < best_cr:
                best_cr = cr
                best_pts = candidate.copy()
                pts = candidate
                lr *= 0.99
            else:
                lr *= 0.95
                if lr < 0.001:
                    break
        
        # Phase 2: Targeted worst-facet refinement
        if worst_simplex is not None and worst_cc is not None:
            cr2, worst_simplex2, worst_cc2 = _covering_radius_deg(pts)
            if worst_simplex2 is not None:
                # Move the three vertices of the worst facet toward the circumcenter
                idx = worst_simplex2
                step = 0.05
                for _ in range(3):  # Try a few different step sizes
                    candidate2 = pts.copy()
                    for i in idx:
                        # Move point i toward the circumcenter
                        direction = worst_cc2 - pts[i]
                        direction = direction / np.linalg.norm(direction)
                        candidate2[i] = pts[i] + step * direction
                        candidate2[i] = candidate2[i] / np.linalg.norm(candidate2[i])
                    
                    cr3, _, _ = _covering_radius_deg(candidate2)
                    if cr3 < best_cr:
                        best_cr = cr3
                        best_pts = candidate2.copy()
                        pts = candidate2
                        break
                    step *= 0.5
    
    return best_pts, best_cr


def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _icosahedral(n):
    """Generate points with icosahedral symmetry."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, 0, t], [1, 0, t], [-1, 0, -t], [1, 0, -t],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [0, t, -1], [0, t, 1], [0, -t, -1], [0, -t, 1],
        [t, -1, 0], [t, 1, 0], [-t, -1, 0], [-t, 1, 0],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Use icosahedral symmetry group to generate more points
    # The icosahedral group has 60 elements. We can use the 12 vertices and
    # their images under the symmetry group.
    
    # For simplicity, just use the Fibonacci spiral as a good starting point
    # and add some random perturbation
    pts = _fibonacci_spiral(n)
    rng = np.random.RandomState(42)
    noise = rng.randn(n, 3) * 0.1
    pts = pts + noise
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _random_init(n, seed):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _polish(pts, deadline, max_iters=200):
    """High-precision polish: iteratively refine the worst facets with small steps."""
    pts = pts.copy()
    best_pts = pts.copy()
    best_cr, _, _ = _covering_radius_deg(pts)
    
    step = 0.02
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        cr, worst_simplex, worst_cc = _covering_radius_deg(pts)
        if worst_simplex is None or worst_cc is None:
            break
        
        # Try moving vertices of the worst facet toward the circumcenter
        improved = False
        for trial_step in [step, step * 0.5, step * 0.25]:
            candidate = pts.copy()
            for i in worst_simplex:
                direction = worst_cc - pts[i]
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-10:
                    direction = direction / norm_dir
                    candidate[i] = pts[i] + trial_step * direction
                    candidate[i] = candidate[i] / np.linalg.norm(candidate[i])
            
            cr_new, _, _ = _covering_radius_deg(candidate)
            if cr_new < best_cr - 1e-12:
                best_cr = cr_new
                best_pts = candidate.copy()
                pts = candidate
                improved = True
                break
        
        if not improved:
            step *= 0.7
            if step < 1e-4:
                break
    
    return best_pts, best_cr


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best_pts = None
    best_cr = float('inf')
    
    # Try multiple initializations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(_fibonacci_spiral(N))
    
    # 2. Random inits with different seeds
    for seed in [1, 2, 3, 42, 123, 999]:
        inits.append(_random_init(N, seed))
    
    # 3. Perturbed Fibonacci
    rng = np.random.RandomState(7)
    fib = _fibonacci_spiral(N)
    for scale in [0.05, 0.1, 0.2]:
        noisy = fib + rng.randn(N, 3) * scale
        noisy = noisy / np.linalg.norm(noisy, axis=1, keepdims=True)
        inits.append(noisy)
    
    # 4. Antipodal pairs + random
    for seed in [10, 20]:
        rng = np.random.RandomState(seed)
        half = N // 2
        pts_half = rng.randn(half, 3)
        pts_half = pts_half / np.linalg.norm(pts_half, axis=1, keepdims=True)
        pts = np.vstack([pts_half, -pts_half])
        if N % 2 == 1:
            extra = rng.randn(1, 3)
            extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
            pts = np.vstack([pts, extra])
        inits.append(pts)
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        # Quick validity checks
        min_dist = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(init_pts[i] - init_pts[j])
                min_dist = min(min_dist, d)
        if min_dist < 0.01:
            continue
        
        # Check if origin is inside convex hull
        try:
            hull = ConvexHull(init_pts)
            inside = True
            for simplex in hull.simplices:
                tri = init_pts[simplex]
                n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-10:
                    continue
                n = n / n_norm
                centroid = tri.mean(axis=0)
                if np.dot(n, centroid) < 0:
                    n = -n
                if np.dot(n, centroid) < 0:
                    inside = False
                    break
            if not inside:
                continue
        except Exception:
            continue
        
        # Optimize
        try:
            result_pts, cr = _optimize(init_pts, max_iters=250, deadline=deadline - 5)
            if cr < best_cr:
                best_cr = cr
                best_pts = result_pts.copy()
        except Exception:
            continue
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    # Final high-precision polish
    if time.time() < deadline - 3:
        try:
            polished_pts, polished_cr = _polish(best_pts, deadline - 2)
            if polished_cr < best_cr:
                best_cr = polished_cr
                best_pts = polished_pts
        except Exception:
            pass
    
    # Final normalization
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
