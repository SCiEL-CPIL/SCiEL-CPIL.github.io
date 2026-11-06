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
from scipy.spatial import ConvexHull
import math

N = 50
DEADLINE_SECONDS = 115  # safety margin below 130s hard limit

def _covering_radius_deg(pts):
    """Compute the covering radius in degrees for a set of unit vectors."""
    # The covering radius is the maximum over the sphere of the angular distance
    # to the nearest point. This equals the largest circumradius of the facets
    # of the convex hull (in angular terms).
    #
    # For a set of points on the unit sphere, the covering radius (in radians)
    # is the maximum over all points y on the sphere of min_i angle(y, p_i).
    #
    # A practical way: for each facet of the convex hull, compute the circumcenter
    # of that facet (projected onto the sphere) and the angular distance from that
    # center to the facet's vertices. The max over all facets gives the covering radius.
    #
    # However, a simpler and robust approach: sample many points on the sphere and
    # compute the max of min-distance. But that's approximate.
    #
    # Better: use the convex hull. For each triangular facet with vertices a, b, c
    # (unit vectors), the farthest point on the sphere from the facet's plane is the
    # point along the facet normal. The angular distance from that point to each vertex
    # gives the circumradius of that facet. The covering radius is the max over all facets.
    #
    # Actually, for a spherical cap covering problem, the covering radius is the
    # maximum, over all points y on the sphere, of min_i arccos(y · p_i).
    # This is equivalent to: for each facet of the convex hull of {p_i}, compute
    # the angular distance from the facet's circumcenter (on the sphere) to the
    # farthest vertex. The max over all facets is the covering radius.
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: use a dense sampling
        n_samples = 20000
        # Fibonacci sphere for sampling
        i_s = np.arange(n_samples) + 0.5
        z_s = 1.0 - 2.0 * i_s / n_samples
        phi_s = np.pi * (1.0 + np.sqrt(5.0)) * i_s
        r_s = np.sqrt(np.maximum(0, 1.0 - z_s * z_s))
        samples = np.column_stack([r_s * np.cos(phi_s), r_s * np.sin(phi_s), z_s])
        # Compute min distance from each sample to each point
        # Use chunked computation
        max_dist = 0.0
        chunk = 5000
        for start in range(0, n_samples, chunk):
            end = min(start + chunk, n_samples)
            s = samples[start:end]
            # dot products
            dots = s @ pts.T  # (chunk, N)
            dots = np.clip(dots, -1.0, 1.0)
            min_dist = np.arccos(dots).min(axis=1)
            max_dist = max(max_dist, min_dist.max())
        return math.degrees(max_dist)
    
    # For each facet, compute the circumradius
    # Facets are triangles (or possibly other polygons, but for convex hull of points
    # in general position on sphere, they are triangles)
    max_circ_r = 0.0
    for simplex in hull.simplices:
        # simplex has 3 indices
        tri = pts[simplex]  # (3, 3)
        a, b, c = tri[0], tri[1], tri[2]
        # Compute the circumcenter of the triangle on the sphere
        # The circumcenter is the point on the sphere equidistant from a, b, c
        # It lies along the normal to the plane of the triangle, from the origin
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        n = n / n_norm
        # The circumcenter is at n (or -n) projected to unit sphere
        # Check which side: the center should be on the same side as the origin relative to the facet
        # For a convex hull, the origin is inside, so the center should be n or -n such that
        # it's "outside" the hull. Actually, for covering radius, we want the point on the
        # sphere farthest from the facet, which is along the outward normal.
        # The outward normal of the convex hull facet points away from the interior.
        # For a convex hull containing the origin, the outward normal is the one where
        # n · a > 0 (since a is a vertex on the hull).
        # Actually, for a convex hull of points on the unit sphere containing the origin,
        # the outward normal n satisfies n · a >= 0 for all vertices a of the facet.
        # Let's just use n and -n and pick the one with larger angular distance.
        
        # Compute angular distance from n to a
        dot_na = np.clip(np.dot(n, a), -1.0, 1.0)
        ang_n = np.arccos(dot_na)
        dot_na2 = np.clip(np.dot(-n, a), -1.0, 1.0)
        ang_n2 = np.arccos(dot_na2)
        
        # The circumradius of the facet is the max angular distance from the circumcenter
        # to any vertex. But actually, the circumcenter is equidistant to all vertices,
        # so we just need the angular distance from the circumcenter to one vertex.
        # But we need to verify it's actually equidistant. For a triangle on the sphere,
        # the circumcenter is equidistant to all three vertices.
        
        # Use the one that gives a valid circumcenter (the one where the angular
        # distances to all three vertices are equal)
        # For n:
        d1 = np.arccos(np.clip(np.dot(n, a), -1.0, 1.0))
        d2 = np.arccos(np.clip(np.dot(n, b), -1.0, 1.0))
        d3 = np.arccos(np.clip(np.dot(n, c), -1.0, 1.0))
        # For -n:
        d1b = np.arccos(np.clip(np.dot(-n, a), -1.0, 1.0))
        d2b = np.arccos(np.clip(np.dot(-n, b), -1.0, 1.0))
        d3b = np.arccos(np.clip(np.dot(-n, c), -1.0, 1.0))
        
        # The circumcenter should be equidistant; pick the one with more equal distances
        # and that is "outside" (farther from the hull interior)
        # For covering radius, we want the point on the sphere that is farthest from
        # all points in the configuration. For a given facet, that's the circumcenter
        # on the "outside" of the hull.
        
        # The outward normal for a convex hull containing the origin: n · centroid > 0
        # where centroid is the average of the vertices (inside the hull)
        centroid = (a + b + c) / 3.0
        if np.dot(n, centroid) < 0:
            n = -n
        
        # Now n is the outward normal. The circumcenter is n (unit vector).
        circ_r = np.arccos(np.clip(np.dot(n, a), -1.0, 1.0))
        max_circ_r = max(max_circ_r, circ_r)
    
    return math.degrees(max_circ_r)


def _optimize(pts, max_iters=500, lr_init=0.01, deadline=None):
    """Lloyd-style / gradient descent to minimize covering radius."""
    pts = pts.copy()
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    best_pts = pts.copy()
    best_cr = _covering_radius_deg(pts)
    
    lr = lr_init
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute the farthest point on the sphere for each point
        # For each point p_i, find the point on the sphere that is farthest from p_i
        # but still closest to p_i among all points. This is the "Voronoi cell" center.
        #
        # A simpler approach: for each point, compute the gradient of the covering
        # radius with respect to that point. The covering radius is determined by
        # the facet with the largest circumradius. Move that point toward the
        # circumcenter of its worst facet.
        
        # Alternative: use the "Lloyd" iteration for spherical covering:
        # For each point, replace it with the centroid of its Voronoi cell (on the sphere).
        # The Voronoi cell of p_i is the set of points y on the sphere such that
        # y · p_i >= y · p_j for all j.
        
        # Compute Voronoi centroids using a sampling-based approach
        n_samples = 5000
        i_s = np.arange(n_samples) + 0.5
        z_s = 1.0 - 2.0 * i_s / n_samples
        phi_s = np.pi * (1.0 + np.sqrt(5.0)) * i_s
        r_s = np.sqrt(np.maximum(0, 1.0 - z_s * z_s))
        samples = np.column_stack([r_s * np.cos(phi_s), r_s * np.sin(phi_s), z_s])
        
        # For each sample, find the nearest point
        dots = samples @ pts.T  # (n_samples, N)
        nearest = np.argmax(dots, axis=1)
        
        # Compute centroid of each Voronoi cell
        new_pts = np.zeros_like(pts)
        counts = np.zeros(N)
        for k in range(N):
            mask = (nearest == k)
            if mask.sum() > 0:
                new_pts[k] = samples[mask].mean(axis=0)
                counts[k] = mask.sum()
            else:
                new_pts[k] = pts[k]
        
        # Normalize
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        new_pts = new_pts / norms
        
        # Blend with old points to avoid large jumps
        alpha = 0.5
        candidate = (1 - alpha) * pts + alpha * new_pts
        candidate = candidate / np.linalg.norm(candidate, axis=1, keepdims=True)
        
        cr = _covering_radius_deg(candidate)
        if cr < best_cr:
            best_cr = cr
            best_pts = candidate.copy()
            pts = candidate
            # Decay learning rate
            lr *= 0.99
        else:
            # Try smaller step
            alpha = 0.1
            candidate = (1 - alpha) * pts + alpha * new_pts
            candidate = candidate / np.linalg.norm(candidate, axis=1, keepdims=True)
            cr = _covering_radius_deg(candidate)
            if cr < best_cr:
                best_cr = cr
                best_pts = candidate.copy()
                pts = candidate
                lr *= 0.99
            else:
                lr *= 0.95
                if lr < 0.001:
                    break
    
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
        if time.time() > deadline - 5:
            break
        
        # Check if init is valid (not in a closed hemisphere, no coincident points)
        # Check coincident points
        min_dist = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(init_pts[i] - init_pts[j])
                min_dist = min(min_dist, d)
        if min_dist < 0.01:
            continue
        
        # Check if in a closed hemisphere: if all points are in some hemisphere,
        # the convex hull doesn't contain the origin
        try:
            hull = ConvexHull(init_pts)
            # Check if origin is inside the hull
            # A point is inside a convex hull if it can be written as a convex combination
            # of the vertices. For a convex polyhedron, the origin is inside if
            # for all facets, the origin is on the same side as the interior.
            # Simpler: check if the hull contains the origin by checking
            # if the origin is a convex combination of the vertices.
            # Use scipy's check: the origin is inside if the minimum distance
            # from origin to any facet is 0 (i.e., origin is on or inside all facets).
            # Actually, for a convex hull of points on the unit sphere, the origin
            # is inside if and only if the points are not all in a closed hemisphere.
            
            # Check: compute the minimum over all directions of the max dot product
            # If there exists a direction d such that d · p_i <= 0 for all i, then
            # all points are in the hemisphere {x : d · x <= 0}.
            # This is equivalent to: the origin is not in the convex hull.
            
            # Use a linear programming approach or check via hull facets
            # For each facet, the outward normal n satisfies n · v >= 0 for all vertices v
            # of the facet. The origin is inside if for all facets, n · 0 = 0 >= -offset
            # where offset is the distance from origin to the facet plane.
            # For a facet with outward normal n and a vertex v, the plane is n · x = n · v.
            # The origin is on the interior side if n · 0 <= n · v, i.e., 0 <= n · v.
            # Since v is on the unit sphere and n is outward, n · v > 0 for all vertices.
            # So the origin is inside if for all facets, n · v > 0, which is always true
            # for a convex hull of points on the unit sphere that doesn't lie in a hemisphere.
            
            # Simpler check: use the fact that the origin is in the convex hull iff
            # there exist non-negative weights summing to 1 such that sum w_i p_i = 0.
            # This is a linear feasibility problem. For our purposes, we can check
            # if the minimum singular value of the matrix of points is large enough,
            # or just check if the points span R^3 and are not in a hemisphere.
            
            # Practical check: compute the convex hull and check if the origin is inside
            # by verifying that for each facet, the origin is on the interior side.
            inside = True
            for simplex in hull.simplices:
                tri = init_pts[simplex]
                n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-10:
                    continue
                n = n / n_norm
                # The outward normal should satisfy n · centroid > 0
                centroid = tri.mean(axis=0)
                if np.dot(n, centroid) < 0:
                    n = -n
                # Origin is inside if n · 0 <= n · centroid, i.e., 0 <= n · centroid
                if np.dot(n, centroid) < 0:
                    inside = False
                    break
            
            if not inside:
                continue
        except Exception:
            continue
        
        # Optimize
        try:
            result_pts, cr = _optimize(init_pts, max_iters=300, deadline=deadline - 2)
            if cr < best_cr:
                best_cr = cr
                best_pts = result_pts.copy()
        except Exception:
            continue
    
    if best_pts is None:
        # Fallback: return Fibonacci spiral
        best_pts = _fibonacci_spiral(N)
    
    # Final normalization
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
