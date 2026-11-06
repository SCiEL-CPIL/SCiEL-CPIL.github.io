"""Improved program for covering the sphere with 130 equal caps."""
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import time

N = 130


def _covering_radius(points):
    """Compute covering radius in degrees using convex hull facets."""
    try:
        hull = ConvexHull(points)
        # For each facet, find the point on the facet farthest from the origin
        # (which is the circumcenter for the spherical cap)
        # Actually, for a facet of the convex hull of points on the unit sphere,
        # the covering radius contribution is the angular distance from the origin
        # to the farthest point in that facet's circumcircle.
        
        # Better approach: for each facet, compute the circumradius of the spherical triangle
        # formed by the facet vertices. The covering radius is the max over all facets.
        
        # For a planar triangle with vertices on the unit sphere, the circumradius
        # in angular measure can be computed.
        
        # Alternative: use the fact that the covering radius is the max over all facets
        # of the distance from the centroid (in some sense) to the farthest vertex.
        
        # Most robust: for each facet, find the point on the unit sphere that is
        # equidistant (in angle) from all three vertices if possible, or the farthest
        # vertex from the facet's "center".
        
        # Simpler and correct: the covering radius is the maximum, over all facets,
        # of the maximum angular distance from any point on that facet's spherical triangle
        # to its nearest vertex. This equals the circumradius of the spherical triangle
        # if the circumcenter lies inside the triangle, otherwise it's the max of the
        # half-arc-lengths.
        
        # Practical approach: sample the sphere densely and find max distance to nearest point.
        # But that's expensive. Instead, use the convex hull approach:
        
        # For each facet (triangle), the farthest point from the vertices within that
        # spherical triangle is either a vertex or the circumcenter (if it's inside).
        
        # Let's compute: for each facet with vertices i, j, k:
        # - The three pairwise angular distances
        # - Check if the circumcenter of the spherical triangle is inside
        # - The covering contribution is min(circumradius, max of half-arcs) if circumcenter outside
        
        # Actually, the simplest correct method: the covering radius is the maximum
        # circumradius over all facets of the convex hull, where for a facet we compute
        # the radius of the smallest circle on the sphere containing the facet vertices.
        
        # For three points on a sphere, the minimal covering circle is either:
        # 1. The circumcircle of the triangle (if acute), or
        # 2. The circle with diameter = longest side (if obtuse)
        
        max_r = 0.0
        for simplex in hull.simplices:
            # simplex contains indices of vertices
            if len(simplex) == 3:
                a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
                # Compute angular distances
                d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
                d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
                d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
                
                # Find the circumradius of the spherical triangle
                # Use the formula: R = (d_ab * d_bc * d_ca) / (4 * area)
                # where area is the spherical excess
                
                # Spherical excess E = A + B + C - pi, where A, B, C are angles at vertices
                # Alternatively, use: sin(E/2) = (sin(d_ab/2) * sin(d_bc/2) * sin(d_ca/2)) / (2 * sin(s) * sin(s-d_ab) * sin(s-d_bc) * sin(s-d_ca))
                # where s = (d_ab + d_bc + d_ca) / 2
                
                s = (d_ab + d_bc + d_ca) / 2.0
                # Check if triangle is valid
                if s <= 0 or s >= np.pi:
                    r_facet = max(d_ab, d_bc, d_ca) / 2.0
                else:
                    sin_s = np.sin(s)
                    sin_sa = np.sin(s - d_ab)
                    sin_sb = np.sin(s - d_bc)
                    sin_sc = np.sin(s - d_ca)
                    denom = 2.0 * sin_s * sin_sa * sin_sb * sin_sc
                    if abs(denom) < 1e-15:
                        r_facet = max(d_ab, d_bc, d_ca) / 2.0
                    else:
                        num = np.sin(d_ab / 2.0) * np.sin(d_bc / 2.0) * np.sin(d_ca / 2.0)
                        sin_e_half = num / denom
                        sin_e_half = np.clip(sin_e_half, -1, 1)
                        E = 2.0 * np.arcsin(abs(sin_e_half))
                        # Circumradius R satisfies: sin(R) = sin(d_ab) * sin(d_bc) * sin(d_ca) / (4 * sin(E/2) * ... )
                        # Actually, for spherical triangles: R = arcsin( (sin(d_ab) * sin(d_bc) * sin(d_ca)) / (4 * sin(E/2) * ... ) )
                        # Simpler: use the planar approximation or the formula:
                        # sin(R) = (sin(d_ab) * sin(d_bc) * sin(d_ca)) / (4 * sin(E/2) * cos(A) * cos(B) * cos(C)) -- too complex
                        
                        # Use a different approach: the circumradius of a spherical triangle
                        # with sides a, b, c and area E is given by:
                        # R = arcsin( (sin(a/2) * sin(b/2) * sin(c/2)) / (2 * sin(E/2)) )
                        # Wait, that's not right either.
                        
                        # Let's use the fact that for a spherical triangle, the circumradius R
                        # satisfies: the distance from the circumcenter to each vertex is R.
                        # We can find the circumcenter by solving for a point x on the sphere
                        # such that |x - a| = |x - b| = |x - c| in angular distance.
                        
                        # This is equivalent to: x · a = x · b = x · c (for unit vectors)
                        # So x is perpendicular to (a-b) and (b-c), i.e., x is along (a-b) × (b-c)
                        
                        cross1 = np.cross(a - b, b - c)
                        if np.linalg.norm(cross1) < 1e-15:
                            r_facet = max(d_ab, d_bc, d_ca) / 2.0
                        else:
                            x = cross1 / np.linalg.norm(cross1)
                            # Check if x or -x is the circumcenter
                            d_xa = np.arccos(np.clip(np.dot(x, a), -1, 1))
                            d_xb = np.arccos(np.clip(np.dot(x, b), -1, 1))
                            d_xc = np.arccos(np.clip(np.dot(x, c), -1, 1))
                            d_negxa = np.arccos(np.clip(-np.dot(x, a), -1, 1))
                            d_negxb = np.arccos(np.clip(-np.dot(x, b), -1, 1))
                            d_negxc = np.arccos(np.clip(-np.dot(x, c), -1, 1))
                            
                            r1 = max(d_xa, d_xb, d_xc)
                            r2 = max(d_negxa, d_negxb, d_negxc)
                            r_circum = min(r1, r2)
                            
                            # Check if the circumcenter is inside the triangle
                            # For now, use the min of circumradius and max half-side
                            r_max_side = max(d_ab, d_bc, d_ca) / 2.0
                            r_facet = min(r_circum, r_max_side)
                
                max_r = max(max_r, r_facet)
            elif len(simplex) == 4:
                # Tetrahedral facet (shouldn't happen for 3D convex hull of points on sphere)
                # Handle by checking all triangles within it
                for i in range(4):
                    for j in range(i+1, 4):
                        for k in range(j+1, 4):
                            a, b, c = points[simplex[i]], points[simplex[j]], points[simplex[k]]
                            d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
                            d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
                            d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
                            r_facet = max(d_ab, d_bc, d_ca) / 2.0
                            max_r = max(max_r, r_facet)
        
        return np.degrees(max_r)
    except Exception:
        # Fallback: dense sampling
        theta = np.linspace(0, np.pi, 100)
        phi = np.linspace(0, 2 * np.pi, 200)
        TH, PH = np.meshgrid(theta, phi)
        x = np.sin(TH) * np.cos(PH)
        y = np.sin(TH) * np.sin(PH)
        z = np.cos(TH)
        samples = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        min_dists = np.min([np.arccos(np.clip(np.dot(samples, p), -1, 1)) for p in points], axis=0)
        return np.degrees(np.max(min_dists))


def _soft_covering_loss(points, temperature=0.05):
    """Soft surrogate for covering radius using log-sum-exp."""
    # Sample points on the sphere
    n_samples = 2000
    rng = np.random.RandomState(42)
    theta = rng.uniform(0, np.pi, n_samples)
    phi = rng.uniform(0, 2 * np.pi, n_samples)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    samples = np.column_stack([x, y, z])
    
    # Compute distance from each sample to each point
    dots = np.clip(samples @ points.T, -1, 1)
    dists = np.arccos(dots)  # shape (n_samples, N)
    
    # For each sample, the min distance to any point
    min_dists = np.min(dists, axis=1)
    
    # Soft max using log-sum-exp
    scaled = min_dists / temperature
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_max = temperature * log_sum_exp
    
    return soft_max


def _pairwise_uniformity_loss(points):
    """Loss encouraging uniform distribution of points."""
    # Average pairwise distance should be maximized
    dots = np.clip(points @ points.T, -1, 1)
    np.fill_diagonal(dots, 1.0)
    dists = np.arccos(dots)
    np.fill_diagonal(dists, 0.0)
    return -np.mean(dists)


def _optimize(points, max_iters=500, lr=0.01, temperature=0.05):
    """Gradient-based optimization to minimize covering radius."""
    points = points.copy()
    best_points = points.copy()
    best_loss = _soft_covering_loss(points, temperature)
    
    for it in range(max_iters):
        # Compute gradient of soft covering loss
        # d/dp of soft_max(min_dist)
        n_samples = 2000
        rng = np.random.RandomState(42)
        theta = rng.uniform(0, np.pi, n_samples)
        phi = rng.uniform(0, 2 * np.pi, n_samples)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        samples = np.column_stack([x, y, z])
        
        dots = np.clip(samples @ points.T, -1, 1)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        
        # Soft max weights
        scaled = min_dists / temperature
        max_scaled = np.max(scaled)
        exp_scaled = np.exp(scaled - max_scaled)
        weights = exp_scaled / np.sum(exp_scaled)  # shape (n_samples,)
        
        # Gradient: for each sample, the gradient w.r.t. points that achieve the min
        # d/dp_j arccos(s · p_j) = -s / (p_j · sqrt(1 - (s·p_j)^2))
        # But we only need it for the min-achieving points
        
        grad = np.zeros_like(points)
        for s_idx in range(n_samples):
            if weights[s_idx] < 1e-6:
                continue
            s = samples[s_idx]
            min_j = np.argmin(dists[s_idx])
            p_j = points[min_j]
            dot_val = np.clip(np.dot(s, p_j), -1, 1)
            denom = np.sqrt(max(1 - dot_val * dot_val, 1e-15))
            # d/dp arccos(s·p) = -s / (p · sqrt(1 - (s·p)^2)) ... wait
            # Let f(p) = arccos(s · p). df/dp = -s / sqrt(1 - (s·p)^2)
            grad[min_j] -= weights[s_idx] * s / denom
        
        # Add pairwise uniformity gradient
        dots_pp = np.clip(points @ points.T, -1, 1)
        np.fill_diagonal(dots_pp, 1.0)
        # d/dp_i mean(arccos(p_i · p_j)) for j != i
        # = -1/(N-1) * sum_{j!=i} p_j / sqrt(1 - (p_i·p_j)^2)
        grad_uniform = np.zeros_like(points)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dot_val = dots_pp[i, j]
                denom = np.sqrt(max(1 - dot_val * dot_val, 1e-15))
                grad_uniform[i] -= points[j] / (denom * (N - 1))
        
        total_grad = grad + 0.1 * grad_uniform
        
        # Update points
        points -= lr * total_grad
        
        # Renormalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / np.maximum(norms, 1e-15)
        
        # Evaluate
        current_loss = _soft_covering_loss(points, temperature)
        if current_loss < best_loss:
            best_loss = current_loss
            best_points = points.copy()
        
        # Adaptive learning rate
        if it % 50 == 49:
            lr *= 0.9
    
    return best_points


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Multi-restart optimization
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            current = points.copy()
        else:
            # Add random perturbation
            rng = np.random.RandomState(restart * 12345)
            noise = rng.randn(N, 3) * 0.1
            current = points + noise
            norms = np.linalg.norm(current, axis=1, keepdims=True)
            current = current / norms
        
        # Optimize
        optimized = _optimize(current, max_iters=200, lr=0.005, temperature=0.03)
        
        # Evaluate
        radius = _covering_radius(optimized)
        if radius < best_radius:
            best_radius = radius
            best_points = optimized.copy()
        
        # If we have time, do a polish pass on the best
        if time.time() > deadline - 20.0:
            break
    
    # Final polish if time permits
    if time.time() < deadline - 10.0:
        polished = _optimize(best_points, max_iters=100, lr=0.002, temperature=0.01)
        radius = _covering_radius(polished)
        if radius < best_radius:
            best_points = polished
    
    # Ensure output is valid
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / np.maximum(norms, 1e-15)
    
    return best_points
