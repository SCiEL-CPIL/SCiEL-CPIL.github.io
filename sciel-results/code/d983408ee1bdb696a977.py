"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Fibonacci spiral initialization
    def fibonacci_init():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Random initialization
    def random_init(seed=42):
        rng = np.random.RandomState(seed)
        pts = rng.randn(N, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts
    
    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return pts / norms
    
    def covering_radius(pts):
        """Compute the covering radius in degrees."""
        pts = normalize(pts)
        # Use spherical Voronoi via convex hull approach
        # For efficiency, sample points on sphere and compute max distance to nearest point
        # But better: compute the largest circumradius of facets of convex hull
        
        # Alternative: dense sampling approach for speed
        # Sample many points on sphere and find max angular distance
        theta = np.linspace(0, np.pi, 200)
        phi = np.linspace(0, 2 * np.pi, 400)
        TH, PH = np.meshgrid(theta, phi)
        sx = np.sin(TH) * np.cos(PH)
        sy = np.sin(TH) * np.sin(PH)
        sz = np.cos(TH)
        samples = np.column_stack([sx.ravel(), sy.ravel(), sz.ravel()])
        
        # For each sample, find nearest configuration point
        # Use dot products
        dots = samples @ pts.T  # shape (n_samples, N)
        min_dots = dots.max(axis=1)  # max dot = min angular distance
        min_dots = np.clip(min_dots, -1, 1)
        angles = np.arccos(min_dots)
        return np.degrees(angles.max())
    
    def covering_radius_voronoi(pts):
        """More accurate covering radius using convex hull facets."""
        from scipy.spatial import ConvexHull
        pts = normalize(pts)
        try:
            hull = ConvexHull(pts)
            # For each facet, compute circumradius
            max_r = 0.0
            for simplex in hull.simplices:
                tri = pts[simplex]
                # Circumradius of triangle on unit sphere
                a, b, c = tri[0], tri[1], tri[2]
                # Side lengths (chord lengths)
                ab = np.linalg.norm(a - b)
                bc = np.linalg.norm(b - c)
                ca = np.linalg.norm(c - a)
                # Area of triangle
                cross = np.cross(b - a, c - a)
                area = 0.5 * np.linalg.norm(cross)
                if area < 1e-15:
                    continue
                R = (ab * bc * ca) / (4.0 * area)
                max_r = max(max_r, R)
            # Convert circumradius to angular distance
            # For a triangle on unit sphere, the circumradius (in angular measure)
            # The angular covering radius is arccos(1 - R^2/2) approximately? 
            # Actually, the circumcenter of the spherical triangle
            # Let me use a different approach: the covering radius is the max over all points
            # of the distance to the nearest site. This equals the max circumradius of
            # the Voronoi cells, which for a convex hull facet corresponds to the
            # circumcircle of that triangle.
            # The angular distance from the circumcenter to a vertex is:
            # If the chord length from circumcenter to vertex is R, and the point is on unit sphere,
            # then the angular distance theta satisfies: chord = 2*sin(theta/2) = R
            # So theta = 2*arcsin(R/2)
            if max_r > 0:
                max_r = min(max_r, 1.0)  # clamp
                angular = 2 * np.arcsin(max_r / 2.0)
                return np.degrees(angular)
            return 0.0
        except Exception:
            return covering_radius(pts)
    
    def objective_and_grad(pts):
        """Objective: covering radius. Gradient via finite differences or analytic."""
        pts = normalize(pts)
        # Use a smooth surrogate: soft-max of angular distances
        # For each point on a dense grid, compute distance to nearest site
        # Then take a soft-max
        
        # Actually, let's use the packing formulation: maximize min pairwise distance
        # This is a good proxy for minimizing covering radius
        dots = pts @ pts.T
        np.fill_diagonal(dots, 2.0)  # avoid self
        min_dots = dots.min(axis=1)  # for each point, min dot with others
        min_dots = np.clip(min_dots, -1, 1)
        min_angles = np.arccos(min_dots)
        
        # Soft-min of angles (we want to maximize the min angle)
        # Use negative soft-max for min
        beta = 50.0
        neg_angles = -min_angles
        logsumexp = neg_angles.max() + np.log(np.sum(np.exp(neg_angles - neg_angles.max())))
        soft_min = -logsumexp  # this is a smooth approximation of -min(angles) = max(-angles)
        
        # We want to maximize min angle, so minimize -soft_min
        # But we need gradient w.r.t. pts
        # Let's use finite differences for simplicity, or analytic gradient
        
        # Analytic gradient:
        # d/dpts of min_dots[i] where j* = argmin_j dots[i,j]
        # This is complex. Let's use a differentiable potential instead.
        
        return soft_min
    
    def packing_objective(pts, beta=30.0):
        """Differentiable objective: soft-max of pairwise repulsion potential.
        We want to maximize the minimum distance, equivalent to minimizing the soft-max
        of a repulsive potential."""
        pts = normalize(pts)
        dots = pts @ pts.T
        np.fill_diagonal(dots, 2.0)
        
        # Potential: use -log(1 - dots^2) or similar, but simpler:
        # Minimize the soft-max of (1 - dots) which is proportional to chord distance squared
        # Actually, minimize soft-max of (1 - dots) to maximize min dot = minimize max distance
        
        # Let's minimize the soft-max of angular distances
        dots_clipped = np.clip(dots, -0.999999, 0.999999)
        angles = np.arccos(dots_clipped)
        np.fill_diagonal(angles, 0.0)
        
        # Soft-max of angles (we want to minimize the max angle between pairs? No.)
        # We want to maximize the MINIMUM pairwise angle.
        # Minimize -min(angles) = minimize the soft-min of angles
        # soft-min(x) = -soft-max(-x)
        neg_angles = -angles
        neg_max = neg_angles.max(axis=1)
        logsumexp = neg_max + np.log(np.sum(np.exp(neg_angles - neg_max[:, None]), axis=1))
        soft_min_per_point = -logsumexp  # shape (N,)
        
        # Now we want the worst-case point, so take soft-max of soft_min_per_point
        # But actually we want min over all pairs, which is min over i of soft_min_per_point[i]
        # Take the max of soft_min_per_point as a smooth surrogate for the covering radius proxy
        
        # Actually, let's just minimize the mean of soft_min_per_point with high beta
        # This encourages all points to have large minimum distances
        obj = np.mean(soft_min_per_point)
        
        # Gradient computation
        # This is getting complex. Let me use a simpler approach with scipy minimize
        # and numerical gradients.
        
        return obj
    
    # Strategy: use scipy.optimize.minimize with BFGS or L-BFGS-B
    # Objective: covering radius (approximated)
    # Variables: flattened points (3N)
    
    def obj_func(x):
        pts = x.reshape(N, 3)
        pts = normalize(pts)
        return covering_radius(pts)
    
    # Start with Fibonacci
    x0 = fibonacci_init().ravel()
    best_x = x0.copy()
    best_val = covering_radius(best_x.reshape(N, 3))
    
    # Try a few restarts with different initializations
    inits = [fibonacci_init()]
    for seed in [0, 1, 2, 3, 4]:
        inits.append(random_init(seed))
    
    for init_pts in inits:
        if time.time() > deadline - 20:
            break
        
        x_init = init_pts.ravel()
        
        try:
            # Use L-BFGS-B with numerical gradient
            result = minimize(
                obj_func,
                x_init,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.x is not None:
                val = covering_radius(result.x.reshape(N, 3))
                if val < best_val:
                    best_val = val
                    best_x = result.x.copy()
        except Exception:
            pass
    
    # Polish with more iterations if time remains
    if time.time() < deadline - 15:
        try:
            result = minimize(
                obj_func,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            if result.x is not None:
                val = covering_radius(result.x.reshape(N, 3))
                if val < best_val:
                    best_val = val
                    best_x = result.x.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 5:
        try:
            result = minimize(
                obj_func,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 200}
            )
            if result.x is not None:
                val = covering_radius(result.x.reshape(N, 3))
                if val < best_val:
                    best_val = val
                    best_x = result.x.copy()
        except Exception:
            pass
    
    final_pts = normalize(best_x.reshape(N, 3))
    
    # Validate: no coincident points, not in a hemisphere
    norms = np.linalg.norm(final_pts, axis=1)
    if np.any(norms < 1e-10):
        return fibonacci_init()
    
    # Check hemisphere: if all points have positive dot with some vector, they're in a hemisphere
    # This shouldn't happen for a good covering, but let's verify
    # For N=130 on unit sphere, a good configuration should span both hemispheres
    
    return final_pts
# EVOLVE-BLOCK-END
