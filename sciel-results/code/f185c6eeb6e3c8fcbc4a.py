"""Program for covering the sphere with 130 equal caps."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
_DEADLINE = None
_BEST_SCORE = None
_BEST_POINTS = None
_MESH = None


def _time_left():
    return _DEADLINE[1] - time.time() if _DEADLINE else 120.0


def _build_mesh(n_theta=300, n_phi=600):
    """Build a high-resolution spherical mesh for coverage evaluation."""
    global _MESH
    if _MESH is not None:
        return _MESH
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    mesh = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    _MESH = mesh
    return mesh


def _covering_radius(points):
    """Compute covering radius in degrees using convex hull facets + mesh sampling."""
    mesh = _build_mesh()
    
    # Mesh-based estimate
    dots = mesh @ points.T
    dots = np.clip(dots, -1, 1)
    max_dots = np.max(dots, axis=1)
    min_angles = np.arccos(max_dots)
    mesh_rad = np.max(min_angles)
    
    # Convex hull facet-based refinement
    try:
        hull = ConvexHull(points)
        hull_rad = 0.0
        for simplex in hull.simplices:
            tri = points[simplex]
            v1 = tri[1] - tri[0]
            v2 = tri[2] - tri[0]
            n = np.cross(v1, v2)
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n = n / nn
            
            # Circumcenter distance
            d0 = np.arccos(np.clip(np.dot(n, tri[0]), -1, 1))
            d1 = np.arccos(np.clip(np.dot(n, tri[1]), -1, 1))
            d2 = np.arccos(np.clip(np.dot(n, tri[2]), -1, 1))
            facet_rad = max(d0, d1, d2)
            
            # Check if circumcenter is the worst point in this facet
            dists = np.arccos(np.clip(points @ n, -1, 1))
            min_dist = np.min(dists)
            
            if min_dist + 1e-10 >= min(d0, d1, d2):
                if min_dist > hull_rad:
                    hull_rad = min_dist
            else:
                # Check edge midpoints
                for i in range(3):
                    for j in range(i + 1, 3):
                        s = tri[i] + tri[j]
                        sn = np.linalg.norm(s)
                        if sn < 1e-12:
                            continue
                        mid = s / sn
                        dists_mid = np.arccos(np.clip(points @ mid, -1, 1))
                        if np.min(dists_mid) > hull_rad:
                            hull_rad = np.min(dists_mid)
        
        total_rad = max(mesh_rad, hull_rad)
    except Exception:
        total_rad = mesh_rad
    
    return total_rad * 180.0 / np.pi


def _soft_max_covering_radius(points, beta=50.0):
    """Differentiable soft-max of the covering radius over mesh points."""
    mesh = _build_mesh()
    dots = mesh @ points.T  # (M, N)
    dots = np.clip(dots, -1, 1)
    angles = np.arccos(dots)  # (M, N)
    
    # Soft min over N for each mesh point
    # Use log-sum-exp trick: min_i a_i = -log(sum(exp(-beta * a_i))) / beta
    neg_beta_angles = -beta * angles  # (M, N)
    # Subtract max for numerical stability
    max_nb = np.max(neg_beta_angles, axis=1, keepdims=True)
    exp_vals = np.exp(neg_beta_angles - max_nb)
    log_sum = np.log(np.sum(exp_vals, axis=1)) + max_nb.ravel()
    soft_min = -log_sum / beta  # (M,)
    
    # Soft max over M mesh points
    beta2 = 10.0
    neg_beta2_soft = -beta2 * soft_min
    max_nb2 = np.max(neg_beta2_soft, keepdims=True)
    exp_vals2 = np.exp(neg_beta2_soft - max_nb2)
    log_sum2 = np.log(np.sum(exp_vals2)) + max_nb2.ravel()[0]
    soft_max = -log_sum2 / beta2
    
    return soft_max * 180.0 / np.pi


def _soft_max_gradient(points, beta=50.0, beta2=10.0):
    """Compute gradient of soft-max covering radius w.r.t. points."""
    mesh = _build_mesh()
    M = len(mesh)
    Npts = len(points)
    
    dots = mesh @ points.T  # (M, N)
    dots = np.clip(dots, -1, 1)
    angles = np.arccos(dots)  # (M, N)
    
    # Soft min over N
    neg_beta_angles = -beta * angles
    max_nb = np.max(neg_beta_angles, axis=1, keepdims=True)
    exp_vals = np.exp(neg_beta_angles - max_nb)
    sum_exp = np.sum(exp_vals, axis=1, keepdims=True)
    weights = exp_vals / sum_exp  # (M, N) - softmax weights
    
    soft_min = -np.log(sum_exp) / beta + max_nb.ravel() / beta  # (M,)
    
    # Soft max over M
    neg_beta2_sm = -beta2 * soft_min
    max_nb2 = np.max(neg_beta2_sm)
    exp_vals2 = np.exp(neg_beta2_sm - max_nb2)
    weights2 = exp_vals2 / np.sum(exp_vals2)  # (M,)
    
    # Gradient of soft_min w.r.t. angles:
    # d(soft_min)/d(angles[m,n]) = -weights[m,n] / beta
    # d(angles[m,n])/d(points[k]) = -mesh[m] / sqrt(1 - (mesh[m]·points[k])^2)
    
    # Gradient of soft_max w.r.t. soft_min[m] = weights2[m] * beta2
    # Combined: d(soft_max)/d(angles[m,n]) = weights2[m] * beta2 * (-weights[m,n] / beta)
    
    grad_angles = np.outer(weights2, -1.0) * weights * (beta2 / beta)  # (M, N)
    
    # Now d(angles)/d(points): for point k, d(angles[m,k])/d(points[k]) = -mesh[m]/sqrt(1-dots[m,k]^2)
    # For point k' != k, d(angles[m,k])/d(points[k']) = 0 (angles[m,k] doesn't depend on points[k'])
    # But angles[m,n] depends on points[n], so:
    # d(soft_max)/d(points[k]) = sum_m grad_angles[m,k] * (-mesh[m] / sqrt(1 - dots[m,k]^2))
    
    # sqrt(1 - dots^2)
    ones_minus_dots_sq = 1.0 - dots * dots
    ones_minus_dots_sq = np.clip(ones_minus_dots_sq, 1e-12, None)
    denom = np.sqrt(ones_minus_dots_sq)  # (M, N)
    
    # grad_points[k] = -sum_m grad_angles[m,k] * mesh[m] / denom[m,k]
    # = -mesh.T @ (grad_angles / denom)
    
    grad_div_denom = grad_angles / denom  # (M, N)
    grad_points = -mesh.T @ grad_div_denom  # (N, 3)
    
    # Scale to degrees
    grad_points = grad_points * 180.0 / np.pi
    
    return grad_points


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    return points / norms


def _fibonacci_spiral():
    """Generate initial Fibonacci spiral configuration."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, None))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(points)


def _repulsion_force(points, strength=1.0):
    """Compute repulsion force to prevent point clustering."""
    dots = points @ points.T
    dots = np.clip(dots, -1, 1)
    angles = np.arccos(dots)
    
    # Repulsion: push points apart when they're too close
    force = np.zeros_like(points)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if angles[i, j] < np.pi / 6:  # Within 30 degrees
                # Push i away from j
                force[i] -= points[j] * (np.pi / 6 - angles[i, j])
    
    return force * strength


def _optimize_with_gradient(points, max_iters=200, lr=0.01):
    """Optimize using gradient descent on soft-max covering radius."""
    pts = points.copy()
    best_local = _covering_radius(pts)
    best_local_pts = pts.copy()
    
    for it in range(max_iters):
        if _time_left() < 10:
            break
        
        # Compute gradient
        grad = _soft_max_gradient(pts, beta=50.0, beta2=10.0)
        
        # Add small repulsion to prevent clustering
        rep = _repulsion_force(pts, strength=0.01)
        total_grad = grad + rep
        
        # Gradient step
        step = lr * total_grad
        new_pts = pts - step
        new_pts = _project_to_sphere(new_pts)
        
        # Check improvement
        new_score = _covering_radius(new_pts)
        if new_score < best_local - 1e-8:
            pts = new_pts
            best_local = new_score
            best_local_pts = pts.copy()
        else:
            # Try smaller step
            new_pts = pts - 0.5 * step
            new_pts = _project_to_sphere(new_pts)
            new_score = _covering_radius(new_pts)
            if new_score < best_local - 1e-8:
                pts = new_pts
                best_local = new_score
                best_local_pts = pts.copy()
            else:
                # No improvement, reduce learning rate
                lr *= 0.9
                if lr < 1e-6:
                    break
    
    return best_local_pts


def _targeted_face_reduction(points, max_iters=100):
    """Targeted reduction of the largest facet circumradius."""
    pts = points.copy()
    
    for it in range(max_iters):
        if _time_left() < 10:
            break
        
        # Find the worst facet
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        worst_rad = 0.0
        worst_simplex = None
        for simplex in hull.simplices:
            tri = pts[simplex]
            v1 = tri[1] - tri[0]
            v2 = tri[2] - tri[0]
            n = np.cross(v1, v2)
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n = n / nn
            d0 = np.arccos(np.clip(np.dot(n, tri[0]), -1, 1))
            d1 = np.arccos(np.clip(np.dot(n, tri[1]), -1, 1))
            d2 = np.arccos(np.clip(np.dot(n, tri[2]), -1, 1))
            facet_rad = max(d0, d1, d2)
            
            dists = np.arccos(np.clip(pts @ n, -1, 1))
            min_dist = np.min(dists)
            
            if min_dist + 1e-10 >= min(d0, d1, d2):
                if min_dist > worst_rad:
                    worst_rad = min_dist
                    worst_simplex = simplex
        
        if worst_simplex is None:
            break
        
        # Move the three vertices of the worst facet toward the circumcenter
        tri = pts[worst_simplex]
        v1 = tri[1] - tri[0]
        v2 = tri[2] - tri[0]
        n = np.cross(v1, v2)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            break
        n = n / nn
        
        # Move each vertex slightly away from the circumcenter (toward each other)
        # to reduce the circumradius
        step = 0.02
        for idx in worst_simplex:
            # Move point idx toward the centroid of the other two
            others = [i for i in worst_simplex if i != idx]
            centroid = (pts[others[0]] + pts[others[1]]) / 2
            centroid = centroid / np.linalg.norm(centroid)
            new_pt = pts[idx] + step * (centroid - pts[idx])
            new_pt = new_pt / np.linalg.norm(new_pt)
            pts[idx] = new_pt
        
        # Check if this improved things
        new_score = _covering_radius(pts)
        if new_score > _covering_radius(points) + 1e-6:
            # Revert
            pts = points.copy()
            break
    
    return pts


def _polish_step(points):
    """Combine gradient descent with targeted face reduction for final polish."""
    pts = points.copy()
    
    # First, targeted face reduction
    pts = _targeted_face_reduction(pts, max_iters=30)
    
    # Then gradient descent
    pts = _optimize_with_gradient(pts, max_iters=50, lr=0.005)
    
    return pts


def construct():
    global _DEADLINE, _BEST_SCORE, _BEST_POINTS
    
    _DEADLINE = [time.time(), time.time() + 145.0]
    
    # Build mesh
    _build_mesh(300, 600)
    
    # Start with Fibonacci spiral
    points = _fibonacci_spiral()
    score = _covering_radius(points)
    _BEST_SCORE = score
    _BEST_POINTS = points.copy()
    
    # Phase 1: Gradient-based optimization
    while _time_left() > 40:
        points = _optimize_with_gradient(points, max_iters=100, lr=0.01)
        score = _covering_radius(points)
        if score < _BEST_SCORE - 1e-8:
            _BEST_SCORE = score
            _BEST_POINTS = points.copy()
        else:
            # Try perturbation and restart
            if _time_left() < 20:
                break
            rng = np.random.default_rng()
            perturbation = rng.normal(0, 0.03, size=points.shape)
            points = points + perturbation
            points = _project_to_sphere(points)
            score = _covering_radius(points)
            if score < _BEST_SCORE - 1e-8:
                _BEST_SCORE = score
                _BEST_POINTS = points.copy()
    
    # Phase 2: Final polish with targeted face reduction
    if _time_left() > 15:
        points = _BEST_POINTS.copy()
        points = _polish_step(points)
        score = _covering_radius(points)
        if score < _BEST_SCORE - 1e-8:
            _BEST_SCORE = score
            _BEST_POINTS = points.copy()
    
    if _time_left() > 8:
        points = _BEST_POINTS.copy()
        points = _polish_step(points)
        score = _covering_radius(points)
        if score < _BEST_SCORE - 1e-8:
            _BEST_SCORE = score
            _BEST_POINTS = points.copy()
    
    # Ensure valid output
    points = _BEST_POINTS
    points = _project_to_sphere(points)
    
    # Final validation
    if not np.all(np.isfinite(points)):
        points = _fibonacci_spiral()
    
    # Check not in a hemisphere
    # Compute convex hull to verify
    try:
        hull = ConvexHull(points)
    except Exception:
        # Points are degenerate, fallback
        points = _fibonacci_spiral()
    
    return points
# EVOLVE-BLOCK-END
