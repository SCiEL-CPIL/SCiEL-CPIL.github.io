import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 130

def _normalize(x):
    """Normalize vectors to unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return x / norms

def _covering_radius_deg(points):
    """Compute covering radius in degrees using convex hull facets."""
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
        max_radius = 0.0
        for simplex in hull.simplices:
            facet_points = points[simplex]
            # Find circumcenter of the facet (triangle)
            a, b, c = facet_points
            # Circumcenter of triangle in 3D
            ab = b - a
            ac = c - a
            bc = c - b
            
            # Edge lengths
            d_ab = np.linalg.norm(ab)
            d_ac = np.linalg.norm(ac)
            d_bc = np.linalg.norm(bc)
            
            # Check for degenerate triangles
            if d_ab < 1e-10 or d_ac < 1e-10 or d_bc < 1e-10:
                continue
            
            # Area using cross product
            cross = np.cross(ab, ac)
            area = 0.5 * np.linalg.norm(cross)
            
            if area < 1e-10:
                continue
            
            # Circumradius R = (a*b*c) / (4*area)
            R = (d_ab * d_ac * d_bc) / (4.0 * area)
            
            # Circumcenter
            # Using the formula: O = a + (|ac|^2 * (b-a) x n + |ab|^2 * n x (c-a)) / (2 * |n|^2 * 2)
            # Simpler: solve for circumcenter
            # The circumcenter O satisfies |O-a|^2 = |O-b|^2 = |O-c|^2
            # This gives linear equations
            # 2*(b-a)·O = |b|^2 - |a|^2
            # 2*(c-a)·O = |c|^2 - |a|^2
            
            A_mat = np.array([2.0 * ab, 2.0 * ac])
            b_vec = np.array([np.dot(b, b) - np.dot(a, a),
                             np.dot(c, c) - np.dot(a, a)])
            
            # Solve in the plane of the triangle
            # Project onto the plane
            n = cross / np.linalg.norm(cross)
            
            # The circumcenter lies on the line through the centroid in the direction of n
            # Actually, let's use the formula directly
            # O = a + alpha*(b-a) + beta*(c-a) where alpha, beta solve the system
            
            # From |O-a|^2 = |O-b|^2:
            # 2*(b-a)·(O-a) = |b-a|^2
            # From |O-a|^2 = |O-c|^2:
            # 2*(c-a)·(O-a) = |c-a|^2
            
            # Let v = O - a = alpha*(b-a) + beta*(c-a)
            # 2*alpha*|ab|^2 + 2*beta*ab·ac = |ab|^2
            # 2*alpha*ab·ac + 2*beta*|ac|^2 = |ac|^2
            
            dot_ab_ac = np.dot(ab, ac)
            A2 = np.array([[2.0 * d_ab**2, 2.0 * dot_ab_ac],
                          [2.0 * dot_ab_ac, 2.0 * d_ac**2]])
            b2 = np.array([d_ab**2, d_ac**2])
            
            try:
                alpha_beta = np.linalg.solve(A2, b2)
            except np.linalg.LinAlgError:
                continue
            
            O = a + alpha_beta[0] * ab + alpha_beta[1] * ac
            
            # Distance from origin to circumcenter
            dist_to_origin = np.linalg.norm(O)
            
            # The cap radius is the angle such that sin(theta) = R / sqrt(1 + dist_to_origin^2)
            # Actually, the covering radius for this facet is the angular distance from the 
            # circumcenter direction to the vertices
            # The angle theta satisfies: cos(theta) = (1 + dist_to_origin^2 - R^2) / (2 * sqrt(1 + dist_to_origin^2))
            # Simpler: the angular radius is arcsin(R / sqrt(1 + dist_to_origin^2))
            
            denom = np.sqrt(1.0 + dist_to_origin**2)
            sin_theta = R / denom
            sin_theta = np.clip(sin_theta, 0, 1)
            theta = np.arcsin(sin_theta)
            
            max_radius = max(max_radius, theta)
        
        return np.degrees(max_radius)
    except Exception:
        return 180.0  # Invalid configuration

def _fast_covering_radius(points, n_test=2000):
    """Fast approximate covering radius using test points."""
    points = _normalize(points)
    # Generate test points on sphere
    j = np.arange(n_test) + 0.5
    z_t = 1.0 - 2.0 * j / n_test
    phi_t = np.pi * (1.0 + np.sqrt(5.0)) * j
    r_t = np.sqrt(np.maximum(0, 1.0 - z_t * z_t))
    y = np.column_stack([r_t * np.cos(phi_t), r_t * np.sin(phi_t), z_t])
    
    # Compute distances
    d2 = np.sum(points**2, axis=1)[None, :] + np.sum(y**2, axis=1)[:, None] - 2.0 * y @ points.T
    min_d2 = np.min(d2, axis=1)
    max_d = np.sqrt(np.max(min_d2))
    
    # Convert to angle: cos(theta) = 1 - d2/2, so theta = arccos(1 - d2/2)
    # Actually, for unit vectors, d^2 = 2(1-cos(theta)), so cos(theta) = 1 - d^2/2
    cos_theta = 1.0 - min_d2 / 2.0
    cos_theta = np.clip(cos_theta, -1, 1)
    theta = np.arccos(cos_theta)
    
    return np.degrees(np.max(theta))

def _fibonacci_sphere(n):
    """Generate points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(x)

def _spectral_energy(points, lmax=8):
    """
    Compute spectral energy based on variance of pair correlation.
    For uniform distribution, the pair correlation function should be flat.
    We measure deviations from uniformity using spherical harmonics.
    """
    points = _normalize(points)
    n = len(points)
    
    # Compute the empirical pair correlation function coefficients
    # c_lm = (1/n) * sum_i Y_lm(x_i)
    # The variance of the pair correlation is related to sum |c_lm|^2 for l > 0
    
    # For efficiency, we'll use a simplified approach:
    # The spectral energy is proportional to sum_{l=1}^{lmax} (2l+1) * |c_lm|^2
    
    # Compute using Legendre polynomials and associated Legendre functions
    # This is complex, so we'll use a simpler proxy:
    # The Thomson-like energy but weighted by spherical harmonic modes
    
    # Alternative: use the fact that for uniform distribution on sphere,
    # the expected value of Y_lm Y_l'm'* is delta_ll' delta_mm* / (4pi)
    # So the "spectral energy" can be computed as:
    # E = sum_{l=1}^{lmax} sum_m |c_lm|^2 * (2l+1)
    
    # For practical purposes, we'll use a combination of:
    # 1. Inverse distance energy (Thomson)
    # 2. Variance of nearest-neighbor distances
    
    # Pairwise distances
    d2 = np.sum(points**2, axis=1)[:, None] + np.sum(points**2, axis=1)[None, :] - 2.0 * points @ points.T
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2)
    
    # Thomson energy
    E_thomson = np.sum(1.0 / d)
    
    # Variance of nearest neighbor distances
    nn_dists = np.min(d, axis=1)
    E_nn_var = np.var(nn_dists)
    
    # Combine
    return E_thomson + 10.0 * E_nn_var

def _combined_loss(points, n_test=1000):
    """Combined loss function for optimization."""
    points = _normalize(points)
    
    # Fast covering radius estimate
    cover_rad = _fast_covering_radius(points, n_test)
    
    # Spectral energy
    spec_energy = _spectral_energy(points)
    
    # Combine: primarily minimize covering radius, secondarily minimize spectral energy
    loss = cover_rad + 0.01 * spec_energy
    
    return loss

def _gradient_step(points, step_size=0.01, n_test=1000):
    """Perform one gradient descent step using finite differences or analytic gradient."""
    points = _normalize(points)
    n = len(points)
    
    # Compute gradient of covering radius surrogate
    # Use a smooth max approximation
    j = np.arange(n_test) + 0.5
    z_t = 1.0 - 2.0 * j / n_test
    phi_t = np.pi * (1.0 + np.sqrt(5.0)) * j
    r_t = np.sqrt(np.maximum(0, 1.0 - z_t * z_t))
    y = np.column_stack([r_t * np.cos(phi_t), r_t * np.sin(phi_t), z_t])
    
    # d2[m, i] = |y_m - x_i|^2
    d2_test = np.sum(y**2, axis=1)[:, None] + np.sum(points**2, axis=1)[None, :] - 2.0 * y @ points.T
    min_d2 = np.min(d2_test, axis=1)
    
    # Smooth max with temperature tau
    tau = 30.0
    max_val = np.max(tau * min_d2)
    exp_vals = np.exp(tau * min_d2 - max_val)
    sum_exp = np.sum(exp_vals)
    w = exp_vals / sum_exp  # weights for each test point
    
    # Gradient with respect to points
    argmins = np.argmin(d2_test, axis=1)
    grad = np.zeros_like(points)
    diffs = y - points[argmins]
    contribs = -2.0 * w[:, None] * diffs
    np.add.at(grad, argmins, contribs)
    
    # Add Thomson energy gradient
    d2_pair = np.sum(points**2, axis=1)[:, None] + np.sum(points**2, axis=1)[None, :] - 2.0 * points @ points.T
    np.fill_diagonal(d2_pair, 1e-10)
    d_pair = np.sqrt(d2_pair)
    inv_d3 = 1.0 / (d2_pair * d_pair)
    np.fill_diagonal(inv_d3, 0.0)
    
    grad_thomson = - (points * np.sum(inv_d3, axis=1)[:, None]) + (inv_d3 @ points)
    
    # Combine gradients
    total_grad = grad + 0.01 * grad_thomson
    
    # Project to tangent space
    dot = np.sum(total_grad * points, axis=1, keepdims=True)
    grad_tangent = total_grad - dot * points
    
    # Update
    new_points = points - step_size * grad_tangent
    new_points = _normalize(new_points)
    
    return new_points

def _local_search(points, max_iter=100, step_size=0.005):
    """Local search to refine the configuration."""
    points = _normalize(points)
    best_points = points.copy()
    best_loss = _combined_loss(points, n_test=500)
    
    for it in range(max_iter):
        new_points = _gradient_step(points, step_size=step_size, n_test=500)
        new_loss = _combined_loss(new_points, n_test=500)
        
        if new_loss < best_loss:
            best_loss = new_loss
            best_points = new_points.copy()
            points = new_points
            # Decrease step size slowly
            step_size *= 0.99
        else:
            # Try smaller step
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return best_points

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # 1. Initialize with Fibonacci spiral
    x = _fibonacci_sphere(N)
    
    # 2. Local search with gradient descent
    x = _local_search(x, max_iter=200, step_size=0.01)
    
    # 3. Try multiple random restarts if time permits
    best_x = x.copy()
    best_score = _fast_covering_radius(x, n_test=2000)
    
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        # Random initialization
        x_r = np.random.randn(N, 3)
        x_r = _normalize(x_r)
        
        # Local search
        x_r = _local_search(x_r, max_iter=100, step_size=0.005)
        
        score_r = _fast_covering_radius(x_r, n_test=2000)
        if score_r < best_score:
            best_score = score_r
            best_x = x_r.copy()
    
    # 4. Final refinement with smaller steps
    if time.time() < deadline - 5:
        best_x = _local_search(best_x, max_iter=50, step_size=0.001)
    
    return best_x
