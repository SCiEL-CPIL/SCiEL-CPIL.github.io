"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 135.0  # safety margin before 150s kill

def _to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _covering_radius(pts):
    """
    Compute the covering radius (in radians) of a set of points on the sphere.
    Uses convex hull facets: for each facet, the circumcenter on the sphere
    gives a candidate for the largest empty circle center.
    """
    pts = _to_sphere(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return _covering_radius_fallback(pts)
    
    max_radius = 0.0
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            continue
        
        x = cross / norm_cross
        
        dot_a = np.dot(x, a)
        dot_a = np.clip(dot_a, -1.0, 1.0)
        r1 = np.arccos(dot_a)
        
        dot_a_neg = -dot_a
        dot_a_neg = np.clip(dot_a_neg, -1.0, 1.0)
        r2 = np.arccos(dot_a_neg)
        
        candidate = max(r1, r2)
        if candidate > max_radius:
            max_radius = candidate
    
    # Check antipodes of each point
    for i in range(len(pts)):
        neg_pt = -pts[i]
        dots = pts @ neg_pt
        dots = np.clip(dots, -1.0, 1.0)
        min_dist = np.arccos(np.min(dots))
        if min_dist > max_radius:
            max_radius = min_dist
    
    return max_radius

def _covering_radius_fallback(pts):
    """Fallback covering radius computation via moderate sampling."""
    n_theta = 120
    n_phi = 240
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    max_radius = 0.0
    for theta in thetas:
        st = np.sin(theta)
        ct = np.cos(theta)
        y_batch = np.column_stack([
            st * np.cos(phis),
            st * np.sin(phis),
            np.full_like(phis, ct)
        ])
        dots = pts @ y_batch.T
        dots = np.clip(dots, -1.0, 1.0)
        min_dists = np.arccos(np.min(dots, axis=0))
        max_radius = max(max_radius, np.max(min_dists))
    return max_radius

def _objective(pts):
    """Compute the objective (covering radius in radians)."""
    return _covering_radius(pts)

def _soft_max_objective_and_grad(pts, beta):
    """
    Compute the soft-max (log-sum-exp) approximation of the covering radius
    and its gradient with respect to the points.
    
    The soft-max of distances d_i is approximated by (1/beta) * log(sum(exp(beta * d_i))).
    Here d_i is the angular distance from point i to its nearest neighbor.
    We use a soft-min for the nearest neighbor distance to make it differentiable.
    """
    pts = _to_sphere(pts)
    n = len(pts)
    
    # Compute pairwise dot products
    dots = pts @ pts.T
    np.fill_diagonal(dots, 1.0)
    dots = np.clip(dots, -1.0, 1.0)
    
    # Compute pairwise angular distances
    dists = np.arccos(dots)
    np.fill_diagonal(dists, 0.0)
    
    # Soft-min for nearest neighbor distance for each point i:
    # d_min_i = - (1/beta) * log(sum_j exp(-beta * d_ij))
    # This is differentiable and approximates min_j d_ij
    
    # For the soft-max of d_min_i:
    # F = (1/beta) * log(sum_i exp(beta * d_min_i))
    # 
    # To make this tractable, we compute the gradient numerically-free via chain rule.
    # d_min_i = - (1/beta) * log(sum_j exp(-beta * d_ij))
    # 
    # Let S_i = sum_j exp(-beta * d_ij)
    # d_min_i = - (1/beta) * log(S_i)
    # 
    # F = (1/beta) * log(sum_i exp(beta * d_min_i))
    #   = (1/beta) * log(sum_i exp(-log(S_i)))
    #   = (1/beta) * log(sum_i 1/S_i)
    #
    # This is elegant! F = (1/beta) * log(sum_i 1/S_i)
    # where S_i = sum_j exp(-beta * d_ij)
    
    # Compute exp(-beta * dists)
    exp_terms = np.exp(-beta * dists)
    
    # S_i = sum_j exp(-beta * d_ij)
    S = exp_terms.sum(axis=1)
    
    # F = (1/beta) * log(sum_i 1/S_i)
    inv_S = 1.0 / S
    F = (1.0 / beta) * np.log(np.sum(inv_S))
    
    # Gradient computation:
    # F = (1/beta) * log(sum_i 1/S_i)
    # dF/dS_i = (1/beta) * (1/(sum_j 1/S_j)) * (-1/S_i^2)
    # 
    # S_i = sum_j exp(-beta * d_ij)
    # dS_i/dd_ij = -beta * exp(-beta * d_ij)
    # 
    # d_ij = arccos(d_ij_dot) where d_ij_dot = pts[i] . pts[j]
    # dd_ij/dd_ij_dot = -1/sqrt(1 - d_ij_dot^2) = -1/sin(d_ij)
    # 
    # dd_ij_dot/dpts[k] = ... complex, so let's use a different approach.
    
    # Actually, let's compute the gradient more directly.
    # The soft-max objective F depends on pts through the distances.
    # We'll compute the gradient using the chain rule carefully.
    
    # Let's use a simpler formulation. Instead of the double soft (soft-min then soft-max),
    # let's directly minimize the maximum distance using a differentiable approximation.
    
    # Alternative: minimize sum of exp(beta * d_min_i) which is equivalent to soft-max.
    # Or even simpler: use the fact that we want to minimize the covering radius,
    # which is max over all y on sphere of min_i d(y, pts[i]).
    
    # A common differentiable approximation is to use:
    # F = (1/beta) * log(sum_i exp(beta * d_min_i))
    # where d_min_i = min_j d(i, j)
    
    # But computing d_min_i exactly is non-differentiable. The soft-min trick gives:
    # d_min_i ≈ -(1/beta) * log(sum_j exp(-beta * d_ij))
    
    # So F ≈ (1/beta) * log(sum_i exp(-(log(sum_j exp(-beta * d_ij)))))
    #      = (1/beta) * log(sum_i 1/(sum_j exp(-beta * d_ij)))
    
    # This is what we have. Now for the gradient:
    # dF/dpts[k] = sum_i (dF/dS_i) * (dS_i/dpts[k])
    # 
    # dF/dS_i = -(1/beta) * (1/S_i^2) / (sum_m 1/S_m)
    # 
    # S_i = sum_j exp(-beta * d_ij)
    # dS_i/dpts[k] = sum_j (-beta * exp(-beta * d_ij)) * (d d_ij / dpts[k])
    # 
    # d d_ij / dpts[k] = 0 if k != i and k != j
    #                   = -pts[j]/sin(d_ij) if k == i  (since d_ij = arccos(pts[i].pts[j]))
    #                   = -pts[i]/sin(d_ij) if k == j
    
    # This is getting complex. Let's use finite differences or a simpler approach.
    # Given time constraints, let's use a direct gradient computation.
    
    # Actually, let's simplify. We'll compute the gradient numerically using a small perturbation
    # but that's expensive. Instead, let's use the analytical gradient.
    
    # Let T = sum_m 1/S_m
    T = np.sum(inv_S)
    
    # dF/dS_i = -(1/beta) * inv_S[i] / (S[i] * T)
    dF_dS = -(1.0 / beta) * inv_S / (S * T)
    
    # Now dS_i/dpts[k]:
    # S_i = sum_j exp(-beta * d_ij)
    # dS_i/dpts[k] = sum_j (-beta * exp(-beta * d_ij)) * dd_ij_ddpts[k]
    # 
    # For k == i: dd_ij/ddpts[i] = -pts[j] / sin(d_ij) for j != i
    # For k == j: dd_ij/ddpts[j] = -pts[i] / sin(d_ij) for i != j
    
    # Let's compute this efficiently.
    # grad_dS[i, k] = sum_j (-beta * exp(-beta * d_ij)) * dd_ij_ddpts[k]
    
    # For each i:
    # grad_dS[i, i] = sum_{j != i} (-beta * exp(-beta * d_ij)) * (-pts[j] / sin(d_ij))
    #                = beta * sum_{j != i} exp(-beta * d_ij) * pts[j] / sin(d_ij)
    # 
    # grad_dS[i, j] (for j != i) = (-beta * exp(-beta * d_ij)) * (-pts[i] / sin(d_ij))
    #                            = beta * exp(-beta * d_ij) * pts[i] / sin(d_ij)
    
    # Compute sin(d_ij)
    sin_dists = np.sin(dists)
    sin_dists[np.diag_indices_from(sin_dists)] = 1.0  # avoid division by zero
    
    # weights_ij = beta * exp(-beta * d_ij) / sin(d_ij)
    weights = beta * exp_terms / sin_dists
    np.fill_diagonal(weights, 0.0)
    
    # grad_dS[i, i] = sum_j weights[i, j] * pts[j]
    grad_dS_diag = weights @ pts  # shape (n, 3)
    
    # grad_dS[i, j] for j != i = weights[i, j] * pts[i]
    # So the full gradient matrix is:
    # grad_dS[i, :] = grad_dS_diag[i] + (sum_j weights[i, j]) * pts[i] - weights[i, i] * pts[i]
    # Since weights[i, i] = 0:
    # grad_dS[i, :] = grad_dS_diag[i] + (weights[i, :].sum()) * pts[i]
    
    weight_sums = weights.sum(axis=1, keepdims=True)
    grad_dS = grad_dS_diag + weight_sums * pts  # shape (n, 3)
    
    # Now dF/dpts[k] = sum_i dF_dS[i] * grad_dS[i, k]
    # In matrix form: dF/dpts = (dF_dS[:, None] * grad_dS).sum(axis=0)
    # Wait, no. dF/dpts[k] = sum_i dF_dS[i] * grad_dS[i, k]
    # So dF/dpts = grad_dS.T @ dF_dS
    
    grad_F = grad_dS.T @ dF_dS  # shape (3,) -- this is wrong, we need per-point gradient
    
    # Actually, dF/dpts[i] = sum_i' dF_dS[i'] * grad_dS[i', i]
    # So dF/dpts = (dF_dS[:, None] * grad_dS).sum(axis=0) is not right either.
    # 
    # Let me redo this. We have:
    # dF/dpts[i] = sum_{i'} (dF/dS_{i'}) * (dS_{i'}/dpts[i])
    #            = sum_{i'} dF_dS[i'] * grad_dS[i', i]
    # 
    # So dF/dpts[i] = (dF_dS * grad_dS[:, i]).sum()
    # In matrix form: dF/dpts = (dF_dS[:, None] * grad_dS).sum(axis=0)
    # No wait, grad_dS has shape (n, 3) where grad_dS[i, k] = dS_i/dpts[k]
    # So dF/dpts[k] = sum_i dF_dS[i] * grad_dS[i, k]
    # dF/dpts = grad_dS.T @ dF_dS
    
    grad_F = grad_dS.T @ dF_dS  # shape (3,) -- this gives a 3-vector, not (n, 3)
    
    # I made an error. Let me think again.
    # dF/dpts[i] is a 3-vector.
    # dF/dpts[i]_k = sum_i' dF_dS[i'] * (dS_i'/dpts[i]_k) = sum_i' dF_dS[i'] * grad_dS[i', i, k]
    # 
    # But grad_dS as I computed it has shape (n, 3) where grad_dS[i, k] = dS_i/dpts[k]
    # This is correct. So dF/dpts[k] = sum_i dF_dS[i] * grad_dS[i, k]
    # This gives a 3-vector for the total gradient, but we need per-point gradients.
    
    # I think I confused myself. Let's use a different approach.
    # The gradient with respect to pts[i] is:
    # dF/dpts[i] = sum_{i'} dF_dS[i'] * (dS_i'/dpts[i])
    # 
    # dS_i'/dpts[i] is a 3-vector.
    # From our computation, grad_dS[i', i] = dS_i'/dpts[i]
    # So dF/dpts[i] = sum_{i'} dF_dS[i'] * grad_dS[i', i]
    # 
    # In matrix notation, if we define G = grad_dS (shape n x 3), then:
    # dF/dpts[i] = sum_{i'} dF_dS[i'] * G[i', i]
    # This is the i-th column of (dF_dS[:, None] * G) summed over rows? No.
    # 
    # dF/dpts[i] = (dF_dS * G[:, i]).sum()
    # So dF/dpts = (dF_dS[:, None] * G).sum(axis=0) -- no, this gives shape (3,)
    
    # Let's just compute it element-wise.
    # dF_dpts = np.zeros((n, 3))
    # for i in range(n):
    #     dF_dpts[i] = np.sum(dF_dS[:, None] * G[:, i, None], axis=0)
    # 
    # In vectorized form:
    # dF_dpts[i] = sum_{i'} dF_dS[i'] * G[i', i]
    # = (dF_dS * G.T).sum(axis=1)  -- wait, G.T has shape (3, n), G.T[k, i] = G[i, k]
    # So (dF_dS * G.T).sum(axis=1) gives for each i: sum_{i'} dF_dS[i'] * G.T[i, i'] = sum_{i'} dF_dS[i'] * G[i', i]
    # Yes! That's correct.
    
    dF_dpts = (dF_dS[:, None] * G).sum(axis=0)  # This is wrong, G is (n, 3)
    
    # Let me just do it properly:
    # dF_dpts[i] = sum_{i'} dF_dS[i'] * G[i', i]
    # G has shape (n, 3), G[i', i] doesn't make sense because i is the point index, not coordinate.
    
    # I think the issue is that G[i, k] = dS_i / dpts[k], where pts[k] is the k-th point.
    # So G has shape (n, n) where G[i, k] = dS_i / dpts[k]? No, that's not right either.
    
    # Let me restart the gradient computation more carefully.
    # 
    # S_i = sum_j exp(-beta * d_ij)
    # dS_i / dpts[k] = sum_j (-beta * exp(-beta * d_ij)) * (d d_ij / dpts[k])
    # 
    # d d_ij / dpts[k] is a 3-vector.
    # If k == i: d d_ij / dpts[i] = -pts[j] / sin(d_ij)
    # If k == j: d d_ij / dpts[j] = -pts[i] / sin(d_ij)
    # Otherwise: 0
    # 
    # So dS_i / dpts[k] is a 3-vector for each k.
    # Let's define H[i, k] = dS_i / dpts[k] as a 3-vector.
    # H has shape (n, n, 3).
    
    # This is getting too complex. Let's use a simpler approach:
    # Compute the gradient numerically using a small perturbation for each point.
    # But that's O(n^2) evaluations, which might be too slow.
    
    # Alternative: use a simpler objective that's easier to differentiate.
    # Let's minimize the sum of squared nearest-neighbor distances, which is differentiable
    # if we use the soft-min approximation.
    
    # Actually, let's just use scipy's minimize with a numerical gradient.
    # Given the time constraints, this might be the most practical approach.
    
    return F, None  # Return objective value, gradient to be computed numerically

def _l_bfgs_optimize(pts, max_time, beta_start=20.0, beta_end=100.0):
    """
    Optimize the points using L-BFGS-B on the soft-max objective.
    Uses backtracking line search for monotonic improvement.
    """
    pts = _to_sphere(pts.copy())
    
    def objective_and_grad(x):
        """Objective and gradient for L-BFGS-B."""
        pts_x = _to_sphere(x.reshape(N, 3))
        
        # Compute soft-max objective
        dots = pts_x @ pts_x.T
        np.fill_diagonal(dots, 1.0)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        np.fill_diagonal(dists, 0.0)
        
        exp_terms = np.exp(-beta_start * dists)
        S = exp_terms.sum(axis=1)
        inv_S = 1.0 / S
        F = (1.0 / beta_start) * np.log(np.sum(inv_S))
        
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            pts_plus = _to_sphere(x_plus.reshape(N, 3))
            
            dots_plus = pts_plus @ pts_plus.T
            np.fill_diagonal(dots_plus, 1.0)
            dots_plus = np.clip(dots_plus, -1.0, 1.0)
            dists_plus = np.arccos(dots_plus)
            np.fill_diagonal(dists_plus, 0.0)
            
            exp_plus = np.exp(-beta_start * dists_plus)
            S_plus = exp_plus.sum(axis=1)
            inv_S_plus = 1.0 / S_plus
            F_plus = (1.0 / beta_start) * np.log(np.sum(inv_S_plus))
            
            grad[i] = (F_plus - F) / eps
        
        return F, grad
    
    result = minimize(
        objective_and_grad,
        pts.flatten(),
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    opt_pts = _to_sphere(result.x.reshape(N, 3))
    return opt_pts

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    # Start with Fibonacci spiral initial configuration
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _to_sphere(pts)
    
    best_pts = pts.copy()
    best_obj = _objective(pts)
    
    deadline = _DEADLINE
    remaining = deadline - time.time()
    
    # Phase 1: L-BFGS optimization from initial config
    if remaining > 10:
        opt_pts = _l_bfgs_optimize(pts, max_time=min(remaining - 5, 30))
        opt_obj = _objective(opt_pts)
        if opt_obj < best_obj:
            best_obj = opt_obj
            best_pts = opt_pts.copy()
    
    # Phase 2: Multiple deterministic restarts with L-BFGS
    remaining = deadline - time.time()
    if remaining > 20:
        np.random.seed(42)
        n_restarts = min(5, int((remaining - 10) / 15))
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            # Perturb best solution
            perturbation = np.random.randn(N, 3) * 0.2
            perturbed = _to_sphere(best_pts + perturbation)
            
            opt_pts = _l_bfgs_optimize(perturbed, max_time=min(deadline - time.time() - 10, 20))
            opt_obj = _objective(opt_pts)
            
            if opt_obj < best_obj:
                best_obj = opt_obj
                best_pts = opt_pts.copy()
    
    # Phase 3: Final polish with small steps
    remaining = deadline - time.time()
    if remaining > 5:
        # Use very small perturbations and L-BFGS
        small_pert = _to_sphere(best_pts + np.random.randn(N, 3) * 0.05)
        opt_pts = _l_bfgs_optimize(small_pert, max_time=min(remaining - 3, 10))
        opt_obj = _objective(opt_pts)
        if opt_obj < best_obj:
            best_obj = opt_obj
            best_pts = opt_pts.copy()
    
    # Ensure all points are on the unit sphere
    best_pts = _to_sphere(best_pts)
    
    # Validate: check for coincident points
    dots = best_pts @ best_pts.T
    np.fill_diagonal(dots, 0)
    if np.max(np.abs(dots)) > 0.9999:
        best_pts = _to_sphere(best_pts + np.random.randn(N, 3) * 0.01)
        best_pts = _to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
