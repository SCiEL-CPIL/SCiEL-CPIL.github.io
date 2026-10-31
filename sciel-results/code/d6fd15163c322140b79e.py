"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as optimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin against 150s limit

    # Precompute all triples
    triples = list(combinations(range(N), 3))
    idx_i = np.array([t[0] for t in triples])
    idx_j = np.array([t[1] for t in triples])
    idx_k = np.array([t[2] for t in triples])
    n_triples = len(triples)

    def _compute_areas(x):
        """Vectorized triangle area computation."""
        x_i = x[idx_i]
        x_j = x[idx_j]
        x_k = x[idx_k]
        v1 = x_j - x_i
        v2 = x_k - x_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return 0.5 * np.abs(cross)

    def _hull_area(x):
        """Compute convex hull area."""
        try:
            hull = ConvexHull(x)
            return hull.volume
        except Exception:
            return 0.0

    def score(x_flat):
        """Returns the minimum triangle area divided by convex hull area."""
        x = x_flat.reshape(-1, 2)
        hull_area = _hull_area(x)
        if hull_area < 1e-10:
            return 0.0
        areas = _compute_areas(x)
        min_area = np.min(areas)
        return min_area / hull_area

    def soft_min_score(x_flat, beta):
        """Soft-min version of the score for smoother optimization."""
        x = x_flat.reshape(-1, 2)
        hull_area = _hull_area(x)
        if hull_area < 1e-10:
            return 0.0
        areas = _compute_areas(x)
        scaled = -areas / beta
        max_scaled = np.max(scaled)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min = -beta * log_sum_exp
        return soft_min / hull_area

    def soft_min_score_gradient(x_flat, beta):
        """Gradient of soft-min score w.r.t. x_flat."""
        x = x_flat.reshape(-1, 2)
        hull_area = _hull_area(x)
        if hull_area < 1e-10:
            return np.zeros_like(x_flat)
        areas = _compute_areas(x)
        scaled = -areas / beta
        max_scaled = np.max(scaled)
        exp_scaled = np.exp(scaled - max_scaled)
        weights = exp_scaled / np.sum(exp_scaled)  # softmax weights
        # Gradient of -beta * log(sum(exp(-areas/beta))) / hull_area
        # d(soft_min)/d(area_m) = -weights[m]
        # d(area_m)/dx: area_m = 0.5*|cross|, for non-degenerate triangles
        # Use signed area for gradient: A_m = 0.5*(x_j-x_i) x (x_k-x_i)
        x_i = x[idx_i]
        x_j = x[idx_j]
        x_k = x[idx_k]
        v1 = x_j - x_i
        v2 = x_k - x_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        signed_areas = 0.5 * cross
        
        # Gradient of signed area w.r.t. each point
        grad_areas = np.zeros_like(x_flat)
        # For each triangle m, d(A_m)/dx_i = -0.5*(x_k-x_j) rotated, etc.
        # A = 0.5*(xj-xi) x (xk-xi) = 0.5*(xj_x*xk_y - xj_y*xk_x - xi_x*xk_y + xi_y*xk_x + xi_x*xj_y - xi_y*xj_x)
        # dA/dxi_x = 0.5*(-xk_y + xj_y)
        # dA/dxi_y = 0.5*(xk_x - xj_x)
        # dA/dxj_x = 0.5*(xk_y - xi_y)
        # dA/dxj_y = 0.5*(-xk_x + xi_x)
        # dA/dxk_x = 0.5*(xi_y - xj_y)
        # dA/dxk_y = 0.5*(-xi_x + xj_x)
        
        for m in range(n_triples):
            w = weights[m]
            if w < 1e-15:
                continue
            i, j, k = triples[m]
            xi, xj, xk = x[i], x[j], x[k]
            grad_areas[2*i]   += -w * 0.5 * (xk[1] - xj[1])
            grad_areas[2*i+1] += -w * 0.5 * (xk[0] - xj[0])
            grad_areas[2*j]   += -w * 0.5 * (xk[1] - xi[1])
            grad_areas[2*j+1] += -w * 0.5 * (xi[0] - xk[0])
            grad_areas[2*k]   += -w * 0.5 * (xi[1] - xj[1])
            grad_areas[2*k+1] += -w * 0.5 * (xj[0] - xi[0])
        
        # Divide by hull_area (treating hull_area as constant for gradient)
        return grad_areas / hull_area

    def annealed_objective(x_flat, beta):
        return -soft_min_score(x_flat, beta)

    def annealed_objective_gradient(x_flat, beta):
        return -soft_min_score_gradient(x_flat, beta)

    def regular_polygon(n=N, cx=0.0, cy=0.0, r=1.0, phase=0.0):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + phase
        return np.column_stack([cx + r*np.cos(angles), cy + r*np.sin(angles)])

    # Define several base grid patterns for the 13 points
    def get_base_patterns():
        """Return list of base point configurations (13x2 arrays)."""
        patterns = []
        
        # Pattern 1: 4x4 grid minus one corner, centered
        grid_4x4 = np.array([
            [-1.5, -1.5], [-0.5, -1.5], [0.5, -1.5], [1.5, -1.5],
            [-1.5, -0.5], [-0.5, -0.5], [0.5, -0.5], [1.5, -0.5],
            [-1.5, 0.5],  [-0.5, 0.5],  [0.5, 0.5],  [1.5, 0.5],
            [-1.5, 1.5]   # 13th point: bottom-left corner
        ])
        patterns.append(grid_4x4)
        
        # Pattern 2: 3x5 grid
        xs = np.array([-1.0, 0.0, 1.0])
        ys = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        grid_3x5 = np.array([[x, y] for x in xs for y in ys])[:13]
        patterns.append(grid_3x5)
        
        # Pattern 3: Regular 13-gon
        patterns.append(regular_polygon(N))
        
        # Pattern 4: Two concentric rings (7 outer, 6 inner)
        outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
        inner_angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
        outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        inner = np.column_stack([0.5*np.cos(inner_angles), 0.5*np.sin(inner_angles)])
        patterns.append(np.vstack([outer, inner]))
        
        # Pattern 5: Hexagonal lattice subset (13 points)
        # Honeycomb-like arrangement
        hex_points = np.array([
            [0.0, 0.0],
            [1.0, 0.0], [-1.0, 0.0],
            [0.5, 0.866], [-0.5, 0.866],
            [0.5, -0.866], [-0.5, -0.866],
            [2.0, 0.0], [-2.0, 0.0],
            [1.5, 0.866], [-1.5, 0.866],
            [1.5, -0.866], [-1.5, -0.866]
        ])
        patterns.append(hex_points)
        
        # Pattern 6: 5-4-4 ring structure
        ring1_angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
        ring2_angles = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
        ring3_angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
        ring1 = np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles)])
        ring2 = np.column_stack([0.65*np.cos(ring2_angles), 0.65*np.sin(ring2_angles)])
        ring3 = np.column_stack([0.35*np.cos(ring3_angles), 0.35*np.sin(ring3_angles)])
        patterns.append(np.vstack([ring1, ring2, ring3]))
        
        # Pattern 7: Perturbed 4x4 grid
        grid_pert = grid_4x4 + np.random.randn(13, 2) * 0.05
        patterns.append(grid_pert)
        
        return patterns

    def affine_transform(base_points, a, b, c, d):
        """Apply affine transformation [[a,b],[c,d]] to base points."""
        M = np.array([[a, b], [c, d]])
        return base_points @ M.T

    def score_affine(base_points, a, b, c, d):
        """Score using affine parameters."""
        x = affine_transform(base_points, a, b, c, d).flatten()
        return score(x)

    def soft_min_score_affine(base_points, a, b, c, d, beta):
        """Soft-min score using affine parameters."""
        x = affine_transform(base_points, a, b, c, d).flatten()
        return soft_min_score(x, beta)

    def soft_min_gradient_affine(base_points, a, b, c, d, beta):
        """Gradient of soft-min score w.r.t. affine params (a,b,c,d)."""
        x = affine_transform(base_points, a, b, c, d)
        # d(x_i)/d(a) = base_points[i, 0], d(x_i)/d(b) = base_points[i, 1], etc.
        # Chain rule: grad_affine = J^T @ grad_x where J is 26x4 Jacobian
        grad_x = soft_min_score_gradient(x.flatten(), beta)
        
        # J[i*2, 0] = base[i,0], J[i*2, 1] = base[i,1]
        # J[i*2+1, 2] = base[i,0], J[i*2+1, 3] = base[i,1]
        grad_a = np.sum(grad_x[0::2] * base_points[:, 0])
        grad_b = np.sum(grad_x[0::2] * base_points[:, 1])
        grad_c = np.sum(grad_x[1::2] * base_points[:, 0])
        grad_d = np.sum(grad_x[1::2] * base_points[:, 1])
        return np.array([grad_a, grad_b, grad_c, grad_d])

    def optimize_affine(base_points, a0=1.0, b0=0.0, c0=0.0, d0=1.0, 
                         beta_start=0.3, beta_end=0.005, n_steps=15, max_iter=200):
        """Optimize affine transformation parameters."""
        params = np.array([a0, b0, c0, d0])
        
        for step in range(n_steps):
            if time.time() > deadline - 5:
                break
            t = step / max(n_steps - 1, 1)
            beta = beta_start * (beta_end / beta_start) ** t
            
            def obj(p):
                return -soft_min_score_affine(base_points, p[0], p[1], p[2], p[3], beta)
            
            def grad(p):
                return -soft_min_gradient_affine(base_points, p[0], p[1], p[2], p[3], beta)
            
            try:
                res = optimize.minimize(
                    obj, params, jac=grad,
                    method='L-BFGS-B',
                    options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-10}
                )
                params = res.x
            except Exception:
                break
        
        return params

    def polish_affine(base_points, params, max_iter=1000):
        """Polish with hard objective using L-BFGS-B."""
        if time.time() > deadline - 3:
            return params
        
        def obj(p):
            return -score_affine(base_points, p[0], p[1], p[2], p[3])
        
        # Numerical gradient for hard objective
        eps = 1e-7
        def grad(p):
            g = np.zeros(4)
            for i in range(4):
                p_plus = p.copy()
                p_plus[i] += eps
                p_minus = p.copy()
                p_minus[i] -= eps
                g[i] = (obj(p_plus) - obj(p_minus)) / (2 * eps)
            return g
        
        try:
            res = optimize.minimize(
                obj, params, jac=grad,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'gtol': 1e-10, 'ftol': 1e-12}
            )
            return res.x
        except Exception:
            return params

    best_x = None
    best_score = -np.inf

    def try_update(x_candidate):
        nonlocal best_x, best_score
        s = score(x_candidate)
        if s > best_score:
            best_score = s
            best_x = x_candidate.copy()
        return s

    # Phase 0: Regular polygon baseline
    x_reg = regular_polygon(N).flatten()
    try_update(x_reg)

    # Phase 1: Affine optimization from multiple base patterns
    base_patterns = get_base_patterns()
    
    for base in base_patterns:
        if time.time() > deadline - 20:
            break
        
        # Try different initial affine parameters
        initial_affines = [
            [1.0, 0.0, 0.0, 1.0],       # Identity
            [1.0, 0.1, 0.0, 1.0],       # Slight shear
            [0.9, 0.0, 0.1, 1.1],       # Slight rotation-like
            [1.1, 0.0, 0.0, 0.9],       # Anisotropic scale
            [1.0, 0.05, -0.05, 1.0],    # Opposite shears
        ]
        
        for init in initial_affines:
            if time.time() > deadline - 15:
                break
            try:
                params = optimize_affine(base, init[0], init[1], init[2], init[3],
                                          beta_start=0.3, beta_end=0.005,
                                          n_steps=12, max_iter=150)
                x_candidate = affine_transform(base, params[0], params[1], params[2], params[3]).flatten()
                try_update(x_candidate)
            except Exception:
                pass

    # Phase 2: Polish best solution with full 26-dim optimization
    if best_x is not None and time.time() < deadline - 15:
        # Use L-BFGS-B with soft-min gradient for polish
        def polish_soft_min(x_start, beta, max_iter=500):
            def obj(x_flat):
                return -soft_min_score(x_flat, beta)
            
            def grad(x_flat):
                return -soft_min_score_gradient(x_flat, beta)
            
            try:
                res = optimize.minimize(
                    obj, x_start, jac=grad,
                    method='L-BFGS-B',
                    options={'maxiter': max_iter, 'gtol': 1e-10, 'ftol': 1e-12}
                )
                return res.x
            except Exception:
                return x_start
        
        # Anneal beta down
        x_curr = best_x.copy()
        for beta in [0.1, 0.05, 0.02, 0.01, 0.005]:
            if time.time() > deadline - 8:
                break
            x_curr = polish_soft_min(x_curr, beta, max_iter=300)
        
        try_update(x_curr)

    # Phase 3: More affine restarts from perturbed best
    for i in range(8):
        if time.time() > deadline - 10:
            break
        if best_x is not None:
            # Extract approximate affine from best solution using first base pattern
            base = base_patterns[0]
            # Simple: just perturb the affine params
            init = [1.0 + np.random.randn()*0.05, 0.0 + np.random.randn()*0.05,
                    0.0 + np.random.randn()*0.05, 1.0 + np.random.randn()*0.05]
        else:
            base = regular_polygon(N)
            init = [1.0, 0.0, 0.0, 1.0]
        
        try:
            params = optimize_affine(base, init[0], init[1], init[2], init[3],
                                      beta_start=0.2, beta_end=0.005,
                                      n_steps=10, max_iter=100)
            x_candidate = affine_transform(base, params[0], params[1], params[2], params[3]).flatten()
            try_update(x_candidate)
        except Exception:
            pass

    # Phase 4: Final polish with hard objective
    if best_x is not None and time.time() < deadline - 5:
        def final_polish(x_start, max_iter=2000):
            def obj(x_flat):
                return -score(x_flat)
            
            eps = 1e-6
            def grad(x_flat):
                g = np.zeros_like(x_flat)
                for i in range(len(x_flat)):
                    x_plus = x_flat.copy()
                    x_minus = x_flat.copy()
                    x_plus[i] += eps
                    x_minus[i] -= eps
                    g[i] = (obj(x_plus) - obj(x_minus)) / (2 * eps)
                return g
            
            try:
                res = optimize.minimize(
                    obj, x_start, jac=grad,
                    method='L-BFGS-B',
                    options={'maxiter': max_iter, 'gtol': 1e-12, 'ftol': 1e-14}
                )
                return res.x
            except Exception:
                return x_start
        
        x_polished = final_polish(best_x, max_iter=1000)
        try_update(x_polished)

    if best_x is None:
        best_x = regular_polygon(N).flatten()

    return best_x.reshape(-1, 2)

# EVOLVE-BLOCK-END
