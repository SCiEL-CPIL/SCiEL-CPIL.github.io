import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 11

# Vertices of equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2

def _bary_to_cart(bary):
    """Convert barycentric coords (u,v) array of shape (N,2) to Cartesian (N,2)."""
    u = bary[:, 0]
    v = bary[:, 1]
    x = u * V1[0] + v * V2[0]
    y = u * V1[1] + v * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(cart):
    """Convert Cartesian (N, 2) to barycentric (u,v) where w=1-u-v."""
    x = cart[:, 0]
    y = cart[:, 1]
    v = y / (np.sqrt(3) / 2)
    u = x - 0.5 * v
    return np.column_stack([u, v])

def _all_areas_from_bary(bary):
    """
    Compute all triangle areas (absolute) from barycentric coords.
    Returns (areas, i_idx, j_idx, k_idx) for all i<j<k triples.
    """
    n = len(bary)
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_f = i_idx[mask]
    j_f = j_idx[mask]
    k_f = k_idx[mask]
    
    cart = _bary_to_cart(bary)
    pi = cart[i_f]
    pj = cart[j_f]
    pk = cart[k_f]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas, i_f, j_f, k_f

def _min_area_ratio(bary):
    """Return minimum triangle area divided by triangle area."""
    areas, _, _, _ = _all_areas_from_bary(bary)
    return np.min(areas) / TRI_AREA

def _objective_and_grad(x):
    """
    Objective: negative min area ratio. Returns (value, gradient).
    x: flat array of length 2*N (u0, v0, u1, v1, ...) in barycentric coords.
    """
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # If any point is outside simplex, return large penalty
    if np.any(u < -1e-6) or np.any(v < -1e-6) or np.any(w < -1e-6):
        # Gradient points toward interior
        grad = np.zeros_like(x)
        bary_g = grad.reshape(N, 2)
        for i in range(N):
            if u[i] < -1e-6:
                bary_g[i, 0] = 1.0
            if v[i] < -1e-6:
                bary_g[i, 1] = 1.0
            if w[i] < -1e-6:
                bary_g[i, 0] = -1.0
                bary_g[i, 1] = -1.0
        return 100.0 + np.sum(np.maximum(0, -u)**2) + np.sum(np.maximum(0, -v)**2) + np.sum(np.maximum(0, -w)**2), grad.flatten()
    
    areas, i_f, j_f, k_f = _all_areas_from_bary(bary)
    min_idx = np.argmin(areas)
    min_area = areas[min_idx]
    value = -min_area / TRI_AREA
    
    # Compute gradient with respect to barycentric coords
    # Area of triangle (i,j,k) = 0.5 * |cross(pj-pi, pk-pi)|
    # In barycentric: pi = ui*V1 + vi*V2, etc.
    # cross = (pj-pi)_x * (pk-pi)_y - (pj-pi)_y * (pk-pi)_x
    # Let's compute analytically
    
    grad = np.zeros((N, 2))
    
    # For the minimum triangle (i,j,k):
    i, j, k = i_f[min_idx], j_f[min_idx], k_f[min_idx]
    
    # Get barycentric coords
    ui, vi = bary[i]
    uj, vj = bary[j]
    uk, vk = bary[k]
    
    # Cartesian coords
    xi = ui * V1[0] + vi * V2[0]
    yi = ui * V1[1] + vi * V2[1]
    xj = uj * V1[0] + vj * V2[0]
    yj = uj * V1[1] + vj * V2[1]
    xk = uk * V1[0] + vk * V2[0]
    yk = uk * V1[1] + vk * V2[1]
    
    # cross = (xj-xi)*(yk-yi) - (yj-yi)*(xk-xi)
    dx_ji = xj - xi
    dy_ji = yj - yi
    dx_ki = xk - xi
    dy_ki = yk - yi
    cross_val = dx_ji * dy_ki - dy_ji * dx_ki
    
    # The area is 0.5*|cross|, so d(area)/d(cross) = 0.5 * sign(cross)
    sign = 1.0 if cross_val >= 0 else -1.0
    
    # d(cross)/d(xi) = -dy_ki
    # d(cross)/d(yi) = dx_ki
    # d(cross)/d(xj) = dy_ki
    # d(cross)/d(yj) = -dx_ki
    # d(cross)/d(xk) = -dy_ji
    # d(cross)/d(yk) = dx_ji
    
    # Now convert Cartesian gradients to barycentric:
    # x = u*V1[0] + v*V2[0] = u + 0.5*v
    # y = u*V1[1] + v*V2[1] = v*sqrt(3)/2
    # dx/du = 1, dx/dv = 0.5
    # dy/du = 0, dy/dv = sqrt(3)/2
    
    # d(cross)/d(u_i) = d(cross)/d(xi)*1 + d(cross)/d(yi)*0 = -dy_ki
    # d(cross)/d(v_i) = d(cross)/d(xi)*0.5 + d(cross)/d(yi)*sqrt(3)/2 = 0.5*(-dy_ki) + (sqrt(3)/2)*dx_ki
    
    # Similarly for j and k
    
    factor = 0.5 * sign / TRI_AREA  # d(value)/d(cross) * d(cross)/d(area)... 
    # Actually: value = -area/TRI_AREA = -0.5*|cross|/TRI_AREA
    # d(value)/d(cross) = -0.5*sign(cross)/TRI_AREA
    
    coeff = -0.5 * sign / TRI_AREA
    
    # For point i:
    grad[i, 0] += coeff * (-dy_ki)  # d(cross)/d(u_i)
    grad[i, 1] += coeff * (0.5 * (-dy_ki) + (np.sqrt(3)/2) * dx_ki)  # d(cross)/d(v_i)
    
    # For point j:
    grad[j, 0] += coeff * dy_ki  # d(cross)/d(u_j)
    grad[j, 1] += coeff * (0.5 * dy_ki + (np.sqrt(3)/2) * (-dx_ki))  # d(cross)/d(v_j)
    
    # For point k:
    grad[k, 0] += coeff * (-dy_ji)  # d(cross)/d(u_k)
    grad[k, 1] += coeff * (0.5 * (-dy_ji) + (np.sqrt(3)/2) * dx_ji)  # d(cross)/d(v_k)
    
    return value, grad.flatten()

def _objective(x):
    """Objective function (value only)."""
    val, _ = _objective_and_grad(x)
    return val

def _clamp_bary(bary):
    """Clamp barycentric coords to valid simplex."""
    u = np.clip(bary[:, 0], 0, 1)
    v = np.clip(bary[:, 1], 0, 1)
    w = 1.0 - u - v
    # If w < 0, project back
    neg_w = w < 0
    if np.any(neg_w):
        for idx in np.where(neg_w)[0]:
            scale = 1.0 / (1.0 - w[idx])
            u[idx] *= scale
            v[idx] *= scale
    return np.column_stack([u, v])

def _fps_init_cart(rng, n_points=N, n_candidates=500):
    """Farthest Point Sampling in Cartesian space within the triangle."""
    # Generate random candidates inside triangle
    candidates = []
    for _ in range(n_candidates):
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        x = r1 * 1.0 + r2 * 0.5
        y = r2 * np.sqrt(3) / 2
        candidates.append([x, y])
    candidates = np.array(candidates)
    
    # Start with a random point
    start_idx = rng.randint(n_candidates)
    selected = [candidates[start_idx]]
    
    for _ in range(n_points - 1):
        # Find candidate farthest from all selected
        best_dist = -1
        best_idx = 0
        for ci in range(n_candidates):
            min_d = np.min(np.linalg.norm(candidates[ci] - np.array(selected), axis=1))
            if min_d > best_dist:
                best_dist = min_d
                best_idx = ci
        selected.append(candidates[best_idx])
    
    cart = np.array(selected)
    bary = _cart_to_bary(cart)
    bary = _clamp_bary(bary)
    return bary.flatten()

def _structured_inits(rng):
    """Generate diverse structured initializations."""
    inits = []
    
    # FPS-based inits
    for _ in range(8):
        inits.append(_fps_init_cart(rng))
    
    # Grid-based init
    grid_u = np.linspace(0, 1, 5)
    grid_v = np.linspace(0, 1, 5)
    grid_pts = []
    for gu in grid_u:
        for gv in grid_v:
            if gu + gv <= 1 + 1e-9:
                grid_pts.append([gu, gv])
    grid_pts = np.array(grid_pts)
    
    # Select N points from grid via FPS
    bary = np.zeros((N, 2))
    bary[0] = grid_pts[0]
    for i in range(1, N):
        best_dist = -1
        best_idx = 0
        for gi in range(len(grid_pts)):
            min_d = np.inf
            for ei in range(i):
                d = np.linalg.norm(grid_pts[gi] - bary[ei])
                if d < min_d:
                    min_d = d
            if min_d > best_dist:
                best_dist = min_d
                best_idx = gi
        bary[i] = grid_pts[best_idx]
    inits.append(bary.flatten())
    
    # Symmetric init: place points at symmetric positions
    # Center
    bary_sym = np.zeros((N, 2))
    bary_sym[0] = [1/3, 1/3]  # center
    # Vertices
    bary_sym[1] = [0.9, 0.05]
    bary_sym[2] = [0.05, 0.9]
    bary_sym[3] = [0.05, 0.05]
    # Midpoints of edges
    bary_sym[4] = [0.5, 0.0]
    bary_sym[5] = [0.0, 0.5]
    bary_sym[6] = [0.5, 0.5]
    # Quarter points
    bary_sym[7] = [0.25, 0.25]
    bary_sym[8] = [0.75, 0.25]
    bary_sym[9] = [0.25, 0.75]
    inits.append(bary_sym.flatten())
    
    # Random inits
    for _ in range(10):
        bary = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1:
                u = rng.random()
                v = rng.random()
            bary[i] = [u, v]
        inits.append(bary.flatten())
    
    return inits

def _local_slsqp(x0, deadline, maxiter=200):
    """Local optimization with SLSQP and explicit simplex constraints."""
    try:
        def cons_u(x):
            return x.reshape(N, 2)[:, 0]
        
        def cons_v(x):
            return x.reshape(N, 2)[:, 1]
        
        def cons_w(x):
            bary = x.reshape(N, 2)
            return 1.0 - bary[:, 0] - bary[:, 1]
        
        constraints = [
            {'type': 'ineq', 'fun': cons_u},
            {'type': 'ineq', 'fun': cons_v},
            {'type': 'ineq', 'fun': cons_w}
        ]
        
        bounds = [(0, 1)] * (2 * N)
        
        res = minimize(
            _objective,
            x0,
            jac=lambda x: _objective_and_grad(x)[1],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-12}
        )
        if time.time() > deadline:
            return x0
        return _clamp_bary(res.x.reshape(N, 2)).flatten()
    except Exception:
        return x0

def _local_lbfgs(x0, deadline, maxiter=200):
    """Local optimization with L-BFGS-B."""
    try:
        bounds = [(0, 1)] * (2 * N)
        res = minimize(
            _objective,
            x0,
            jac=lambda x: _objective_and_grad(x)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if time.time() > deadline:
            return x0
        return _clamp_bary(res.x.reshape(N, 2)).flatten()
    except Exception:
        return x0

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    rng = np.random.RandomState(42)
    
    best_x = None
    best_score = -float('inf')
    
    def evaluate(x):
        """Evaluate a configuration, return score and validity."""
        bary = x.reshape(N, 2)
        bary = _clamp_bary(bary)
        cart = _bary_to_cart(bary)
        
        # Check validity
        valid = True
        for i in range(N):
            xi, yi = cart[i]
            if yi < -1e-9 or yi > np.sqrt(3) * (1.0 - xi) + 1e-9:
                if not (0 <= xi <= 1 and 0 <= yi <= np.sqrt(3)/2):
                    valid = False
                    break
            # More precise check
            if yi < -1e-9:
                valid = False
                break
            if xi < -1e-9 or xi > 1 + 1e-9:
                valid = False
                break
            # Check if inside triangle: y <= sqrt(3)*x and y <= sqrt(3)*(1-x)
            if yi > np.sqrt(3) * xi + 1e-9 and xi <= 0.5:
                valid = False
                break
            if yi > np.sqrt(3) * (1.0 - xi) + 1e-9 and xi >= 0.5:
                valid = False
                break
        
        if not valid:
            return -float('inf'), False
        
        score = _min_area_ratio(bary)
        return score, True
    
    # Phase 1: Differential Evolution for global search
    if time.time() < deadline - 80:
        try:
            bounds_de = [(0, 1)] * (2 * N)
            de_result = differential_evolution(
                _objective,
                bounds_de,
                seed=42,
                maxiter=30,
                popsize=15,
                tol=1e-8,
                mutation=(0.5, 1.5),
                recombination=0.7,
                polish=False,
                init='sobol'
            )
            if time.time() < deadline - 60:
                x_de = _clamp_bary(de_result.x.reshape(N, 2)).flatten()
                score, valid = evaluate(x_de)
                if valid and score > best_score:
                    best_score = score
                    best_x = x_de
        except Exception:
            pass
    
    # Phase 2: Local optimization from multiple inits
    inits = _structured_inits(rng)
    
    for init in inits:
        if time.time() > deadline - 40:
            break
        
        # Try SLSQP
        x_opt = _local_slsqp(init, deadline, maxiter=150)
        score, valid = evaluate(x_opt)
        if valid and score > best_score:
            best_score = score
            best_x = x_opt
        
        # Try L-BFGS-B from SLSQP result
        if time.time() < deadline - 20:
            x_opt2 = _local_lbfgs(x_opt, deadline, maxiter=100)
            score2, valid2 = evaluate(x_opt2)
            if valid2 and score2 > best_score:
                best_score = score2
                best_x = x_opt2
    
    # Phase 3: Polish best solution with multiple passes
    if best_x is not None and time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            x_polished = _local_slsqp(best_x, deadline, maxiter=100)
            score, valid = evaluate(x_polished)
            if valid and score > best_score:
                best_score = score
                best_x = x_polished
    
    # Phase 4: Perturbation-based diversification
    if best_x is not None and time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            noise_scale = 0.05
            noise = rng.randn(2 * N) * noise_scale
            x_perturbed = best_x + noise
            x_perturbed = _clamp_bary(x_perturbed.reshape(N, 2)).flatten()
            
            x_opt = _local_slsqp(x_perturbed, deadline, maxiter=80)
            score, valid = evaluate(x_opt)
            if valid and score > best_score:
                best_score = score
                best_x = x_opt
    
    # Final result
    if best_x is not None:
        bary = _clamp_bary(best_x.reshape(N, 2))
        return _bary_to_cart(bary)
    
    # Fallback
    bary = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.33, 0.33], [0.33, 0.0], [0.0, 0.33],
        [0.66, 0.0], [0.66, 0.33], [0.33, 0.66],
        [0.5, 0.25], [0.25, 0.5]
    ])
    bary = _clamp_bary(bary)
    return _bary_to_cart(bary)
