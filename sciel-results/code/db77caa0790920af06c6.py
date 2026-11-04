"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area.
Barycentric representation + epigraph soft-min with analytic gradient + L-BFGS-B.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
TRI_AREA = np.sqrt(3) / 4.0

# Barycentric-to-Cartesian: p = lam0*V0 + lam1*V1 + lam2*V2
# V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)
# So x = lam1 + 0.5*lam2, y = (sqrt(3)/2)*lam2
_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2.0]])

# Precompute all triples (i,j,k) with i<j<k
_triples = [(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)]
_TRIPLES = np.array(_triples, dtype=np.int64)  # shape (M, 3)
_M = len(_TRIPLES)

# For each triple (i,j,k), the area = 0.5 * |det(P_j - P_i, P_k - P_i)|
# In barycentric coords, P = lam @ _VERTS (each point is a row of lam)
# P_j - P_i = (lam_j - lam_i) @ _VERTS
# det((lam_j-lam_i)@V, (lam_k-lam_i)@V) = det(V) * det(bary_j - bary_i, bary_k - bary_i)
# where bary vectors are (lam1, lam2) since lam0 = 1 - lam1 - lam2
# Actually, let's work directly: for point with bary (a,b,c) where a+b+c=1,
# cartesian = b*V1 + c*V2 (since V0=origin)
# So we can use 2D coords: (b + 0.5*c, (sqrt(3)/2)*c)
# The area of triangle with bary (a1,b1,c1),(a2,b2,c2),(a3,b3,c3) is:
# 0.5 * |det( [b2-b1, c2-c1; b3-b1, c3-c1] )| * (sqrt(3)/2)  ... let me verify
# Actually: P = b*V1 + c*V2. V1=(1,0), V2=(0.5, sqrt(3)/2)
# P_x = b + 0.5*c, P_y = (sqrt(3)/2)*c
# det(P2-P1, P3-P1) = (b2-b1+0.5*(c2-c1)) * ((sqrt(3)/2)*(c3-c1)) - ((sqrt(3)/2)*(c2-c1)) * (b3-b1+0.5*(c3-c1))
# = (sqrt(3)/2) * [ (b2-b1+0.5*(c2-c1))*(c3-c1) - (c2-c1)*(b3-b1+0.5*(c3-c1)) ]
# = (sqrt(3)/2) * [ (b2-b1)*(c3-c1) + 0.5*(c2-c1)*(c3-c1) - (c2-c1)*(b3-b1) - 0.5*(c2-c1)*(c3-c1) ]
# = (sqrt(3)/2) * [ (b2-b1)*(c3-c1) - (c2-c1)*(b3-b1) ]
# = (sqrt(3)/2) * det( [b2-b1, c2-c1; b3-b1, c3-c1] )  ... wait, that's not quite right
# Let me redo: det([x2-x1, y2-y1], [x3-x1, y3-x1]) = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
# x2-x1 = (b2-b1) + 0.5*(c2-c1)
# y2-y1 = (sqrt(3)/2)*(c2-c1)
# x3-x1 = (b3-b1) + 0.5*(c3-c1)
# y3-y1 = (sqrt(3)/2)*(c3-c1)
# det = [(b2-b1)+0.5*(c2-c1)] * (sqrt(3)/2)*(c3-c1) - (sqrt(3)/2)*(c2-c1) * [(b3-b1)+0.5*(c3-c1)]
# = (sqrt(3)/2) * { [(b2-b1)+0.5*(c2-c1)]*(c3-c1) - (c2-c1)*[(b3-b1)+0.5*(c3-c1)] }
# = (sqrt(3)/2) * { (b2-b1)(c3-c1) + 0.5(c2-c1)(c3-c1) - (c2-c1)(b3-b1) - 0.5(c2-c1)(c3-c1) }
# = (sqrt(3)/2) * { (b2-b1)(c3-c1) - (c2-c1)(b3-b1) }
# = (sqrt(3)/2) * det( [[b2-b1, c2-c1], [b3-b1, c3-c1]] )  ... NO
# det([[A,B],[C,D]]) = AD - BC
# Here A=(b2-b1), B=(c2-c1), C=(b3-b1), D=(c3-c1)
# AD-BC = (b2-b1)(c3-c1) - (c2-c1)(b3-b1)  ✓
# So area = 0.5 * (sqrt(3)/2) * |det( [b2-b1, c2-c1; b3-b1, c3-c1] )|
# = (sqrt(3)/4) * |det|
# And TRI_AREA = sqrt(3)/4, so area/TRI_AREA = |det|
# Great! So the ratio area/TRI_AREA = |det( [b2-b1, c2-c1; b3-b1, c3-c1] )|
# where for each point we use the (b, c) components of barycentric coords.

# So for each point, store (b, c) where b = lam1, c = lam2, a = 1-b-c >= 0
# The constraint is: b >= 0, c >= 0, b + c <= 1

# Precompute for each triple the coefficient vectors for the determinant
# det = (b_j - b_i)*(c_k - c_i) - (c_j - c_i)*(b_k - b_i)
# = b_j*c_k - b_j*c_i - b_i*c_k + b_i*c_i - c_j*b_k + c_j*b_i + c_i*b_k - c_i*b_i
# = b_j*c_k - b_j*c_i - b_i*c_k - c_j*b_k + c_j*b_i + c_i*b_k
# (b_i*c_i - c_i*b_i cancels)
# Let's expand as linear in each b_m, c_m:
# Coefficient of b_i: -c_k + c_j
# Coefficient of b_j: c_k - c_i
# Coefficient of b_k: -c_j + c_i
# Coefficient of c_i: -b_j + b_k
# Coefficient of c_j: b_i - b_k
# Coefficient of c_k: b_j - b_i

# So det = sum over m of (db_m * b_m + dc_m * c_m) where:
# db_i = c_j - c_k, dc_i = -b_j + b_k
# db_j = c_k - c_i, dc_j = b_i - b_k
# db_k = c_i - c_j, dc_k = b_j - b_i

# We can precompute for each triple the 6x? coefficients... but they depend on the other points' b,c values.
# So we compute them on the fly but vectorized over all M triples.

def _bary_to_bc(params):
    """params is (N, 2) array of (b, c) for each point. Return as-is."""
    return params

def _compute_dets(params):
    """
    params: (N, 2) array, each row is (b, c) for a point.
    Returns array of shape (M,) with det values (signed) for each triple.
    """
    b = params[:, 0]  # (N,)
    c = params[:, 1]  # (N,)
    
    i_idx = _TRIPLES[:, 0]
    j_idx = _TRIPLES[:, 1]
    k_idx = _TRIPLES[:, 2]
    
    bi = b[i_idx]
    bj = b[j_idx]
    bk = b[k_idx]
    ci = c[i_idx]
    cj = c[j_idx]
    ck = c[k_idx]
    
    # det = (bj - bi)*(ck - ci) - (cj - ci)*(bk - bi)
    det = (bj - bi) * (ck - ci) - (cj - ci) * (bk - bi)
    return det

def _min_area_ratio(params):
    """Return min |det| over all triples (this equals min_area / TRI_AREA)."""
    det = _compute_dets(params)
    return np.min(np.abs(det))

def _softmin_objective_and_grad(params, beta=80.0):
    """
    Smooth approximation of -min(|det|) using log-sum-exp.
    We want to maximize min(|det|), so minimize -min(|det|).
    Soft-min: -logsumexp(-beta * |det|) / beta  approximates -min(|det|) for large beta.
    
    Actually, let's use: f = -(1/beta) * log(sum(exp(-beta * |det|)))
    This approximates -min(|det|) as beta -> inf.
    
    Gradient: df/dparams = (1/beta) * sum(exp(-beta*|det|) * beta * d|det|/dparams) / sum(exp(-beta*|det|))
    = sum(w_m * d|det_m|/dparams) where w_m = exp(-beta*|det_m|) / sum(exp(-beta*|det|))
    
    d|det_m|/dparams = sign(det_m) * d(det_m)/dparams
    """
    det = _compute_dets(params)
    abs_det = np.abs(det)
    sign_det = np.sign(det)
    # Avoid issues with zero
    sign_det[abs_det < 1e-15] = 1.0
    
    # Compute weights
    neg_beta_abs = -beta * abs_det
    max_val = np.max(neg_beta_abs)
    shifted = neg_beta_abs - max_val
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp  # (M,)
    
    # Objective value
    obj = -(1.0 / beta) * (max_val + np.log(sum_exp))
    
    # Gradient: for each point m, grad = sum over triples containing m of weight * sign_det * coeff
    # We need d(det_t)/d(b_m) and d(det_t)/d(c_m) for each triple t and point m.
    # From the expansion:
    # For triple (i,j,k):
    # db_i = c_j - c_k, dc_i = -b_j + b_k
    # db_j = c_k - c_i, dc_j = b_i - b_k
    # db_k = c_i - c_j, dc_k = b_j - b_i
    
    b = params[:, 0]
    c = params[:, 1]
    i_idx = _TRIPLES[:, 0]
    j_idx = _TRIPLES[:, 1]
    k_idx = _TRIPLES[:, 2]
    
    bi = b[i_idx]
    bj = b[j_idx]
    bk = b[k_idx]
    ci = c[i_idx]
    cj = c[j_idx]
    ck = c[k_idx]
    
    # Coefficients for each triple
    db_i = cj - ck
    dc_i = -bj + bk
    db_j = ck - ci
    dc_j = bi - bk
    db_k = ci - cj
    dc_k = bj - bi
    
    # Weighted contributions
    w = weights * sign_det  # (M,)
    
    grad_b = np.zeros(N)
    grad_c = np.zeros(N)
    
    np.add.at(grad_b, i_idx, w * db_i)
    np.add.at(grad_b, j_idx, w * db_j)
    np.add.at(grad_b, k_idx, w * db_k)
    
    np.add.at(grad_c, i_idx, w * dc_i)
    np.add.at(grad_c, j_idx, w * dc_j)
    np.add.at(grad_c, k_idx, w * dc_k)
    
    grad = np.column_stack([grad_b, grad_c])
    
    return obj, grad

def _clamp_bc(params):
    """Ensure b >= 0, c >= 0, b + c <= 1. Project onto the simplex."""
    b = params[:, 0].copy()
    c = params[:, 1].copy()
    
    for m in range(N):
        bm = b[m]
        cm = c[m]
        # Project (bm, cm) onto {b>=0, c>=0, b+c<=1}
        if bm < 0:
            bm = 0.0
        if cm < 0:
            cm = 0.0
        if bm + cm > 1.0:
            # Scale to boundary
            s = 1.0 / (bm + cm)
            bm *= s
            cm *= s
        b[m] = bm
        c[m] = cm
    
    return np.column_stack([b, c])

def _fps_init(rng, n=N):
    """Farthest-point sampling on the barycentric simplex (b,c) plane."""
    points = []
    # Start with a vertex or random point
    u = rng.random()
    v = rng.random()
    while u + v > 1.0:
        u = rng.random()
        v = rng.random()
    points.append(np.array([u, v]))
    
    while len(points) < n:
        pts = np.array(points)
        # Find point in a grid that is farthest from existing points
        # Use a random set of candidates
        best_dist = -1
        best_pt = None
        for _ in range(500):
            u = rng.random()
            v = rng.random()
            while u + v > 1.0:
                u = rng.random()
                v = rng.random()
            cand = np.array([u, v])
            dists = np.linalg.norm(pts - cand, axis=1)
            min_d = np.min(dists)
            if min_d > best_dist:
                best_dist = min_d
                best_pt = cand
        points.append(best_pt)
    
    return np.array(points)

def _structured_init():
    """Symmetric structured initialization in (b,c) coordinates."""
    points = []
    # Vertices: (0,0), (1,0), (0,1)
    points.append(np.array([0.0, 0.0]))
    points.append(np.array([1.0, 0.0]))
    points.append(np.array([0.0, 1.0]))
    # Edge midpoints
    points.append(np.array([0.5, 0.0]))
    points.append(np.array([0.5, 0.5]))
    points.append(np.array([0.0, 0.5]))
    # Centroid
    points.append(np.array([1.0/3, 1.0/3]))
    # Hexagonal pattern around centroid
    cx, cy = 1.0/3, 1.0/3
    r = 0.12
    for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        if px >= -1e-9 and py >= -1e-9 and px + py <= 1 + 1e-9:
            points.append(np.array([max(0,px), max(0,py)]))
    
    # Trim or pad
    points = points[:N]
    rng = np.random.default_rng(0)
    while len(points) < N:
        u = rng.random()
        v = rng.random()
        while u + v > 1.0:
            u = rng.random()
            v = rng.random()
        points.append(np.array([u, v]))
    
    return np.array(points[:N])

def _random_init(rng, n=N):
    """Random points in the barycentric simplex."""
    points = []
    for _ in range(n):
        u = rng.random()
        v = rng.random()
        while u + v > 1.0:
            u = rng.random()
            v = rng.random()
        points.append(np.array([u, v]))
    return np.array(points)

def _optimize_from(params, deadline, beta_start=20.0, beta_end=150.0):
    """Run L-BFGS-B with increasing beta (annealing)."""
    current = params.copy()
    
    # Animate beta from beta_start to beta_end
    n_stages = 5
    betas = np.geomspace(beta_start, beta_end, n_stages)
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        def obj_and_grad(x):
            p = x.reshape(-1, 2)
            p = _clamp_bc(p)
            obj, grad = _softmin_objective_and_grad(p, beta=beta)
            # Gradient of clamping is identity in interior; for simplicity, ignore boundary gradient
            return obj, grad.flatten()
        
        x0 = current.flatten()
        try:
            res = minimize(
                obj_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
            )
            if res.fun < obj_and_grad(x0)[0]:
                current = _clamp_bc(res.x.reshape(-1, 2))
        except Exception:
            pass
    
    return current

def _refine_nm(params, deadline):
    """Nelder-Mead refinement on the exact (non-smooth) objective."""
    def obj(x):
        p = x.reshape(-1, 2)
        p = _clamp_bc(p)
        # Return negative of min area ratio (we minimize)
        det = _compute_dets(p)
        return -np.min(np.abs(det))
    
    x0 = params.flatten()
    try:
        res = minimize(
            obj,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        return _clamp_bc(res.x.reshape(-1, 2))
    except Exception:
        return params

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 * 0.88  # safety margin
    
    best_params = None
    best_score = -np.inf
    
    def _evaluate(params):
        nonlocal best_params, best_score
        params = _clamp_bc(params)
        score = _min_area_ratio(params)
        if score > best_score:
            best_score = score
            best_params = params.copy()
        return score
    
    # Generate diverse initializations
    inits = []
    
    # Structured
    inits.append(_structured_init())
    
    # FPS with different seeds
    for seed in range(6):
        rng = np.random.default_rng(seed * 123 + 7)
        inits.append(_fps_init(rng))
    
    # Random
    for seed in range(4):
        rng = np.random.default_rng(seed * 456 + 11)
        inits.append(_random_init(rng))
    
    # Evaluate all initializations
    for init in inits:
        if time.time() > deadline:
            break
        _evaluate(init)
    
    # Optimize from each initialization
    for init in inits:
        if time.time() > deadline - 5:
            break
        
        # L-BFGS-B with soft-min
        opt_params = _optimize_from(init, deadline)
        _evaluate(opt_params)
        
        # Nelder-Mead refinement
        if time.time() > deadline - 3:
            break
        nm_params = _refine_nm(opt_params, deadline)
        _evaluate(nm_params)
    
    # If we have time, do a few more rounds from the best
    if best_params is not None and time.time() < deadline - 10:
        for round_i in range(3):
            if time.time() > deadline - 5:
                break
            # Perturb best and re-optimize
            rng = np.random.default_rng(1000 + round_i)
            perturbed = best_params + rng.normal(0, 0.05, size=best_params.shape)
            perturbed = _clamp_bc(perturbed)
            
            opt_params = _optimize_from(perturbed, deadline)
            _evaluate(opt_params)
            
            if time.time() > deadline - 3:
                break
            nm_params = _refine_nm(opt_params, deadline)
            _evaluate(nm_params)
    
    # Convert best barycentric params to Cartesian
    if best_params is not None:
        b = best_params[:, 0]
        c = best_params[:, 1]
        # a = 1 - b - c
        # Cartesian: x = b*1 + c*0.5, y = c*(sqrt(3)/2)
        x = b + 0.5 * c
        y = (np.sqrt(3) / 2.0) * c
        points = np.column_stack([x, y])
        
        # Final sanity check: project any points slightly outside
        V0 = np.array([0.0, 0.0])
        V1 = np.array([1.0, 0.0])
        V2 = np.array([0.5, np.sqrt(3) / 2.0])
        
        for i in range(len(points)):
            p = points[i]
            # Barycentric check
            d0 = (V1[0] - V0[0]) * (p[1] - V0[1]) - (V1[1] - V0[1]) * (p[0] - V0[0])
            d1 = (V2[0] - V1[0]) * (p[1] - V1[1]) - (V2[1] - V1[1]) * (p[0] - V1[0])
            d2 = (V0[0] - V2[0]) * (p[1] - V2[1]) - (V0[1] - V2[1]) * (p[0] - V2[0])
            has_neg = (d0 < -1e-10) or (d1 < -1e-10) or (d2 < -1e-10)
            has_pos = (d0 > 1e-10) or (d1 > 1e-10) or (d2 > 1e-10)
            if has_neg and has_pos:
                # Project to closest edge
                def dist_to_segment(a, b_pt, p_pt):
                    ab = b_pt - a
                    t = np.dot(p_pt - a, ab) / np.dot(ab, ab)
                    t = np.clip(t, 0, 1)
                    proj = a + t * ab
                    return np.linalg.norm(p_pt - proj), proj
                
                best_d = np.inf
                best_proj = p.copy()
                for ii in range(3):
                    a = [V0, V1, V2][ii]
                    b_pt = [V0, V1, V2][(ii + 1) % 3]
                    d, proj = dist_to_segment(a, b_pt, p)
                    if d < best_d:
                        best_d = d
                        best_proj = proj
                points[i] = best_proj
        
        return points
    else:
        # Fallback
        return _structured_init()  # This returns (b,c) coords, need to convert
        # Actually, let's just return a valid set of Cartesian points
        b = _structured_init()[:, 0]
        c = _structured_init()[:, 1]
        x = b + 0.5 * c
        y = (np.sqrt(3) / 2.0) * c
        return np.column_stack([x, y])
