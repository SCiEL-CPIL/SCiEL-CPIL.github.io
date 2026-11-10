import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def _score(pts):
    """Compute the score: Delta / n^n where Delta is product of ordered-pair distances after scaling to diameter 2."""
    n = len(pts)
    if n != 8:
        return -np.inf
    
    # Check distinctness and compute all pairwise distances
    dists = np.zeros((n, n))
    min_dist = np.inf
    max_dist = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-10:
                return -np.inf
            dists[i, j] = d
            dists[j, i] = d
            if d < min_dist:
                min_dist = d
            if d > max_dist:
                max_dist = d
    
    if max_dist < 1e-10:
        return -np.inf
    
    # Scale so that diameter = 2
    scale = 2.0 / max_dist
    scaled = pts * scale
    
    # Compute log of product of all ordered-pair distances (i != j)
    log_prod = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                d = np.linalg.norm(scaled[i] - scaled[j])
                if d < 1e-15:
                    return -np.inf
                log_prod += np.log(d)
    
    # Score = exp(log_prod) / n^n
    log_score = log_prod - n * np.log(n)
    return np.exp(log_score)

def _neg_log_score(pts):
    """Negative log of score for optimization (minimize this)."""
    s = _score(pts)
    if not np.isfinite(s) or s <= 0:
        return 1e10
    return -np.log(s)

def _roots_from_coeffs(coeffs):
    """
    Given coefficients of a degree-8 polynomial in complex form,
    return 8 real 2D points (roots as complex numbers -> (real, imag)).
    coeffs: array of 8 floats representing real/imag parts of the 8 non-leading coefficients.
    We use a monic polynomial: z^8 + c7*z^7 + ... + c0
    where c_i are complex. We parameterize each c_i by 2 floats (real, imag).
    So coeffs has 16 elements: [re(c7), im(c7), re(c6), im(c6), ..., re(c0), im(c0)]
    """
    n = 8
    # Build complex coefficients
    complex_coeffs = []
    for i in range(n):
        re = coeffs[2*i]
        im = coeffs[2*i+1]
        complex_coeffs.append(re + 1j*im)
    
    # Polynomial: z^8 + c7*z^7 + ... + c0
    poly_coeffs = [1.0] + complex_coeffs
    roots = np.roots(poly_coeffs)
    
    # Convert to 2D points
    pts = np.column_stack([roots.real, roots.imag])
    return pts

def _poly_neglog(coeffs):
    """
    Objective for polynomial coefficient optimization.
    coeffs: 16 floats (re/im of 8 complex coefficients of monic degree-8 poly)
    Returns negative log score, with penalty if roots are not distinct or diameter is degenerate.
    """
    try:
        pts = _roots_from_coeffs(coeffs)
        if not np.all(np.isfinite(pts)):
            return 1e10
        s = _score(pts)
        if not np.isfinite(s) or s <= 0:
            return 1e10
        return -np.log(s)
    except Exception:
        return 1e10

def _centrally_symmetric_model(params):
    """
    Centrally symmetric model: 4 points and their negatives.
    params: 8 floats = (x0,y0,x1,y1,x2,y2,x3,y3)
    Points: (x0,y0), (-x0,-y0), (x1,y1), (-x1,-y1), (x2,y2), (-x2,-y2), (x3,y3), (-x3,-y3)
    """
    pts = np.array([
        [params[0], params[1]],
        [-params[0], -params[1]],
        [params[2], params[3]],
        [-params[2], -params[3]],
        [params[4], params[5]],
        [-params[4], -params[5]],
        [params[6], params[7]],
        [-params[6], -params[7]]
    ])
    return pts

def _centrally_symmetric_neglog(params):
    pts = _centrally_symmetric_model(params)
    s = _score(pts)
    if not np.isfinite(s) or s <= 0:
        return 1e10
    return -np.log(s)

def _two_squares_model(params):
    """
    Two squares model: two squares centered at origin, rotated by angle theta.
    params: [r1, r2, theta] where r1, r2 are circumradii, theta is rotation of second square.
    Square 1 vertices: r1*(cos(k*pi/2), sin(k*pi/2)) for k=0,1,2,3
    Square 2 vertices: r2*(cos(k*pi/2 + theta), sin(k*pi/2 + theta)) for k=0,1,2,3
    """
    r1, r2, theta = params
    pts = []
    for k in range(4):
        angle = k * np.pi / 2
        pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    for k in range(4):
        angle = k * np.pi / 2 + theta
        pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    return np.array(pts)

def _two_squares_neglog(params):
    r1, r2, theta = params
    if r1 < 0: r1 = abs(r1)
    if r2 < 0: r2 = abs(r2)
    pts = _two_squares_model([r1, r2, theta])
    s = _score(pts)
    if not np.isfinite(s) or s <= 0:
        return 1e10
    return -np.log(s)

def _polish_16d(x0):
    """Polish a 16D point configuration using Powell."""
    def obj(p):
        pts = p.reshape(8, 2)
        s = _score(pts)
        if not np.isfinite(s) or s <= 0:
            return 1e10
        return -np.log(s)
    
    try:
        res = minimize(obj, x0, method='Powell', tol=1e-12, options={'maxiter': 500, 'maxfev': 2000})
        if res.fun < 1e9:
            return res.x.reshape(8, 2), -np.exp(-res.fun)
    except Exception:
        pass
    return None, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_score = 1.0
    best_points = None
    
    def update_best(score, points):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points
    
    def time_left():
        return deadline - time.time()
    
    # Fallback: regular octagon
    def regular_octagon():
        return np.array([[np.cos(2 * np.pi * k / 8), np.sin(2 * np.pi * k / 8)] for k in range(8)])
    
    oct_pts = regular_octagon()
    oct_score = _score(oct_pts)
    if oct_score > best_score:
        best_score = oct_score
        best_points = oct_pts.copy()
    
    # === Phase 1: Centrally symmetric model (8D) ===
    if time_left() > 40:
        try:
            bounds_cs = [(-2, 2)] * 8
            result_cs = differential_evolution(
                _centrally_symmetric_neglog,
                bounds_cs,
                maxiter=50,
                popsize=12,
                tol=1e-8,
                seed=42
            )
            if result_cs.fun < 1e9:
                score_cs = np.exp(-result_cs.fun)
                pts_cs = _centrally_symmetric_model(result_cs.x)
                update_best(score_cs, pts_cs)
                
                # Polish
                if time_left() > 20:
                    pts_pol, score_pol = _polish_16d(result_cs.x.reshape(8, 2).flatten() if False else pts_cs.flatten())
                    if pts_pol is not None:
                        update_best(score_pol, pts_pol)
        except Exception:
            pass
    
    # === Phase 2: Two squares model (3D) ===
    if time_left() > 30:
        try:
            bounds_ts = [(0, 2), (0, 2), (0, np.pi)]
            result_ts = differential_evolution(
                _two_squares_neglog,
                bounds_ts,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                seed=123
            )
            if result_ts.fun < 1e9:
                score_ts = np.exp(-result_ts.fun)
                pts_ts = _two_squares_model(result_ts.x)
                update_best(score_ts, pts_ts)
                
                # Polish in 16D
                if time_left() > 15:
                    pts_pol, score_pol = _polish_16d(pts_ts.flatten())
                    if pts_pol is not None:
                        update_best(score_pol, pts_pol)
        except Exception:
            pass
    
    # === Phase 3: Polynomial coefficient optimization (16D) ===
    if time_left() > 50:
        try:
            # Initial guess: regular octagon as roots of z^8 - 1
            # z^8 - 1 = 0 => coefficients: [0,0,0,0,0,0,0,-1] in complex
            # In our format: [re(c7),im(c7),...,re(c0),im(c0)]
            # c7=0, c6=0, ..., c1=0, c0=-1
            init_poly = np.zeros(16)
            init_poly[14] = -1.0  # re(c0) = -1
            init_poly[15] = 0.0   # im(c0) = 0
            
            bounds_poly = [(-2, 2)] * 16
            result_poly = differential_evolution(
                _poly_neglog,
                bounds_poly,
                maxiter=30,
                popsize=10,
                tol=1e-6,
                seed=777,
                init='sobol'
            )
            if result_poly.fun < 1e9:
                score_poly = np.exp(-result_poly.fun)
                pts_poly = _roots_from_coeffs(result_poly.x)
                if np.all(np.isfinite(pts_poly)):
                    update_best(score_poly, pts_poly)
                    
                    # Polish in 16D
                    if time_left() > 15:
                        pts_pol, score_pol = _polish_16d(pts_poly.flatten())
                        if pts_pol is not None:
                            update_best(score_pol, pts_pol)
        except Exception:
            pass
    
    # === Phase 4: General 16D DE + polish ===
    if time_left() > 40:
        try:
            # Start from best found so far
            if best_points is not None:
                init_16d = best_points.flatten()
            else:
                init_16d = oct_pts.flatten()
            
            bounds_16d = [(-2, 2)] * 16
            result_16d = differential_evolution(
                lambda p: _neg_log_score(p.reshape(8, 2)),
                bounds_16d,
                maxiter=20,
                popsize=8,
                tol=1e-6,
                seed=999,
                init='sobol'
            )
            if result_16d.fun < 1e9:
                score_16d = np.exp(-result_16d.fun)
                pts_16d = result_16d.x.reshape(8, 2)
                update_best(score_16d, pts_16d)
                
                # Polish
                if time_left() > 10:
                    pts_pol, score_pol = _polish_16d(result_16d.x)
                    if pts_pol is not None:
                        update_best(score_pol, pts_pol)
        except Exception:
            pass
    
    # === Phase 5: Multi-start local optimization from diverse initializations ===
    if time_left() > 20:
        try:
            # Try several diverse starting points
            starts = []
            if best_points is not None:
                starts.append(best_points.flatten())
            starts.append(oct_pts.flatten())
            
            # Random starts
            rng = np.random.RandomState(42)
            for _ in range(3):
                rand_pts = rng.uniform(-1, 1, (8, 2))
                starts.append(rand_pts.flatten())
            
            # Centrally symmetric random
            for _ in range(2):
                base = rng.uniform(-1, 1, (4, 2))
                cs_pts = np.vstack([base, -base])
                starts.append(cs_pts.flatten())
            
            for s in starts:
                if time_left() < 5:
                    break
                pts_pol, score_pol = _polish_16d(s)
                if pts_pol is not None:
                    update_best(score_pol, pts_pol)
        except Exception:
            pass
    
    # Final: return best points found
    if best_points is not None:
        # Ensure coordinates are within limits
        if np.any(np.abs(best_points) >= 1e6):
            # Scale down
            max_coord = np.max(np.abs(best_points))
            best_points = best_points * (1e5 / max_coord)
        return best_points.tolist()
    else:
        return oct_pts.tolist()
