"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 145.0  # safety margin before 150s kill

def _check_validity(circles, W, H, tol=1e-8):
    """Check if circles are valid: inside rectangle, no overlaps, positive radii."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -tol or y - r < -tol or x + r > W + tol or y + r > H + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True

def _constraints(params, W, H):
    """
    Define inequality constraints for SLSQP.
    Constraint: g(params) <= 0
    """
    circles = params.reshape(N, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    cons = []
    
    # Boundary constraints:
    # x - r >= 0  =>  r - x <= 0
    # x + r <= W  =>  x + r - W <= 0
    # y - r >= 0  =>  r - y <= 0
    # y + r <= H  =>  y + r - H <= 0
    
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+2] - p[3*i], 'jac': lambda p, i=i: _jac_boundary(p, i, 0, W, H)})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] + p[3*i+2] - W, 'jac': lambda p, i=i: _jac_boundary(p, i, 1, W, H)})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+2] - p[3*i+1], 'jac': lambda p, i=i: _jac_boundary(p, i, 2, W, H)})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] + p[3*i+2] - H, 'jac': lambda p, i=i: _jac_boundary(p, i, 3, W, H)})
        
    # Non-overlap constraints:
    # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    # (r_i + r_j)^2 - (x_i - x_j)^2 - (y_i - y_j)^2 <= 0
    
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({
                'type': 'ineq', 
                'fun': lambda p, i=i, j=j: _overlap_val(p, i, j), 
                'jac': lambda p, i=i, j=j: _jac_overlap(p, i, j)
            })
            
    return cons

def _jac_boundary(p, i, type, W, H):
    """Jacobian for boundary constraints."""
    jac = np.zeros(3*N)
    if type == 0: # r - x <= 0
        jac[3*i+2] = 1.0
        jac[3*i] = -1.0
    elif type == 1: # x + r - W <= 0
        jac[3*i] = 1.0
        jac[3*i+2] = 1.0
    elif type == 2: # r - y <= 0
        jac[3*i+2] = 1.0
        jac[3*i+1] = -1.0
    elif type == 3: # y + r - H <= 0
        jac[3*i+1] = 1.0
        jac[3*i+2] = 1.0
    return jac

def _overlap_val(p, i, j):
    """Value of overlap constraint: (r_i + r_j)^2 - dist^2"""
    xi, yi, ri = p[3*i], p[3*i+1], p[3*i+2]
    xj, yj, rj = p[3*j], p[3*j+1], p[3*j+2]
    dx = xi - xj
    dy = yi - yj
    dist_sq = dx*dx + dy*dy
    return (ri + rj)**2 - dist_sq

def _jac_overlap(p, i, j):
    """Jacobian of overlap constraint."""
    xi, yi, ri = p[3*i], p[3*i+1], p[3*i+2]
    xj, yj, rj = p[3*j], p[3*j+1], p[3*j+2]
    dx = xi - xj
    dy = yi - yj
    dist_sq = dx*dx + dy*dy
    if dist_sq < 1e-10:
        dist = 1e-10
    else:
        dist = np.sqrt(dist_sq)
        
    # f = (ri+rj)^2 - (xi-xj)^2 - (yi-yj)^2
    # df/dri = 2(ri+rj)
    # df/dxj = -2(xi-xj)
    # df/dyi = 2(yi-yj)
    
    jac = np.zeros(3*N)
    jac[3*i+2] = 2*(ri+rj)
    jac[3*j+2] = 2*(ri+rj)
    
    jac[3*i] = -2*dx
    jac[3*j] = 2*dx
    jac[3*i+1] = -2*dy
    jac[3*j+1] = 2*dy
    
    return jac

def _objective(params, W, H):
    """Objective: minimize -sum(radii)"""
    r = params[2::3]
    return -np.sum(r)

def _hex_initialization(W, H):
    """Create a hexagonal grid initialization."""
    circles = []
    # Approximate radius for hex packing in rectangle
    # Area per circle ~ sqrt(3)/2 * (2r)^2 = 2*sqrt(3)*r^2
    # Total Area = W*H
    # N * 2*sqrt(3)*r^2 <= W*H  => r <= sqrt(W*H / (2*sqrt(3)*N))
    r_est = np.sqrt(W*H / (2*np.sqrt(3)*N)) * 0.95
    
    # Adjust grid to fit
    spacing = 2 * r_est
    dx = spacing
    dy = spacing * np.sqrt(3) / 2
    
    rows = int(H / dy) + 1
    cols = int(W / dx) + 1
    
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= N:
                break
            x = (j + 0.5) * dx
            y = (i + 0.5) * dy
            if i % 2 == 1:
                x += dx / 2.0
            
            # Clip to bounds
            x = np.clip(x, r_est, W - r_est)
            y = np.clip(y, r_est, H - r_est)
            
            circles.append([x, y, r_est])
            count += 1
            
    # If we don't have enough, fill randomly or shrink
    while len(circles) < N:
        r = r_est * 0.5
        x = np.random.uniform(r, W - r)
        y = np.random.uniform(r, H - r)
        circles.append([x, y, r])
        
    return np.array(circles[:N])

def _random_initialization(W, H, rng):
    """Create a random initialization."""
    circles = []
    for _ in range(N):
        r = rng.uniform(0.02, 0.1)
        x = rng.uniform(r, W - r)
        y = rng.uniform(r, H - r)
        circles.append([x, y, r])
    return np.array(circles)

def _refine_slsqp(circles, W, H, max_iter=200):
    """Refine using SLSQP with constraints."""
    params = circles.flatten()
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, W)) # x
        bounds.append((0.0, H)) # y
        bounds.append((0.0, min(W, H)/2)) # r
        
    try:
        result = minimize(
            _objective, 
            params, 
            args=(W, H),
            method='SLSQP',
            bounds=bounds,
            constraints=_constraints(params, W, H),
            options={'maxiter': max_iter, 'ftol': 1e-12}
        )
        return result.x.reshape(N, 3)
    except Exception:
        return circles

def _polish(circles, W, H, n_iter=50):
    """Simple iterative push-apart to fix small violations."""
    circles = circles.copy()
    for _ in range(n_iter):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-10:
                    dx, dy = 1e-5, 1e-5
                    dist = np.sqrt(dx*dx + dy*dy)
                
                overlap = circles[i][2] + circles[j][2] - dist
                if overlap > 1e-8:
                    push = overlap / 2.0 + 1e-5
                    ux, uy = dx / dist, dy / dist
                    circles[i][0] += push * ux
                    circles[i][1] += push * uy
                    circles[j][0] -= push * ux
                    circles[j][1] -= push * uy
                    moved = True
        
        # Clamp to bounds
        for i in range(N):
            r = circles[i][2]
            circles[i][0] = np.clip(circles[i][0], r, W - r)
            circles[i][1] = np.clip(circles[i][1], r, H - r)
            
        if not moved:
            break
    return circles

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    
    # Perimeter = 2(W+H) = 4 => W+H = 2
    # Try aspect ratios
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
    
    rng = np.random.default_rng(42)
    
    for ar in aspect_ratios:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)
        
        # Try hex init
        if time.time() - start_time < DEADLINE - 20:
            circ_hex = _hex_initialization(W, H)
            circ_ref = _refine_slsqp(circ_hex, W, H, max_iter=100)
            circ_ref = _polish(circ_ref, W, H)
            if _check_validity(circ_ref, W, H):
                score = np.sum(circ_ref[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = circ_ref.copy()
                    
        # Try random inits
        for _ in range(3):
            if time.time() - start_time > DEADLINE - 10:
                break
            circ_rand = _random_initialization(W, H, rng)
            circ_ref = _refine_slsqp(circ_rand, W, H, max_iter=100)
            circ_ref = _polish(circ_ref, W, H)
            if _check_validity(circ_ref, W, H):
                score = np.sum(circ_ref[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = circ_ref.copy()
                    
    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
        best_score = np.sum(best_result[:, 2])
        
    return best_result
# EVOLVE-BLOCK-END
