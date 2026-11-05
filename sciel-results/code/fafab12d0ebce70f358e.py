"""Packing 12 unit hexagons in a regular hexagon using smooth optimization."""
import numpy as np
import time

N = 12
# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _hexagon_support(theta, cx, cy, angle_deg, side=1.0):
    """Support function of a regular hexagon with side length `side` centered at (cx,cy) rotated by angle_deg.
    The support function h(theta) = max over vertices of (v - center) dot (cos theta, sin theta)."""
    angle = np.radians(angle_deg)
    # Unit hexagon vertices (side=1) at angles 0, 60, 120, 180, 240, 300 degrees
    # For a regular hexagon with side s, vertices are at distance s from center
    # at angles 0, 60, 120, 180, 240, 300
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    vertices = side * np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    # Rotate by angle
    ca, sa = np.cos(angle), np.sin(angle)
    R = np.array([[ca, -sa], [sa, ca]])
    vertices = vertices @ R.T
    # Support: max of v dot (cos theta, sin theta)
    u = np.array([np.cos(theta), np.sin(theta)])
    return np.max(vertices @ u)

def _hex_support_fast(theta, cx, cy, angle_deg, side=1.0):
    """Fast support function for unit hexagon.
    A regular hexagon with side s has support function:
    h(theta) = s * max_k cos(theta - alpha_k) where alpha_k are the outward normal angles
    For a regular hexagon, outward normals are at 0, 60, 120, 180, 240, 300 degrees.
    Actually, for a regular hexagon with flat top (vertex at angle 0), the edge normals are at 30, 90, 150, 210, 270, 330.
    Let me think: a regular hexagon with vertices at angles 0,60,120,180,240,300 has edges between consecutive vertices.
    The outward normal of the edge from vertex at angle 0 to vertex at angle 60 is at angle 30.
    So outward normals are at 30, 90, 150, 210, 270, 330 degrees.
    The support function in direction theta is:
    h(theta) = max over edges of (vertex dot n) where n is unit normal... 
    Actually simpler: h(theta) = max_k (v_k dot u) where v_k are vertices, u = (cos theta, sin theta).
    For a regular hexagon centered at origin with side s, vertices at s*(cos(60k), sin(60k)) for k=0..5.
    h(theta) = s * max_k cos(theta - 60k * pi/180)
    """
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    angle = np.radians(angle_deg)
    # h(theta) = s * max_k cos(theta - angle - 60k*pi/180)
    cos_vals = np.cos(theta - angle - base_angles)
    return side * np.max(cos_vals)

def _separation_distance(params, i, j, theta_samples):
    """Compute smooth separation between hexagon i and j.
    We want to ensure hex_i and hex_j don't overlap.
    Two convex sets A and B are disjoint iff for some direction theta,
    h_A(theta) + h_B(theta + pi) < 0 (in relative coordinates)
    Or equivalently, the Minkowski difference A - B contains the origin iff they overlap.
    A - B = {a - b : a in A, b in B}. Origin in A-B iff they intersect.
    h_{A-B}(theta) = h_A(theta) + h_B(theta + pi)
    A and B are disjoint iff h_{A-B}(theta) < 0 for some theta... no.
    A and B are disjoint iff the origin is NOT in A - B, i.e., min_theta h_{A-B}(theta) < 0.
    Actually, origin is in A-B iff for all theta, h_{A-B}(theta) >= 0.
    So A and B are disjoint iff min_theta [h_A(theta) + h_B(theta + pi)] < 0.
    
    For smooth optimization, we use a soft minimum:
    soft_min(f) = -logsumexp(-f) / beta (for large beta, approximates min)
    
    Here f(theta) = h_A(theta) + h_B(theta + pi) evaluated at sample points.
    We want f_min < 0 (disjoint). We penalize when f_min > 0 (overlap).
    """
    # params layout: [x0,y0,a0, x1,y1,a1, ...]
    xi, yi, ai = params[3*i], params[3*i+1], params[3*i+2]
    xj, yj, aj = params[3*j], params[3*j+1], params[3*j+2]
    
    # For each theta sample, compute h_A(theta) and h_B(theta + pi)
    # h_A(theta) for hex centered at (xi,yi) with angle ai:
    # The support of (A translated to origin) in direction theta, plus the center offset
    # h_A(theta) = h_unit(theta - ai) + (xi,yi) dot (cos theta, sin theta)
    # where h_unit is support of unit hex centered at origin.
    
    u = np.column_stack([np.cos(theta_samples), np.sin(theta_samples)])
    # h_A(theta) = h_unit(theta - ai) + (xi*cos(theta) + yi*sin(theta))
    # h_B(theta + pi) = h_unit(theta + pi - aj) + (xj*cos(theta+pi) + yj*sin(theta+pi))
    #                = h_unit(theta + pi - aj) + (-xj*cos(theta) - yj*sin(theta))
    
    # h_unit(phi) = max_k cos(phi - 60k*pi/180) for unit hex
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    
    # Compute h_unit for all theta samples at once
    # For h_A: phi = theta - ai
    phi_A = theta_samples - np.radians(ai)
    cos_A = np.cos(phi_A[:, None] - base_angles[None, :])  # (n_theta, 6)
    h_unit_A = np.max(cos_A, axis=1)  # (n_theta,)
    
    # For h_B: phi = theta + pi - aj
    phi_B = theta_samples + np.pi - np.radians(aj)
    cos_B = np.cos(phi_B[:, None] - base_angles[None, :])
    h_unit_B = np.max(cos_B, axis=1)
    
    # h_A(theta) = h_unit_A + xi*cos(theta) + yi*sin(theta)
    # h_B(theta+pi) = h_unit_B - xj*cos(theta) - yj*sin(theta)
    
    hA = h_unit_A + xi * np.cos(theta_samples) + yi * np.sin(theta_samples)
    hB = h_unit_B - xj * np.cos(theta_samples) - yj * np.sin(theta_samples)
    
    f = hA + hB  # This is h_{A-B}(theta)
    
    # Soft minimum of f: if min(f) < 0, disjoint. We want to penalize max(0, soft_min(f)).
    # Use log-sum-exp for soft min: soft_min(f) = -logsumexp(-f) / beta
    beta = 10.0
    # For numerical stability
    f_max = np.max(-f)
    logsumexp_neg_f = f_max + np.log(np.sum(np.exp(-f - f_max)))
    soft_min_f = -logsumexp_neg_f / beta
    
    # Penalty: we want soft_min_f <= 0 (with margin)
    # Return the violation: max(0, soft_min_f)
    return np.maximum(0.0, soft_min_f)

def _outer_constraint(params, outer_side, outer_center, outer_angle, theta_samples):
    """Check that all inner hexagons are inside the outer hexagon.
    Outer hexagon centered at (ocx, ocy) with side S, angle outer_angle.
    For inner hex i at (xi, yi) with angle ai:
    We need: for all theta, h_inner_i(theta) <= h_outer(theta)
    Equivalently: max_theta [h_inner_i(theta) - h_outer(theta)] <= 0
    
    h_inner_i(theta) = h_unit(theta - ai) + xi*cos(theta) + yi*sin(theta)
    h_outer(theta) = S * h_unit(theta - outer_angle) + ocx*cos(theta) + ocy*sin(theta)
    
    So: h_unit(theta - ai) + xi*cos(theta) + yi*sin(theta) <= S*h_unit(theta - outer_angle) + ocx*cos(theta) + ocy*sin(theta)
    """
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    u = np.column_stack([np.cos(theta_samples), np.sin(theta_samples)])
    
    ocx, ocy = outer_center
    oa = np.radians(outer_angle)
    
    # h_outer(theta)
    phi_outer = theta_samples - oa
    cos_outer = np.cos(phi_outer[:, None] - base_angles[None, :])
    h_unit_outer = np.max(cos_outer, axis=1)
    h_outer = outer_side * h_unit_outer + ocx * np.cos(theta_samples) + ocy * np.sin(theta_samples)
    
    total_violation = 0.0
    for i in range(N):
        xi, yi, ai = params[3*i], params[3*i+1], params[3*i+2]
        phi_i = theta_samples - np.radians(ai)
        cos_i = np.cos(phi_i[:, None] - base_angles[None, :])
        h_unit_i = np.max(cos_i, axis=1)
        h_inner = h_unit_i + xi * np.cos(theta_samples) + yi * np.sin(theta_samples)
        
        # Violation: h_inner - h_outer, we want max of this to be <= 0
        diff = h_inner - h_outer
        # Soft max: logsumexp(diff) / beta
        beta = 10.0
        d_max = np.max(diff)
        logsumexp_diff = d_max + np.log(np.sum(np.exp(diff - d_max)))
        soft_max_diff = logsumexp_diff / beta
        total_violation += np.maximum(0.0, soft_max_diff)
    
    return total_violation

def _objective(params, outer_side, outer_center, outer_angle, theta_samples):
    """Total objective: minimize outer_side + penalty for violations."""
    # Separation constraints
    sep_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            sep_penalty += _separation_distance(params, i, j, theta_samples)
    
    # Outer constraint
    outer_pen = _outer_constraint(params, outer_side, outer_center, outer_angle, theta_samples)
    
    return outer_side + 100.0 * (sep_penalty + outer_pen)

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    np.random.seed(42)
    
    # Theta samples for support function evaluation
    n_theta = 72
    theta_samples = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    
    # Initial guess: place hexagons in a grid-like pattern
    # For 12 hexagons, try a 3x4 or 4x3 arrangement
    # Unit hexagons have width sqrt(3) (flat-to-flat) and height 2 (point-to-point)
    # Let's start with a reasonable configuration
    
    params = np.zeros(3 * N)
    # Arrange in rows: 4 rows of 3, or 3 rows of 4
    # Hexagonal packing: offset rows
    positions = []
    # Try a compact arrangement
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons? No, that's 12.
    # Actually let's do:
    # Row 0: 4
    # Row 1: 4  
    # Row 2: 4
    # With hexagonal offset
    
    idx = 0
    for row in range(3):
        for col in range(4):
            x = col * 1.75 + (0.875 if row % 2 == 1 else 0.0)
            y = row * 1.5
            positions.append((x, y, 0.0))
            idx += 1
    
    # Center the configuration
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    
    for i, (x, y, a) in enumerate(positions):
        params[3*i] = x - cx
        params[3*i+1] = y - cy
        params[3*i+2] = a
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    outer_side = 5.0  # Initial guess
    
    best_params = params.copy()
    best_side = outer_side
    best_score = _objective(params, outer_side, outer_center, outer_angle, theta_samples)
    
    # Use scipy L-BFGS-B
    from scipy.optimize import minimize
    
    def obj_func(p):
        return _objective(p, outer_side, outer_center, outer_angle, theta_samples)
    
    def time_check():
        return (time.time() - _START_TIME) > TIME_LIMIT
    
    # Optimize inner hexagon positions and angles
    if not time_check():
        result = minimize(obj_func, params, method='L-BFGS-B', 
                         options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
        if result.success or result.fun < best_score:
            best_params = result.x.copy()
            best_score = result.fun
            # Extract outer side from objective: outer_side + penalty
            # The actual outer side is still outer_side, but we need to find the minimal valid one
    
    # Now optimize the outer side: binary search for minimal outer_side that allows valid packing
    # For a given outer_side, check if there's a valid configuration
    
    def check_valid(params, outer_side, outer_center, outer_angle):
        """Check if configuration is valid (no overlaps, all inside)."""
        # Check separation
        for i in range(N):
            for j in range(i+1, N):
                sep = _separation_distance(params, i, j, theta_samples)
                if sep > 0.01:  # Small tolerance
                    return False
        # Check outer constraint
        outer_pen = _outer_constraint(params, outer_side, outer_center, outer_angle, theta_samples)
        if outer_pen > 0.01:
            return False
        return True
    
    # Try to reduce outer_side
    # Binary search
    lo, hi = 2.0, 10.0
    
    # First, make sure hi is valid
    while not check_valid(best_params, hi, outer_center, outer_angle) and hi < 20.0:
        hi *= 1.5
    
    if not check_valid(best_params, hi, outer_center, outer_angle):
        # Fallback: return a valid but not optimal solution
        # Place hexagons far apart in a large hexagon
        params_fb = np.zeros(3 * N)
        for i in range(N):
            angle = 2 * np.pi * i / N
            r = 3.0
            params_fb[3*i] = r * np.cos(angle)
            params_fb[3*i+1] = r * np.sin(angle)
            params_fb[3*i+2] = 0.0
        side_fb = 15.0
        return [[params_fb[3*i], params_fb[3*i+1], params_fb[3*i+2]] for i in range(N)], (0.0, 0.0), side_fb, 0.0
    
    if time_check():
        # Return best found
        side = hi
        return [[best_params[3*i], best_params[3*i+1], best_params[3*i+2]] for i in range(N)], outer_center, side, outer_angle
    
    # Binary search for minimal outer_side
    for _ in range(20):
        if time_check():
            break
        mid = (lo + hi) / 2
        if check_valid(best_params, mid, outer_center, outer_angle):
            hi = mid
        else:
            lo = mid
    
    final_side = hi
    
    # Try to optimize further with the constrained outer side
    if not time_check():
        def obj_func_constrained(p):
            sep_penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    sep_penalty += _separation_distance(p, i, j, theta_samples)
            outer_pen = _outer_constraint(p, final_side, outer_center, outer_angle, theta_samples)
            return sep_penalty + outer_pen
        
        result2 = minimize(obj_func_constrained, best_params, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
        if result2.success or result2.fun < 0.01:
            best_params = result2.x.copy()
        
        # Re-check and possibly tighten
        if check_valid(best_params, final_side, outer_center, outer_angle):
            # Try to tighten a bit more
            for _ in range(10):
                if time_check():
                    break
                new_side = final_side * 0.99
                if check_valid(best_params, new_side, outer_center, outer_angle):
                    final_side = new_side
                else:
                    break
    
    # Final validation
    if not check_valid(best_params, final_side, outer_center, outer_angle):
        # Expand slightly to ensure validity
        while not check_valid(best_params, final_side, outer_center, outer_angle) and final_side < 30.0:
            final_side *= 1.01
    
    return [[best_params[3*i], best_params[3*i+1], best_params[3*i+2]] for i in range(N)], outer_center, final_side, outer_angle

_START_TIME = time.time()
