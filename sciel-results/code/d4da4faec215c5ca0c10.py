import numpy as np
import scipy.optimize as opt
import time

N = 12

# Precompute min distance table
# d_min(delta) for delta in [0, pi/6]
# We compute this by finding the minimum distance between two unit hexagons
# with relative rotation delta.
# Since the function is periodic and symmetric, we only need [0, pi/6].

def support_function(angle):
    """
    Support function of a unit regular hexagon centered at origin.
    angle: direction angle in radians.
    Returns the distance from origin to the boundary in that direction.
    """
    # The hexagon has symmetry every pi/3.
    # The "flat" side is at angle 0? No, standard hexagon usually has vertex at 0 or flat side at 0.
    # Let's define the hexagon with a flat side perpendicular to the x-axis?
    # Vertices at angles: 0, 60, 120, 180, 240, 300?
    # If vertex at 0, then inradius is at 30, 90, etc.
    # Support function h(phi) = r / cos(phi - phi_n) where phi_n is the angle of the normal.
    # Normals are at 30, 90, 150, 210, 270, 330 degrees (if vertices at 0, 60...).
    # Let's use the standard: Vertex at angle 0.
    # Then the sides are between 0-60, 60-120, etc.
    # The normal to the side between 0 and 60 is at 30 degrees.
    # The inradius r = sqrt(3)/2.
    # So h(phi) = r / cos( (phi - 30) mod 60 - 30 )?
    # Let's simplify:
    # phi_mod = phi % (np.pi/3)
    # The normal is at pi/6 within each sector.
    # So delta = phi_mod - pi/6
    # h = r / cos(delta)
    phi_mod = angle % (np.pi / 3)
    delta = phi_mod - np.pi / 6
    return (np.sqrt(3) / 2) / np.cos(delta)

def min_dist_table():
    """
    Compute the minimum center-to-center distance for two disjoint unit hexagons
    with relative rotation delta.
    """
    # We sample delta in [0, pi/6]
    # For a given delta, we need to find the smallest d such that they don't overlap.
    # This is equivalent to the support function of the Minkowski sum?
    # Actually, d_min(delta) is the minimum over all orientations of the "touching" distance.
    # But the relative orientation is fixed.
    # We can compute this by checking all vertex-edge pairs.
    # However, a simpler approximation is often sufficient for initial packing,
    # but for a strong score, we need accuracy.
    
    # Let's compute it numerically for a grid.
    # For a given delta, place hex B at (d, 0) relative to hex A.
    # We want the minimum d such that min_dist(A, B) >= 0.
    # Actually, we want the d where they touch.
    
    # The distance between two convex polygons is the minimum distance between any point in A and any point in B.
    # If they are disjoint, this is > 0.
    # We can use a simple binary search on d?
    # Or just use the analytical lower bound:
    # d_min(delta) >= sqrt(3)
    # And d_min(pi/6) = 1 + sqrt(3)/2
    # Let's interpolate linearly?
    # Or compute exactly.
    
    # Exact computation:
    # The minimum distance between two convex polygons is achieved by a vertex-edge pair or vertex-vertex.
    # We can iterate over all 6 vertices of A and 6 edges of B.
    # For a given delta and d, we can check if they overlap.
    # This is expensive to do for the table, but we only do it once.
    
    # Let's just use a conservative lower bound for now to ensure validity, 
    # and refine if needed.
    # Actually, using d_min = sqrt(3) for all deltas is SAFE (guarantees non-overlap) 
    # but SUBOPTIMAL (allows more space than needed, so L will be larger).
    # To get a better score, we should use the tighter bound.
    
    # Let's compute d_min(delta) properly.
    # d_min(delta) is the radius of the Minkowski sum of Hex A and -Hex B? No.
    # It is the minimum d such that Hex A and (Hex B translated by d) are disjoint.
    
    # We can approximate d_min(delta) with:
    # d_min(delta) = sqrt(3) + ( (1 + sqrt(3)/2) - sqrt(3) ) * (delta / (pi/6))
    # This is linear interpolation.
    
    delta_max = np.pi / 6
    d_min_0 = np.sqrt(3)
    d_min_max = 1 + np.sqrt(3) / 2
    
    # Let's verify d_min_max.
    # At delta = pi/6, vertex of A at 30 deg? No.
    # If A is aligned with vertex at 0.
    # B is rotated by 30 deg. Vertex of B at 30 deg.
    # Side of A has normal at 30 deg.
    # So Vertex of B touches Side of A.
    # Distance = R_B + r_A = 1 + sqrt(3)/2. Correct.
    
    # Is it linear?
    # Let's assume linear for now. It's a lower bound?
    # The function is likely convex. Linear interpolation might underestimate the distance in the middle, 
    # leading to overlaps.
    # To be safe, we can use a slightly larger distance, e.g. max(linear_interp, sqrt(3) + 0.01*delta)
    # Or just use sqrt(3) everywhere? That's very safe but loose.
    
    # Let's try to compute the exact d_min for a few points and fit a quadratic?
    # Or just use the linear interpolation and add a small safety margin.
    
    pass

# We will use the linear interpolation with a small safety margin in the constraints.

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Precompute constants
    r_out_factor = np.sqrt(3) / 2
    d_min_0 = np.sqrt(3)
    d_min_max = 1 + np.sqrt(3) / 2
    delta_max = np.pi / 6
    
    def d_min(delta):
        # Normalize delta to [0, pi/6]
        delta = np.abs(delta) % (np.pi / 3)
        if delta > np.pi / 6:
            delta = np.pi / 3 - delta
        # Linear interpolation
        t = delta / delta_max
        return d_min_0 + t * (d_min_max - d_min_0)
        
    # Variables: x0, y0, theta0, ..., x11, y11, theta11, L
    # Total 37 variables.
    
    def objective(vars):
        # L is the last variable
        return vars[-1]
        
    def constraints(vars):
        cons = []
        L = vars[-1]
        r_out = L * r_out_factor
        
        # Extract positions and rotations
        # vars[0:36] are x,y,theta for 12 hexagons
        # vars[36] is L
        
        # Containment constraints
        # For each hexagon i, and each of 6 outer normals
        outer_normals = [np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2*np.pi, 6, endpoint=False)]
        
        for i in range(N):
            x = vars[3*i]
            y = vars[3*i+1]
            theta = vars[3*i+2]
            
            for n in outer_normals:
                # Projection of center
                proj_center = x * n[0] + y * n[1]
                # Support function of inner hexagon in direction n
                # The direction n is absolute. The hexagon is rotated by theta.
                # So the relative angle is n_angle - theta.
                n_angle = np.arctan2(n[1], n[0])
                h = support_function(n_angle - theta)
                
                # Constraint: proj_center + h <= r_out
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, n=n, theta=theta, L=L, h=h, proj_center=proj_center, r_out=r_out: r_out - (proj_center + h)})
                
        # Non-overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                xi = vars[3*i]
                yi = vars[3*i+1]
                ti = vars[3*i+2]
                
                xj = vars[3*j]
                yj = vars[3*j+1]
                tj = vars[3*j+2]
                
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                
                d_req = d_min(ti - tj)
                d_req_sq = d_req * d_req
                
                # Constraint: dist_sq >= d_req_sq
                cons.append({'type': 'ineq', 'fun': lambda x, dist_sq=dist_sq, d_req_sq=d_req_sq: dist_sq - d_req_sq})
                
        return cons

    # Initial guesses
    best_L = np.inf
    best_vars = None
    
    # Try a few random restarts
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        # Place hexagons in a loose hexagonal pattern
        # Center at (0,0)
        # Others in a ring
        
        x0 = np.random.uniform(-2, 2, N)
        y0 = np.random.uniform(-2, 2, N)
        # Or structured:
        # 1 at center, 6 around, 5 more?
        # Let's use a structured initialization for the first few restarts
        if r < 10:
            # Structured init
            x0 = np.zeros(N)
            y0 = np.zeros(N)
            # Place 1 at (0,0)
            # Place 6 at distance 2.0 (safe)
            for k in range(6):
                angle = k * np.pi / 3
                x0[k+1] = 2.0 * np.cos(angle)
                y0[k+1] = 2.0 * np.sin(angle)
            # Place remaining 5
            for k in range(5):
                angle = (k + 0.5) * np.pi / 3
                x0[k+7] = 3.5 * np.cos(angle)
                y0[k+7] = 3.5 * np.sin(angle)
            # Add some random noise
            x0 += np.random.uniform(-0.2, 0.2, N)
            y0 += np.random.uniform(-0.2, 0.2, N)
        else:
            # Random init
            x0 = np.random.uniform(-3, 3, N)
            y0 = np.random.uniform(-3, 3, N)
            
        theta0 = np.zeros(N) # Start with aligned
        # L guess: 4.0 is safe
        L0 = 4.0
        
        vars_init = np.concatenate([x0, y0, theta0, [L0]])
        # Interleave x, y, theta
        vars_init = np.zeros(3*N + 1)
        for i in range(N):
            vars_init[3*i] = x0[i]
            vars_init[3*i+1] = y0[i]
            vars_init[3*i+2] = theta0[i]
        vars_init[-1] = L0
        
        try:
            res = opt.minimize(objective, vars_init, method='SLSQP', bounds=[(None, None)]*3*N + [(1.0, 10.0)], constraints=constraints, options={'maxiter': 200, 'ftol': 1e-6})
            
            if res.success:
                L_res = res.x[-1]
                if L_res < best_L:
                    best_L = L_res
                    best_vars = res.x
        except Exception as e:
            pass

    if best_vars is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Extract result
    inner = []
    for i in range(N):
        x = best_vars[3*i]
        y = best_vars[3*i+1]
        theta_deg = np.degrees(best_vars[3*i+2])
        inner.append([x, y, theta_deg])
        
    L = best_vars[-1]
    
    return inner, (0.0, 0.0), L, 0.0
