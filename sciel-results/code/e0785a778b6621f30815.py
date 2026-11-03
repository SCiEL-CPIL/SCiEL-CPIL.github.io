import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Variables:
    # 0: Cx, 1: Cy, 2: S, 3: Theta (Outer)
    # 4..: For each of 12 inner hexes: x, y, phi
    
    # Initial guess
    # Arrange 12 hexes in a 2x6 grid? Or hexagonal?
    # Let's try a hexagonal pattern.
    # Centers of a hex grid with spacing d = sqrt(3) * s_inner? 
    # Distance between centers of touching unit hexes is sqrt(3).
    
    # Let's place them in a cluster.
    # 1 center, 6 around, 5 more?
    # Or just a rectangle 2x6.
    # Let's use a grid for simplicity of initialization.
    # Spacing x: sqrt(3), y: 1.5
    # 12 hexes: 2 rows of 6.
    
    init_x = []
    init_y = []
    init_phi = []
    
    # 2 rows, 6 cols
    for r in range(2):
        for c in range(6):
            x = (c - 2.5) * np.sqrt(3)
            y = (r - 0.5) * 1.5
            init_x.append(x)
            init_y.append(y)
            init_phi.append(0.0)
            
    # Outer hex guess
    # Width of 2x6 grid: 5 * sqrt(3) + 2 * 0.5 (radius) ?
    # Approx side length 4.0
    init_Cx = 0.0
    init_Cy = 0.0
    init_S = 4.5
    init_Theta = 0.0
    
    x0 = np.array([init_Cx, init_Cy, init_S, init_Theta] + init_x + init_y + init_phi)
    
    beta = 10.0 # Smoothness parameter for LogSumExp
    
    def get_hex_vertices(cx, cy, s, phi):
        angles = phi + np.arange(6) * np.pi / 3.0
        return np.array([cx + s * np.cos(angles), cy + s * np.sin(angles)]).T
        
    def get_hex_normals(cx, cy, s, phi):
        # Normals to edges, pointing outwards
        # Edge k is between vertex k and k+1.
        # Normal angle is phi + k*pi/3 + pi/6
        angles = phi + np.arange(6) * np.pi / 3.0 + np.pi / 6.0
        return np.array([np.cos(angles), np.sin(angles)]).T
        
    def smooth_max(vals, beta=10.0):
        # Approximate max(vals) using logsumexp
        # To avoid overflow, subtract max
        v_max = np.max(vals)
        if v_max == -np.inf: return -np.inf
        return v_max + np.log(np.sum(np.exp(beta * (vals - v_max)))) / beta
        
    def smooth_min(vals, beta=10.0):
        return -smooth_max(-vals, beta)
        
    def calculate_gaps(x, y, phi_i, s_i, x_j, y_j, phi_j, s_j):
        # Returns max gap between hex i and hex j
        # Normals from i and j
        ni = get_hex_normals(x, y, s_i, phi_i)
        nj = get_hex_normals(x_j, y_j, s_j, phi_j)
        all_normals = np.vstack([ni, nj])
        
        vi = get_hex_vertices(x, y, s_i, phi_i)
        vj = get_hex_vertices(x_j, y_j, s_j, phi_j)
        
        # Project vi and vj onto each normal
        # proj_i = vi @ all_normals.T (shape 6 x 12)
        # proj_j = vj @ all_normals.T (shape 6 x 12)
        
        # Max proj of i, Min proj of j
        max_i = np.max(vi @ all_normals.T, axis=0)
        min_j = np.min(vj @ all_normals.T, axis=0)
        
        gaps = min_j - max_i
        # We want the max of these gaps to be >= 0
        # smooth_max(gaps) approximates the max gap
        return smooth_max(gaps, beta)
        
    def calculate_containment_violation(cx, cy, s, phi, ox, oy, os, ophi):
        # Check if hex (cx,cy,s,phi) is inside hex (ox,oy,os,ophi)
        # For each normal of outer, check if max proj of inner <= center dot normal + apothem
        no = get_hex_normals(ox, oy, os, ophi)
        vi = get_hex_vertices(cx, cy, s, phi)
        
        # Apothem of outer
        apothem = os * np.sqrt(3) / 2.0
        
        # Center projection
        c_proj = ox * no[:, 0] + oy * no[:, 1]
        
        # Max projection of inner vertices
        max_i = np.max(vi @ no.T, axis=0)
        
        # Violation is max_i - (c_proj + apothem)
        # We want this to be <= 0
        # Violation magnitude is max(0, max_i - limit)
        # Use smooth max of the individual violations
        viol = max_i - (c_proj + apothem)
        return smooth_max(viol, beta)
        
    def objective(vars):
        Cx, Cy, S, Theta = vars[0], vars[1], vars[2], vars[3]
        if S <= 0: return 1e10
        
        inner_x = vars[4::3]
        inner_y = vars[5::3]
        inner_phi = vars[6::3]
        
        penalty = 0.0
        
        # Containment
        for i in range(N):
            v = calculate_containment_violation(inner_x[i], inner_y[i], 1.0, inner_phi[i], Cx, Cy, S, Theta)
            # v is the smooth max of violations. If v > 0, it's a violation.
            # Penalty is square of positive part.
            # Since smooth_max is always defined, we just square it if it's positive?
            # Actually, if the hex is inside, max viol is negative.
            # We want to penalize only if it's outside.
            # max(0, v)^2
            if v > 0:
                penalty += v * v
            else:
                # Small gradient push to keep it inside? No, SLSQP will handle minimization of S.
                # Just 0 penalty if inside.
                pass
                
        # Non-overlap
        for i in range(N):
            for j in range(i + 1, N):
                gap = calculate_gaps(inner_x[i], inner_y[i], inner_phi[i], 1.0, 
                                     inner_x[j], inner_y[j], inner_phi[j], 1.0)
                # gap is smooth max of gaps.
                # If gap < 0, they overlap.
                # Penalty is max(0, -gap)^2
                if gap < 0:
                    penalty += (-gap) * (-gap)
                    
        return S + 100.0 * penalty
        
    # Bounds
    # S > 0
    # Others unbounded, but we can add loose bounds to help SLSQP
    bounds = [(0.0, 100.0)] * 40 # Loose bounds
    
    # Run optimization
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-8})
    
    if res.success or res.fun < 100:
        final_vars = res.x
    else:
        final_vars = x0 # Fallback to initial guess if it fails badly
        
    Cx, Cy, S, Theta = final_vars[0], final_vars[1], final_vars[2], final_vars[3]
    inner_x = final_vars[4::3]
    inner_y = final_vars[5::3]
    inner_phi = final_vars[6::3]
    
    inner = []
    for i in range(N):
        # Convert radians to degrees
        inner.append([inner_x[i], inner_y[i], np.degrees(inner_phi[i])])
        
    return inner, (Cx, Cy), S, np.degrees(Theta)
# EVOLVE-BLOCK-END
