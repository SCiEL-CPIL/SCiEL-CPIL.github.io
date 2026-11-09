import numpy as np
from scipy.optimize import minimize, brentq

def construct():
    # Regular octagon edge length for diameter 2 (radius 1)
    # side = 2 * sin(pi/8)
    l_reg = 2 * np.sin(np.pi / 8)
    # We optimize 7 lengths. The 8th is determined by the sum.
    # Fix perimeter to 8 * l_reg for consistency with regular octagon scale?
    # Actually, scale is normalized later. So perimeter value doesn't matter, only ratios.
    # Let's fix sum of lengths to 8.
    P = 8.0
    l_reg_scaled = l_reg * (P / (8 * l_reg)) # This is just 1.0?
    # Wait, if we fix P=8, and regular octagon has P_reg = 8 * 2 sin(pi/8) approx 6.12.
    # If we set P=8, the regular octagon in this parametrization would have l_i = 1.0.
    # Let's check if l_i=1.0 forms a regular octagon in this cyclic model.
    # If l_i=1, sum arcs = 2pi.
    # 8 * 2 arcsin(1 / 2R) = 2pi => arcsin(1/2R) = pi/8 => 1/2R = sin(pi/8) => R = 1/(2 sin(pi/8)).
    # This is the circumradius of a regular octagon with side 1.
    # So yes, l_i=1.0 corresponds to the regular octagon shape.
    
    x0 = np.ones(7)
    
    def objective(l_flat):
        l = np.append(l_flat, P - np.sum(l_flat))
        if np.any(l <= 0):
            return 1e10
        
        # Find R such that sum(2 * arcsin(l_i / 2R)) = 2 * pi
        # Function to solve for R
        def g(R):
            if np.any(l >= 2 * R):
                return -1 # Invalid, R too small
            return np.sum(2 * np.arcsin(l / (2 * R))) - 2 * np.pi
        
        # R must be > max(l)/2
        R_min = np.max(l) / 2 + 1e-6
        # As R -> inf, g(R) -> -2pi
        # As R -> R_min, one term -> pi, others > 0. Sum > 2pi?
        # If l is very uneven, it might not close.
        # Let's check range.
        # If g(R_min) < 0, no solution (polygon can't close with these lengths? 
        # Actually, if sum of lengths is too large for the radius, it opens up?
        # No, for fixed lengths, if R is small, arcs are large.
        # If R is large, arcs are small.
        # We need sum arcs = 2pi.
        # If at R_min, sum arcs < 2pi, then even the smallest possible radius gives too small a sum?
        # Wait. arcsin(x) increases with x.
        # l/2R decreases as R increases.
        # So arcs decrease as R increases.
        # So g(R) decreases as R increases.
        # g(R_min) is the max value of the sum of arcs.
        # If g(R_min) < 2pi, then we can never reach 2pi (sum is always < 2pi). No solution.
        # If g(R_min) > 2pi, there is a solution.
        
        try:
            val_min = g(R_min)
            if val_min < 0:
                return 1e10 # Cannot form closed polygon
            # Find upper bound for R
            R_max = R_min * 100
            while g(R_max) > 0:
                R_max *= 2
                if R_max > 1e6:
                    return 1e10
            
            R_sol = brentq(g, R_min, R_max)
            
            # Calculate angles
            alphas = 2 * np.arcsin(l / (2 * R_sol))
            
            # Cumulative angles
            theta = np.cumsum(np.concatenate(([0], alphas)))
            
            # Points on circle of radius R_sol
            pts = np.column_stack((R_sol * np.cos(theta[:-1]), R_sol * np.sin(theta[:-1])))
            
            # Scale to diameter 2
            # Diameter is 2 * R_sol
            # We need diameter 2, so scale factor is 1/R_sol
            # But wait, the problem says "scaling the set to diameter 2".
            # The points are already on a circle of radius R_sol. Diameter is 2 R_sol.
            # So scaling by 1/R_sol gives diameter 2.
            pts_scaled = pts / R_sol
            
            # Calculate product of distances
            # Use log sum
            # Pairwise distances
            # n=8, 28 pairs
            dists = []
            for i in range(8):
                for j in range(i+1, 8):
                    d = np.linalg.norm(pts_scaled[i] - pts_scaled[j])
                    if d < 1e-12:
                        return 1e10
                    dists.append(d)
            
            log_prod = np.sum(np.log(dists))
            return -log_prod
            
        except Exception:
            return 1e10

    # Optimize
    # Use Nelder-Mead for robustness
    result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    l_opt = np.append(result.x, P - np.sum(result.x))
    
    # Reconstruct points
    def g(R):
        return np.sum(2 * np.arcsin(l_opt / (2 * R))) - 2 * np.pi
        
    R_min = np.max(l_opt) / 2 + 1e-6
    R_max = R_min * 100
    while g(R_max) > 0:
        R_max *= 2
    R_sol = brentq(g, R_min, R_max)
    
    alphas = 2 * np.arcsin(l_opt / (2 * R_sol))
    theta = np.cumsum(np.concatenate(([0], alphas)))
    pts = np.column_stack((R_sol * np.cos(theta[:-1]), R_sol * np.sin(theta[:-1])))
    
    # Scale to diameter 2
    pts_scaled = pts / R_sol
    
    return pts_scaled.tolist()
