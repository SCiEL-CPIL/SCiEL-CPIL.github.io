import numpy as np
from scipy.optimize import minimize, brentq
from scipy.spatial.distance import pdist

def construct():
        # Perimeter fixed to 8.0 for simplicity
        P = 8.0
        
        def get_vertices_from_lengths(lengths):
            # lengths: array of 8 positive floats
            # Returns array of 8 (x,y) points
            l_max = np.max(lengths)
            l_sum = np.sum(lengths)
            
            # Check validity for cyclic polygon
            if l_max >= l_sum - l_max: # l_max >= sum(others)
                return None
            
            # Find R
            # h(R) = sum(2 * arcsin(li / 2R)) - 2*pi
            def h(R):
                # Avoid domain error if li/2R > 1
                # R must be > l_max/2
                if R <= l_max / 2.0:
                    return np.inf
                args = lengths / (2.0 * R)
                # Clip to [-1, 1] just in case of floating point
                args = np.clip(args, -1.0, 1.0)
                angles = 2.0 * np.arcsin(args)
                return np.sum(angles) - 2.0 * np.pi
            
            # Find bracket for brentq
            # Lower bound: slightly above l_max/2
            R_low = l_max / 2.0 + 1e-6
            # Upper bound: start with l_sum / 2 and increase if needed
            R_high = l_sum / 2.0
            
            # Ensure h(R_low) > 0 and h(R_high) < 0
            # If not, expand range
            try:
                # If h(R_low) is not positive, the polygon might be degenerate or invalid in this branch
                # But for convex cyclic, h(R_low) should be > 0 if valid.
                # Let's just try brentq
                R = brentq(h, R_low, R_high, xtol=1e-10)
            except ValueError:
                # Might need to expand R_high
                # If h(R_high) > 0, increase R_high
                # If h(R_low) < 0, decrease R_low (impossible, it's the boundary)
                # Actually, if h(R_low) < 0, it means sum of angles < 2pi at the "flattest" possible circle?
                # This implies the polygon cannot be cyclic? No, it implies my assumption about h(R_low) > 0 is wrong for some cases?
                # Let's just return a penalty or regular octagon if it fails.
                return None

            # Calculate angles
            angles = 2.0 * np.arcsin(lengths / (2.0 * R))
            
            # Construct vertices
            # Start at angle 0
            theta = 0
            pts = np.zeros((8, 2))
            for i in range(8):
                pts[i, 0] = R * np.cos(theta)
                pts[i, 1] = R * np.sin(theta)
                theta += angles[i]
                
            return pts

        def objective(u):
            # u: 7 floats, log-lengths
            logs = u
            l_1_to_7 = np.exp(logs)
            l_sum_7 = np.sum(l_1_to_7)
            l_8 = P - l_sum_7
            
            if l_8 <= 1e-6:
                return 1e10
            
            lengths = np.concatenate([l_1_to_7, [l_8]])
            
            # Check max length constraint
            l_max = np.max(lengths)
            l_sum_others = P - l_max
            if l_max >= l_sum_others:
                return 1e10
            
            pts = get_vertices_from_lengths(lengths)
            if pts is None:
                return 1e10
            
            # Calculate distances
            dists = pdist(pts)
            if np.any(dists < 1e-10):
                return 1e10
            
            # Maximize product of distances (unordered)
            # Minimize negative log sum
            return -np.sum(np.log(dists))

        # Initial guess: Regular octagon
        # Side length 1.0
        # log(1.0) = 0
        u0 = np.zeros(7)
        
        # Bounds for u?
        # l_i > 0 => u can be -inf.
        # l_i < P => u < log(P).
        # Let's use reasonable bounds to keep optimizer stable
        bounds = [(-10, 10)] * 7
        
        # Optimize
        res = minimize(objective, u0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12})
        
        # Get optimal lengths
        logs_opt = res.x
        l_1_to_7 = np.exp(logs_opt)
        l_8 = P - np.sum(l_1_to_7)
        lengths = np.concatenate([l_1_to_7, [l_8]])
        
        pts = get_vertices_from_lengths(lengths)
        
        if pts is None:
            # Fallback to regular octagon
            angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
            pts = np.column_stack((np.cos(angles), np.sin(angles)))
            
        # Scale to diameter 2
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max > 1e-12:
            pts = pts * (2.0 / d_max)
            
        return pts.tolist()
