import numpy as np
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Maximizes the sum of side lengths of 10 axis-aligned squares in the unit square.
    Uses a barrier-based optimization with diverse initial configurations.
    """
    n = 10
    tol = 1e-7
    
    def objective(z):
        # z shape: (30,) -> [cx1, cy1, s1, cx2, cy2, s2, ...]
        cx = z[0::3]
        cy = z[1::3]
        s = z[2::3]
        
        # Maximize sum of sides -> Minimize negative sum
        obj = -np.sum(s)
        
        penalty = 0.0
        
        # Boundary penalties: squares must be inside [0,1]^2
        # cx - s/2 >= 0, cx + s/2 <= 1, same for cy
        for i in range(n):
            # Left boundary
            v = max(0.0, s[i]/2.0 - cx[i])
            penalty += v * v
            # Right boundary
            v = max(0.0, cx[i] + s[i]/2.0 - 1.0)
            penalty += v * v
            # Bottom boundary
            v = max(0.0, s[i]/2.0 - cy[i])
            penalty += v * v
            # Top boundary
            v = max(0.0, cy[i] + s[i]/2.0 - 1.0)
            penalty += v * v
        
        # Overlap penalties
        for i in range(n):
            for j in range(i + 1, n):
                # Required center distance to avoid overlap
                req = (s[i] + s[j]) / 2.0
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                
                # Overlap if dx < req AND dy < req
                # Use a smooth approximation
                ox = max(0.0, req - dx)
                oy = max(0.0, req - dy)
                
                # Penalty is active only if BOTH dimensions overlap
                # Use product to ensure both must be violated
                if ox > 0 and oy > 0:
                    penalty += (ox * oy) ** 2
                else:
                    # Small penalty for near-overlap to guide optimizer
                    penalty += max(0.0, ox - 1e-4) ** 2 + max(0.0, oy - 1e-4) ** 2
        
        obj += 5000.0 * penalty
        return obj
    
    def is_valid(cx, cy, s):
        """Check if configuration is valid within tolerance."""
        # Check bounds
        for i in range(n):
            if cx[i] - s[i]/2.0 < -tol:
                return False
            if cx[i] + s[i]/2.0 > 1.0 + tol:
                return False
            if cy[i] - s[i]/2.0 < -tol:
                return False
            if cy[i] + s[i]/2.0 > 1.0 + tol:
                return False
            if s[i] <= 0:
                return False
        
        # Check overlaps
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                if t <= 0:
                    continue
                if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                    return False
        return True
    
    def get_initial_config(type_idx):
        """Generate diverse initial configurations."""
        if type_idx == 0:
            # 4x3 grid (12 cells, use 10)
            cols, rows = 4, 3
            s = np.full(n, 1.0 / cols)
            cx = np.zeros(n)
            cy = np.zeros(n)
            for i in range(n):
                c = i % cols
                r = i // cols
                cx[i] = (c + 0.5) / cols
                cy[i] = (r + 0.5) / rows
            return np.concatenate([cx, cy, s]).reshape(-1, 3).T.flatten()
        
        elif type_idx == 1:
            # 5x2 grid
            cols, rows = 5, 2
            s = np.full(n, 1.0 / cols)
            cx = np.zeros(n)
            cy = np.zeros(n)
            for i in range(n):
                c = i % cols
                r = i // cols
                cx[i] = (c + 0.5) / cols
                cy[i] = (r + 0.5) / rows
            return np.concatenate([cx, cy, s]).reshape(-1, 3).T.flatten()
        
        elif type_idx == 2:
            # 2 large in corners, 8 small in a 4x2 grid in remaining space
            # Large squares: side 0.4, placed at corners
            s = np.zeros(n)
            cx = np.zeros(n)
            cy = np.zeros(n)
            
            # Two large squares
            s[0] = 0.4
            s[1] = 0.4
            cx[0] = 0.2
            cy[0] = 0.2
            cx[1] = 0.8
            cy[1] = 0.8
            
            # 8 small squares in the middle region
            # Available region: x in [0.4, 0.6] is narrow, so use full width
            # Actually, let's place them in a 4x2 grid in [0,1]x[0,1] avoiding large ones
            # Simpler: 4x2 grid with side 0.25, but adjust
            for i in range(2, n):
                idx = i - 2
                c = idx % 4
                r = idx // 4
                s[i] = 0.22
                cx[i] = (c + 0.5) * 0.25
                cy[i] = 0.5 + (r - 0.5) * 0.25
            
            return np.concatenate([cx, cy, s]).reshape(-1, 3).T.flatten()
        
        elif type_idx == 3:
            # Random with reasonable sizes
            s = np.random.uniform(0.15, 0.35, n)
            cx = np.random.uniform(0.1, 0.9, n)
            cy = np.random.uniform(0.1, 0.9, n)
            return np.concatenate([cx, cy, s]).reshape(-1, 3).T.flatten()
        
        elif type_idx == 4:
            # 3x4 grid
            cols, rows = 3, 4
            s = np.full(n, 1.0 / cols)
            cx = np.zeros(n)
            cy = np.zeros(n)
            for i in range(n):
                c = i % cols
                r = i // cols
                cx[i] = (c + 0.5) / cols
                cy[i] = (r + 0.5) / rows
            return np.concatenate([cx, cy, s]).reshape(-1, 3).T.flatten()
        
        else:
            # Perturbed grid
            cols, rows = 4, 3
            s = np.full(n, 1.0 / cols * 0.95)
            cx = np.zeros(n)
            cy = np.zeros(n)
            for i in range(n):
                c = i % cols
                r = i // cols
                cx[i] = (c + 0.5) / cols + np.random.uniform(-0.01, 0.01)
                cy[i] = (r + 0.5) / rows + np.random.uniform(-0.01, 0.01)
            return np.concatenate([cx, cy, s]).reshape(-1, 3).T.flatten()
    
    best_score = -np.inf
    best_result = None
    
    bounds = [(0, 1)] * (n * 3)
    
    # Try multiple initial configurations
    for init_type in range(6):
        z0 = get_initial_config(init_type)
        
        try:
            res = minimize(objective, z0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            z = res.x
            cx = z[0::3]
            cy = z[1::3]
            s = z[2::3]
            
            if is_valid(cx, cy, s):
                score = np.sum(s)
                if score > best_score:
                    best_score = score
                    best_result = (cx.copy(), cy.copy(), s.copy())
                    
        except Exception:
            pass
    
    # If no valid result found, use fallback
    if best_result is not None:
        cx, cy, s = best_result
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = s.astype(np.float64)
        return centers, sides
    else:
        # Fallback: 4x3 grid
        cols, rows = 4, 3
        s = np.full(n, 1.0 / cols)
        cx = np.zeros(n)
        cy = np.zeros(n)
        for i in range(n):
            c = i % cols
            r = i // cols
            cx[i] = (c + 0.5) / cols
            cy[i] = (r + 0.5) / rows
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = s.astype(np.float64)
        return centers, sides
