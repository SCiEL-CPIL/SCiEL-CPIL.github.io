import numpy as np
from scipy.optimize import differential_evolution

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7

    # Analytic seed: 4 corners (1/3), 4 edges (1/3), 2 center (1/6)
    # Sum = 4*(1/3) + 4*(1/3) + 2*(1/6) = 3.0
    a = 1.0 / 3.0
    b = 1.0 / 3.0
    c = 1.0 / 6.0

    centers = np.zeros((n, 2))
    sides = np.zeros(n)

    # Corners (0-3)
    centers[0] = [a/2, a/2]
    centers[1] = [1-a/2, a/2]
    centers[2] = [a/2, 1-a/2]
    centers[3] = [1-a/2, 1-a/2]
    sides[0:4] = a

    # Edges (4-7)
    centers[4] = [0.5, b/2]      # Bottom
    centers[5] = [0.5, 1-b/2]    # Top
    centers[6] = [b/2, 0.5]      # Left
    centers[7] = [1-b/2, 0.5]    # Right
    sides[4:8] = b

    # Center (8-9)
    # Vertical space between edges: 1 - 2*b = 1/3.
    # Two squares of side c=1/6 fit exactly.
    # Square 8: y in [b, b+c] = [1/3, 1/2]. Center y = 5/12.
    # Square 9: y in [1-b-c, 1-b] = [1/2, 2/3]. Center y = 7/12.
    centers[8] = [0.5, 5.0/12.0]
    centers[9] = [0.5, 7.0/12.0]
    sides[8:10] = c

    # Validation function
    def is_valid(c, s):
        for i in range(n):
            if not (np.isfinite(c[i,0]) and np.isfinite(c[i,1]) and np.isfinite(s[i])):
                return False
            if s[i] <= 0:
                return False
            if c[i,0] - s[i]/2 < -tol or c[i,0] + s[i]/2 > 1 + tol:
                return False
            if c[i,1] - s[i]/2 < -tol or c[i,1] + s[i]/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j])/2 - tol
                if abs(c[i,0] - c[j,0]) < t and abs(c[i,1] - c[j,1]) < t:
                    return False
        return True

    # Objective function for optimization
    # We maximize sum of sides, so minimize -sum(sides) + penalty
    def objective(x):
        c = x[:2*n].reshape(n, 2)
        s = x[2*n:]
        
        # Penalty for invalidity
        pen = 0.0
        for i in range(n):
            # Bounds
            if c[i,0] - s[i]/2 < 0: pen += (c[i,0] - s[i]/2)**2
            if c[i,0] + s[i]/2 > 1: pen += (c[i,0] + s[i]/2 - 1)**2
            if c[i,1] - s[i]/2 < 0: pen += (c[i,1] - s[i]/2)**2
            if c[i,1] + s[i]/2 > 1: pen += (c[i,1] + s[i]/2 - 1)**2
            
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(c[i,0] - c[j,0])
                dy = abs(c[i,1] - c[j,1])
                t = (s[i] + s[j])/2
                if dx < t and dy < t:
                    # Overlap penalty: square of overlap area
                    ox = t - dx
                    oy = t - dy
                    pen += ox * oy

        return -np.sum(s) + 100.0 * pen

    # Bounds for differential evolution
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # cx
        bounds.append((0.0, 1.0)) # cy
    for i in range(n):
        bounds.append((1e-4, 0.5)) # s (max side is 0.5 for 2x2 grid, but 10 squares likely smaller)

    x0 = np.concatenate([centers.flatten(), sides])
    best_x = x0
    best_val = np.sum(sides)

    # Run differential evolution
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=10,
            tol=1e-8,
            polish=True
        )
        if res.success or res.fun < objective(x0):
            x_new = res.x
            c_new = x_new[:2*n].reshape(n, 2)
            s_new = x_new[2*n:]
            if is_valid(c_new, s_new):
                val_new = np.sum(s_new)
                if val_new > best_val:
                    best_x = x_new
                    best_val = val_new
    except Exception:
        pass

    c_opt = best_x[:2*n].reshape(n, 2)
    s_opt = best_x[2*n:]

    # Final safety check and shrink if needed
    if not is_valid(c_opt, s_opt):
        # Fallback to analytic seed if optimization failed
        c_opt = centers
        s_opt = sides
        
    return c_opt, s_opt
