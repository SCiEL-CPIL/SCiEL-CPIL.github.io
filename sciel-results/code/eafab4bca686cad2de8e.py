import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    time_limit = 590.0  # Safety margin for 600s limit
    
    def is_valid(cx, cy, s, tol=1e-7):
        # Check bounds
        if np.any(s <= 0):
            return False
        if np.any(cx - s/2 < -tol) or np.any(cx + s/2 > 1 + tol):
            return False
        if np.any(cy - s/2 < -tol) or np.any(cy + s/2 > 1 + tol):
            return False
        
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j])/2.0 - tol
                if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                    return False
        return True

    def objective(params):
        # Minimize negative sum of sides
        # We assume params are valid or we handle invalidity by returning a large penalty
        # However, Nelder-Mead doesn't handle constraints well.
        # Better: use a penalty function that is very high if invalid.
        p = params.reshape(n, 3)
        cx = p[:, 0]
        cy = p[:, 1]
        s = p[:, 2]
        
        # Basic bounds check for penalty
        penalty = 0.0
        # Boundary
        if np.any(cx - s/2 < -1e-6) or np.any(cx + s/2 > 1 + 1e-6):
            penalty += 1000.0
        if np.any(cy - s/2 < -1e-6) or np.any(cy + s/2 > 1 + 1e-6):
            penalty += 1000.0
        if np.any(s <= 0.01):
            penalty += 1000.0
            
        # Overlaps
        for i in range(n):
            for j in range(i+1, n):
                ox = (s[i] + s[j])/2.0 - abs(cx[i] - cx[j])
                oy = (s[i] + s[j])/2.0 - abs(cy[i] - cy[j])
                if ox > 0 and oy > 0:
                    penalty += ox * oy * 1000.0 + 100.0
        
        return -np.sum(s) + penalty

    # Strategy:
    # 1. Generate multiple initial configurations based on known good structures.
    # 2. Run Nelder-Mead on each.
    # 3. Return the best valid result.
    
    best_params = None
    best_score = -np.inf
    
    # Known optimal-ish structure: 2 large squares (side ~0.5) and 8 small squares (side ~0.25)
    # Arrangement:
    # Large 1 at (0.25, 0.25), s=0.5
    # Large 2 at (0.75, 0.25), s=0.5
    # This leaves top half empty.
    # Small squares can fill the top half or interleave.
    # Actually, a better known packing for 10 squares:
    # 2 squares of side 0.5, 8 squares of side 0.25.
    # Sum = 2*0.5 + 8*0.25 = 1.0 + 2.0 = 3.0.
    # Can we do better?
    # If we use 1 square of side 0.5, 4 of 0.25, 5 of ...?
    # Let's try to maximize sum.
    
    # Initial guesses:
    # Guess 1: The "2 large + 8 small" grid.
    # Large squares at bottom left and bottom right.
    # Small squares in the remaining space.
    
    def create_guess_1():
        # 2 large squares
        cx = np.zeros(n)
        cy = np.zeros(n)
        s = np.zeros(n)
        
        # Square 0: Large
        cx[0] = 0.25; cy[0] = 0.25; s[0] = 0.5
        # Square 1: Large
        cx[1] = 0.75; cy[1] = 0.25; s[1] = 0.5
        
        # Remaining 8 small squares (side 0.25)
        # Top left quadrant: (0.25, 0.75)
        # Top right quadrant: (0.75, 0.75)
        # We can fit 4 small squares in each top quadrant? No, 0.25*4 = 1.0 width.
        # Top left: x in [0, 0.5], y in [0.5, 1.0].
        # Small squares of 0.25:
        # (0.125, 0.625), (0.375, 0.625), (0.125, 0.875), (0.375, 0.875)
        # (0.625, 0.625), (0.875, 0.625), (0.625, 0.875), (0.875, 0.875)
        
        small_positions = [
            (0.125, 0.625), (0.375, 0.625), (0.125, 0.875), (0.375, 0.875),
            (0.625, 0.625), (0.875, 0.625), (0.625, 0.875), (0.875, 0.875)
        ]
        
        for i in range(2, 10):
            cx[i] = small_positions[i-2][0]
            cy[i] = small_positions[i-2][1]
            s[i] = 0.25
            
        return np.array([cx, cy, s]).T.flatten()

    def create_guess_2():
        # Try a different arrangement, maybe 1 large, 9 small?
        # Or random perturbation of guess 1
        g1 = create_guess_1()
        # Add small noise
        noise = np.random.normal(0, 0.01, size=n*3)
        g2 = g1 + noise
        # Clamp s to positive
        for i in range(n):
            if g2[i*3+2] < 0.1:
                g2[i*3+2] = 0.1
        return g2

    def create_guess_3():
        # Random valid-ish start
        # Just place 10 small squares in a grid
        cx = np.zeros(n)
        cy = np.zeros(n)
        s = np.full(n, 0.3)
        
        # 3x3 grid + 1?
        # Let's just do a 4x4 grid of 0.25 squares, pick 10
        positions = []
        for r in range(4):
            for c in range(4):
                positions.append((c*0.25 + 0.125, r*0.25 + 0.125))
        
        # Pick first 10
        for i in range(10):
            cx[i] = positions[i][0]
            cy[i] = positions[i][1]
            s[i] = 0.25
            
        return np.array([cx, cy, s]).T.flatten()

    guesses = [create_guess_1(), create_guess_2(), create_guess_3()]
    
    # Add a few more random starts
    for _ in range(5):
        cx = np.random.rand(n) * 0.8 + 0.1
        cy = np.random.rand(n) * 0.8 + 0.1
        s = np.random.rand(n) * 0.4 + 0.1
        # Ensure they don't overlap initially? Hard to guarantee.
        # Just feed to optimizer.
        guesses.append(np.array([cx, cy, s]).T.flatten())

    for i, guess in enumerate(guesses):
        if time.time() - start_time > time_limit - 30:
            break
            
        try:
            # Nelder-Mead is good for this low-dimensional, non-smooth-ish problem
            # if we use a penalty function.
            res = minimize(
                objective, 
                guess, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            # Check if result is valid
            p = res.x.reshape(n, 3)
            cx = p[:, 0]
            cy = p[:, 1]
            s = p[:, 2]
            
            if is_valid(cx, cy, s):
                score = np.sum(s)
                if score > best_score:
                    best_score = score
                    best_params = res.x
                    
        except Exception:
            continue

    if best_params is None:
        # Fallback to a very simple valid configuration
        # 4x4 grid of 0.25 squares, take 10
        cx = np.zeros(n)
        cy = np.zeros(n)
        s = np.full(n, 0.25)
        for i in range(10):
            r = i // 4
            c = i % 4
            cx[i] = c * 0.25 + 0.125
            cy[i] = r * 0.25 + 0.125
        centers = np.stack([cx, cy], axis=1)
        sides = s
        return centers, sides

    p = best_params.reshape(n, 3)
    cx = p[:, 0]
    cy = p[:, 1]
    s = p[:, 2]
    
    centers = np.stack([cx, cy], axis=1)
    sides = s
    
    # Final sanity check
    if not is_valid(cx, cy, s):
        # Fallback
        cx = np.zeros(n)
        cy = np.zeros(n)
        s = np.full(n, 0.25)
        for i in range(10):
            r = i // 4
            c = i % 4
            cx[i] = c * 0.25 + 0.125
            cy[i] = r * 0.25 + 0.125
        centers = np.stack([cx, cy], axis=1)
        sides = s
        
    return centers, sides
