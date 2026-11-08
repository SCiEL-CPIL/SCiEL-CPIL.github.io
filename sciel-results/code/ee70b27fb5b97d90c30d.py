import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    # We will try to maximize the sum of radii in a 1x1 square (Perimeter 4).
    # If we find a configuration that fits in a 1x1 square, it is valid.
    # We can also try to optimize the aspect ratio, but 1x1 is a strong baseline.
    
    def objective(vars):
        # vars: [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        
        # Objective: Maximize sum(r) -> Minimize -sum(r)
        obj = -np.sum(r)
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    penalty += (min_dist - dist)**2
        
        # Penalty for boundaries (must stay within [0,1]x[0,1])
        # x[i] must be >= r[i] and <= 1 - r[i]
        # y[i] must be >= r[i] and <= 1 - r[i]
        for i in range(N):
            if x[i] < r[i]:
                penalty += (r[i] - x[i])**2
            if x[i] > 1 - r[i]:
                penalty += (x[i] - (1 - r[i]))**2
            if y[i] < r[i]:
                penalty += (r[i] - y[i])**2
            if y[i] > 1 - r[i]:
                penalty += (y[i] - (1 - r[i]))**2
        
        # Penalty for negative radii (though bounded below in optimizer)
        for i in range(N):
            if r[i] < 0:
                penalty += r[i]**2
                
        return obj + 100 * penalty

    def run_single_trial(seed):
        np.random.seed(seed)
        # Initial guess: Grid-like distribution
        side = int(np.ceil(np.sqrt(N)))
        x0 = []
        y0 = []
        r0 = []
        for i in range(N):
            col = i % side
            row = i // side
            # Spread them out
            xi = (col + 0.5) / side
            yi = (row + 0.5) / side
            # Add some noise
            xi += np.random.uniform(-0.05, 0.05)
            yi += np.random.uniform(-0.05, 0.05)
            x0.append(xi)
            y0.append(yi)
            r0.append(0.05) # Small initial radius
            
        vars0 = []
        for i in range(N):
            vars0.extend([x0[i], y0[i], r0[i]])
            
        # Bounds:
        # x, y in [0, 1]
        # r in [0, 0.5] (max possible in 1x1)
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.0, 0.5)) # r
            
        # Use L-BFGS-B
        res = minimize(objective, vars0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-10})
        
        # Check validity
        x = res.x[0::3]
        y = res.x[1::3]
        r = res.x[2::3]
        
        # Verify constraints strictly
        valid = True
        for i in range(N):
            if x[i] < r[i] - 1e-5 or x[i] > 1 - r[i] + 1e-5:
                valid = False
            if y[i] < r[i] - 1e-5 or y[i] > 1 - r[i] + 1e-5:
                valid = False
            if r[i] <= 0:
                valid = False
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dist < r[i] + r[j] - 1e-5:
                    valid = False
                    
        if valid:
            return res.x, np.sum(r)
        else:
            # Return a valid fallback or None
            return None, 0.0

    best_vars = None
    best_score = 0.0
    
    # Time limit: 120s. Let's run as many trials as possible.
    start_time = time.time()
    trial = 0
    while time.time() - start_time < 100: # Leave 20s buffer
        trial += 1
        # Try different random seeds
        seed = 42 + trial
        # Also try a "hexagonal" initial guess every few trials
        if trial % 5 == 0:
             # Hexagonal packing initial guess
             x0 = []
             y0 = []
             r0 = []
             spacing = 0.2
             for row in range(6):
                 for col in range(6):
                     if len(x0) >= N: break
                     xi = col * spacing * 0.866 + (0.5 * spacing * 0.866 if row % 2 else 0)
                     yi = row * spacing * 0.75
                     # Scale to fit in 1x1 roughly
                     xi = xi / 1.2 
                     yi = yi / 1.2
                     x0.append(xi)
                     y0.append(yi)
                     r0.append(0.1)
             # Pad if necessary
             while len(x0) < N:
                 x0.append(np.random.uniform(0.1, 0.9))
                 y0.append(np.random.uniform(0.1, 0.9))
                 r0.append(0.1)
             vars0 = []
             for i in range(N):
                 vars0.extend([x0[i], y0[i], r0[i]])
             bounds = []
             for i in range(N):
                 bounds.append((0.0, 1.0))
                 bounds.append((0.0, 1.0))
                 bounds.append((0.0, 0.5))
             res = minimize(objective, vars0, method='L-BFGS-B', bounds=bounds, 
                        options={'maxiter': 1000, 'ftol': 1e-10})
             x = res.x[0::3]
             y = res.x[1::3]
             r = res.x[2::3]
             valid = True
             for i in range(N):
                 if x[i] < r[i] - 1e-5 or x[i] > 1 - r[i] + 1e-5: valid = False
                 if y[i] < r[i] - 1e-5 or y[i] > 1 - r[i] + 1e-5: valid = False
             for i in range(N):
                 for j in range(i+1, N):
                     dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                     if dist < r[i] + r[j] - 1e-5: valid = False
             if valid:
                 score = np.sum(r)
                 if score > best_score:
                     best_score = score
                     best_vars = res.x
        else:
            vars, score = run_single_trial(seed)
            if vars is not None and score > best_score:
                best_score = score
                best_vars = vars

    if best_vars is None:
        # Fallback to seed solution
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Format output: (x, y, r)
    result = []
    for i in range(N):
        result.append([best_vars[3*i], best_vars[3*i+1], best_vars[3*i+2]])
        
    return np.array(result)
