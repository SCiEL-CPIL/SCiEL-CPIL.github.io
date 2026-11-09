"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    max_time = 145.0  # Safety margin before 150s kill
    
    # Rectangle perimeter is 4, so 2(w+h) = 4 => w+h=2.
    # We fix aspect ratio w=1, h=1 for simplicity, or we can optimize w too.
    # Let's fix w=1, h=1 first to keep dimensionality lower (2N + 1 vs 2N).
    # Actually, optimizing aspect ratio can help. Let's optimize w as well.
    # Variables: x_i, y_i, r_i for i=0..N-1, and w.
    # h = 2 - w.
    # Constraints:
    # 1. x_i >= r_i, x_i + r_i <= w
    # 2. y_i >= r_i, y_i + r_i <= h
    # 3. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    # 4. r_i > 0
    # 5. w > 0, h > 0 => 0 < w < 2
    
    best_score = 0.0
    best_solution = None
    
    # Initial guess: Uniform grid
    side = int(np.ceil(np.sqrt(N)))
    # We will use a few different initial configurations
    
    def get_initial_guesses():
        guesses = []
        # 1. Uniform grid in a square
        w0 = 1.0
        h0 = 1.0
        xs = []
        ys = []
        rs = []
        for i in range(side):
            for j in range(side):
                if len(xs) < N:
                    xs.append((i + 0.5) / side)
                    ys.append((j + 0.5) / side)
                    rs.append(0.5 / side * 0.9) # Slightly smaller to avoid initial violation
        # Add w to the vector
        guesses.append(np.array(xs + ys + rs + [w0]))
        
        # 2. Random jittered grid
        for _ in range(5):
            xs = np.random.rand(N) * 0.8 + 0.1
            ys = np.random.rand(N) * 0.8 + 0.1
            rs = np.full(N, 0.1)
            w0 = 1.0
            guesses.append(np.array(xs + ys + rs + [w0]))
            
        # 3. Different aspect ratio initial
        w0 = 1.2
        h0 = 0.8
        xs = np.random.rand(N) * (w0 - 0.2) + 0.1
        ys = np.random.rand(N) * (h0 - 0.2) + 0.1
        rs = np.full(N, 0.1)
        guesses.append(np.array(xs + ys + rs + [w0]))
        
        return guesses

    def unpack(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        w = v[3*N]
        h = 2.0 - w
        return x, y, r, w, h

    def objective(v):
        x, y, r, w, h = unpack(v)
        # Penalize if w or h are out of bounds or invalid
        if w <= 0.1 or h <= 0.1:
            return 1e6
        # Maximize sum of radii => Minimize negative sum
        return -np.sum(r)

    def constraints(v):
        x, y, r, w, h = unpack(v)
        cons = []
        # Boundary constraints
        cons.append(w - 0.1) # w >= 0.1
        cons.append(h - 0.1) # h >= 0.1
        cons.append(1.9 - w) # w <= 1.9
        for i in range(N):
            cons.append(x[i] - r[i])
            cons.append(w - x[i] - r[i])
            cons.append(y[i] - r[i])
            cons.append(h - y[i] - r[i])
            cons.append(r[i] - 0.01) # r >= 0.01
        # Non-overlapping
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                sum_r = r[i] + r[j]
                cons.append(dist_sq - sum_r**2)
        return cons

    def is_valid(v):
        x, y, r, w, h = unpack(v)
        if w <= 0 or h <= 0:
            return False
        for i in range(N):
            if x[i] < r[i] - 1e-9 or x[i] + r[i] > w + 1e-9:
                return False
            if y[i] < r[i] - 1e-9 or y[i] + r[i] > h + 1e-9:
                return False
            if r[i] <= 0:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                sum_r = r[i] + r[j]
                if dist_sq < sum_r**2 - 1e-9:
                    return False
        return True

    def score(v):
        if not is_valid(v):
            return 0.0
        x, y, r, w, h = unpack(v)
        return np.sum(r)

    guesses = get_initial_guesses()
    
    for guess in guesses:
        if time.time() - start_time > max_time:
            break
            
        try:
            res = minimize(objective, guess, method='SLSQP', constraints=[{'type': 'ineq', 'fun': constraints}], 
                           options={'maxiter': 200, 'ftol': 1e-9, 'disp': False})
            
            if res.success:
                val = score(res.x)
                if val > best_score:
                    best_score = val
                    best_solution = res.x
        except Exception:
            pass

    # If no valid solution found, fallback to seed
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    x, y, r, w, h = unpack(best_solution)
    # Construct the final array
    result = np.column_stack((x, y, r))
    return result

# EVOLVE-BLOCK-END
