import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_val = 0.0
    best_sol = None
    
    # Helper to check validity and calculate score
    def evaluate(sol):
        # sol: [x1...xN, y1...yN, r1...rN, w]
        x = sol[:N]
        y = sol[N:2*N]
        r = sol[2*N:3*N]
        w = sol[3*N]
        h = 2.0 - w
        
        if w <= 0 or h <= 0:
            return 0.0, False
        if np.any(r <= 0):
            return 0.0, False
        if np.any(x < r) or np.any(x > w - r) or np.any(y < r) or np.any(y > h - r):
            return 0.0, False
            
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                rad_sum = r[i] + r[j]
                if dist_sq < rad_sum**2 - 1e-9:
                    return 0.0, False
        
        return np.sum(r), True

    def objective(params):
        # params: [x, y, r, w]
        val, valid = evaluate(params)
        if not valid:
            return 1e6
        return -val

    def constraints(params):
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        w = params[3*N]
        h = 2.0 - w
        
        cons = []
        # Boundary constraints
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2*N+i]}) # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*N] - p[i] - p[2*N+i]}) # w - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[N+i] - p[2*N+i]}) # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: 2.0 - p[3*N] - p[N+i] - p[2*N+i]}) # h - y_i - r_i >= 0
        
        # Non-overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                cons.append({'type': 'ineq', 'fun': lambda p, i=i, j=j: (p[i]-p[j])**2 + (p[N+i]-p[N+j])**2 - (p[2*N+i]+p[2*N+j])**2})
                
        return cons

    def run_optimization(x0):
        bounds = []
        for i in range(N):
            bounds.append((1e-4, 2.0)) # x
        for i in range(N):
            bounds.append((1e-4, 2.0)) # y
        for i in range(N):
            bounds.append((1e-4, 1.0)) # r
        bounds.append((0.1, 1.9)) # w
        
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
        
        if res.success or (res.fun < 1e5):
            val, valid = evaluate(res.x)
            if valid and val > 0:
                return res.x, val
        return None, 0.0

    # Initial Guess 1: 5x4 Grid + 1
    # 5 cols, 4 rows.
    # w=10/9, h=8/9 approx.
    # Let's start with w=1.0, h=1.0 and let it adjust.
    x0_1 = np.zeros(3*N + 1)
    x0_1[3*N] = 1.0 # w
    # 5x4 grid
    idx = 0
    for i in range(5):
        for j in range(4):
            x0_1[idx] = (i + 0.5) / 5.0
            x0_1[N + idx] = (j + 0.5) / 4.0
            x0_1[2*N + idx] = 0.09 # slightly less than 0.1
            idx += 1
    # 21st circle in center
    x0_1[idx] = 0.5
    x0_1[N + idx] = 0.5
    x0_1[2*N + idx] = 0.05
    
    # Initial Guess 2: Hexagonal
    x0_2 = np.zeros(3*N + 1)
    x0_2[3*N] = 1.0
    # Rows: 4, 5, 4, 4, 4
    rows = [4, 5, 4, 4, 4]
    idx = 0
    dy = 0.2
    for i, count in enumerate(rows):
        y_pos = 0.2 + i * dy
        x_start = (1.0 - count * 0.2) / 2.0 + 0.1
        for j in range(count):
            x0_2[idx] = x_start + j * 0.2
            x0_2[N + idx] = y_pos
            x0_2[2*N + idx] = 0.09
            idx += 1

    # Initial Guess 3: Random
    np.random.seed(42)
    x0_3 = np.random.rand(3*N + 1) * 0.8 + 0.1
    x0_3[3*N] = 1.0
    for i in range(N):
        x0_3[2*N + i] = 0.08

    guesses = [x0_1, x0_2, x0_3]
    
    for guess in guesses:
        if time.time() > deadline:
            break
        sol, val = run_optimization(guess)
        if val > best_val:
            best_val = val
            best_sol = sol
            
    if best_sol is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Extract result
    x = best_sol[:N]
    y = best_sol[N:2*N]
    r = best_sol[2*N:3*N]
    
    return np.column_stack((x, y, r))
