import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_score = -np.inf
    best_x = None
    
    # Initial guesses
    # 1. Square grid 5x5 (first 21)
    # 2. Hexagonal packing
    # 3. Random uniform
    
    def initial_grid():
        side = 5
        w = 1.0
        h = 1.0
        r = 0.18
        xs = []
        ys = []
        rs = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x = (i + 0.5) * (w / side)
                y = (j + 0.5) * (h / side)
                xs.append(x)
                ys.append(y)
                rs.append(r)
                count += 1
        # Add w as variable
        return np.array(xs + ys + rs + [w])

    def initial_hex():
        # Hexagonal packing in a 1x1 square
        w = 1.0
        h = 1.0
        r = 0.15
        xs = []
        ys = []
        rs = []
        count = 0
        dy = 2 * r * np.sqrt(3) / 2
        row = 0
        while count < N:
            y = r + row * dy
            if y + r > h:
                break
            offset = 0 if row % 2 == 0 else r
            x = r + offset
            col = 0
            while x + r <= w and count < N:
                xs.append(x)
                ys.append(y)
                rs.append(r)
                count += 1
                x += 2 * r
            row += 1
        if count < N:
            # Fill remaining with grid
            for i in range(N - count):
                xs.append(0.5)
                ys.append(0.5)
                rs.append(0.1)
        return np.array(xs + ys + rs + [w])

    def initial_random():
        w = 1.0
        xs = np.random.rand(N)
        ys = np.random.rand(N)
        rs = np.full(N, 0.1)
        return np.array(xs + ys + rs + [w])

    def objective(x):
        xs = x[:N]
        ys = x[N:2*N]
        rs = x[2*N:3*N]
        w = x[3*N]
        h = 2.0 - w
        return -np.sum(rs)

    def constraints(x):
        xs = x[:N]
        ys = x[N:2*N]
        rs = x[2*N:3*N]
        w = x[3*N]
        h = 2.0 - w
        
        cons = []
        # Boundary constraints
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*N+i]}) # x_i >= r_i
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*N] - x[i] - x[2*N+i]}) # w - x_i >= r_i
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[N+i] - x[2*N+i]}) # y_i >= r_i
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: (2.0 - x[3*N]) - x[N+i] - x[2*N+i]}) # h - y_i >= r_i
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*N+i]}) # r_i >= 0
            
        # Pairwise constraints
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq', 
                    'fun': lambda x, i=i, j=j: (x[i]-x[j])**2 + (x[N+i]-x[N+j])**2 - (x[2*N+i]+x[2*N+j])**2
                })
        
        # Aspect ratio constraints (optional, to prevent degenerate rectangles)
        cons.append({'type': 'ineq', 'fun': lambda x: x[3*N] - 0.5}) # w >= 0.5
        cons.append({'type': 'ineq', 'fun': lambda x: 1.5 - x[3*N]}) # w <= 1.5
        
        return cons

    def bounds():
        bounds = []
        for i in range(N):
            bounds.append((0.0, 2.0)) # x
        for i in range(N):
            bounds.append((0.0, 2.0)) # y
        for i in range(N):
            bounds.append((0.0, 1.0)) # r
        bounds.append((0.5, 1.5)) # w
        return bounds

    initializers = [initial_grid, initial_hex, initial_random, initial_random, initial_random]
    
    for init_func in initializers:
        if time.time() > deadline:
            break
            
        x0 = init_func()
        
        # Run SLSQP
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds(), constraints=constraints, 
                           options={'maxiter': 500, 'ftol': 1e-8})
            if res.success:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_x = res.x
        except Exception:
            pass
            
    if best_x is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Extract result
    xs = best_x[:N]
    ys = best_x[N:2*N]
    rs = best_x[2*N:3*N]
    
    result = np.column_stack((xs, ys, rs))
    return result
