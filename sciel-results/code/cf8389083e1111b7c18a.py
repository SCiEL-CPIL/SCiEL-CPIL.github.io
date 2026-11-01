"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit, leave 10s margin

    best_score = -np.inf
    best_config = None

    # Objective and Gradient
    def objective_and_grad(params):
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        obj = -np.sum(r)
        grad = np.zeros(3*N)
        grad[2*N:3*N] = -1.0
        
        penalty = 0.0
        grad_pen = np.zeros(3*N)
        
        # Overlap
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    p = overlap * overlap
                    penalty += p
                    # Gradient of p = 2 * overlap * (grad of overlap)
                    # grad overlap wrt x_i = -dx/dist
                    # grad overlap wrt x_j = dx/dist
                    # grad overlap wrt r_i = 1
                    # grad overlap wrt r_j = 1
                    
                    factor = 2.0 * overlap
                    inv_dist = 1.0 / dist
                    
                    grad_pen[i] += -factor * dx * inv_dist
                    grad_pen[N+j] += factor * dx * inv_dist
                    grad_pen[2*N+i] += factor
                    grad_pen[2*N+j] += factor
                    
                    grad_pen[N+i] += -factor * dy * inv_dist
                    grad_pen[N+j] += factor * dy * inv_dist
                    
        # Boundary
        for i in range(N):
            # Left: r > x
            viol = r[i] - x[i]
            if viol > 0:
                p = viol * viol
                penalty += p
                factor = 2.0 * viol
                grad_pen[i] += -factor
                grad_pen[2*N+i] += factor
            # Right: r > 1-x
            viol = r[i] - (1 - x[i])
            if viol > 0:
                p = viol * viol
                penalty += p
                factor = 2.0 * viol
                grad_pen[i] += factor
                grad_pen[2*N+i] += factor
            # Bottom: r > y
            viol = r[i] - y[i]
            if viol > 0:
                p = viol * viol
                penalty += p
                factor = 2.0 * viol
                grad_pen[N+i] += -factor
                grad_pen[2*N+i] += factor
            # Top: r > 1-y
            viol = r[i] - (1 - y[i])
            if viol > 0:
                p = viol * viol
                penalty += p
                factor = 2.0 * viol
                grad_pen[N+i] += factor
                grad_pen[2*N+i] += factor
                
        obj += 100.0 * penalty
        grad += 100.0 * grad_pen
        
        return obj, grad

    def objective(params):
        return objective_and_grad(params)[0]

    def is_valid(config):
        x, y, r = config[:, 0], config[:, 1], config[:, 2]
        if np.any(x < r - 1e-9) or np.any(1-x < r - 1e-9) or np.any(y < r - 1e-9) or np.any(1-y < r - 1e-9):
            return False
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9:
                    return False
        return True

    def score(config):
        if not is_valid(config):
            return -np.inf
        return np.sum(config[:, 2])

    bounds = [(0,1)]*N + [(0,1)]*N + [(0,0.5)]*N

    # Generate Seeds
    seeds = []
    
    # 1. Random
    for _ in range(30):
        x = np.random.rand(N)
        y = np.random.rand(N)
        r = np.random.rand(N) * 0.05 + 0.01
        seeds.append(np.concatenate([x, y, r]))
        
    # 2. Hexagonal
    # Create a hex grid
    hex_seeds = []
    dx = 0.1
    dy = dx * np.sqrt(3) / 2
    for i in range(15):
        for j in range(15):
            x = i * dx + (0.05 if j % 2 else 0)
            y = j * dy
            if 0 <= x <= 1 and 0 <= y <= 1:
                hex_seeds.append((x, y))
    # Shuffle and take N
    np.random.shuffle(hex_seeds)
    if len(hex_seeds) >= N:
        hx = np.array([p[0] for p in hex_seeds[:N]])
        hy = np.array([p[1] for p in hex_seeds[:N]])
        hr = np.full(N, 0.02)
        seeds.append(np.concatenate([hx, hy, hr]))

    # 3. Grid
    side = int(np.ceil(np.sqrt(N)))
    grid_x = []
    grid_y = []
    for i in range(side):
        for j in range(side):
            grid_x.append((i + 0.5) / side)
            grid_y.append((j + 0.5) / side)
    gx = np.array(grid_x[:N])
    gy = np.array(grid_y[:N])
    gr = np.full(N, 0.02)
    seeds.append(np.concatenate([gx, gy, gr]))

    # Optimization Loop
    for seed in seeds:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                objective, 
                seed, 
                jac=lambda p: objective_and_grad(p)[1],
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            config = res.x.reshape(N, 3)
            s = score(config)
            
            if s > best_score:
                best_score = s
                best_config = config
        except Exception:
            pass

    if best_config is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_config
# EVOLVE-BLOCK-END
