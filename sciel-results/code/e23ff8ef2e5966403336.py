import numpy as np
import time
from scipy.optimize import minimize

N = 26
DEADLINE = 100.0  # Safety margin before 120s limit

def construct():
    start_time = time.time()
    best_value = -np.inf
    best_result = None

    # Initial configurations
    def make_grid(n):
        side = int(np.ceil(np.sqrt(n)))
        # Slightly inset to avoid boundary issues initially
        xs = np.linspace(0.5/side, 1 - 0.5/side, side)
        ys = np.linspace(0.5/side, 1 - 0.5/side, side)
        points = []
        for i in range(side):
            for j in range(side):
                if len(points) < n:
                    points.append([xs[i], ys[j], 0.4/side])
        return np.array(points)

    def make_hex(n):
        # Hexagonal packing approximation
        points = []
        r = 0.1
        dx = 2 * r * np.sqrt(3) / 2
        dy = 2 * r * 1.5
        i = 0
        row = 0
        while len(points) < n:
            x_offset = (row % 2) * dx / 2
            for col in range(int(1 / dx) + 1):
                if len(points) >= n:
                    break
                x = x_offset + col * dx + r
                y = row * dy + r
                if 0 <= x <= 1 and 0 <= y <= 1:
                    points.append([x, y, r])
            row += 1
        return np.array(points)

    def make_random(n, seed=42):
        np.random.seed(seed)
        # Random positions, small initial radius
        x = np.random.uniform(0.1, 0.9, n)
        y = np.random.uniform(0.1, 0.9, n)
        r = np.full(n, 0.05)
        return np.column_stack((x, y, r))

    initializers = [
        make_grid(N),
        make_hex(N),
        make_random(N, 1),
        make_random(N, 2),
        make_random(N, 3),
        make_random(N, 4),
        make_random(N, 5)
    ]

    def objective_and_constraints(vars):
        # vars: [x1, y1, r1, x2, y2, r2, ...]
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        
        # Objective: Maximize sum of radii -> Minimize -sum(r)
        obj = -np.sum(r)
        
        # Constraints:
        # 1. Boundary: x >= r, x <= 1-r, y >= r, y <= 1-r
        # 2. Non-overlap: dist(i,j) >= ri + rj
        
        constraints = []
        
        # Boundary constraints (inequality: c(x) >= 0)
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})       # x_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i] - v[3*i+2]})   # 1 - x_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})     # y_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i+1] - v[3*i+2]}) # 1 - y_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+2] - 1e-6})         # r_i > 0

        # Non-overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                def overlap_constraint(v, i=i, j=j):
                    xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
                    xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
                    dist_sq = (xi - xj)**2 + (yi - yj)**2
                    sum_r = ri + rj
                    return dist_sq - sum_r**2
                constraints.append({'type': 'ineq', 'fun': overlap_constraint})
        
        return obj, constraints

    def run_optimizer(init_config):
        x0 = init_config.flatten()
        
        # Bounds for variables
        # x, y in [0, 1], r in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
            bounds.append((1e-6, 0.5)) # r
            
        try:
            res = minimize(
                lambda v: objective_and_constraints(v)[0],
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=lambda v: [c['fun'](v) for c in objective_and_constraints(v)[1]],
                options={'maxiter': 1000, 'ftol': 1e-8, 'disp': False}
            )
            if res.success:
                val = -res.fun
                # Verify validity
                x = res.x[0::3]
                y = res.x[1::3]
                r = res.x[2::3]
                
                # Check boundaries
                if np.any(x < r - 1e-5) or np.any(x > 1 - r + 1e-5) or \
                   np.any(y < r - 1e-5) or np.any(y > 1 - r + 1e-5) or \
                   np.any(r < 1e-6):
                    return None, -np.inf
                
                # Check overlaps
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        if dist < r[i] + r[j] - 1e-5:
                            return None, -np.inf
                
                return np.column_stack((x, y, r)), val
            else:
                return None, -np.inf
        except Exception as e:
            return None, -np.inf

    # Multi-start search
    for i, init in enumerate(initializers):
        if time.time() - start_time > DEADLINE:
            break
        
        result, val = run_optimizer(init)
        if result is not None and val > best_value:
            best_value = val
            best_result = result
            
        # If we have a very good result, we might stop early or continue
        # Let's do a few more random restarts if time permits
        if i == 0:
             # Add a few more random starts dynamically
             for k in range(5):
                 if time.time() - start_time > DEADLINE:
                     break
                 new_init = make_random(N, seed=10+k)
                 result, val = run_optimizer(new_init)
                 if result is not None and val > best_value:
                     best_value = val
                     best_result = result

    if best_result is None:
        # Fallback to seed if optimizer fails completely
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Sum of radii: {np.sum(res[:, 2])}")
