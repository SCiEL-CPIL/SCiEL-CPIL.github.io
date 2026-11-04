import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    time_limit = 120.0
    
    best_score = 0.0
    best_result = None

    def make_random_init(rng):
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        r = np.full(N, 0.05)
        return np.hstack([x, y, r])

    def make_grid_init():
        x, y = np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5))
        x = x.flatten()
        y = y.flatten()
        x = np.append(x, 0.5)
        y = np.append(y, 0.5)
        r = np.full(N, 0.08)
        return np.hstack([x, y, r])

    inits = [make_grid_init()]
    rng = np.random.default_rng(42)
    for _ in range(10):
        inits.append(make_random_init(rng))

    def penalty_objective(z, lambda_val):
        z = z.reshape(N, 3)
        x = z[:, 0]
        y = z[:, 1]
        r = z[:, 2]
        
        sum_r = np.sum(r)
        
        # Boundary penalties
        # r_i > x_i, 1-x_i, y_i, 1-y_i
        # overlap = max(0, r_i - dist_to_boundary)
        dist_b = np.min(np.stack([x, 1-x, y, 1-y], axis=1), axis=1)
        b_overlap = np.maximum(0, r - dist_b)
        b_pen = np.sum(b_overlap**2)
        
        # Circle-circle penalties
        c_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                overlap = np.maximum(0, r[i] + r[j] - d)
                c_pen += overlap**2
        
        return -sum_r + lambda_val * (b_pen + c_pen)

    def run_optimization(z0):
        # Increase lambda over iterations
        # We can do this by calling minimize multiple times with increasing lambda
        # Or just use a large lambda from the start?
        # Large lambda makes the landscape very steep.
        # Let's use a moderate lambda and refine.
        
        current_z = z0
        for lam in [10, 100, 1000, 10000]:
            res = minimize(
                penalty_objective, 
                current_z, 
                args=(lam,),
                method='L-BFGS-B', 
                bounds=[(0, 1)] * (2*N) + [(0, 0.5)] * N,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            current_z = res.x
            if time.time() - start_time > time_limit - 10:
                break
        
        # Check validity and score
        z_final = current_z.reshape(N, 3)
        x, y, r = z_final[:,0], z_final[:,1], z_final[:,2]
        
        # Verify constraints
        valid = True
        if np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1) or np.any(r < 0):
            valid = False
        for i in range(N):
            if x[i] - r[i] < -1e-9 or 1 - x[i] - r[i] < -1e-9 or y[i] - r[i] < -1e-9 or 1 - y[i] - r[i] < -1e-9:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    if d < r[i] + r[j] - 1e-9:
                        valid = False
                        break
                if not valid: break
        
        if valid:
            score = np.sum(r)
            return score, z_final
        else:
            return 0, None

    for init in inits:
        if time.time() - start_time > time_limit - 10:
            break
        
        score, result = run_optimization(init)
        if score > best_score:
            best_score = score
            best_result = result

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_result
