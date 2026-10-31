import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    best_score = 0.0
    best_result = None

    def evaluate_solution(sol):
        if len(sol) != N * 3 + 2:
            return -np.inf, None
        circles = sol[:N*3].reshape(N, 3)
        W = sol[N*3]
        H = sol[N*3+1]
        
        # Check perimeter
        if 2 * (W + H) > 4.0 + 1e-9:
            return -np.inf, None
            
        # Check boundaries and overlaps
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        
        for i in range(N):
            if xs[i] < rs[i] - 1e-9 or xs[i] > W - rs[i] + 1e-9:
                return -np.inf, None
            if ys[i] < rs[i] - 1e-9 or ys[i] > H - rs[i] + 1e-9:
                return -np.inf, None
            if rs[i] < 1e-9:
                return -np.inf, None
                
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (xs[i] - xs[j])**2 + (ys[i] - ys[j])**2
                rad_sum = rs[i] + rs[j]
                if dist_sq < rad_sum**2 - 1e-9:
                    return -np.inf, None
                    
        score = np.sum(rs)
        return score, np.column_stack([xs, ys, rs])

    def cost_function(v):
        circles = v[:N*3].reshape(N, 3)
        W = v[N*3]
        H = v[N*3+1]
        
        if W < 0 or H < 0 or np.any(circles[:, 2] < 0):
            return 1e10
            
        perim_violation = max(0, 2 * (W + H) - 4.0)
        cost = -np.sum(circles[:, 2]) + 100.0 * perim_violation**2
        
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        
        bound_pen = 0
        for i in range(N):
            if xs[i] < rs[i]:
                bound_pen += (rs[i] - xs[i])**2
            if xs[i] > W - rs[i]:
                bound_pen += (xs[i] - (W - rs[i]))**2
            if ys[i] < rs[i]:
                bound_pen += (rs[i] - ys[i])**2
            if ys[i] > H - rs[i]:
                bound_pen += (ys[i] - (H - rs[i]))**2
        cost += 10.0 * bound_pen
        
        overlap_pen = 0
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                rad_sum = rs[i] + rs[j]
                if dist < rad_sum:
                    overlap_pen += (rad_sum - dist)**2
        cost += 10.0 * overlap_pen
        
        return cost

    def run_optimization(initial_guess, maxiter=300, maxfev=5000):
        options = {'maxiter': maxiter, 'maxfev': maxfev}
        res = minimize(cost_function, initial_guess, method='Powell', options=options)
        return res.x

    # 1. Grid initialization
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.4 / side])
                
    while len(cells) < N:
        cells.append([0.5, 0.5, 0.1])
        
    init_grid = np.array(cells[:N]).flatten()
    init_grid = np.append(init_grid, [1.0, 1.0])
    
    if time.time() < deadline:
        v_opt = run_optimization(init_grid)
        score, result = evaluate_solution(v_opt)
        if score > best_score and result is not None:
            best_score = score
            best_result = result

    # 2. Random restarts
    def random_init():
        W = 1.0
        H = 1.0
        xs = np.random.uniform(0.1, 0.9, N)
        ys = np.random.uniform(0.1, 0.9, N)
        rs = np.random.uniform(0.05, 0.15, N)
        v = np.zeros(N*3+2)
        v[0::3] = xs
        v[1::3] = ys
        v[2::3] = rs
        v[-2] = W
        v[-1] = H
        return v

    restarts = 10
    for i in range(restarts):
        if time.time() > deadline - 10:
            break
            
        np.random.seed(i) # Deterministic restarts for reproducibility in testing, or use global seed
        v_init = random_init()
        v_opt = run_optimization(v_init, maxiter=200, maxfev=3000)
        score, result = evaluate_solution(v_opt)
        if score > best_score and result is not None:
            best_score = score
            best_result = result
            
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
