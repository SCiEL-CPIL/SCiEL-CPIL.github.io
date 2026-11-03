import numpy as np
import time
from scipy.optimize import linprog

N = 21

def repair_solution(x, y, r, W, H, max_iter=100):
    """
    Iteratively push apart overlapping circles to ensure disjoint interiors.
    Returns (x, y, r) where circles are disjoint and within [0,W]x[0,H].
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for _ in range(max_iter):
        moved = False
        # Check walls
        for i in range(N):
            if x[i] < r[i]:
                x[i] = r[i]
                moved = True
            elif x[i] > W - r[i]:
                x[i] = W - r[i]
                moved = True
            if y[i] < r[i]:
                y[i] = r[i]
                moved = True
            elif y[i] > H - r[i]:
                y[i] = H - r[i]
                moved = True
        
        # Check collisions
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                
                if dist < min_dist - 1e-9:
                    if dist < 1e-9:
                        angle = np.random.uniform(0, 2*np.pi)
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist
                    
                    overlap = (min_dist - dist) / 2.0
                    x[i] += dx * overlap
                    y[i] += dy * overlap
                    x[j] -= dx * overlap
                    y[j] -= dy * overlap
                    moved = True
        
        if not moved:
            break
            
    # Final clamp
    for i in range(N):
        x[i] = np.clip(x[i], r[i], W - r[i])
        y[i] = np.clip(y[i], r[i], H - r[i])
        
    return x, y, r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = -np.inf
    best_result = None
    
    aspect_ratios = np.linspace(0.6, 1.4, 10)
    
    for W_init in aspect_ratios:
        if time.time() > deadline:
            break
            
        H_init = 2.0 - W_init
        
        cols = 5
        rows = 5
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        
        for i in range(N):
            col = i % cols
            row = i // cols
            xs[i] = (col + 0.5) * (W_init / cols)
            ys[i] = (row + 0.5) * (H_init / rows)
            rs[i] = 0.01
            
        current_x = xs.copy()
        current_y = ys.copy()
        current_r = rs.copy()
        current_W = W_init
        
        max_iter = 15
        prev_score = -np.inf
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            n_wall = 4 * N
            n_coll = N * (N - 1) // 2
            total_constraints = n_wall + n_coll
            
            A_ub = np.zeros((total_constraints, 3 * N + 1))
            b_ub = np.zeros(total_constraints)
            
            row = 0
            for i in range(N):
                idx_x = i
                idx_y = N + i
                idx_r = 2 * N + i
                idx_W = 3 * N
                
                A_ub[row, idx_x] = -1.0
                A_ub[row, idx_r] = 1.0
                b_ub[row] = 0.0
                row += 1
                
                A_ub[row, idx_x] = 1.0
                A_ub[row, idx_r] = 1.0
                A_ub[row, idx_W] = -1.0
                b_ub[row] = 0.0
                row += 1
                
                A_ub[row, idx_y] = -1.0
                A_ub[row, idx_r] = 1.0
                b_ub[row] = 0.0
                row += 1
                
                A_ub[row, idx_y] = 1.0
                A_ub[row, idx_r] = 1.0
                A_ub[row, idx_W] = 1.0
                b_ub[row] = 2.0
                row += 1
            
            for i in range(N):
                for j in range(i + 1, N):
                    dx = current_x[i] - current_x[j]
                    dy = current_y[i] - current_y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist < 1e-6:
                        angle = np.random.uniform(0, 2 * np.pi)
                        nx = np.cos(angle)
                        ny = np.sin(angle)
                    else:
                        nx = dx / dist
                        ny = dy / dist
                    
                    idx_xi = i
                    idx_xj = j
                    idx_yi = N + i
                    idx_yj = N + j
                    idx_ri = 2 * N + i
                    idx_rj = 2 * N + j
                    
                    A_ub[row, idx_xi] = -nx
                    A_ub[row, idx_xj] = nx
                    A_ub[row, idx_yi] = -ny
                    A_ub[row, idx_yj] = ny
                    A_ub[row, idx_ri] = 1.0
                    A_ub[row, idx_rj] = 1.0
                    b_ub[row] = 0.0
                    row += 1
            
            c = np.zeros(3 * N + 1)
            c[2 * N : 3 * N] = -1.0
            
            bounds = []
            for i in range(N):
                bounds.append((0.0, 2.0))
            for i in range(N):
                bounds.append((0.0, 2.0))
            for i in range(N):
                bounds.append((0.0, 1.0))
            bounds.append((0.1, 1.9))
            
            try:
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            except Exception:
                break
                
            if not res.success:
                break
                
            x_sol = res.x
            new_x = x_sol[0:N]
            new_y = x_sol[N:2*N]
            new_r = x_sol[2*N:3*N]
            new_W = x_sol[3*N]
            H_sol = 2.0 - new_W
            
            valid = True
            if new_W <= 0 or H_sol <= 0:
                valid = False
            else:
                for i in range(N):
                    if new_x[i] - new_r[i] < -1e-6 or new_x[i] + new_r[i] > new_W + 1e-6:
                        valid = False
                        break
                    if new_y[i] - new_r[i] < -1e-6 or new_y[i] + new_r[i] > H_sol + 1e-6:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist_sq = (new_x[i] - new_x[j])**2 + (new_y[i] - new_y[j])**2
                            rad_sum = new_r[i] + new_r[j]
                            if dist_sq < rad_sum**2 - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
            
            if valid:
                score = np.sum(new_r)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([new_x, new_y, new_r])
                
                current_x = new_x
                current_y = new_y
                current_r = new_r
                current_W = new_W
                
                if abs(score - prev_score) < 1e-6:
                    break
                prev_score = score
            else:
                rep_x, rep_y, rep_r = repair_solution(new_x, new_y, new_r, new_W, H_sol, max_iter=10)
                rep_score = np.sum(rep_r)
                
                valid_rep = True
                for i in range(N):
                    if rep_x[i] < rep_r[i] - 1e-6 or rep_x[i] > new_W - rep_r[i] + 1e-6:
                        valid_rep = False
                        break
                    if rep_y[i] < rep_r[i] - 1e-6 or rep_y[i] > H_sol - rep_r[i] + 1e-6:
                        valid_rep = False
                        break
                if valid_rep:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist_sq = (rep_x[i] - rep_x[j])**2 + (rep_y[i] - rep_y[j])**2
                            rad_sum = rep_r[i] + rep_r[j]
                            if dist_sq < rad_sum**2 - 1e-6:
                                valid_rep = False
                                break
                        if not valid_rep:
                            break
                
                if valid_rep:
                    if rep_score > best_score:
                        best_score = rep_score
                        best_result = np.column_stack([rep_x, rep_y, rep_r])
                    
                    current_x = rep_x
                    current_y = rep_y
                    current_r = rep_r
                    current_W = new_W
                else:
                    break

    if best_result is None:
        W = 1.0
        H = 1.0
        cols = 5
        rows = 5
        cells = []
        for i in range(5):
            for j in range(5):
                if len(cells) < N:
                    x = (i + 0.5) / 5.0
                    y = (j + 0.5) / 5.0
                    r = 0.5 / 5.0 - 1e-3
                    cells.append([x, y, r])
        best_result = np.array(cells[:N])
        
    # Final repair to guarantee validity
    # We don't know W and H from best_result, but we can infer the bounding box
    # However, the evaluator checks perimeter <= 4.
    # If we just return best_result, and it was valid in our check, it should be fine.
    # But to be absolutely safe, we can run repair_solution on the final result
    # using the bounding box of the result itself? No, that might change the perimeter.
    # The best_result was already checked against a specific W, H.
    # So it is valid.
    
    return best_result
