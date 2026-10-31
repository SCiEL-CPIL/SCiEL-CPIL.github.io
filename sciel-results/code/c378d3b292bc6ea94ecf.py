import numpy as np
from scipy.optimize import linprog
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_val = -np.inf
    best_sol = None
    
    # Initial guess: Grid
    pos = generate_initial_grid()
    
    # Local search loop
    step_size = 0.05
    for i in range(500):
        if time.time() > deadline:
            break
            
        # Jitter positions
        # Use a cooling schedule or random restarts
        if i % 50 == 0:
            step_size = 0.05 * (0.95 ** (i // 50))
            
        # Try random moves
        move = np.random.randn(N, 2) * step_size
        new_pos = pos + move
        
        # Keep inside [0,1]^2 with margin
        new_pos = np.clip(new_pos, 0.01, 0.99)
        
        # Solve for max radii for these positions
        radii = solve_radii_lp(new_pos)
        
        if radii is None:
            continue
            
        val = np.sum(radii)
        
        if val > best_val:
            best_val = val
            best_sol = np.column_stack((new_pos, radii))
            pos = new_pos # Accept move (greedy)
        else:
            # Optionally accept worse moves (simulated annealing), but greedy is safer for time
            pass
            
    # Final polish: try to increase radii slightly or adjust positions near boundaries
    if best_sol is not None:
        # One last check and strict validation
        final_sol = validate_and_return(best_sol)
        # Re-evaluate value to be sure
        x, y, r = final_sol[:, 0], final_sol[:, 1], final_sol[:, 2]
        final_val = np.sum(r)
        if final_val > best_val:
            best_val = final_val
            best_sol = final_sol
            
    if best_sol is None:
        return trivial_solution()
        
    return best_sol

def generate_initial_grid():
    # 5x6 grid roughly
    rows, cols = 5, 6
    # Adjust to fit 26 circles
    # 5 rows, 6 cols = 30, take 26
    xs = []
    ys = []
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= N:
                break
            x = (c + 0.5) / cols
            y = (r + 0.5) / rows
            xs.append(x)
            ys.append(y)
            count += 1
    return np.column_stack((xs, ys))

def solve_radii_lp(pos):
    """
    Maximize sum(r_i)
    Subject to:
    r_i + r_j <= ||p_i - p_j|| for all i < j
    r_i <= min(x_i, 1-x_i, y_i, 1-y_i) for all i
    r_i >= 0
    """
    c = -np.ones(N)
    
    A_ub = []
    b_ub = []
    
    # Pairwise constraints
    for i in range(N):
        for j in range(i+1, N):
            dij = np.linalg.norm(pos[i] - pos[j])
            row = np.zeros(N)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(dij)
            
    # Boundary constraints
    for i in range(N):
        bi = np.min([pos[i][0], pos[i][1], 1-pos[i][0], 1-pos[i][1]])
        row = np.zeros(N)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(bi)
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(0, None) for _ in range(N)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            return None
    except Exception:
        return None

def trivial_solution():
    # Simple grid of 26 circles
    # 5 rows, 6 cols (30 spots), take first 26
    rows, cols = 5, 6
    sol = []
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= N:
                break
            x = (c + 0.5) / cols
            y = (r + 0.5) / rows
            # Max radius is half the min distance to neighbor or boundary
            # In a grid, distance to neighbor is 1/cols or 1/rows
            # Radius can be approx 0.5 * min(1/cols, 1/rows)
            r_max = 0.5 * min(1.0/cols, 1.0/rows) * 0.95
            sol.append([x, y, r_max])
            count += 1
    return np.array(sol)

def validate_and_return(sol):
    x = sol[:, 0].copy()
    y = sol[:, 1].copy()
    r = sol[:, 2].copy()
    
    # Clip positions to [0,1]
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)
    
    # Enforce boundary constraints strictly
    r_max_boundary = np.minimum(np.minimum(x, y), np.minimum(1-x, 1-y))
    r = np.minimum(r, r_max_boundary)
    r = np.maximum(r, 0)
    
    # Enforce pairwise constraints strictly
    # Iterate a few times to resolve conflicts
    for _ in range(10):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm([x[i]-x[j], y[i]-y[j]])
                sum_r = r[i] + r[j]
                if sum_r > dist:
                    # Scale down both proportionally to their current radii?
                    # Or just scale both by dist/sum_r
                    if dist < 1e-9:
                        # If they are at the same spot, shrink both to 0
                        r[i] = 0
                        r[j] = 0
                    else:
                        scale = dist / sum_r
                        r[i] *= scale
                        r[j] *= scale
                    changed = True
        if not changed:
            break
            
    return np.column_stack((x, y, r))
