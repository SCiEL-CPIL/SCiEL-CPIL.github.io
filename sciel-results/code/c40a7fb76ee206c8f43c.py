# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import time

def squares_in_square10() -> tuple:
    n = 10
    deadline = time.time() + 590 # Safety margin
    
    # Variables: x_0..x_9, y_0..y_9, s_0..s_9
    def solve_lp(sep_constraints):
        c = np.zeros(3*n)
        for i in range(n):
            c[2*n + i] = -1.0
            
        A_ub = []
        b_ub = []
        
        # Boundaries
        for i in range(n):
            # x_i - s_i/2 >= 0
            row = np.zeros(3*n); row[i] = -1.0; row[2*n+i] = 0.5
            A_ub.append(row); b_ub.append(0.0)
            # x_i + s_i/2 <= 1
            row = np.zeros(3*n); row[i] = 1.0; row[2*n+i] = 0.5
            A_ub.append(row); b_ub.append(1.0)
            # y_i - s_i/2 >= 0
            row = np.zeros(3*n); row[n+i] = -1.0; row[2*n+i] = 0.5
            A_ub.append(row); b_ub.append(0.0)
            # y_i + s_i/2 <= 1
            row = np.zeros(3*n); row[n+i] = 1.0; row[2*n+i] = 0.5
            A_ub.append(row); b_ub.append(1.0)
            
        # Separations
        for i, j, typ in sep_constraints:
            row = np.zeros(3*n)
            if typ == 0: # i left
                row[i] = 1.0; row[2*n+i] = 0.5; row[j] = -1.0; row[2*n+j] = 0.5
            elif typ == 1: # i right
                row[i] = -1.0; row[2*n+i] = 0.5; row[j] = 1.0; row[2*n+j] = -0.5
            elif typ == 2: # i below
                row[n+i] = 1.0; row[2*n+i] = 0.5; row[n+j] = -1.0; row[2*n+j] = 0.5
            elif typ == 3: # i above
                row[n+i] = -1.0; row[2*n+i] = 0.5; row[n+j] = 1.0; row[2*n+j] = -0.5
            A_ub.append(row)
            b_ub.append(0.0)
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        bounds = [(-1.0, 2.0)] * (2*n) + [(1e-6, 1.0)] * n
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                x = res.x[:n]
                y = res.x[n:2*n]
                s = res.x[2*n:3*n]
                # Validation
                tol = 1e-7
                if np.any(s <= 0) or np.any(~np.isfinite(s)):
                    return -np.inf, None, None, None
                if np.any(x - s/2 < -tol) or np.any(x + s/2 > 1 + tol):
                    return -np.inf, None, None, None
                if np.any(y - s/2 < -tol) or np.any(y + s/2 > 1 + tol):
                    return -np.inf, None, None, None
                for i in range(n):
                    for j in range(i+1, n):
                        t = (s[i] + s[j]) / 2 - tol
                        if abs(x[i] - x[j]) < t and abs(y[i] - y[j]) < t:
                            return -np.inf, None, None, None
                return -res.fun, x, y, s
        except Exception:
            pass
        return -np.inf, None, None, None

    def get_sep_type(x, y, s, i, j):
        left = x[i] + s[i]/2 <= x[j] - s[j]/2 + 1e-9
        right = x[j] + s[j]/2 <= x[i] - s[i]/2 + 1e-9
        below = y[i] + s[i]/2 <= y[j] - s[j]/2 + 1e-9
        above = y[j] + s[j]/2 <= y[i] - s[i]/2 + 1e-9
        if left: return 0
        if right: return 1
        if below: return 2
        if above: return 3
        return -1

    # Initialize Grid
    k = 3
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    x_init = (cols + 0.5) * side
    y_init = (rows + 0.5) * side
    s_init = np.full(n, side)
    
    init_constraints = []
    for i in range(n):
        for j in range(i+1, n):
            t = get_sep_type(x_init, y_init, s_init, i, j)
            if t == -1: t = 0
            init_constraints.append((i, j, t))
            
    score, x, y, s = solve_lp(init_constraints)
    best_score = score
    best_centers = np.column_stack([x, y])
    best_sides = s
    current_constraints = init_constraints.copy()
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    alpha = 0.99
    np.random.seed(42)
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        i, j = np.random.choice(n, 2, replace=False)
        new_type = np.random.randint(4)
        
        new_constraints = []
        for c in current_constraints:
            if c[0] == i and c[1] == j:
                new_constraints.append((i, j, new_type))
            else:
                new_constraints.append(c)
                
        new_score, nx, ny, ns = solve_lp(new_constraints)
        
        if new_score > best_score:
            best_score = new_score
            best_centers = np.column_stack([nx, ny])
            best_sides = ns
            current_constraints = new_constraints.copy()
            score = new_score
        elif new_score > -np.inf:
            delta = new_score - score
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                score = new_score
                current_constraints = new_constraints.copy()
                if x is not None: # Update current geometry for reference? 
                    # We don't strictly need current x,y for SA on topology, 
                    # but we use score.
                    pass
        
        T *= alpha
        
    return best_centers.astype(np.float64), best_sides.astype(np.float64)

# EVOLVE-BLOCK-END
