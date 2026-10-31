import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    def is_valid(x, y, r, tol=1e-7):
        if np.any(r < tol):
            return False
        if np.any(x < r - tol) or np.any(x > 1 - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < r[i] + r[j] - tol:
                    return False
        return True

    def score(x, y, r):
        return np.sum(r)

    def get_initial_guess(method='grid'):
        if method == 'grid':
            side = int(np.ceil(np.sqrt(N)))
            xs = np.linspace(0.1, 0.9, side)
            ys = np.linspace(0.1, 0.9, side)
            x, y = np.meshgrid(xs, ys)
            x = x.flatten()
            y = y.flatten()
            # Take first N
            x = x[:N]
            y = y[:N]
            r = np.full(N, 0.15) # Start with reasonable radius
            return x, y, r
        elif method == 'hex':
            # Simple hex-like pattern
            x = []
            y = []
            r = []
            # 5 rows
            for row in range(5):
                for col in range(6):
                    if len(x) >= N:
                        break
                    xx = 0.1 + col * 0.15 + (0.075 if row % 2 == 1 else 0)
                    yy = 0.1 + row * 0.18
                    if xx > 0.9:
                        continue
                    x.append(xx)
                    y.append(yy)
                    r.append(0.08)
            # Pad if necessary
            while len(x) < N:
                x.append(0.5)
                y.append(0.5)
                r.append(0.05)
            return np.array(x[:N]), np.array(y[:N]), np.array(r[:N])
        else: # random
            rng = np.random.RandomState(42 + hash(method) % 100)
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)
            r = rng.uniform(0.05, 0.1, N)
            return x, y, r

    best_result = None
    best_score = -np.inf

    def grow_and_push(x0, y0, r0, max_iter=1000, time_limit=20.0):
        """
        Hard-constraint grow-and-push: 
        1. Try to grow a random circle's radius
        2. If it violates constraints (boundary or overlap), push other circles away
        3. Repeat until no more growth is possible or time limit reached
        """
        x = x0.copy().astype(float)
        y = y0.copy().astype(float)
        r = r0.copy().astype(float)
        
        t_start = time.time()
        rng = np.random.RandomState(42)
        
        for iteration in range(max_iter):
            if time.time() - t_start > time_limit:
                break
            
            # Pick a random circle to grow
            i = rng.randint(N)
            
            # Try to grow circle i by a small amount
            grow_amount = 0.001
            r[i] += grow_amount
            
            # Check if still valid; if not, push others away
            needs_push = False
            for j in range(N):
                if i == j:
                    continue
                # Check overlap with i
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = r[i] + r[j]
                if d < min_d - 1e-9:
                    needs_push = True
                    # Push j away from i
                    if d < 1e-9:
                        # Random direction if coincident
                        angle = rng.uniform(0, 2*np.pi)
                        push_x = np.cos(angle) * (min_d - d + 1e-6)
                        push_y = np.sin(angle) * (min_d - d + 1e-6)
                    else:
                        push_x = dx / d * (min_d - d + 1e-6)
                        push_y = dy / d * (min_d - d + 1e-6)
                    
                    x[j] += push_x
                    y[j] += push_y
            
            # Check boundary violations for all circles and fix
            for j in range(N):
                # Fix x boundary
                if x[j] < r[j]:
                    x[j] = r[j]
                if x[j] > 1 - r[j]:
                    x[j] = 1 - r[j]
                # Fix y boundary
                if y[j] < r[j]:
                    y[j] = r[j]
                if y[j] > 1 - r[j]:
                    y[j] = 1 - r[j]
            
            # Final validity check; if invalid, shrink r[i] back
            valid = True
            for j in range(N):
                if x[j] < r[j] - 1e-8 or x[j] > 1 - r[j] + 1e-8:
                    valid = False
                    break
                if y[j] < r[j] - 1e-8 or y[j] > 1 - r[j] + 1e-8:
                    valid = False
                    break
            if valid:
                for j in range(N):
                    for k in range(j+1, N):
                        dx = x[j] - x[k]
                        dy = y[j] - y[k]
                        d = np.sqrt(dx*dx + dy*dy)
                        if d < r[j] + r[k] - 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
            
            if not valid:
                # Revert growth
                r[i] -= grow_amount
        
        return np.concatenate([x, y, r])

    def local_refinement(x, y, r, time_limit=10.0):
        """
        Local refinement using SLSQP to fine-tune positions and radii
        while maintaining feasibility.
        """
        params = np.concatenate([x, y, r])
        
        def objective(p):
            x, y, r = p[:N], p[N:2*N], p[2*N:3*N]
            return -np.sum(r)
            
        cons = []
        # Boundary constraints
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i] - p[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[N+i] - p[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[N+i] - p[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*N+i] - 1e-6})

        # Overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                cons.append({'type': 'ineq', 'fun': lambda p, i=i, j=j: 
                             (p[i]-p[j])**2 + (p[N+i]-p[N+j])**2 - (p[2*N+i]+p[2*N+j])**2})
                             
        bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N

        t_start = time.time()
        try:
            res = minimize(
                objective, 
                params, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            if time.time() - t_start > time_limit:
                return None
            return res.x
        except Exception:
            return None

    # Strategies: grow-and-push with different initializations
    init_methods = ['grid', 'hex', 'random']
    
    for init_type in init_methods:
        if time.time() > deadline:
            break
            
        x0, y0, r0 = get_initial_guess(init_type)
        
        # Grow and push
        p = grow_and_push(x0, y0, r0, max_iter=500, time_limit=25.0)
        
        if p is None:
            continue
            
        x, y, r = p[:N], p[N:2*N], p[2*N:3*N]
        
        # Local refinement
        if time.time() < deadline - 5:
            p_refined = local_refinement(x, y, r, time_limit=10.0)
            if p_refined is not None:
                x, y, r = p_refined[:N], p_refined[N:2*N], p_refined[2*N:3*N]
        
        if is_valid(x, y, r):
            s = score(x, y, r)
            if s > best_score:
                best_score = s
                best_result = np.column_stack((x, y, r))

    if best_result is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        best_result = np.array(cells[:N])
        
    return best_result
