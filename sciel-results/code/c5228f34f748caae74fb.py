"""Rewritten program for circle packing in unit square."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_score = -np.inf
    best_result = None
    
    # Try multiple random initializations and local optimizations
    np.random.seed(42)
    
    # Number of restarts to try
    max_restarts = 50
    restarts_done = 0
    
    # Initial guess: distribute points in a grid-like pattern with some randomness
    def _initial_guess():
        # Use a stratified sampling approach
        n_per_side = int(np.ceil(np.sqrt(N)))
        positions = []
        for i in range(n_per_side):
            for j in range(n_per_side):
                if len(positions) >= N:
                    break
                x = (i + 0.5) / n_per_side + np.random.uniform(-0.1, 0.1)
                y = (j + 0.5) / n_per_side + np.random.uniform(-0.1, 0.1)
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                positions.append([x, y])
        
        # Fill remaining with random positions
        while len(positions) < N:
            positions.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
        
        # Initial radii: small
        radii = np.full(N, 0.05)
        return np.array(positions), radii
    
    def _objective_and_constraints(params):
        """
        params: 3N array [x1, y1, r1, x2, y2, r2, ...]
        We want to maximize sum(r), so minimize -sum(r)
        Constraints:
        - Circles must be inside [0,1]^2: x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
        - Circles must not overlap: dist(i,j) >= r_i + r_j
        """
        n = len(params) // 3
        xs = params[0::3]
        ys = params[1::3]
        rs = params[2::3]
        
        # Objective: maximize sum of radii
        obj = -np.sum(rs)
        
        # Constraints (all should be >= 0)
        constraints = []
        
        # Boundary constraints
        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})  # x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i] - p[3*i+2]})  # 1 - x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})  # y - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i+1] - p[3*i+2]})  # 1 - y - r >= 0
        
        # Overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda p, i=i, j=j: np.sqrt((p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2) - p[3*i+2] - p[3*j+2]
                })
        
        return obj, constraints
    
    def _optimize_from_initial(pos, rad):
        """Run SLSQP optimization from given initial positions and radii."""
        n = len(pos)
        params0 = np.concatenate([pos.flatten(), rad])
        
        # Bounds: x, y in [0, 1], r in [0, 0.5]
        bounds = []
        for i in range(n):
            bounds.append((0, 1))  # x
            bounds.append((0, 1))  # y
            bounds.append((0, 0.5))  # r
        
        # Use a simplified approach with penalty method for better performance
        def _penalized_objective(params):
            xs = params[0::3]
            ys = params[1::3]
            rs = params[2::3]
            
            obj = -np.sum(rs)
            
            # Penalty for boundary violations
            penalty = 0.0
            for i in range(n):
                if xs[i] - rs[i] < 0:
                    penalty += (xs[i] - rs[i])**2
                if 1 - xs[i] - rs[i] < 0:
                    penalty += (1 - xs[i] - rs[i])**2
                if ys[i] - rs[i] < 0:
                    penalty += (ys[i] - rs[i])**2
                if 1 - ys[i] - rs[i] < 0:
                    penalty += (1 - ys[i] - rs[i])**2
            
            # Penalty for overlaps
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    overlap = rs[i] + rs[j] - dist
                    if overlap > 0:
                        penalty += overlap**2
            
            return obj + 1000.0 * penalty
        
        # Use Nelder-Mead first for robustness, then SLSQP for polish
        result1 = minimize(_penalized_objective, params0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
        
        params1 = result1.x
        
        # Polish with SLSQP
        def _slsqp_objective(params):
            xs = params[0::3]
            ys = params[1::3]
            rs = params[2::3]
            return -np.sum(rs)
        
        def _slsqp_constraints(params):
            xs = params[0::3]
            ys = params[1::3]
            rs = params[2::3]
            
            cons = []
            for i in range(n):
                cons.append(xs[i] - rs[i])
                cons.append(1 - xs[i] - rs[i])
                cons.append(ys[i] - rs[i])
                cons.append(1 - ys[i] - rs[i])
            
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    cons.append(dist - rs[i] - rs[j])
            
            return cons
        
        from scipy.optimize import NonlinearConstraint
        # This is getting complex; let's use a simpler iterative refinement
        
        # Iterative refinement: try to increase radii while maintaining feasibility
        for iteration in range(50):
            if time.time() > deadline - 10:
                break
            
            xs = params1[0::3].copy()
            ys = params1[1::3].copy()
            rs = params1[2::3].copy()
            
            improved = False
            for i in range(n):
                # Try to increase radius of circle i
                for delta in [0.005, 0.002, 0.001, 0.0005]:
                    new_r = rs[i] + delta
                    if new_r > 0.5:
                        continue
                    
                    # Check boundary
                    if xs[i] - new_r < -1e-9 or 1 - xs[i] - new_r < -1e-9:
                        continue
                    if ys[i] - new_r < -1e-9 or 1 - ys[i] - new_r < -1e-9:
                        continue
                    
                    # Check overlaps
                    feasible = True
                    for j in range(n):
                        if i == j:
                            continue
                        dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                        if dist < new_r + rs[j] - 1e-9:
                            feasible = False
                            break
                    
                    if feasible:
                        rs[i] = new_r
                        improved = True
                        break
            
            if not improved:
                break
            
            params1 = np.concatenate([xs, ys, rs])
        
        return params1
    
    # Main loop: multiple restarts
    for restart in range(max_restarts):
        if time.time() > deadline - 20:
            break
        
        pos, rad = _initial_guess()
        
        # Add some randomness to positions
        pos = pos + np.random.uniform(-0.05, 0.05, pos.shape)
        pos = np.clip(pos, 0.05, 0.95)
        
        try:
            params = _optimize_from_initial(pos, rad)
            
            # Extract result
            xs = params[0::3]
            ys = params[1::3]
            rs = params[2::3]
            
            # Verify feasibility
            feasible = True
            for i in range(N):
                if xs[i] - rs[i] < -1e-6 or 1 - xs[i] - rs[i] < -1e-6:
                    feasible = False
                    break
                if ys[i] - rs[i] < -1e-6 or 1 - ys[i] - rs[i] < -1e-6:
                    feasible = False
                    break
            
            if feasible:
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                        if dist < rs[i] + rs[j] - 1e-6:
                            feasible = False
                            break
                    if not feasible:
                        break
            
            if feasible:
                score = np.sum(rs)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([xs, ys, rs])
            
            restarts_done += 1
            
        except Exception as e:
            continue
    
    # If no feasible solution found, return a trivial valid packing
    if best_result is None:
        # Grid of small circles
        side = 6
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result

def construct():
    """Entry point that calls the internal construct function."""
    return _construct()
# EVOLVE-BLOCK-END
