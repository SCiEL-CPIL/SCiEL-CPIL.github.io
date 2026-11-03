"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

def _check_validity(arr, tol=1e-7):
    """Check if circles are valid (inside unit square, non-overlapping)."""
    n = len(arr)
    # Check boundaries
    for i in range(n):
        x, y, r = arr[i]
        if x < -tol or y < -tol or x + 2*r > 1 + tol or y + 2*r > 1 + tol:
            return False
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = arr[i]
            xj, yj, rj = arr[j]
            dist = np.hypot(xi - xj, yi - yj)
            if dist < ri + rj - tol:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    # Initial guess: grid
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4])
    
    best_arr = np.array(cells[:N])
    best_score = np.sum(best_arr[:, 2])
    
    # Objective: maximize sum of radii
    # Constraints:
    # 1. x_i >= r_i, y_i >= r_i, x_i + r_i <= 1, y_i + r_i <= 1
    # 2. dist(i,j) >= r_i + r_j for all i != j
    
    def objective(params):
        # params: x1, y1, r1, x2, y2, r2, ...
        radii = params[2::3]
        return -np.sum(radii)
    
    # We will use a sequential optimization approach with dynamic contact graph
    # First, get a good initial layout
    
    # Use a simple iterative relaxation to get a better starting point
    arr = best_arr.copy()
    for iteration in range(1000):
        if time.time() > deadline - 60:
            break
        improved = False
        for i in range(N):
            # Try to expand circle i
            best_r = arr[i, 2]
            # Check constraints with other circles
            max_r = 0.5
            for j in range(N):
                if i == j:
                    continue
                # dist(i,j) >= r_i + r_j
                # arr[i,0] - arr[j,0] squared + ... >= (r_i + r_j)^2
                # This is complex, so we just check current feasibility and try small steps
                pass
            
            # Simple gradient-free step: try to increase r_i slightly
            for dr in [0.001, 0.0005, 0.0001]:
                new_r = arr[i, 2] + dr
                if new_r > 0.5:
                    continue
                # Check boundary
                if arr[i, 0] < new_r or arr[i, 1] < new_r or arr[i, 0] + new_r > 1 or arr[i, 1] + new_r > 1:
                    continue
                # Check overlaps
                valid = True
                for j in range(N):
                    if i == j:
                        continue
                    dist = np.hypot(arr[i, 0] - arr[j, 0], arr[i, 1] - arr[j, 1])
                    if dist < new_r + arr[j, 2] - 1e-9:
                        valid = False
                        break
                if valid:
                    arr[i, 2] = new_r
                    improved = True
                    break
        if not improved:
            break
    
    if _check_validity(arr):
        best_arr = arr.copy()
        best_score = np.sum(arr[:, 2])
    
    # Now use scipy minimize with SLSQP for local optimization
    # We'll optimize positions and radii jointly
    
    def make_objective_and_constraints(arr):
        n = len(arr)
        
        def obj(x):
            # x: [x1, y1, r1, x2, y2, r2, ...]
            radii = x[2::3]
            return -np.sum(radii)
        
        def obj_grad(x):
            grad = np.zeros_like(x)
            grad[2::3] = -1.0
            return grad
        
        # Inequality constraints: g(x) >= 0
        # Boundary: x_i - r_i >= 0
        #            y_i - r_i >= 0
        #            1 - x_i - r_i >= 0
        #            1 - y_i - r_i >= 0
        # Overlap: dist(i,j)^2 - (r_i + r_j)^2 >= 0
        
        def boundary_constraints(x):
            con = []
            for i in range(n):
                xi = x[3*i]
                yi = x[3*i+1]
                ri = x[3*i+2]
                con.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2], 'jac': lambda x, i=i: np.array([1, 0, -1] if i==0 else [0,0,0])})
                # This lambda approach is tricky, let's use a vectorized constraint function instead
            return con
        
        # Better approach: define constraint functions that return arrays
        def boundary_con(x):
            # Returns array of boundary constraint values (all should be >= 0)
            n = len(x) // 3
            vals = np.zeros(4*n)
            for i in range(n):
                vals[4*i] = x[3*i] - x[3*i+2]       # x_i - r_i >= 0
                vals[4*i+1] = x[3*i+1] - x[3*i+2]   # y_i - r_i >= 0
                vals[4*i+2] = 1 - x[3*i] - x[3*i+2] # 1 - x_i - r_i >= 0
                vals[4*i+3] = 1 - x[3*i+1] - x[3*i+2] # 1 - y_i - r_i >= 0
            return vals
        
        def boundary_jac(x):
            # Jacobian of boundary constraints
            n = len(x) // 3
            jac = np.zeros((4*n, 3*n))
            for i in range(n):
                jac[4*i, 3*i] = 1
                jac[4*i, 3*i+2] = -1
                jac[4*i+1, 3*i+1] = 1
                jac[4*i+1, 3*i+2] = -1
                jac[4*i+2, 3*i] = -1
                jac[4*i+2, 3*i+2] = -1
                jac[4*i+3, 3*i+1] = -1
                jac[4*i+3, 3*i+2] = -1
            return jac
        
        def overlap_con(x):
            # Returns array of overlap constraint values (all should be >= 0)
            n = len(x) // 3
            m = n * (n-1) // 2
            vals = np.zeros(m)
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    xi = x[3*i]
                    yi = x[3*i+1]
                    ri = x[3*i+2]
                    xj = x[3*j]
                    yj = x[3*j+1]
                    rj = x[3*j+2]
                    dist2 = (xi - xj)**2 + (yi - yj)**2
                    vals[idx] = dist2 - (ri + rj)**2
                    idx += 1
            return vals
        
        def overlap_jac(x):
            # Jacobian of overlap constraints
            n = len(x) // 3
            m = n * (n-1) // 2
            jac = np.zeros((m, 3*n))
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    xi = x[3*i]
                    yi = x[3*i+1]
                    ri = x[3*i+2]
                    xj = x[3*j]
                    yj = x[3*j+1]
                    rj = x[3*j+2]
                    dx = xi - xj
                    dy = yi - yj
                    dist2 = dx*dx + dy*dy
                    # Avoid division by zero
                    if dist2 < 1e-12:
                        dist = 1e-6
                        dxn = 1.0
                        dyn = 0.0
                    else:
                        dist = np.sqrt(dist2)
                        dxn = dx / dist
                        dyn = dy / dist
                    
                    sum_r = ri + rj
                    
                    jac[idx, 3*i] = 2 * dxn
                    jac[idx, 3*i+1] = 2 * dyn
                    jac[idx, 3*i+2] = -2 * sum_r
                    jac[idx, 3*j] = -2 * dxn
                    jac[idx, 3*j+1] = -2 * dyn
                    jac[idx, 3*j+2] = -2 * sum_r
                    idx += 1
            return jac
        
        return obj, obj_grad, boundary_con, boundary_jac, overlap_con, overlap_jac
    
    # Try optimization from the current best point
    if time.time() < deadline - 30:
        x0 = best_arr.flatten()
        try:
            obj, obj_grad, boundary_con, boundary_jac, overlap_con, overlap_jac = make_objective_and_constraints(best_arr)
            
            # Combine constraints
            def all_con(x):
                b = boundary_con(x)
                o = overlap_con(x)
                return np.concatenate([b, o])
            
            def all_jac(x):
                jb = boundary_jac(x)
                jo = overlap_jac(x)
                return np.vstack([jb, jo])
            
            res = minimize(obj, x0, method='SLSQP', jac=obj_grad,
                          constraints=[{'type': 'ineq', 'fun': all_con, 'jac': all_jac}],
                          bounds=[(0, 1)] * (2*N) + [(1e-6, 0.5)] * N,
                          options={'maxiter': 500, 'ftol': 1e-10})
            
            if res.success or _check_validity(res.x.reshape(-1, 3)):
                new_arr = res.x.reshape(-1, 3)
                if _check_validity(new_arr):
                    new_score = np.sum(new_arr[:, 2])
                    if new_score > best_score:
                        best_arr = new_arr
                        best_score = new_score
        except Exception:
            pass
    
    # Try multiple random restarts if time permits
    np.random.seed(42)
    for restart in range(10):
        if time.time() > deadline - 20:
            break
        
        # Random initial positions
        arr = np.zeros((N, 3))
        arr[:, 0] = np.random.uniform(0.1, 0.9, N)
        arr[:, 1] = np.random.uniform(0.1, 0.9, N)
        arr[:, 2] = 0.02
        
        # Quick relaxation
        for _ in range(200):
            if time.time() > deadline - 20:
                break
            for i in range(N):
                for dr in [0.002, 0.001, 0.0005]:
                    new_r = arr[i, 2] + dr
                    if new_r > 0.5:
                        continue
                    if arr[i, 0] < new_r or arr[i, 1] < new_r or arr[i, 0] + new_r > 1 or arr[i, 1] + new_r > 1:
                        continue
                    valid = True
                    for j in range(N):
                        if i == j:
                            continue
                        dist = np.hypot(arr[i, 0] - arr[j, 0], arr[i, 1] - arr[j, 1])
                        if dist < new_r + arr[j, 2] - 1e-9:
                            valid = False
                            break
                    if valid:
                        arr[i, 2] = new_r
                        break
        
        if not _check_validity(arr):
            continue
        
        score = np.sum(arr[:, 2])
        if score > best_score:
            best_arr = arr.copy()
            best_score = score
        
        # Optimize from this point
        if time.time() < deadline - 15:
            x0 = best_arr.flatten()
            try:
                obj, obj_grad, boundary_con, boundary_jac, overlap_con, overlap_jac = make_objective_and_constraints(best_arr)
                
                def all_con(x):
                    b = boundary_con(x)
                    o = overlap_con(x)
                    return np.concatenate([b, o])
                
                def all_jac(x):
                    jb = boundary_jac(x)
                    jo = overlap_jac(x)
                    return np.vstack([jb, jo])
                
                res = minimize(obj, x0, method='SLSQP', jac=obj_grad,
                              constraints=[{'type': 'ineq', 'fun': all_con, 'jac': all_jac}],
                              bounds=[(0, 1)] * (2*N) + [(1e-6, 0.5)] * N,
                              options={'maxiter': 300, 'ftol': 1e-10})
                
                if res.success or _check_validity(res.x.reshape(-1, 3)):
                    new_arr = res.x.reshape(-1, 3)
                    if _check_validity(new_arr):
                        new_score = np.sum(new_arr[:, 2])
                        if new_score > best_score:
                            best_arr = new_arr
                            best_score = new_score
            except Exception:
                pass
    
    # Final validation
    if not _check_validity(best_arr):
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4])
        best_arr = np.array(cells[:N])
    
    return best_arr

# EVOLVE-BLOCK-END
