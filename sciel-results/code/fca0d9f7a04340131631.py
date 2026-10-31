"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Use SLSQP with hard constraints for tight packing
    # Variables: x_1..x_N, y_1..y_N, r_1..r_N (3*N variables)
    
    def objective(v):
        # Maximize sum of radii = minimize negative sum
        radii = v[2*N:]
        return -np.sum(radii)
    
    def constraints_fn(v):
        """Return list of constraint dictionaries."""
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:]
        
        constraints = []
        
        # Boundary constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
        # These can be written as: x - r >= 0, 1 - x - r >= 0, etc.
        for i in range(N):
            # x_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[i] - v[2*N + i],
                'jac': lambda v, i=i: np.concatenate([np.eye(1)[i], np.zeros(N), -np.eye(1)[i]])
            })
            # 1 - x_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: 1.0 - v[i] - v[2*N + i],
                'jac': lambda v, i=i: np.concatenate([-np.eye(1)[i], np.zeros(N), -np.eye(1)[i]])
            })
            # y_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[N + i] - v[2*N + i],
                'jac': lambda v, i=i: np.concatenate([np.zeros(N), np.eye(1)[i], -np.eye(1)[i]])
            })
            # 1 - y_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: 1.0 - v[N + i] - v[2*N + i],
                'jac': lambda v, i=i: np.concatenate([np.zeros(N), -np.eye(1)[i], -np.eye(1)[i]])
            })
        
        # Non-overlap: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        # Equivalently: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        for i in range(N):
            for j in range(i+1, N):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (v[i]-v[j])**2 + (v[N+i]-v[N+j])**2 - (v[2*N+i]+v[2*N+j])**2,
                    'jac': lambda v, i=i, j=j: _nonoverlap_jac(v, i, j)
                })
        
        # Radius positivity: r_i >= 0 (small epsilon to avoid zero-radius circles)
        for i in range(N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[2*N + i] - 0.01,
                'jac': lambda v, i=i: np.concatenate([np.zeros(2*N), np.eye(1)[i]])
            })
        
        return constraints
    
    def _nonoverlap_jac(v, i, j):
        """Jacobian for non-overlap constraint between circles i and j."""
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:]
        
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        s = r[i] + r[j]
        
        jac = np.zeros(3*N)
        # d/dx_i: 2*dx
        jac[i] = 2.0 * dx
        # d/dx_j: -2*dx
        jac[j] = -2.0 * dx
        # d/dy_i: 2*dy
        jac[N + i] = 2.0 * dy
        # d/dy_j: -2*dy
        jac[N + j] = -2.0 * dy
        # d/dr_i: -2*s
        jac[2*N + i] = -2.0 * s
        # d/dr_j: -2*s
        jac[2*N + j] = -2.0 * s
        
        return jac
    
    def initial_guess():
        """Create an initial feasible guess using a grid layout."""
        side = int(np.ceil(np.sqrt(N)))
        cells_per_side = side
        margin = 0.05
        spacing = (1.0 - 2*margin) / cells_per_side
        
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        for i in range(N):
            row = i // cells_per_side
            col = i % cells_per_side
            x[i] = margin + (col + 0.5) * spacing
            y[i] = margin + (row + 0.5) * spacing
            # Radius limited by distance to nearest neighbor and boundary
            r[i] = min(spacing * 0.45, margin * 0.8)
        
        v = np.concatenate([x, y, r])
        return v
    
    def check_feasibility(v):
        """Check if a solution is feasible (all constraints satisfied)."""
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:]
        
        # Check boundaries
        if np.any(x < r - 1e-6) or np.any(x > 1 - r + 1e-6):
            return False
        if np.any(y < r - 1e-6) or np.any(y > 1 - r + 1e-6):
            return False
        if np.any(r < 0.01 - 1e-6):
            return False
        
        # Check non-overlap
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                min_dist = r[i] + r[j]
                if dist_sq < min_dist**2 - 1e-6:
                    return False
        
        return True
    
    # Try multiple random restarts
    best_v = None
    best_obj = np.inf
    
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            v0 = initial_guess()
        else:
            # Perturbed initial guess
            v0 = initial_guess()
            noise_scale = 0.05
            v0[:N] += np.random.randn(N) * noise_scale
            v0[N:2*N] += np.random.randn(N) * noise_scale
            # Keep radii same for now
            # Clip to valid ranges
            v0[:N] = np.clip(v0[:N], 0.01, 0.99)
            v0[N:2*N] = np.clip(v0[N:2*N], 0.01, 0.99)
        
        try:
            result = minimize(
                objective,
                v0,
                method='SLSQP',
                constraints=constraints_fn(v0),
                bounds=[(0.01, 0.99)] * (2*N) + [(0.01, 0.5)] * N,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if result.success or result.status == 0:
                if check_feasibility(result.x):
                    obj = -result.fun
                    if obj < best_obj:
                        best_obj = obj
                        best_v = result.x.copy()
            
        except Exception as e:
            pass
        
        if time.time() > deadline - 10.0:
            break
    
    # If no feasible solution found, use the initial guess
    if best_v is None:
        v0 = initial_guess()
        if check_feasibility(v0):
            best_v = v0
        else:
            # Fallback: very small circles in grid
            side = int(np.ceil(np.sqrt(N)))
            cells = []
            for i in range(N):
                row = i // side
                col = i % side
                cx = (col + 0.5) / side
                cy = (row + 0.5) / side
                r = 0.5 / side - 0.02
                cells.append([cx, cy, r])
            return np.array(cells)
    
    # Extract final solution
    x = best_v[:N]
    y = best_v[N:2*N]
    r = best_v[2*N:]
    
    result_array = np.column_stack([x, y, r])
    
    # Final feasibility check and repair if needed
    if not check_feasibility(best_v):
        # Slightly shrink all radii to ensure feasibility
        for i in range(N):
            result_array[i, 2] *= 0.99
        # Re-check
        x = result_array[:, 0]
        y = result_array[:, 1]
        r = result_array[:, 2]
        if not check_feasibility(np.concatenate([x, y, r])):
            # More aggressive shrink
            for i in range(N):
                result_array[i, 2] *= 0.95
    
    return result_array
# EVOLVE-BLOCK-END
