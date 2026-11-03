"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

def _initial_grid(n):
    """Generate an initial grid of circles."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < n:
                cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3])
    return np.array(cells)

def _contact_graph(positions, radii, eps=1e-3):
    """
    Compute contact graph using Delaunay triangulation.
    Returns list of (i, j) pairs that are in contact or near contact.
    """
    n = len(positions)
    if n < 3:
        # For small n, just check all pairs
        contacts = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < radii[i] + radii[j] + eps:
                    contacts.append((i, j))
        return contacts
    
    try:
        tri = Delaunay(positions)
        contacts = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    dist = np.linalg.norm(positions[a] - positions[b])
                    if dist < radii[a] + radii[b] + eps:
                        contacts.add((a, b))
        return list(contacts)
    except Exception:
        # Fallback: all pairs
        contacts = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < radii[i] + radii[j] + eps:
                    contacts.append((i, j))
        return contacts

def _objective_and_constraints(x, n, max_iter=100):
    """
    Objective: maximize sum of radii (minimize negative sum)
    Constraints: 
      - Circles inside unit square: x_i - r_i >= 0, y_i - r_i >= 0, x_i + r_i <= 1, y_i + r_i <= 1
      - Non-overlap: ||p_i - p_j|| >= r_i + r_j for all i < j
    """
    x = x.reshape(-1, 3)
    positions = x[:, :2]
    radii = x[:, 2]
    
    # Objective: minimize -sum(radii)
    obj = -np.sum(radii)
    
    # Inequality constraints (must be >= 0)
    cons = []
    
    # Boundary constraints
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})  # x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})  # y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[i*3] - x[i*3+2]})  # 1 - x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[i*3+1] - x[i*3+2]})  # 1 - y - r >= 0
    
    # Non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sum((x[i*3:i*3+2] - x[j*3:j*3+2])**2) - (x[i*3+2] + x[j*3+2])**2
            })
    
    return obj, cons

def construct():
    deadline = time.time() + 140  # 140s to be safe
    
    # Initial solution
    x0 = _initial_grid(N)
    x0 = x0.flatten()
    
    best_x = x0.copy()
    best_val = -np.sum(x0[2::3])
    
    # Try SLSQP optimization
    try:
        res = minimize(
            lambda x: -np.sum(x[2::3]),
            x0,
            method='SLSQP',
            bounds=[(0, 1)] * (3 * N),
            constraints=_objective_and_constraints(x0, N),
            options={'maxiter': 500, 'ftol': 1e-8}
        )
        if res.success and res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_x.reshape(-1, 3)
    
    # Try L-BFGS-B with penalty method as alternative
    try:
        def penalized_obj(x, mu=10.0):
            x = x.reshape(-1, 3)
            positions = x[:, :2]
            radii = x[:, 2]
            
            # Penalty for boundary violations
            pen = 0
            for i in range(N):
                pen += max(0, positions[i, 0] - radii[i]) ** 2
                pen += max(0, positions[i, 1] - radii[i]) ** 2
                pen += max(0, positions[i, 0] + radii[i] - 1) ** 2
                pen += max(0, positions[i, 1] + radii[i] - 1) ** 2
            
            # Penalty for overlaps
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    overlap = radii[i] + radii[j] - dist
                    if overlap > 0:
                        pen += overlap ** 2
            
            return -np.sum(radii) + mu * pen
        
        res2 = minimize(
            penalized_obj,
            best_x,
            method='L-BFGS-B',
            bounds=[(0, 1)] * (3 * N),
            options={'maxiter': 300}
        )
        if res2.fun < best_val:
            best_x = res2.x
            best_val = res2.fun
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_x.reshape(-1, 3)
    
    # Local refinement with SLSQP from best solution
    try:
        res3 = minimize(
            lambda x: -np.sum(x[2::3]),
            best_x,
            method='SLSQP',
            bounds=[(0, 1)] * (3 * N),
            constraints=_objective_and_constraints(best_x, N),
            options={'maxiter': 300, 'ftol': 1e-9}
        )
        if res3.success and res3.fun < best_val:
            best_x = res3.x
            best_val = res3.fun
    except Exception:
        pass
    
    return best_x.reshape(-1, 3)

# EVOLVE-BLOCK-END
