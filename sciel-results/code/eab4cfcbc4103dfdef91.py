"""Scipy SLSQP solver with SAT-based non-overlap constraints for hexagon packing."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START

# Precompute unit hexagon vertex directions (for angle 0)
_VERT_DIRS = np.array([[math.cos(math.pi / 6 + i * math.pi / 3),
                        math.sin(math.pi / 6 + i * math.pi / 3)] for i in range(6)])

# Precompute edge vectors (from vertex k to vertex k+1)
_EDGE_VECTORS = np.array([_VERT_DIRS[(k + 1) % 6] - _VERT_DIRS[k] for k in range(6)])

# Edge normals (perpendicular to edges, pointing outward)
_EDGE_NORMALS = np.array([np.array([-e[1], e[0]]) for e in _EDGE_VECTORS])


def _hex_vertices(center, angle_deg):
    """Get vertices of a unit hexagon given center and angle in degrees."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    R = np.array([[ca, -sa], [sa, ca]])
    return center + _VERT_DIRS @ R.T


def _edge_normals_at_angle(angle_deg):
    """Get edge normals for a hexagon at given angle."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    R = np.array([[ca, -sa], [sa, ca]])
    return _EDGE_NORMALS @ R.T


def _sat_separation(v1, v2):
    """
    Check if two hexagons are separated using SAT.
    Returns the minimum separation distance (positive = separated, negative = overlapping).
    Returns (is_separated, min_sep)
    """
    min_sep = float('inf')
    # Edge directions of both hexagons
    axes = []
    for k in range(6):
        axes.append(v1[k] - v1[(k + 1) % 6])
        axes.append(v2[k] - v2[(k + 1) % 6])
    
    for e in axes:
        n = np.array([-e[1], e[0]])
        p1 = v1 @ n
        p2 = v2 @ n
        # Separation: how much p1's max < p2's min or vice versa
        sep1 = p2.min() - p1.max()
        sep2 = p1.min() - p2.max()
        sep = max(sep1, sep2)
        min_sep = min(min_sep, sep)
    
    return min_sep > 0, min_sep


def _build_initial_honeycomb():
    """Honeycomb: center + 6 ring + 5 ring-2. Returns centers (12,2)."""
    def axial(q, r):
        return np.array([math.sqrt(3) * (q + r / 2.0), 1.5 * r])
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pts = [(0, 0)] + list(ring1)
    ring2 = []
    for (q, r) in ring1:
        for (dq, dr) in ring1:
            nq, nr = q + dq, r + dr
            if (nq, nr) not in pts and (nq, nr) not in ring2:
                ring2.append((nq, nr))
    pts.extend(ring2[:5])
    C = np.array([axial(q, r) for (q, r) in pts])
    C -= C.mean(axis=0)
    return C


def _objective_and_grad(x):
    """
    Objective: outer hexagon side length.
    x = [c0x, c0y, a0, c1x, c1y, a1, ...] (36 variables)
    Returns (obj, grad)
    """
    C = x[0:24].reshape(12, 2)
    angles = x[24:36]  # in radians
    
    # Outer hexagon at origin, angle 0
    outer_normals = _EDGE_NORMALS
    
    # Compute max projection of all inner hexagon vertices onto outer normals
    max_proj = 0.0
    grad = np.zeros_like(x)
    
    for i in range(N):
        verts = _hex_vertices(C[i], math.degrees(angles[i]))
        for nk in outer_normals:
            proj = verts @ nk
            if proj.max() > max_proj:
                max_proj = proj.max()
    
    # Side length S = 2/sqrt(3) * max_proj
    S = (2.0 / math.sqrt(3.0)) * max_proj
    
    # Gradient computation (numerical or analytical)
    # For simplicity, use numerical gradient
    eps = 1e-7
    for j in range(len(x)):
        x2 = x.copy()
        x2[j] += eps
        C2 = x2[0:24].reshape(12, 2)
        angles2 = x2[24:36]
        max_proj2 = 0.0
        for i in range(N):
            verts = _hex_vertices(C2[i], math.degrees(angles2[i]))
            for nk in outer_normals:
                proj = verts @ nk
                if proj.max() > max_proj2:
                    max_proj2 = proj.max()
        S2 = (2.0 / math.sqrt(3.0)) * max_proj2
        grad[j] = (S2 - S) / eps
    
    return S, grad


def _nonoverlap_constraints(x, time_budget=100.0):
    """
    Check non-overlap constraints for all pairs.
    Returns list of constraint values (positive = satisfied).
    """
    C = x[0:24].reshape(12, 2)
    angles = x[24:36]  # radians
    angles_deg = np.degrees(angles)
    
    constraints = []
    for i in range(N):
        for j in range(i + 1, N):
            v1 = _hex_vertices(C[i], angles_deg[i])
            v2 = _hex_vertices(C[j], angles_deg[j])
            is_sep, min_sep = _sat_separation(v1, v2)
            # Constraint: min_sep >= 0 (with small margin for numerical stability)
            constraints.append(min_sep + 0.001)
    
    return constraints


def construct():
    start = time.time()
    deadline = start + 140.0  # Leave 10s safety margin
    
    # Initial honeycomb configuration
    C0 = _build_initial_honeycomb()
    angles0 = np.zeros(N)
    
    # Pack into x vector
    x0 = np.concatenate([C0.flatten(), angles0])
    
    # Initial objective
    obj0, _ = _objective_and_grad(x0)
    
    best_x = x0.copy()
    best_obj = obj0
    
    # Multi-start with perturbed initial configurations
    np.random.seed(42)
    
    # Try a few different initial configurations
    for trial in range(3):
        if time.time() > deadline - 30:
            break
        
        if trial == 0:
            x_start = x0.copy()
        elif trial == 1:
            # Slightly expanded honeycomb
            x_start = x0.copy()
            x_start[0:24] = C0 * 1.05
        else:
            # Random perturbation
            x_start = x0.copy()
            x_start[0:24] += np.random.normal(0, 0.1, size=(12, 2))
            x_start[24:36] += np.random.normal(0, 0.1, size=N)
        
        # Define constraints for SLSQP
        def constraints_func(x):
            cons = _nonoverlap_constraints(x)
            return [{'type': 'ineq', 'fun': lambda x, c=cons[c_idx]: c[c_idx]} for c_idx in range(len(cons))]
        
        # Simpler: use a single constraint function returning array
        def all_constraints(x):
            return np.array(_nonoverlap_constraints(x))
        
        # Use SLSQP with multiple inequality constraints
        # SLSQP doesn't directly support vector constraints, so we need to handle differently
        
        # Alternative: use penalty method or sequential constraint handling
        # For simplicity, use a penalty approach with scipy minimize
        
        def penalized_objective(x, penalty=1000.0):
            obj, grad = _objective_and_grad(x)
            cons = _nonoverlap_constraints(x)
            # Penalize violations
            violation = max(0, -min(cons)) if cons else 0
            penalty_term = penalty * violation ** 2
            return obj + penalty_term
        
        def penalized_grad(x, penalty=1000.0):
            obj, grad = _objective_and_grad(x)
            cons = _nonoverlap_constraints(x)
            # Approximate gradient of penalty term
            if cons:
                min_cons = min(cons)
                if min_cons < 0:
                    # Add gradient contribution from penalty
                    # This is approximate; for better results, compute analytical gradient
                    grad_penalty = np.zeros_like(x)
                    # Find which pair is most violated
                    worst_idx = np.argmin(cons)
                    # Compute gradient of separation for that pair
                    # For simplicity, use finite difference
                    eps = 1e-5
                    for j in range(len(x)):
                        x2 = x.copy()
                        x2[j] += eps
                        cons2 = _nonoverlap_constraints(x2)
                        if cons2:
                            sep_change = cons2[worst_idx] - cons[worst_idx]
                            if cons[worst_idx] < 0 and cons2[worst_idx] < 0:
                                grad_penalty[j] += penalty * 2 * cons[worst_idx] * sep_change / eps
                    grad = grad + grad_penalty
            return obj, grad
        
        # Run SLSQP
        try:
            res = minimize(
                penalized_objective,
                x_start,
                jac=penalized_grad,
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            
            if res.success and res.fun < best_obj:
                # Verify feasibility
                C_final = res.x[0:24].reshape(12, 2)
                angles_final = res.x[24:36]
                angles_final_deg = np.degrees(angles_final)
                
                # Check all pairs are non-overlapping
                feasible = True
                for i in range(N):
                    for j in range(i + 1, N):
                        v1 = _hex_vertices(C_final[i], angles_final_deg[i])
                        v2 = _hex_vertices(C_final[j], angles_final_deg[j])
                        is_sep, min_sep = _sat_separation(v1, v2)
                        if not is_sep:
                            feasible = False
                            break
                    if not feasible:
                        break
                
                if feasible:
                    best_x = res.x.copy()
                    best_obj = res.fun
        
        except Exception as e:
            pass  # Skip failed trials
        
        # Check time
        if time.time() > deadline - 20:
            break
    
    # Final validation and extraction
    C_final = best_x[0:24].reshape(12, 2)
    angles_final = best_x[24:36]
    angles_final_deg = np.degrees(angles_final)
    
    # Compute final side length
    outer_normals = _EDGE_NORMALS
    max_proj = 0.0
    for i in range(N):
        verts = _hex_vertices(C_final[i], angles_final_deg[i])
        for nk in outer_normals:
            proj = verts @ nk
            if proj.max() > max_proj:
                max_proj = proj.max()
    final_side = (2.0 / math.sqrt(3.0)) * max_proj
    
    # Build inner hexagon list
    inner = []
    for i in range(N):
        inner.append([float(C_final[i, 0]), float(C_final[i, 1]), float(angles_final_deg[i])])
    
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
