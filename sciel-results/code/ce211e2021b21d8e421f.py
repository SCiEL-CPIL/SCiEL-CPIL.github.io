"""Packing unit hexagons in a regular hexagon using non-convex optimization."""
import numpy as np
from scipy.optimize import minimize

N = 11

def _hex_support(direction, angle_deg, scale=1.0):
    """Support function of a regular hexagon in a given direction."""
    angles = np.radians(angle_deg) + np.arange(6) * np.pi / 3
    vertices = scale * np.column_stack([np.cos(angles), np.sin(angles)])
    return np.max(np.dot(vertices, direction))

def _collision_violation(p1, a1, p2, a2, n_dirs=72):
    """
    Compute overlap violation between two unit hexagons.
    Returns max(0, required_separation - actual_separation) over all directions.
    """
    angles = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    dirs = np.column_stack([np.cos(angles), np.sin(angles)])
    
    supports1 = np.array([_hex_support(d, a1) for d in dirs])
    supports2 = np.array([_hex_support(d, a2) for d in dirs])
    
    diff = p2 - p1
    actual = np.dot(dirs, diff)
    
    violations = supports1 + supports2 - actual
    return np.max(violations)

def _boundary_violation(p, angle, outer_center, outer_side, outer_angle, n_dirs=36):
    """
    Compute how much the inner hexagon sticks out of the outer hexagon.
    Returns max(0, penetration) over outer hexagon edges.
    """
    outer_inradius = outer_side * np.sqrt(3) / 2
    
    # Outer hexagon edge normals (outward)
    max_viol = 0.0
    for i in range(6):
        angle_rad = np.radians(outer_angle) + i * np.pi / 3
        n = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        
        center_dist = np.dot(p - outer_center, n)
        inner_support = _hex_support(n, angle)
        
        viol = center_dist + inner_support - outer_inradius
        max_viol = max(max_viol, viol)
    
    return max_viol

def _objective(state, N):
    """Objective: outer_side + penalty * sum of squared violations."""
    inner = []
    for i in range(N):
        inner.append([state[3*i], state[3*i+1], state[3*i+2]])
    
    outer_cx = state[3*N]
    outer_cy = state[3*N+1]
    outer_side = state[3*N+2]
    outer_angle = state[3*N+3]
    
    outer_center = np.array([outer_cx, outer_cy])
    
    penalty = 0.0
    
    # Inner-inner collisions
    for i in range(N):
        p1 = np.array([inner[i][0], inner[i][1]])
        a1 = inner[i][2]
        for j in range(i+1, N):
            p2 = np.array([inner[j][0], inner[j][1]])
            a2 = inner[j][2]
            viol = _collision_violation(p1, a1, p2, a2)
            if viol > 0:
                penalty += viol * viol
    
    # Inner-outer boundary
    for i in range(N):
        p = np.array([inner[i][0], inner[i][1]])
        a = inner[i][2]
        viol = _boundary_violation(p, a, outer_center, outer_side, outer_angle)
        if viol > 0:
            penalty += viol * viol
    
    return outer_side + 50.0 * penalty

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N unit hexagons inside a regular hexagon, minimizing outer side length.
    Returns: inner (list of [x, y, angle_degrees]), outer_center (x, y), outer_side, outer_angle
    """
    # Generate initial hexagonal lattice positions
    dx = np.sqrt(3)
    dy = 1.5
    
    positions = []
    positions.append([0.0, 0.0])
    for i in range(6):
        angle = i * np.pi / 3
        positions.append([dx * np.cos(angle), dx * np.sin(angle)])
    # That's 7, need 4 more for N=11
    for i in range(4):
        angle = i * np.pi / 2 + np.pi / 6
        positions.append([2.0 * dx * np.cos(angle), 2.0 * dx * np.sin(angle)])
    
    positions = positions[:N]
    
    # Build initial state
    state0 = []
    for i in range(N):
        state0.append(positions[i][0])
        state0.append(positions[i][1])
        state0.append(0.0)  # angle
    state0.append(0.0)  # outer_cx
    state0.append(0.0)  # outer_cy
    state0.append(8.0)  # outer_side initial guess
    state0.append(0.0)  # outer_angle
    
    state0 = np.array(state0)
    
    best_state = state0.copy()
    best_val = _objective(state0, N)
    
    # Multi-start local optimization
    rng = np.random.RandomState(42)
    for restart in range(5):
        if restart == 0:
            s = state0.copy()
        else:
            s = state0 + rng.randn(len(state0)) * 0.3
        
        try:
            result = minimize(
                _objective,
                s,
                args=(N,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-7}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_state = result.x
        except Exception:
            pass
    
    # Unpack
    inner = []
    for i in range(N):
        inner.append([best_state[3*i], best_state[3*i+1], best_state[3*i+2]])
    
    outer_cx = best_state[3*N]
    outer_cy = best_state[3*N+1]
    outer_side = best_state[3*N+2]
    outer_angle = best_state[3*N+3]
    
    outer_side = max(outer_side, 1.0)
    
    return inner, (outer_cx, outer_cy), outer_side, outer_angle
# EVOLVE-BLOCK-END
