"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START

def _pairwise_distances(pos):
    """Return (n, n) matrix of Euclidean distances."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))

def _constraints(pos, radii):
    """
    Returns list of constraint functions for scipy minimize.
    Constraints:
      1) x_i >= r_i
      2) y_i >= r_i
      3) x_i + r_i <= 1
      4) y_i + r_i <= 1
      5) dist(i,j) >= r_i + r_j for all i < j
    """
    n = len(radii)
    cons = []
    # Boundary constraints
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda i=i: pos[2*i] - radii[i]})
        cons.append({'type': 'ineq', 'fun': lambda i=i: pos[2*i+1] - radii[i]})
        cons.append({'type': 'ineq', 'fun': lambda i=i: 1.0 - pos[2*i] - radii[i]})
        cons.append({'type': 'ineq', 'fun': lambda i=i: 1.0 - pos[2*i+1] - radii[i]})
    # Pairwise constraints
    for i in range(n):
        for j in range(i+1, n):
            cons.append({'type': 'ineq', 'fun': lambda i=i, j=j: 
                         np.sqrt((pos[2*i]-pos[2*j])**2 + (pos[2*i+1]-pos[2*j+1])**2) - radii[i] - radii[j]})
    return cons

def _make_objective(radii):
    """Objective: maximize sum of radii, i.e., minimize -sum(radii)."""
    def obj(pos):
        return -np.sum(radii)
    return obj

def _project_to_valid(pos, radii, tol=1e-10):
    """
    Given positions and radii, adjust to ensure validity.
    Returns (pos, radii, valid) where valid is a boolean.
    """
    n = len(radii)
    pos = pos.copy()
    radii = radii.copy()
    
    # Clamp positions and radii to be within [0, 1]
    for i in range(n):
        # Ensure radius is positive
        if radii[i] < 1e-10:
            radii[i] = 1e-10
        # Clamp center so circle is inside [0,1]^2
        pos[2*i] = np.clip(pos[2*i], radii[i], 1.0 - radii[i])
        pos[2*i+1] = np.clip(pos[2*i+1], radii[i], 1.0 - radii[i])
    
    # Iteratively resolve overlaps
    for iteration in range(50):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[2*i] - pos[2*j]
                dy = pos[2*i+1] - pos[2*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist - tol:
                    # Overlap detected; push apart
                    if dist < 1e-12:
                        # Random perturbation
                        dx = np.random.rand() - 0.5
                        dy = np.random.rand() - 0.5
                        dist = np.sqrt(dx*dx + dy*dy)
                    overlap = min_dist - dist
                    # Push each circle half the overlap apart
                    push_x = (dx / dist) * overlap / 2
                    push_y = (dy / dist) * overlap / 2
                    pos[2*i] += push_x
                    pos[2*i+1] += push_y
                    pos[2*j] -= push_x
                    pos[2*j+1] -= push_y
                    # Re-clamp to boundaries
                    pos[2*i] = np.clip(pos[2*i], radii[i], 1.0 - radii[i])
                    pos[2*i+1] = np.clip(pos[2*i+1], radii[i], 1.0 - radii[i])
                    pos[2*j] = np.clip(pos[2*j], radii[j], 1.0 - radii[j])
                    pos[2*j+1] = np.clip(pos[2*j+1], radii[j], 1.0 - radii[j])
                    moved = True
        if not moved:
            break
    
    # Final check for validity
    valid = True
    for i in range(n):
        if pos[2*i] < radii[i] - tol or pos[2*i] > 1.0 - radii[i] + tol:
            valid = False
            break
        if pos[2*i+1] < radii[i] - tol or pos[2*i+1] > 1.0 - radii[i] + tol:
            valid = False
            break
    if valid:
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[2*i] - pos[2*j]
                dy = pos[2*i+1] - pos[2*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - tol:
                    valid = False
                    break
            if not valid:
                break
    
    return pos, radii, valid

def _compute_score(pos, radii):
    """Compute the sum of radii, which is the objective value."""
    return np.sum(radii)

def _grid_initialization():
    """Create a grid of equal circles as initial guess."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append((x, y, r))
    cells = cells[:N]
    pos = np.array([item[0] for item in cells] + [item[1] for item in cells])
    radii = np.array([item[2] for item in cells])
    return pos, radii

def _random_initialization():
    """Create random initial positions and radii."""
    # Start with small radii and random positions
    radii = np.full(N, 0.02)
    pos = np.random.rand(2*N)
    # Clamp to be within bounds
    for i in range(N):
        pos[2*i] = np.clip(pos[2*i], radii[i], 1.0 - radii[i])
        pos[2*i+1] = np.clip(pos[2*i+1], radii[i], 1.0 - radii[i])
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Start with grid initialization
    pos, radii = _grid_initialization()
    pos, radii, valid = _project_to_valid(pos, radii)
    best_score = _compute_score(pos, radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    if not valid:
        # Fall back to trivial grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Try to improve using local optimization
    # We'll use a simple gradient-free approach: try increasing radii and re-projecting
    
    # Strategy: iteratively increase radii slightly and project
    for attempt in range(50):
        if time.time() > deadline - 5.0:
            break
        
        # Try to increase radii
        new_radii = radii * (1.0 + 0.01)
        new_pos, new_radii, new_valid = _project_to_valid(pos, new_radii)
        
        if new_valid:
            new_score = _compute_score(new_pos, new_radii)
            if new_score > best_score:
                best_score = new_score
                best_pos = new_pos.copy()
                best_radii = new_radii.copy()
                pos, radii = new_pos, new_radii
        else:
            # Try smaller increase
            new_radii = radii * (1.0 + 0.001)
            new_pos, new_radii, new_valid = _project_to_valid(pos, new_radii)
            if new_valid:
                new_score = _compute_score(new_pos, new_radii)
                if new_score > best_score:
                    best_score = new_score
                    best_pos = new_pos.copy()
                    best_radii = new_radii.copy()
                    pos, radii = new_pos, new_radii
    
    # Try different initializations
    for restart in range(5):
        if time.time() > deadline - 10.0:
            break
        
        pos, radii = _random_initialization()
        pos, radii, valid = _project_to_valid(pos, radii)
        
        if not valid:
            continue
        
        score = _compute_score(pos, radii)
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            best_radii = radii.copy()
        
        # Try to improve from this starting point
        for attempt in range(30):
            if time.time() > deadline - 5.0:
                break
            
            new_radii = radii * (1.0 + 0.02)
            new_pos, new_radii, new_valid = _project_to_valid(pos, new_radii)
            
            if new_valid:
                new_score = _compute_score(new_pos, new_radii)
                if new_score > best_score:
                    best_score = new_score
                    best_pos = new_pos.copy()
                    best_radii = new_radii.copy()
                    pos, radii = new_pos, new_radii
                elif new_score > score:
                    pos, radii = new_pos, new_radii
                    score = new_score
    
    # Final result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_pos[2*i]
        result[i, 1] = best_pos[2*i+1]
        result[i, 2] = best_radii[i]
    
    return result

# EVOLVE-BLOCK-END
