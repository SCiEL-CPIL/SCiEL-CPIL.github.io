import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 110.0

def hex_vertices(cx, cy, r, angle_deg):
    angles = np.deg2rad(angle_deg + np.arange(0, 360, 60))
    x = cx + r * np.cos(angles)
    y = cy + r * np.sin(angles)
    return np.column_stack((x, y))

def l_req(x, y):
    """
    Calculate the minimum side length L of a regular hexagon (centered at 0,0, angle 0)
    that contains the point (x, y).
    The hexagon is defined by vertices at (L, 0), (L/2, L*sqrt(3)/2), etc.
    Inequalities:
    |x| <= L
    |x/2 + y*sqrt(3)/2| <= L
    |x/2 - y*sqrt(3)/2| <= L
    """
    term1 = np.abs(x)
    term2 = np.abs(x * 0.5 + y * (np.sqrt(3) / 2))
    term3 = np.abs(x * 0.5 - y * (np.sqrt(3) / 2))
    return np.maximum(np.maximum(term1, term2), term3)

def overlap_penalty(x1, y1, x2, y2):
    """
    Penalty for two unit hexagons (side 1, angle 0) centered at (x1,y1) and (x2,y2).
    They overlap if the vector d=(x2-x1, y2-y1) is inside the Minkowski difference,
    which is a regular hexagon of side 2.
    Condition for being inside hexagon of side 2:
    max(|dx|, |dx/2 + dy*sqrt(3)/2|, |dx/2 - dy*sqrt(3)/2|) < 2
    We use a smooth penalty based on the violation.
    """
    dx = x2 - x1
    dy = y2 - y1
    
    # Calculate "distance" in hexagonal metric
    term1 = np.abs(dx)
    term2 = np.abs(dx * 0.5 + dy * (np.sqrt(3) / 2))
    term3 = np.abs(dx * 0.5 - dy * (np.sqrt(3) / 2))
    
    dist_metric = np.maximum(np.maximum(term1, term2), term3)
    
    # If dist_metric < 2, they overlap.
    # We want to penalize when dist_metric is small.
    # Penalty = max(0, 2 - dist_metric)^2
    violation = 2.0 - dist_metric
    if violation > 0:
        return violation * violation
    return 0.0

def objective(params):
    start_time = time.time()
    if start_time > DEADLINE:
        return 1e9
        
    positions = params.reshape(N, 2)
    
    # Calculate L_config: max L_req over all vertices of all hexagons
    max_l = 0.0
    for i in range(N):
        x, y = positions[i]
        # Vertices of unit hexagon at (x,y) with angle 0
        # Offsets: (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
        # We can just calculate l_req for the center + offsets
        # But l_req is not linear. We must check all 6 vertices.
        offsets = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
        verts = offsets + np.array([x, y])
        l_vals = l_req(verts[:, 0], verts[:, 1])
        max_l = max(max_l, np.max(l_vals))
        
    # Calculate overlap penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            penalty += overlap_penalty(positions[i, 0], positions[i, 1], positions[j, 0], positions[j, 1])
            
    # Return L + Penalty
    # We use a large weight for penalty to ensure non-overlap
    return max_l + 100.0 * penalty

def generate_lattice_guesses():
    guesses = []
    # Hexagonal lattice spacing sqrt(3)
    d = np.sqrt(3)
    # Generate a grid of lattice points
    # i, j from -5 to 5
    points = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i * d + (j % 2) * (d / 2) # Shifted rows
            y = j * (d * np.sqrt(3) / 2) # Vertical spacing
            # Actually, standard hex lattice:
            # u = (d, 0), v = (d/2, d*sqrt(3)/2)
            # x = i*d + j*d/2
            # y = j*d*sqrt(3)/2
            x = i * d + j * d * 0.5
            y = j * d * np.sqrt(3) * 0.5
            points.append([x, y])
            
    points = np.array(points)
    
    # Select 12 points closest to origin
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:12]
    guess1 = points[idx].flatten()
    guesses.append(guess1)
    
    # Shifted lattice
    points_shifted = points + np.array([0.1, 0.1])
    dists = np.linalg.norm(points_shifted, axis=1)
    idx = np.argsort(dists)[:12]
    guess2 = points_shifted[idx].flatten()
    guesses.append(guess2)
    
    # Another shift
    points_shifted2 = points + np.array([0.5, 0.2])
    dists = np.linalg.norm(points_shifted2, axis=1)
    idx = np.argsort(dists)[:12]
    guess3 = points_shifted2[idx].flatten()
    guesses.append(guess3)
    
    # Random guess
    guess4 = np.random.uniform(-3, 3, 24)
    guesses.append(guess4)
    
    return guesses

def construct():
    start_time = time.time()
    best_params = None
    best_cost = np.inf
    
    guesses = generate_lattice_guesses()
    
    # Add more random restarts if time permits
    for _ in range(5):
        if time.time() > DEADLINE - 10:
            break
        guess = np.random.uniform(-3, 3, 24)
        guesses.append(guess)
        
    for guess in guesses:
        if time.time() > DEADLINE - 5:
            break
            
        try:
            # Use Nelder-Mead for robustness with non-smooth functions
            res = minimize(objective, guess, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False})
            
            if res.fun < best_cost:
                best_cost = res.fun
                best_params = res.x
        except Exception as e:
            pass
            
    if best_params is None:
        # Fallback: simple grid
        inner = np.zeros((12, 3))
        # Place in a circle
        for i in range(12):
            angle = i * 2 * np.pi / 12
            r = 2.0
            inner[i, 0] = r * np.cos(angle)
            inner[i, 1] = r * np.sin(angle)
            inner[i, 2] = 0.0
        return inner, (0.0, 0.0), 5.0, 0.0

    inner_positions = best_params.reshape(12, 2)
    
    # Calculate the actual L for this configuration
    max_l = 0.0
    for i in range(12):
        x, y = inner_positions[i]
        offsets = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
        verts = offsets + np.array([x, y])
        l_vals = l_req(verts[:, 0], verts[:, 1])
        max_l = max(max_l, np.max(l_vals))
        
    # Format output
    inner = np.zeros((12, 3))
    for i in range(12):
        inner[i, 0] = inner_positions[i, 0]
        inner[i, 1] = inner_positions[i, 1]
        inner[i, 2] = 0.0 # Angle 0
        
    return inner, (0.0, 0.0), max_l, 0.0

