import numpy as np
import time
import itertools

N = 11
H = np.sqrt(3) / 2

# Pre-compute indices for combinations of 3 from 11
# We need to compute areas for all C(11,3) = 165 triangles
_tri_indices = np.array(list(itertools.combinations(range(N), 3)))

# Triangle vertices in cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

def _bary_to_cart(bary):
    """
    Convert barycentric coordinates to cartesian.
    bary: shape (N, 3), where each row sums to 1.
    Returns: shape (N, 2) cartesian coordinates.
    """
    # p = a*V0 + b*V1 + c*V2
    # V0 = (0,0), V1 = (1,0), V2 = (0.5, H)
    x = bary[:, 1] * 1.0 + bary[:, 2] * 0.5
    y = bary[:, 2] * H
    return np.column_stack([x, y])

def _cart_to_bary(pts):
    """
    Convert cartesian points to barycentric coordinates.
    pts: shape (N, 2)
    Returns: shape (N, 3) barycentric coordinates.
    """
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Solve:
    # x = b + 0.5c
    # y = H*c  => c = y/H
    c = y / H
    b = x - 0.5 * c
    a = 1.0 - b - c
    
    return np.column_stack([a, b, c])

def _min_area_from_bary(bary):
    """
    Compute the minimum area of any triangle formed by the points.
    bary: shape (N, 3)
    Returns: minimum area (float)
    """
    # Convert to cartesian for area calculation (vectorized)
    pts = _bary_to_cart(bary)
    
    # Get points for all triangles
    pi = pts[_tri_indices[:, 0]]
    pj = pts[_tri_indices[:, 1]]
    pk = pts[_tri_indices[:, 2]]
    
    # Vectorized area calculation: 0.5 * |det(pj-pi, pk-pi)|
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    return np.min(areas)

def _penalty_objective(bary, alpha, penalty_weight):
    """
    Objective function for Simulated Annealing.
    We want to maximize the minimum area.
    Let's minimize the penalty: sum(max(0, alpha - area_ijk)^2) + small term to keep alpha high?
    Actually, standard SA minimizes energy.
    Energy = -alpha + penalty_weight * sum(max(0, alpha - area_ijk)^2)
    
    If we increase alpha, the penalty increases if constraints are violated.
    If we decrease alpha, energy decreases (more negative -alpha) but we want high alpha.
    So we minimize: penalty_weight * sum(violations^2) - alpha.
    """
    pts = _bary_to_cart(bary)
    pi = pts[_tri_indices[:, 0]]
    pj = pts[_tri_indices[:, 1]]
    pk = pts[_tri_indices[:, 2]]
    
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    
    violations = np.maximum(0, alpha - areas)
    penalty = np.sum(violations * violations)
    
    return penalty_weight * penalty - alpha

def _clamp_bary(bary):
    """
    Ensure barycentric coordinates are valid (non-negative and sum to 1).
    Simple projection: clip negatives to 0, then renormalize.
    """
    bary = np.clip(bary, 1e-12, None)
    sums = np.sum(bary, axis=1, keepdims=True)
    bary = bary / sums
    return bary

def _random_bary_point(rng):
    """Generate a random point in barycentric coordinates inside the triangle."""
    r1 = rng.random()
    r2 = rng.random()
    if r1 + r2 > 1:
        r1 = 1 - r1
        r2 = 1 - r2
    # a = 1 - r1 - r2, b = r1, c = r2
    return np.array([1.0 - r1 - r2, r1, r2])

def _init_structured_bary():
    """
    Create a structured initial configuration in barycentric coordinates.
    Vertices: (1,0,0), (0,1,0), (0,0,1)
    Midpoints: (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    Centroid: (1/3, 1/3, 1/3)
    """
    pts = []
    pts.append([1.0, 0.0, 0.0])
    pts.append([0.0, 1.0, 0.0])
    pts.append([0.0, 0.0, 1.0])
    
    pts.append([0.5, 0.5, 0.0])
    pts.append([0.5, 0.0, 0.5])
    pts.append([0.0, 0.5, 0.5])
    
    pts.append([1/3, 1/3, 1/3])
    
    # 4 more points: place them in a small circle around the centroid in barycentric space?
    # Or just randomize slightly from centroid.
    # Let's use a regular tetrahedron-like distribution in 2D barycentric plane.
    # Angles: 0, pi/2, pi, 3pi/2
    # Radius in barycentric "distance" is tricky. 
    # Let's just perturb the centroid.
    cx, cy, cz = 1/3, 1/3, 1/3
    # Small perturbations
    for i in range(4):
        angle = i * np.pi / 2
        # Move in the plane a+b+c=1
        # Basis vectors for the plane: e1 = (1, -1, 0), e2 = (1, 0, -1)
        # Or simpler: just add noise and normalize
        p = np.array([cx, cy, cz])
        p += np.array([np.cos(angle), np.sin(angle), 0]) * 0.1
        p = np.clip(p, 1e-6, None)
        p /= np.sum(p)
        pts.append(p)
        
    return np.array(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def time_left():
        return deadline - time.time()
    
    best_pts_bary = None
    best_alpha = -1.0
    
    # Initial Configuration
    current_bary = _init_structured_bary()
    current_alpha = _min_area_from_bary(current_bary)
    best_pts_bary = current_bary.copy()
    best_alpha = current_alpha
    
    # Simulated Annealing Parameters
    T = 0.1  # Initial Temperature
    T_min = 1e-6
    cooling_rate = 0.995
    penalty_weight = 100.0
    
    # We will also track alpha in the SA. 
    # Actually, optimizing alpha as a variable is hard in SA. 
    # Better: Fix alpha to be the current min area, and just move points to increase min area?
    # No, that's just local search.
    # Standard approach for max-min: 
    # Energy = -min_area. But min_area is non-differentiable.
    # Let's stick to the penalty method where alpha is a target.
    # Start with a low alpha, increase it slowly, and optimize points to satisfy constraints.
    
    # Phase 1: Slowly increase alpha and optimize points using SA
    # We start with alpha = current_min_area * 0.9
    alpha = current_alpha * 0.9
    
    # We will run SA for a fixed number of steps or time, increasing alpha periodically.
    
    rng = np.random.default_rng(42)
    
    current_energy = _penalty_objective(current_bary, alpha, penalty_weight)
    
    step_size = 0.05
    
    while time_left() > 10.0:
        # Generate neighbor
        # Perturb one point's barycentric coordinates
        i = rng.integers(0, N)
        delta = rng.normal(0, step_size, size=3)
        
        new_bary = current_bary.copy()
        new_bary[i] += delta
        new_bary[i] = np.clip(new_bary[i], 1e-12, None)
        # Renormalize only the changed point? 
        # If we change one point, we should renormalize that point to stay in the simplex.
        s = np.sum(new_bary[i])
        if s < 1e-12:
            new_bary[i] = np.array([1/3, 1/3, 1/3])
        else:
            new_bary[i] /= s
            
        new_energy = _penalty_objective(new_bary, alpha, penalty_weight)
        
        delta_E = new_energy - current_energy
        
        if delta_E < 0 or (T > 0 and rng.random() < np.exp(-delta_E / T)):
            current_bary = new_bary
            current_energy = new_energy
            
            # Check if we improved the actual min area
            # We can check every few steps to save time
            if rng.random() < 0.1:
                actual_min = _min_area_from_bary(current_bary)
                if actual_min > best_alpha:
                    best_alpha = actual_min
                    best_pts_bary = current_bary.copy()
        
        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Increase alpha if we are making progress
        # If the penalty is low (constraints satisfied), we can try a higher alpha
        # Estimate penalty magnitude
        # If current_energy is close to -alpha (penalty ~ 0), we can increase alpha
        if current_energy < -alpha * 0.99: # Penalty is very small
            alpha *= 1.01 # Increase target alpha by 1%
            # Reset temperature slightly to explore new landscape
            T = max(T, 0.05)
            
        # Adapt step size?
        # If we are accepting many moves, maybe step size is good.
        # If not, maybe reduce it?
        # For simplicity, we keep step size constant or slowly decrease it.
        # Let's decrease it slightly over time to allow convergence.
        step_size *= 0.999

    # Final Polish: Use the best found barycentric points and run a few steps of Nelder-Mead on the hard min-area objective in Cartesian space
    # Convert best bary to cartesian
    best_pts_cart = _bary_to_cart(best_pts_bary)
    
    # Define hard objective for Nelder-Mead
    def _objective_hard_cart(flat_pts):
        pts = flat_pts.reshape(-1, 2)
        # Clamp to triangle
        bary = _cart_to_bary(pts)
        bary = _clamp_bary(bary)
        pts_clamped = _bary_to_cart(bary)
        return -_min_area_from_bary(bary)
        
    if time_left() > 5.0:
        try:
            from scipy.optimize import minimize
            res = minimize(
                _objective_hard_cart,
                best_pts_cart.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            # Convert result back to barycentric for consistency, though we return cartesian
            final_pts = _clamp_bary(_cart_to_bary(res.x.reshape(-1, 2)))
            final_pts_cart = _bary_to_cart(final_pts)
            final_score = _min_area_from_bary(final_pts)
            
            if final_score > best_alpha:
                best_pts_cart = final_pts_cart
                best_alpha = final_score
        except Exception:
            pass

    # Ensure we return cartesian points
    return best_pts_cart

if __name__ == "__main__":
    pts = construct()
    # Calculate final score for printing
    bary = _cart_to_bary(pts)
    bary = _clamp_bary(bary)
    score = _min_area_from_bary(bary)
    print(f"Best score: {score}")
