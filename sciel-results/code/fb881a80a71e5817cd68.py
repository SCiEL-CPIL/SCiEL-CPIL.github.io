"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def hex_norm(d):
        """
        Calculate the 'hexagonal norm' of a vector d.
        This is the maximum distance to the edges of a unit hexagon (side 1) centered at origin.
        If hex_norm(d) <= 1, d is inside the unit hexagon.
        """
        dx, dy = d
        # Normals for hexagon with vertices at 0, 60, 120...
        # Edge midpoints (normals) are at 30, 90, 150, 210, 270, 330.
        # We only need 3 unique axes due to symmetry.
        # n1 = (sqrt(3)/2, 1/2)
        # n2 = (0, 1)
        # n3 = (-sqrt(3)/2, 1/2)
        v1 = np.sqrt(3)/2 * dx + 0.5 * dy
        v2 = dy
        v3 = -np.sqrt(3)/2 * dx + 0.5 * dy
        return np.max(np.abs([v1, v2, v3]))

    def is_valid_config(positions, S):
        """
        Check if the configuration is valid.
        positions: (N, 2) array of centers.
        S: outer side length.
        Assumes all hexagons are aligned (angle 0).
        """
        # Check disjointness: distance between any two centers must be outside hexagon of side 2.
        # Hexagon of side 2 has apothem 2 * sqrt(3)/2 = sqrt(3).
        # So hex_norm(d) > sqrt(3) is required for disjointness.
        for i in range(N):
            for j in range(i + 1, N):
                d = positions[i] - positions[j]
                if hex_norm(d) <= np.sqrt(3) + 1e-9:
                    return False
        
        # Check inside outer: distance from center to any inner center must be inside hexagon of side S-1.
        # Hexagon of side S-1 has apothem (S-1)*sqrt(3)/2.
        # So hex_norm(d) <= (S-1)*sqrt(3)/2 is required.
        limit = (S - 1) * np.sqrt(3) / 2
        for i in range(N):
            d = positions[i]
            if hex_norm(d) > limit + 1e-9:
                return False
        
        return True

    def objective(params):
        """
        Objective function for optimization.
        Returns S + penalty.
        """
        S = params[0]
        positions = params[1:].reshape((N, 2))
        
        penalty = 0.0
        
        # Disjointness penalty
        for i in range(N):
            for j in range(i + 1, N):
                d = positions[i] - positions[j]
                dist = hex_norm(d)
                if dist <= np.sqrt(3):
                    # Overlap penalty: proportional to how deep they are
                    penalty += (np.sqrt(3) - dist) ** 2
        
        # Inside penalty
        limit = (S - 1) * np.sqrt(3) / 2
        for i in range(N):
            d = positions[i]
            dist = hex_norm(d)
            if dist > limit:
                penalty += (dist - limit) ** 2
        
        return S + penalty * 10.0

    # Initial guess: Hexagonal lattice block
    # Generate 12 points in a 3x4 grid on hex lattice
    # x = sqrt(3) * (i + j/2), y = 1.5 * j
    # Let's center this block
    pts = []
    for i in range(4):
        for j in range(3):
            x = np.sqrt(3) * (i + j / 2.0)
            y = 1.5 * j
            pts.append([x, y])
    pts = np.array(pts)
    
    # Center the points
    center = np.mean(pts, axis=0)
    pts -= center
    
    # Initial S estimate
    # The max hex_norm of any point gives a lower bound for (S-1)*sqrt(3)/2
    max_norm = np.max([hex_norm(p) for p in pts])
    S_init = 1 + max_norm * 2 / np.sqrt(3) + 0.5 # Add margin
    
    initial_params = np.concatenate(([S_init], pts.flatten()))
    
    # Bounds for DE
    # S between 2 and 8
    # Positions between -8 and 8
    bounds = [(2.0, 8.0)] + [(-8.0, 8.0)] * (2 * N)
    
    best_params = initial_params
    best_score = objective(initial_params)
    
    # Run Differential Evolution
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=20, 
            tol=0.01, 
            seed=42,
            workers=1
        )
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
    except Exception as e:
        pass
        
    # Parse best params
    S_best = best_params[0]
    positions_best = best_params[1:].reshape((N, 2))
    
    # Validate
    if is_valid_config(positions_best, S_best):
        inner = [(pos[0], pos[1], 0.0) for pos in positions_best]
        outer_center = (0.0, 0.0)
        outer_side = S_best
        outer_angle = 0.0
        return inner, outer_center, outer_side, outer_angle
    else:
        # Fallback: Return a known valid configuration, even if not optimal
        # Use the initial guess but expand S until valid
        S_fallback = S_init
        while not is_valid_config(pts, S_fallback):
            S_fallback += 0.1
            if S_fallback > 10:
                break
        
        inner = [(pos[0], pos[1], 0.0) for pos in pts]
        outer_center = (0.0, 0.0)
        outer_side = S_fallback
        outer_angle = 0.0
        return inner, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-END
