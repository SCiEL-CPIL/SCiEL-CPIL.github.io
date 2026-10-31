import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    # Precompute base hexagon vertices (side 1, angle 0)
    # Vertices at angles 0, 60, 120, 180, 240, 300
    base_verts = []
    for i in range(6):
        theta = math.radians(60 * i)
        base_verts.append((math.cos(theta), math.sin(theta)))
    base_verts = np.array(base_verts)

    def get_hex_verts(pos, angle_deg=0.0):
        """
        pos: (x, y) center
        angle_deg: rotation of the hexagon
        Returns: (6, 2) array of vertices
        """
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        R = np.array([[ca, -sa], [sa, ca]])
        # Rotate base vertices
        rotated = base_verts @ R.T
        # Translate
        return rotated + pos

    def compute_S_and_validity(positions, outer_angle_deg=0.0, check_overlap=True):
        """
        positions: (N, 2) array of centers.
        Returns (S, is_valid).
        S is the side length of the minimum enclosing regular hexagon.
        is_valid is True if hexagons are disjoint and fit inside.
        """
        # 1. Compute the outer hexagon size S required to contain all inner hexagons
        # The outer hexagon has center (0,0) and angle outer_angle_deg.
        # For a regular hexagon of side S, the apothem (distance from center to side midpoint) is S * sqrt(3)/2.
        # The vertices are at distance S from center.
        # We need to find the minimum S such that all inner vertices are inside.
        
        # The condition for a point p to be inside a regular hexagon centered at origin with angle a and side S:
        # The hexagon is the intersection of 6 half-planes.
        # The normals to the sides point at angles a + 30 + 60*k.
        # Distance from origin to side is apothem = S * sqrt(3)/2.
        # So for normal n_k, we need dot(p - center, n_k) <= apothem.
        # Here center is (0,0).
        # So dot(p, n_k) <= S * sqrt(3)/2.
        # S >= 2/sqrt(3) * dot(p, n_k).
        
        # We have N hexagons. Each has 6 vertices. Total 6N points.
        # For each point p, and each of the 6 outer normals n_k, we need S >= 2/sqrt(3) * dot(p, n_k).
        # So S = max over all p, k of (2/sqrt(3) * dot(p, n_k)).
        
        # Outer normals
        a_outer = math.radians(outer_angle_deg)
        normals = []
        for k in range(6):
            theta = a_outer + math.radians(30 + 60 * k)
            normals.append((math.cos(theta), math.sin(theta)))
        normals = np.array(normals) # (6, 2)
        
        # Get all inner vertices
        all_verts = []
        for i in range(N):
            # Assume all inner hexagons are aligned (angle 0) for this optimization phase?
            # The task allows rotation. However, for packing, aligning is often optimal or near-optimal.
            # To keep the optimization tractable, we fix inner hexagon angles to 0 initially.
            # If we need to allow rotation, the dimension increases. Let's stick to fixed orientation first.
            verts = get_hex_verts(positions[i], 0.0)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts) # (6N, 2)
        
        # Compute projections
        # proj[i, k] = dot(all_verts[i], normals[k])
        # Using matrix multiplication: (6N, 2) @ (2, 6) -> (6N, 6)
        projections = all_verts @ normals.T
        max_proj = np.max(projections)
        
        S = max_proj * 2.0 / math.sqrt(3)
        
        # 2. Check disjointness
        if check_overlap:
            # Check if any two hexagons overlap.
            # Since they are regular and same size, we can check distance between centers.
            # If centers are too close, they overlap.
            # Minimum distance for disjointness of two unit hexagons with same orientation:
            # The distance between centers must be at least 2 * apothem = sqrt(3).
            # Wait, is this true for any orientation?
            # If orientations are the same, yes. The minimum distance between two identical convex bodies is related to the support function.
            # For regular hexagons with same orientation, the minimum center distance for disjointness is indeed 2 * apothem = sqrt(3) if they touch along a side, or potentially larger if they touch at vertices?
            # Actually, the "width" of a hexagon is 2*side = 2? No, width across flats is sqrt(3)*side. Width across corners is 2*side.
            # If two hexagons are aligned, the minimum distance to touch is when they touch side-to-side. Distance = sqrt(3).
            # If they are rotated, the minimum distance can be smaller?
            # Let's assume fixed orientation (0) for all inner hexagons. Then distance >= sqrt(3) is necessary and sufficient for disjointness?
            # Yes, for two identical regular polygons with same orientation, the minimum center distance for disjointness is the distance between parallel supporting lines, which is the width of one polygon.
            # Width of regular hexagon side 1 is sqrt(3).
            
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < math.sqrt(3) - 1e-9:
                        return S, False
        return S, True

    def objective(params):
        """
        params: (2N,) array of x, y coordinates for N hexagons.
        Returns a scalar cost.
        """
        positions = params.reshape(N, 2)
        
        # Soft penalty for overlaps
        # We want to minimize S, but also ensure no overlaps.
        # If overlaps exist, return a large penalty.
        # To make it smooth for differential_evolution, we can use a soft penalty.
        # However, differential_evolution is derivative-free and can handle discontinuities,
        # but a smooth landscape helps.
        
        S, is_valid = compute_S_and_validity(positions, 0.0, check_overlap=True)
        
        if not is_valid:
            # Calculate overlap penalty
            # Sum of squares of violations
            penalty = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < math.sqrt(3):
                        violation = math.sqrt(3) - d
                        penalty += violation ** 2
            # Return a large value to push out of this region
            return S + 100.0 * penalty
        return S

    # Initial guess: Hexagonal packing
    # 1 in center, 6 around, 4 more?
    # Standard hexagonal cluster of 19 has side 3 (1+2+3+2+1 points? No, 1+6+12=19).
    # We need 11.
    # 1 center, 6 neighbors (dist sqrt(3)), 4 more at dist 2*sqrt(3)?
    # Let's generate a few initial points on a triangular lattice.
    
    sp = math.sqrt(3)
    init_positions = []
    # Center
    init_positions.append([0.0, 0.0])
    # 6 neighbors
    for i in range(6):
        theta = math.radians(60 * i)
        init_positions.append([sp * math.cos(theta), sp * math.sin(theta)])
    # 4 more. The next ring is at distance 2*sqrt(3) or sqrt(3)*sqrt(3)=3?
    # Triangular lattice vectors: (sp, 0), (sp/2, sp*sqrt(3)/2)
    # Points at distance ~2*apothem from center?
    # Let's just pick 4 more from the next shell.
    # Shell 2 has 12 points.
    # Vectors: 2*(sp,0), 2*(sp/2, sp*sqrt(3)/2), etc.
    # (2sp, 0), (sp, sp*sqrt(3)), (-sp, sp*sqrt(3)), (-2sp, 0), etc.
    # Let's add (2sp, 0), (-2sp, 0), (sp, sp*sqrt(3)), (-sp, -sp*sqrt(3))
    init_positions.append([2 * sp, 0.0])
    init_positions.append([-2 * sp, 0.0])
    init_positions.append([sp, sp * math.sqrt(3)])
    init_positions.append([-sp, -sp * math.sqrt(3)])
    
    # Flatten
    x0 = np.array(init_positions).flatten()
    
    # Bounds for differential evolution
    # The outer hexagon side S is likely around 3-4.
    # So positions should be within +/- 5.
    bounds = [(-5, 5)] * (2 * N)
    
    # Run differential evolution
    # We have ~140s. DE can be slow.
    # maxiter=100, popsize=15 (150 points).
    # Each evaluation is cheap (N=11, O(N^2) overlap check).
    
    start_time = time.time()
    
    try:
        # Use a fixed seed for reproducibility in the search, but randomize slightly if we run multiple times?
        # The task says "construct()". It's called once.
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=10, 
            tol=1e-6, 
            seed=42,
            polish=True, # Use L-BFGS-B for final polish
            disp=False
        )
        
        if res.success or res.fun < best_S:
            positions = res.x.reshape(N, 2)
            S, is_valid = compute_S_and_validity(positions, 0.0, check_overlap=True)
            if is_valid:
                best_S = S
                best_inner = []
                for i in range(N):
                    best_inner.append([positions[i][0], positions[i][1], 0.0])
                best_center = (0.0, 0.0)
                best_angle = 0.0
                best_valid = True
    except Exception as e:
        pass

    # If DE failed or didn't find a valid solution, try a few random restarts with local search
    if not best_valid:
        # Fallback: simple random search + local refinement
        for seed in range(10):
            if time.time() > deadline:
                break
            rng = random.Random(seed)
            positions = np.array([[rng.uniform(-3, 3), rng.uniform(-3, 3)] for _ in range(N)])
            
            # Local search
            for _ in range(1000):
                if time.time() > deadline:
                    break
                idx = rng.randrange(N)
                old_x, old_y = positions[idx]
                positions[idx] += [rng.gauss(0, 0.2), rng.gauss(0, 0.2)]
                
                S, is_valid = compute_S_and_validity(positions, 0.0, check_overlap=True)
                if is_valid:
                    if S < best_S:
                        best_S = S
                        best_inner = [[positions[i][0], positions[i][1], 0.0] for i in range(N)]
                        best_center = (0.0, 0.0)
                        best_angle = 0.0
                        best_valid = True
                else:
                    # Revert if invalid
                    positions[idx] = [old_x, old_y]

    # Final Fallback: Linear arrangement
    if not best_valid:
        sp = math.sqrt(3)
        fallback_inner = []
        for k in range(N):
            fallback_inner.append([(k - N // 2) * sp, 0.0, 0.0])
        S, _ = compute_S_and_validity(np.array(fallback_inner)[:, :2], 0.0, check_overlap=False)
        best_S = S
        best_inner = fallback_inner
        best_center = (0.0, 0.0)
        best_angle = 0.0
        best_valid = True

    return best_inner, best_center, best_S, best_angle
# EVOLVE-BLOCK-END
