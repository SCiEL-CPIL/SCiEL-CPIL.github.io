import numpy as np
import time
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Precompute constants
    SQRT3 = np.sqrt(3.0)
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def required_outer_side(positions):
        """
        Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
        that contains all unit hexagons at given positions.
        """
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
        verts_flat = verts.reshape(-1, 2)
        
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        
        L1 = x + y / SQRT3
        L2 = y * 2.0 / SQRT3
        L3 = -x + y / SQRT3
        L4 = -x - y / SQRT3
        L5 = -y * 2.0 / SQRT3
        L6 = x - y / SQRT3
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        return np.max(all_L)

    def is_valid(positions, tol=1e-7):
        """
        Check if all hexagons are pairwise disjoint.
        Uses Separating Axis Theorem (SAT).
        Since all inner hexagons are axis-aligned (angle 0), the only axes to check are the 6 normals of a unit hexagon.
        This is much faster than general SAT.
        """
        # Normals for a unit hexagon with flat top (angle 0):
        # The edges are at angles 0, 60, 120, 180, 240, 300.
        # The normals are at 90, 150, 210, 270, 330, 30.
        # Actually, let's just use the 6 projection directions used in required_outer_side.
        # If the projections on these 6 axes don't overlap, they are disjoint.
        
        # Projections of a unit hexagon at origin onto axis u:
        # The width of a unit hexagon along direction u depends on u.
        # However, SAT requires checking if the intervals [min_i, max_i] and [min_j, max_j] are disjoint.
        
        # To speed up, we can precompute the vertices for all hexagons once per check? 
        # No, positions change.
        
        # Let's compute the 6 projections for each hexagon.
        # For a hexagon at (cx, cy) with side 1 and angle 0:
        # Vertices are (cx + cos(a), cy + sin(a)) for a in 0, 60, ...
        
        # Instead of full SAT, we can use a simpler distance metric for a quick reject,
        # but for exact validity, we need SAT.
        
        # Optimization: Only check pairs that are close.
        # But N=12 is small, so O(N^2) is fine if the constant is low.
        
        # Precomputed vertex offsets for unit hexagon
        # Angles: 0, 60, 120, 180, 240, 300
        offsets = np.array([
            [1.0, 0.0],
            [0.5, SQRT3/2],
            [-0.5, SQRT3/2],
            [-1.0, 0.0],
            [-0.5, -SQRT3/2],
            [0.5, -SQRT3/2]
        ])
        
        # Normals for SAT (perpendicular to edges)
        # Edge 0-1: vector (0.5, SQRT3/2) - (1, 0) = (-0.5, SQRT3/2). Normal: (-SQRT3/2, -0.5) or (SQRT3/2, 0.5)
        # Let's use the 6 directions: 30, 90, 150, 210, 270, 330 degrees.
        # cos(30)=SQRT3/2, sin(30)=0.5
        # cos(90)=0, sin(90)=1
        # cos(150)=-SQRT3/2, sin(150)=0.5
        # etc.
        normals = np.array([
            [SQRT3/2, 0.5],
            [0, 1],
            [-SQRT3/2, 0.5],
            [-SQRT3/2, -0.5],
            [0, -1],
            [SQRT3/2, -0.5]
        ])
        
        # For each pair, check all 6 axes.
        for i in range(N):
            cx1, cy1 = positions[i]
            v1 = offsets + np.array([cx1, cy1])
            
            # Projections of hex i onto all 6 normals
            # proj1 = v1.dot(normals.T) -> shape (6, 6) ? No.
            # v1 is (6,2). normals is (6,2).
            # We want projection of each vertex onto each normal.
            # proj[i, k] = v1[i] . normals[k]
            proj1 = v1.dot(normals.T) # (6, 6) where row is vertex, col is normal?
            # Wait, v1 is (6,2). normals.T is (2,6). Result (6,6).
            # min/max per normal (column)
            min1 = proj1.min(axis=0) # (6,)
            max1 = proj1.max(axis=0) # (6,)
            
            for j in range(i + 1, N):
                cx2, cy2 = positions[j]
                v2 = offsets + np.array([cx2, cy2])
                proj2 = v2.dot(normals.T)
                min2 = proj2.min(axis=0)
                max2 = proj2.max(axis=0)
                
                # Check if any normal separates them
                # If max1[k] < min2[k] - tol OR max2[k] < min1[k] - tol
                if np.any(max1 < min2 - tol) or np.any(max2 < min1 - tol):
                    continue # Disjoint
                
                # If no axis separated them, they overlap
                return False
        
        return True

    def get_initial_config():
        """
        Generate a valid initial configuration.
        Uses a hexagonal lattice packing, scaled to fit, then slightly jittered.
        """
        # Hex lattice coordinates for 12 items
        # Center at 0,0
        # Ring 0: 1
        # Ring 1: 6
        # Ring 2: 15 -> take 5 from here
        # Total 12.
        
        # Coordinates in axial system (q, r)
        # x = SQRT3 * (q + r/2)
        # y = 1.5 * r
        
        coords = []
        # Center
        coords.append((0, 0))
        # Ring 1
        for q, r in [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]:
            coords.append((q, r))
        # Ring 2 (first 5)
        # Directions: (2,0), (2,-1), (1,-2), (0,-2), (-1,-1) ... let's just pick 5 neighbors
        # (2,0), (2,-1), (1,-2), (0,-2), (-1,-1) is not quite right.
        # Let's just use a spiral.
        # (2, 0), (2, -1), (1, -2), (0, -2), (-1, -1) -> wait, (-1,-1) is dist sqrt(3)? No, axial dist.
        # Axial dist (q1, r1) to (q2, r2): max(|q1-q2|, |r1-r2|, |q1-r1 - (q2-r2)|)
        # Let's just hardcode 5 points from ring 2 that are valid.
        # (2,0), (2,-1), (1,-2), (0,-2), (-1,-1)
        # Check (-1,-1): q=-1, r=-1. x = SQRT3(-1+0.5) = -0.5*SQRT3. y = -1.5.
        # Is it disjoint from (0,0)? Dist = sqrt(0.75 + 2.25) = sqrt(3) ~ 1.732.
        # Min dist for unit hexes is SQRT3. So they touch. That's valid (disjoint usually means no interior overlap).
        # The task says "disjoint", which usually allows touching.
        
        ring2_pts = [(2, 0), (2, -1), (1, -2), (0, -2), (-1, -1)]
        for q, r in ring2_pts:
            coords.append((q, r))
            
        positions = np.zeros((N, 2))
        for i, (q, r) in enumerate(coords):
            positions[i, 0] = SQRT3 * (q + r / 2.0)
            positions[i, 1] = 1.5 * r
            
        # Add small random jitter to break symmetry
        positions += np.random.uniform(-0.01, 0.01, (N, 2))
        
        # Ensure validity. If not, scale up.
        if not is_valid(positions):
             for scale in [1.1, 1.2, 1.5]:
                 if is_valid(positions * scale):
                     positions = positions * scale
                     break
        return positions

    # Simulated Annealing
    best_positions = get_initial_config()
    if not is_valid(best_positions):
        # Fallback if init fails
        best_positions = np.array([[i * 2.0, 0.0] for i in range(N)])
        
    current_positions = best_positions.copy()
    current_side = required_outer_side(current_positions)
    best_side = current_side
    
    T = 0.5 # Initial temperature
    T_min = 1e-5
    alpha = 0.9995 # Cooling rate
    
    steps = 0
    max_steps = 500000
    
    while time.time() < deadline and steps < max_steps:
        if time.time() > deadline - 5.0:
            break
            
        # Choose a random hexagon to move
        i = np.random.randint(N)
        
        # Generate a random move
        # Step size should decrease with temperature? Or fixed small step?
        # Let's use a step size proportional to T
        step_size = 0.1 * T + 0.01
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        
        new_positions = current_positions.copy()
        new_positions[i, 0] += dx
        new_positions[i, 1] += dy
        
        # Check validity
        if not is_valid(new_positions):
            # Reject invalid moves. 
            # In standard SA, we might accept with low prob, but for hard constraints, rejection is safer.
            # However, to escape local minima, we might need to allow some "tunneling" or larger moves.
            # Since N is small, let's stick to rejection for now to maintain validity.
            steps += 1
            continue
            
        new_side = required_outer_side(new_positions)
        delta = new_side - current_side
        
        # Acceptance probability
        if delta < 0:
            accept = True
        else:
            # Boltzmann factor
            # Avoid overflow
            if T < 1e-10:
                accept = False
            else:
                prob = np.exp(-delta / T)
                accept = np.random.random() < prob
                
        if accept:
            current_positions = new_positions
            current_side = new_side
            if current_side < best_side:
                best_side = current_side
                best_positions = current_positions.copy()
                
        T *= alpha
        steps += 1
        
    # Final validation
    if not is_valid(best_positions):
        # This shouldn't happen if we only accept valid moves, but just in case
        # Fall back to a known valid config
        fallback = np.array([[i * 2.0, 0.0] for i in range(N)])
        if is_valid(fallback):
            best_positions = fallback
            best_side = required_outer_side(best_positions)
        else:
            # Last resort: spread out
            best_positions = np.array([[i * 3.0, 0.0] for i in range(N)])
            best_side = required_outer_side(best_positions)

    final_side = required_outer_side(best_positions)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), 0.0])
        
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
