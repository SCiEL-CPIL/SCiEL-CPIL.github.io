import numpy as np
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # --- Geometry Helpers ---
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        """Calculate vertices of a regular hexagon."""
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def required_outer_side(positions, angles):
        """
        Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
        that contains all unit hexagons at given positions and angles.
        """
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        verts_flat = verts.reshape(-1, 2)
        
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        # The 6 support directions for a regular hexagon aligned with axes (angle 0)
        # Normals are at 30, 90, 150, 210, 270, 330 degrees
        # n_k = (cos(30+60k), sin(30+60k))
        # Projection of point p onto n_k is p.x * nx + p.y * ny
        # For a hexagon of side S centered at origin, max projection is S * sqrt(3) / 2
        # So S >= 2/sqrt(3) * max_projection
        
        # Let's compute projections onto the 6 normal directions of the OUTER hexagon
        # Outer hexagon angle is 0.
        # Normal directions:
        # 30 deg: (sqrt(3)/2, 1/2)
        # 90 deg: (0, 1)
        # 150 deg: (-sqrt(3)/2, 1/2)
        # 210 deg: (-sqrt(3)/2, -1/2)
        # 270 deg: (0, -1)
        # 330 deg: (sqrt(3)/2, -1/2)
        
        nx = np.array([np.sqrt(3)/2, 0, -np.sqrt(3)/2, -np.sqrt(3)/2, 0, np.sqrt(3)/2])
        ny = np.array([1/2, 1, 1/2, -1/2, -1, -1/2])
        
        # Project all vertices onto these 6 directions
        # proj_k = x * nx_k + y * ny_k
        proj = np.column_stack([x * nx, y * ny])
        
        max_proj = np.max(proj, axis=0)
        
        # The side length S must satisfy S * sqrt(3) / 2 >= max_proj_k for all k
        # So S >= 2/sqrt(3) * max(max_proj_k)
        side = (2.0 / np.sqrt(3.0)) * np.max(max_proj)
        
        return side

    def min_separation(v1, v2):
        """
        Compute the minimum distance between two convex polygons (hexagons).
        Returns 0 if they intersect or touch, positive if disjoint.
        Uses the fact that the Minkowski difference of two convex polygons is convex.
        Distance is the minimum distance from origin to the boundary of (v1 - v2) if origin is not inside.
        However, for speed and robustness in SA, we can use a simplified SAT-based distance or 
        just check intersection and use a penalty based on 'depth'.
        
        For Simulated Annealing, we need a smooth-ish penalty or a hard constraint.
        Let's use a robust intersection check and a distance estimate for penalty.
        
        A faster approach for hexagons:
        The distance between two regular hexagons can be approximated or calculated exactly.
        Given time constraints, we will use a robust SAT intersection check.
        If they intersect, we estimate penetration depth as the minimum overlap in any SAT axis.
        """
        # SAT Check
        # Get edges of v1
        min_sep = float('inf')
        is_intersecting = False
        
        # Check axes from v1
        for i in range(6):
            j = (i + 1) % 6
            edge = v1[j] - v1[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10: continue
            normal /= norm
            proj1 = v1.dot(normal)
            proj2 = v2.dot(normal)
            sep = np.min(proj2) - np.max(proj1)
            min_sep = min(min_sep, sep)
            if sep > 0:
                return sep # Disjoint along this axis, this is the distance
        
        # Check axes from v2
        for i in range(6):
            j = (i + 1) % 6
            edge = v2[j] - v2[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10: continue
            normal /= norm
            proj1 = v1.dot(normal)
            proj2 = v2.dot(normal)
            sep = np.min(proj2) - np.max(proj1)
            min_sep = min(min_sep, sep)
            if sep > 0:
                return sep
        
        # If we reached here, they intersect.
        # min_sep is the maximum of the "separations" which are negative.
        # The "penetration depth" is roughly -min_sep (but this is not exact geometric distance,
        # it's the overlap in the most overlapping direction).
        # For penalty, we want to push them apart.
        return min_sep # Negative value

    def compute_penalty(positions, angles):
        """
        Calculate total overlap penalty and outer side length.
        """
        side = required_outer_side(positions, angles)
        
        verts_list = [hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i]) for i in range(N)]
        
        penalty = 0.0
        # Only check pairs that are close to save time
        # Bounding box check
        for i in range(N):
            for j in range(i + 1, N):
                # Quick distance check between centers
                dist_sq = (positions[i,0]-positions[j,0])**2 + (positions[i,1]-positions[j,1])**2
                # Max distance for overlap is roughly 2 * circumradius = 2.0
                # If dist > 2.2, they definitely don't overlap
                if dist_sq > 2.2**2:
                    continue
                
                sep = min_separation(verts_list[i], verts_list[j])
                if sep < 0:
                    # Overlap
                    penalty += sep * sep # Quadratic penalty for depth
                    # Or penalty += -sep
                    # Quadratic is smoother for SA
        
        return side, penalty

    # --- Simulated Annealing ---
    
    def run_sa(initial_positions, initial_angles, time_budget):
        """
        Run Simulated Annealing.
        """
        current_pos = initial_positions.copy()
        current_ang = initial_angles.copy()
        
        current_side, current_pen = compute_penalty(current_pos, current_ang)
        current_cost = current_side + 100.0 * current_pen
        
        best_pos = current_pos.copy()
        best_ang = current_ang.copy()
        best_side = current_side
        best_cost = current_cost
        
        T = 1.0 # Initial temperature
        cooling = 0.995
        min_T = 1e-4
        
        while time.time() < start_time + time_budget and T > min_T:
            # Generate neighbor
            new_pos = current_pos.copy()
            new_ang = current_ang.copy()
            
            # Move a random hexagon
            idx = np.random.randint(N)
            # Random step size
            step = T * 0.5
            new_pos[idx, 0] += np.random.normal(0, step)
            new_pos[idx, 1] += np.random.normal(0, step)
            # Small angle perturbation
            new_ang[idx] += np.random.normal(0, step * 10)
            
            # Optional: Occasionally move two hexagons to escape local minima
            if np.random.rand() < 0.1:
                idx2 = np.random.randint(N)
                if idx2 != idx:
                    new_pos[idx2, 0] += np.random.normal(0, step)
                    new_pos[idx2, 1] += np.random.normal(0, step)
                    new_ang[idx2] += np.random.normal(0, step * 10)
                    
            new_side, new_pen = compute_penalty(new_pos, new_ang)
            new_cost = new_side + 100.0 * new_pen
            
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / T):
                current_pos = new_pos
                current_ang = new_ang
                current_side = new_side
                current_pen = new_pen
                current_cost = new_cost
                
                # Update best
                # We only update best if it's valid (penalty ~ 0) or if it's the best invalid one so far?
                # Ideally, we want a valid solution.
                if new_pen < 1e-6 and new_side < best_side:
                    best_pos = new_pos.copy()
                    best_ang = new_ang.copy()
                    best_side = new_side
                    best_cost = new_cost
                elif best_pen >= 1e-6 and new_cost < best_cost:
                    # If we haven't found a valid solution yet, track the best invalid one
                    best_pos = new_pos.copy()
                    best_ang = new_ang.copy()
                    best_side = new_side
                    best_cost = new_cost
                    best_pen = new_pen
                else:
                    # If we have a valid best, don't replace with invalid
                    pass
            
            T *= cooling
        
        return best_pos, best_ang, best_side

    # --- Initial Configurations ---
    
    def generate_hex_lattice():
        """
        Generate a compact hexagonal lattice arrangement.
        """
        coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
        ]
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i, (q, r) in enumerate(coords):
            # Hex grid to cartesian
            x = np.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            positions[i] = [x, y]
        return positions, angles

    def generate_circular():
        """
        Generate a circular ring arrangement.
        """
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            angle = 2 * np.pi * i / N
            r = 2.5
            positions[i, 0] = r * np.cos(angle)
            positions[i, 1] = r * np.sin(angle)
            angles[i] = angle * 180 / np.pi # Align hexagons radially?
        return positions, angles

    def generate_random():
        positions = np.random.uniform(-4, 4, (N, 2))
        angles = np.random.uniform(0, 60, N) # Hexagons have 6-fold symmetry, 0-60 is enough
        return positions, angles

    # --- Main Logic ---
    
    best_overall_side = float('inf')
    best_overall_pos = None
    best_overall_ang = None
    
    # List of initial configurations
    inits = []
    inits.append(generate_hex_lattice())
    inits.append(generate_circular())
    inits.append(generate_random())
    inits.append(generate_random())
    
    # Allocate time budget
    # 145s total. 
    # Let's give each SA run a portion of the time.
    # 4 runs * 30s = 120s. Leaves 25s for overhead/verification.
    time_per_run = 30.0
    
    for i, (pos, ang) in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
            
        # Run SA
        run_deadline = time.time() + time_per_run
        pos_res, ang_res, side_res = run_sa(pos, ang, run_deadline - time.time())
        
        # Verify the result
        if pos_res is not None:
            # Re-calculate to be sure
            final_side, final_pen = compute_penalty(pos_res, ang_res)
            if final_pen < 1e-5: # Valid
                if final_side < best_overall_side:
                    best_overall_side = final_side
                    best_overall_pos = pos_res
                    best_overall_ang = ang_res
            else:
                # If no valid solution found in this run, check if this invalid one is "less invalid" 
                # but we prefer valid. If we end up with only invalid, we return the best invalid?
                # No, the task requires disjoint. We must return a valid one.
                # If SA fails to find valid, we might need a fallback.
                pass

    # Fallback: If SA didn't find a valid solution (which shouldn't happen with these inits),
    # try a simple grid or spread out.
    if best_overall_pos is None:
        # Use the hex lattice, even if it overlaps? 
        # The lattice generated above might overlap if not scaled properly.
        # Let's check the lattice validity.
        pos_l, ang_l = generate_hex_lattice()
        side_l, pen_l = compute_penalty(pos_l, ang_l)
        
        if pen_l < 1e-5:
            best_overall_side = side_l
            best_overall_pos = pos_l
            best_overall_ang = ang_l
        else:
            # Spread them out manually to ensure validity, even if side is large
            best_overall_pos = np.zeros((N, 2))
            best_overall_ang = np.zeros(N)
            for i in range(N):
                best_overall_pos[i, 0] = i * 2.5
                best_overall_pos[i, 1] = 0
            best_overall_side, _ = compute_penalty(best_overall_pos, best_overall_ang)

    # Final Output
    inner = []
    for i in range(N):
        inner.append([float(best_overall_pos[i, 0]), float(best_overall_pos[i, 1]), float(best_overall_ang[i])])
        
    return inner, (0.0, 0.0), float(best_overall_side), 0.0

# EVOLVE-BLOCK-END
