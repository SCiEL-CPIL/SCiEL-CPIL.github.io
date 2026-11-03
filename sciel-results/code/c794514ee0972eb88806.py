import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def compute_min_enclosing_hexagon(centers, angles):
        """
        Finds the smallest regular hexagon (side length S) that contains all 
        unit hexagons defined by centers and angles.
        Returns (cx, cy, angle_deg, side_length)
        """
        # Precompute vertices of inner hexagons
        all_verts = []
        for i in range(len(centers)):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        def objective_outer(params):
            cx, cy, theta_deg = params
            theta = math.radians(theta_deg)
            # Normals of the outer hexagon faces
            # A regular hexagon with 'pointy top' (vertex at 90 deg) has normals at 0, 60, 120...
            # Let's define the outer hexagon by its center, rotation, and side length.
            # The distance from center to face (apothem) is S * sqrt(3) / 2.
            # We want to minimize S, which is equivalent to minimizing the max projection 
            # onto the normals, given that the hexagon is centered and rotated.
            
            max_proj = 0.0
            for k in range(6):
                # Normal direction for face k
                # If rotation is 0, vertices are at 0, 60... degrees. 
                # Normals are at 30, 90... degrees? 
                # Let's check: Vertex at angle phi. Edge between phi and phi+60.
                # Midpoint angle is phi+30. Normal is perpendicular to edge.
                # For a regular hexagon, the normal to the face is along the direction of the vertex of the dual triangle.
                # If vertices are at angles alpha + i*60, the face normals are at alpha + 30 + i*60.
                normal_angle = theta + math.pi/6 + k * math.pi / 3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                
                # Project all vertices onto this normal
                # Relative to center
                projs = (all_verts[:, 0] - cx) * nx + (all_verts[:, 1] - cy) * ny
                max_p = np.max(projs)
                if max_p > max_proj:
                    max_proj = max_p
            
            # Side length S is related to apothem a by a = S * sqrt(3) / 2
            # So S = 2 * a / sqrt(3)
            # Here max_proj is the max distance from center to a face normal line.
            # This must be >= apothem. To minimize S, we set apothem = max_proj.
            # Wait, the hexagon must *contain* the points.
            # So for all normals, the projection of any point must be <= apothem.
            # Thus, apothem = max over all normals and all points of the projection.
            # S = 2 * apothem / sqrt(3)
            
            return 2.0 * max_proj / math.sqrt(3)

        # Initial guess: center of mass, rotation 0
        cx0 = np.mean([c[0] for c in centers])
        cy0 = np.mean([c[1] for c in centers])
        
        best_val = float('inf')
        best_params = (cx0, cy0, 0.0)
        
        # Grid search for rotation first (coarse)
        for theta_guess in np.linspace(0, 60, 13):
            res = minimize(objective_outer, [cx0, cy0, theta_guess], method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        
        # Refine
        res = minimize(objective_outer, best_params, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
            
        return best_params[0], best_params[1], best_params[2], best_val

    def simulate_packing(initial_centers, initial_angles, time_limit):
        """
        Runs a soft-body simulation to pack hexagons.
        Returns optimal centers and angles.
        """
        # State: [x0, y0, a0, x1, y1, a1, ...]
        state = []
        for i in range(N):
            state.append(initial_centers[i][0])
            state.append(initial_centers[i][1])
            state.append(initial_angles[i])
        state = np.array(state)
        
        velocities = np.zeros_like(state)
        
        # Simulation parameters
        dt = 0.01
        damping = 0.9
        repulsion_strength = 50.0
        attraction_strength = 0.5
        max_steps = int(time_limit / dt)
        
        # Precompute vertices indices for fast access? 
        # It's easier to just compute on the fly for 12 hexagons.
        
        def get_state_verts(state):
            verts = []
            for i in range(N):
                cx = state[i*3]
                cy = state[i*3+1]
                ang = state[i*3+2]
                v = get_hex_vertices(cx, cy, 1.0, ang)
                verts.append(v)
            return verts
        
        def compute_forces(state):
            verts = get_state_verts(state)
            forces = np.zeros_like(state)
            
            # 1. Repulsion between pairs
            for i in range(N):
                for j in range(i+1, N):
                    v1 = verts[i]
                    v2 = verts[j]
                    
                    # Simple overlap check: min distance between vertices
                    # For speed, we can use center distance as a proxy for "close"
                    dx = state[i*3] - state[j*3]
                    dy = state[i*3+1] - state[j*3+1]
                    dist_sq = dx*dx + dy*dy
                    min_dist = math.sqrt(3) # 1.732
                    if dist_sq < (min_dist * 1.5) ** 2:
                        # Detailed check
                        # Approximate repulsion force based on center distance
                        # This is a "soft" body approach.
                        dist = math.sqrt(dist_sq)
                        if dist < 1e-6:
                            # Random kick
                            fx = random.uniform(-1, 1)
                            fy = random.uniform(-1, 1)
                            fa = random.uniform(-1, 1)
                        else:
                            # Force magnitude increases as distance decreases
                            # Overlap depth
                            overlap = min_dist - dist
                            if overlap > 0:
                                # Normalize direction
                                nx = dx / dist
                                ny = dy / dist
                                # Force on i is +n, on j is -n
                                f_mag = repulsion_strength * overlap
                                forces[i*3] += f_mag * nx
                                forces[i*3+1] += f_mag * ny
                                forces[j*3] -= f_mag * nx
                                forces[j*3+1] -= f_mag * ny
                                
                                # Torque: try to rotate to reduce overlap
                                # Calculate cross product of vector from i to j and relative orientation?
                                # Simpler: add random small torque if overlapping
                                forces[i*3+2] += random.uniform(-0.1, 0.1)
                                forces[j*3+2] += random.uniform(-0.1, 0.1)
            
            # 2. Attraction to center
            for i in range(N):
                cx = state[i*3]
                cy = state[i*3+1]
                # Force towards (0,0)
                forces[i*3] += -attraction_strength * cx
                forces[i*3+1] += -attraction_strength * cy
                # Damping on rotation to keep them aligned if possible?
                # Or just let them rotate freely.
                
            return forces

        start_time = time.time()
        for step in range(max_steps):
            if time.time() > deadline - 10:
                break
                
            forces = compute_forces(state)
            
            # Update velocities
            velocities = velocities * damping + forces * dt
            # Limit velocity
            max_vel = 0.5
            velocities = np.clip(velocities, -max_vel, max_vel)
            
            # Update positions
            state += velocities * dt
            
            # Optional: Add small noise to escape local minima
            if step % 100 == 0:
                noise = np.random.normal(0, 0.01, state.shape)
                state += noise
                
        return state

    def is_valid(centers, angles):
        verts = []
        for i in range(N):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
            
        eps = 1e-7
        for i in range(N):
            for j in range(i+1, N):
                v1 = verts[i]
                v2 = verts[j]
                # Separation Axis Theorem (SAT) for convex polygons
                # Check edges of v1
                found_sep = False
                for k in range(6):
                    p1 = v1[k]
                    p2 = v1[(k+1)%6]
                    edge = p2 - p1
                    nx = -edge[1]
                    ny = edge[0]
                    # Project
                    proj1 = np.dot(v1, [nx, ny])
                    proj2 = np.dot(v2, [nx, ny])
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    if max1 < min2 + eps or max2 < min1 + eps:
                        found_sep = True
                        break
                if found_sep:
                    continue
                # Check edges of v2
                for k in range(6):
                    p1 = v2[k]
                    p2 = v2[(k+1)%6]
                    edge = p2 - p1
                    nx = -edge[1]
                    ny = edge[0]
                    proj1 = np.dot(v1, [nx, ny])
                    proj2 = np.dot(v2, [nx, ny])
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    if max1 < min2 + eps or max2 < min1 + eps:
                        found_sep = True
                        break
                if not found_sep:
                    return False
        return True

    best_side = float('inf')
    best_inner = None
    best_outer = None
    
    # Generate multiple initial configurations
    num_restarts = 10
    random.seed(42)
    
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # 1. Lattice start (compressed)
        if r == 0:
            d = math.sqrt(3) * 0.98
            u = np.array([d, 0.0])
            v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
            centers_init = [np.array([0.0, 0.0])]
            ring1 = [u, v, v-u, -u, -v, u-v]
            for vec in ring1:
                centers_init.append(vec.copy())
            # Add 5 more from ring 2
            ring2_cands = []
            basis = [u, v, v-u, -u, -v, u-v]
            for p1 in ring1:
                for b in basis:
                    p2 = p1 + b
                    if not np.allclose(p2, [0,0]) and not any(np.allclose(p2, p1j) for p1j in ring1):
                        if not any(np.allclose(p2, pc) for pc in ring2_cands):
                            ring2_cands.append(p2)
            ring2_cands.sort(key=lambda p: np.linalg.norm(p))
            for i in range(5):
                centers_init.append(ring2_cands[i].copy())
            angles_init = [30.0] * 12
        else:
            # Random start in a circle
            centers_init = []
            for i in range(N):
                # Random position in a hexagon-like shape
                r = random.uniform(0, 2.5)
                theta = random.uniform(0, 2*math.pi)
                centers_init.append([r * math.cos(theta), r * math.sin(theta)])
            angles_init = [random.uniform(0, 60) for _ in range(N)]
            
        # Run simulation
        sim_time = 10.0
        if r > 0:
            sim_time = 5.0 # Shorter for random starts
        
        final_state = simulate_packing(centers_init, angles_init, sim_time)
        
        # Extract results
        centers_final = []
        angles_final = []
        for i in range(N):
            centers_final.append([final_state[i*3], final_state[i*3+1]])
            angles_final.append(final_state[i*3+2])
            
        # Validate
        if not is_valid(centers_final, angles_final):
            # Try to repair by a quick local optimization with hard constraints?
            # Or just skip this restart.
            # Let's try a quick Nelder-Mead on the final state to fix overlaps
            def obj_repair(params):
                cs = []
                angs = []
                for i in range(N):
                    cs.append([params[i*3], params[i*3+1]])
                    angs.append(params[i*3+2])
                verts = []
                for i in range(N):
                    verts.append(get_hex_vertices(cs[i][0], cs[i][1], 1.0, angs[i]))
                penalty = 0.0
                for i in range(N):
                    for j in range(i+1, N):
                        # Simple center distance penalty
                        dx = cs[i][0] - cs[j][0]
                        dy = cs[i][1] - cs[j][1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < math.sqrt(3):
                            penalty += (math.sqrt(3) - dist)**2
                # Attraction to center
                for c in cs:
                    penalty += 0.1 * (c[0]**2 + c[1]**2)
                return penalty
            
            res = minimize(obj_repair, final_state, method='Nelder-Mead', options={'maxiter': 500})
            final_state = res.x
            centers_final = []
            angles_final = []
            for i in range(N):
                centers_final.append([final_state[i*3], final_state[i*3+1]])
                angles_final.append(final_state[i*3+2])
                
            if not is_valid(centers_final, angles_final):
                continue
                
        # Compute outer hexagon
        ocx, ocy, otheta, side = compute_min_enclosing_hexagon(centers_final, angles_final)
        
        if side < best_side:
            best_side = side
            best_inner = [[c[0], c[1], a] for c, a in zip(centers_final, angles_final)]
            best_outer = (ocx, ocy, side, otheta)
            
    # Fallback if simulation failed
    if best_inner is None:
        # Use the initial lattice from r=0
        d = math.sqrt(3)
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        centers_init = [np.array([0.0, 0.0])]
        ring1 = [u, v, v-u, -u, -v, u-v]
        for vec in ring1:
            centers_init.append(vec.copy())
        ring2_cands = []
        basis = [u, v, v-u, -u, -v, u-v]
        for p1 in ring1:
            for b in basis:
                p2 = p1 + b
                if not np.allclose(p2, [0,0]) and not any(np.allclose(p2, p1j) for p1j in ring1):
                    if not any(np.allclose(p2, pc) for pc in ring2_cands):
                        ring2_cands.append(p2)
        ring2_cands.sort(key=lambda p: np.linalg.norm(p))
        for i in range(5):
            centers_init.append(ring2_cands[i].copy())
        angles_init = [30.0] * 12
        ocx, ocy, otheta, side = compute_min_enclosing_hexagon(centers_init, angles_init)
        best_inner = [[c[0], c[1], a] for c, a in zip(centers_init, angles_init)]
        best_outer = (ocx, ocy, side, otheta)
        best_side = side

    inner = best_inner
    outer_center, outer_side, outer_angle = best_outer[0], best_outer[1], best_outer[3]
    
    return inner, (outer_center[0], outer_center[1]), outer_side, outer_angle

# EVOLVE-BLOCK-END
