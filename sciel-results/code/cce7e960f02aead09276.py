# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _hex_vertices(cx, cy, s, angle_deg):
    angles = np.radians(angle_deg) + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + s * np.cos(angles),
        cy + s * np.sin(angles)
    ])

def _all_vertices(positions, angles, side=1.0):
    n = len(positions)
    verts = np.zeros((n, 6, 2))
    base_angles = np.radians(angles)[:, None] + np.radians(60) * np.arange(6)[None, :]
    verts[:, :, 0] = positions[:, 0:1] + side * np.cos(base_angles)
    verts[:, :, 1] = positions[:, 1:2] + side * np.sin(base_angles)
    return verts

def _check_separation_fast(positions, angles, tol=1e-6):
    n = len(positions)
    all_verts = _all_vertices(positions, angles)
    hex_normals = np.zeros((n, 6, 2))
    base = np.radians(angles)[:, None] + np.radians(30) + np.radians(60) * np.arange(6)[None, :]
    hex_normals[:, :, 0] = np.cos(base)
    hex_normals[:, :, 1] = np.sin(base)

    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]
            normals = np.vstack([hex_normals[i], hex_normals[j]])
            proj_i = vi @ normals.T
            proj_j = vj @ normals.T
            max_i = np.max(proj_i, axis=0)
            min_i = np.min(proj_i, axis=0)
            max_j = np.max(proj_j, axis=0)
            min_j = np.min(proj_j, axis=0)
            if np.any(max_i < min_j - tol) or np.any(max_j < min_i - tol):
                continue
            return False
    return True

def _min_enclosing_side(positions, angles):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    normals = np.array([[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)])
    apothem_factor = np.sqrt(3) / 2
    projs = all_verts @ normals.T
    max_proj = np.max(projs)
    return max_proj / apothem_factor

def _check_containment_fast(positions, angles, side, tol=-1e-6):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    outer_verts = _hex_vertices(0.0, 0.0, side, 0.0)
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        e = p2 - p1
        v = all_verts - p1
        cross = e[0] * v[:, 1] - e[1] * v[:, 0]
        if np.min(cross) < tol:
            return False
    return True

def _simulate_packing(start_time, deadline, seed):
    rng = np.random.RandomState(seed)
    
    # 11 hexagons
    N = 11
    # Initial positions: roughly a hexagonal cluster
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    velocities = np.zeros((N, 2))
    angular_velocities = np.zeros(N)
    
    # Place center at 0,0
    positions[0] = [0, 0]
    # Place 6 around center at distance ~2 (touching)
    for k in range(6):
        a = np.radians(30 + 60 * k)
        positions[k+1] = [2.0 * np.cos(a), 2.0 * np.sin(a)]
    # Place remaining 4 further out
    for k in range(4):
        a = np.radians(0 + 60 * k)
        positions[7+k] = [3.5 * np.cos(a), 3.5 * np.sin(a)]
    
    # Add some randomness to break symmetry
    positions += rng.uniform(-0.1, 0.1, (N, 2))
    angles += rng.uniform(-10, 10, N)
    velocities += rng.uniform(-0.1, 0.1, (N, 2))
    angular_velocities += rng.uniform(-0.1, 0.1, N)
    
    dt = 0.01
    repulsion_strength = 100.0
    center_pull_strength = 0.5
    damping = 0.98
    angular_damping = 0.95
    
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_side = _min_enclosing_side(positions, angles)
    
    steps = 0
    while time.time() < deadline - 10:
        # Compute forces
        forces = np.zeros((N, 2))
        torques = np.zeros(N)
        
        for i in range(N):
            for j in range(i+1, N):
                # Check separation
                vi = _all_vertices(positions[i:i+1], angles[i:i+1])[0]
                vj = _all_vertices(positions[j:j+1], angles[j:j+1])[0]
                
                # Compute min distance between edges? 
                # Approximation: Use center distance and a "soft" potential based on overlap
                # For hexagons, if centers are closer than ~2*apothem (sqrt(3)), they might overlap.
                # But orientation matters.
                # Let's use a simpler heuristic: if centers are close, repel.
                # Actually, let's use the vertex overlap check for force direction?
                # Too slow.
                
                # Use center distance with a "radius" estimate
                # Effective radius of hexagon is between apothem (sqrt(3)/2 ~ 0.866) and circumradius (1.0)
                # If dist < 2 * 0.9, they are likely overlapping or close.
                r_eff = 0.9
                min_dist = 2 * r_eff
                
                d = positions[j] - positions[i]
                dist = np.linalg.norm(d)
                
                if dist < min_dist and dist > 1e-6:
                    # Repulsive force
                    # Quadratic potential
                    overlap = min_dist - dist
                    f_mag = repulsion_strength * overlap
                    f_dir = d / dist
                    forces[i] -= f_mag * f_dir
                    forces[j] += f_mag * f_dir
                    
                    # Torque to align?
                    # Simple heuristic: try to align edges by rotating towards each other?
                    # This is tricky. Let's just let them slide.
                    
                # Center pull
                forces[i] -= center_pull_strength * positions[i]
                forces[j] -= center_pull_strength * positions[j]
                
        # Update velocities and positions
        velocities += forces * dt
        velocities *= damping
        angular_velocities *= angular_damping
        
        positions += velocities * dt
        angles += angular_velocities * dt
        
        # Check validity and update best
        if _check_separation_fast(positions, angles):
            side = _min_enclosing_side(positions, angles)
            if side < best_side:
                best_side = side
                best_positions = positions.copy()
                best_angles = angles.copy()
                
        # Occasionally re-center to prevent drift
        if steps % 100 == 0:
            center = np.mean(positions, axis=0)
            positions -= center
            
        steps += 1
        
    return best_positions, best_angles, best_side

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 620.0
    
    best_positions = None
    best_angles = None
    best_side = float('inf')
    
    # Run multiple simulations with different seeds
    # Budget: ~100s per simulation, 5 simulations
    num_sims = 5
    sim_time_budget = (deadline - start_time - 60) / num_sims
    
    for i in range(num_sims):
        if time.time() > deadline - 60:
            break
            
        sim_deadline = time.time() + sim_time_budget
        pos, ang, side = _simulate_packing(start_time, sim_deadline, seed=i*100)
        
        if side < best_side:
            best_side = side
            best_positions = pos
            best_angles = ang
            
    # Fallback if all failed
    if best_positions is None:
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k)
            positions[k + 1] = [2 * np.cos(a), 2 * np.sin(a)]
        for k in range(4):
            a = np.radians(0 + 60 * k)
            positions[7 + k] = [3.5 * np.cos(a), 3.5 * np.sin(a)]
        best_positions = positions
        best_angles = angles
        best_side = _min_enclosing_side(best_positions, best_angles)
        
    # Final verification
    if not _check_separation_fast(best_positions, best_angles):
        # If dynamics failed to find a valid state, use the seed configuration
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k)
            positions[k + 1] = [2 * np.cos(a), 2 * np.sin(a)]
        for k in range(4):
            a = np.radians(0 + 60 * k)
            positions[7 + k] = [3.5 * np.cos(a), 3.5 * np.sin(a)]
        best_positions = positions
        best_angles = angles
        best_side = _min_enclosing_side(best_positions, best_angles)
        
    # Containment check
    if not _check_containment_fast(best_positions, best_angles, best_side):
        best_side *= 1.001
        
    inner = np.column_stack([best_positions, best_angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    side_length = float(best_side)
    
    return inner, outer_hex_data, side_length

# EVOLVE-BLOCK-END
