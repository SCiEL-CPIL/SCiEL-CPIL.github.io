import math
import time
import numpy as np

N = 12
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def time_left():
        return deadline - time.time()

    # Precompute unit hexagon vertices (pointy-top, vertices at 0,60,...)
    unit_verts = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def hex_verts(cx, cy, side, angle_deg):
        a = math.radians(angle_deg)
        c, s = math.cos(a), math.sin(a)
        rot = np.array([[c, -s], [s, c]])
        return unit_verts @ rot.T * side + np.array([cx, cy])

    def all_verts(positions, angles, side=1.0):
        """Vectorized: positions (N,2), angles (N,) -> (N,6,2)"""
        n = len(positions)
        verts = np.zeros((n, 6, 2))
        for i in range(n):
            a = math.radians(angles[i])
            c, s = math.cos(a), math.sin(a)
            rot = np.array([[c, -s], [s, c]])
            verts[i] = unit_verts @ rot.T * side + positions[i]
        return verts

    def min_outer_side(positions, angles, outer_angle=0.0):
        """Minimum side length of regular hexagon at origin with rotation outer_angle containing all hexes."""
        theta = math.radians(outer_angle)
        # 6 outward normals of the container hexagon
        normals = np.array([[math.cos(theta + math.pi/6 + k*math.pi/3),
                             math.sin(theta + math.pi/6 + k*math.pi/3)] for k in range(6)])
        # For each inner hex, compute max projection onto each normal
        # Support function of a regular hexagon of side s at angle phi: s*(cos(phi) + cos(phi-60°))... 
        # Actually for a regular hexagon with vertices at angles phi, phi+60, ..., the support in direction u is:
        # max over vertices of (v . u). For unit hex with vertices at 0,60,...:
        # support(u) = max_k cos(angle(u) - k*60°)
        # For side length s: s * max_k cos(angle(u) - k*60°)
        # The max of cos over 6 directions spaced 60° apart is: cos(min_angle_diff) where min_angle_diff is in [0,30°]
        # So support = s * cos(min_k |angle(u) - k*60°| mod 360, wrapped to [0,180])
        
        # Simpler: just compute vertex projections
        verts = all_verts(positions, angles, 1.0)  # (N,6,2)
        # Project all vertices onto each normal: (N,6,6)
        projs = np.einsum('nki,kj->nij', verts, normals)  # (N, 6, 6)
        max_projs = projs.max(axis=1)  # (N, 6)
        global_max = max_projs.max()
        # For a regular hexagon, the distance from center to edge (apothem) = side * sqrt(3)/2
        # So side = global_max / (sqrt(3)/2) = 2*global_max/sqrt(3)
        return 2.0 * global_max / SQRT3

    def separation_distance(v1, v2, a1, a2, side=1.0):
        """
        Compute the minimum distance between two hexagons using SAT.
        Returns positive if separated, 0 if touching, negative if overlapping.
        v1,v2: centers; a1,a2: angles in degrees.
        """
        verts1 = hex_verts(v1[0], v1[1], side, a1)
        verts2 = hex_verts(v2[0], v2[1], side, a2)
        
        # Get edge normals (12 axes)
        axes = []
        for poly in [verts1, verts2]:
            for i in range(6):
                edge = poly[(i+1)%6] - poly[i]
                n = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(n)
                if norm > 1e-12:
                    axes.append(n / norm)
        
        min_sep = float('inf')
        for axis in axes:
            p1 = verts1 @ axis
            p2 = verts2 @ axis
            sep = max(np.min(p2) - np.max(p1), np.min(p1) - np.max(p2))
            if sep < min_sep:
                min_sep = sep
            if min_sep > 1e-9:
                break  # Already separated on this axis
        
        return min_sep

    def total_overlap_penalty(positions, angles, eps=1e-4):
        """Sum of squared penetration depths for all pairs."""
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                sep = separation_distance(positions[i], positions[j], angles[i], angles[j])
                if sep < 0:
                    penalty += (sep - eps) ** 2
        return penalty

    def is_valid(positions, angles, tol=1e-6):
        for i in range(N):
            for j in range(i+1, N):
                sep = separation_distance(positions[i], positions[j], angles[i], angles[j])
                if sep < -tol:
                    return False
        return True

    # === Phase 1: Rigid-body dynamics with shrinking container ===
    # Initialize with honeycomb lattice
    lattice = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * 1.5
            y = i * SQRT3 / 2.0 + j * SQRT3
            lattice.append((x, y))
    lattice.sort(key=lambda p: p[0]**2 + p[1]**2)
    pos0 = np.array(lattice[:N], dtype=float)
    pos0 -= pos0.mean(axis=0)
    ang0 = np.zeros(N)

    def simulate_shrink(pos_init, ang_init, dt=0.005, n_steps=5000):
        """
        Simulate 12 hexagons with repulsive forces inside a shrinking hexagonal container.
        Returns (positions, angles, container_side).
        """
        pos = pos_init.copy().astype(float)
        ang = ang_init.copy().astype(float)
        vel = np.zeros_like(pos)
        ang_vel = np.zeros_like(ang)
        
        # Container starts large and shrinks
        container_side = min_outer_side(pos, ang) * 1.3
        target_side = 3.5  # Will be refined later
        
        for step in range(n_steps):
            if time.time() > deadline - 30:
                break
            
            # Gradually shrink container
            progress = step / n_steps
            container_side = container_side * (1.0 - 0.001) + target_side * 0.001
            
            forces = np.zeros_like(pos)
            torques = np.zeros_like(ang)
            
            # Pairwise repulsion (hard-core)
            for i in range(N):
                for j in range(i+1, N):
                    sep = separation_distance(pos[i], pos[j], ang[i], ang[j])
                    if sep < 0.1:  # Close or overlapping
                        # Repulsive force
                        strength = 50.0 * (0.1 - sep) ** 2
                        direction = pos[j] - pos[i]
                        dist = np.linalg.norm(direction)
                        if dist < 1e-9:
                            direction = np.array([1.0, 0.0])
                            dist = 1.0
                        direction /= dist
                        forces[i] -= strength * direction
                        forces[j] += strength * direction
                        # Torque to rotate apart (simplified)
                        torque_mag = 5.0 * (0.1 - sep)
                        torques[i] += torque_mag * 0.1
                        torques[j] -= torque_mag * 0.1
            
            # Container repulsion
            theta = 0.0
            normals = np.array([[math.cos(math.pi/6 + k*math.pi/3),
                                 math.sin(math.pi/6 + k*math.pi/3)] for k in range(6)])
            apothem = container_side * SQRT3 / 2.0
            
            for i in range(N):
                verts = hex_verts(pos[i][0], pos[i][1], 1.0, ang[i])
                for k in range(6):
                    proj = np.max(verts @ normals[k])
                    if proj > apothem - 0.1:
                        strength = 100.0 * (proj - (apothem - 0.1)) ** 2
                        forces[i] -= strength * normals[k]
                        # Torque
                        center_proj = pos[i] @ normals[k]
                        torque = strength * (proj - center_proj) * 0.05
                        torques[i] += torque
            
            # Damping
            damping = 0.95
            vel = vel * damping + forces * dt
            ang_vel = ang_vel * damping + torques * dt
            
            pos += vel * dt
            ang += ang_vel * dt
            ang = np.mod(ang, 360.0)
        
        return pos, ang, container_side

    # Run dynamics simulation
    pos_dyn, ang_dyn, side_dyn = simulate_shrink(pos0, ang0, dt=0.005, n_steps=8000)

    # === Phase 2: Local optimization with SLSQP ===
    from scipy.optimize import minimize

    def objective(params):
        if time.time() > deadline - 10:
            return 1e6
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        side = min_outer_side(positions, angles, 0.0)
        penalty = total_overlap_penalty(positions, angles)
        return side + 10.0 * penalty

    # Multiple starting points
    best_params = None
    best_val = float('inf')
    
    start_points = [
        pos0,
        pos_dyn,
        pos0 + np.random.normal(0, 0.3, pos0.shape),
        pos_dyn + np.random.normal(0, 0.3, pos_dyn.shape),
    ]
    start_angles = [
        ang0,
        ang_dyn,
        np.random.uniform(0, 60, N),
        ang_dyn + np.random.normal(0, 10, N),
    ]

    for sp, sa in zip(start_points, start_angles):
        if time.time() > deadline - 20:
            break
        x0 = np.concatenate([sp.flatten(), sa])
        try:
            res = minimize(objective, x0, method='SLSQP',
                          options={'maxiter': 500, 'ftol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

    # === Phase 3: Nelder-Mead polish ===
    if best_params is not None and time.time() < deadline - 15:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

    # === Phase 4: Refine outer hexagon rotation ===
    if best_params is not None:
        positions = best_params[:2*N].reshape(N, 2)
        angles = best_params[2*N:]
        
        # Try different outer rotations
        best_side = float('inf')
        best_outer_ang = 0.0
        for oa in range(0, 30, 2):
            s = min_outer_side(positions, angles, float(oa))
            if s < best_side:
                best_side = s
                best_outer_ang = float(oa)
        
        # Fine-tune
        for oa in np.arange(best_outer_ang - 2, best_outer_ang + 2, 0.5):
            s = min_outer_side(positions, angles, float(oa))
            if s < best_side:
                best_side = s
                best_outer_ang = float(oa)
        
        # Final check
        if not is_valid(positions, angles):
            # Fallback
            positions = pos0
            angles = ang0
            best_side = min_outer_side(positions, angles, 0.0)
            best_outer_ang = 0.0
    else:
        positions = pos0
        angles = ang0
        best_side = min_outer_side(positions, angles, 0.0)
        best_outer_ang = 0.0

    inner = []
    for i in range(N):
        inner.append([float(positions[i][0]), float(positions[i][1]), float(angles[i])])

    return inner, (0.0, 0.0), float(best_side), float(best_outer_ang)
