"""Combined program: hexagonal lattice packing + Nelder-Mead graph layout optimization."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 135.0  # 150s limit, keep 15s margin

    # Lattice vectors for hexagonal packing
    # Distance between centers of touching unit hexagons is sqrt(3)
    eps = 0.005
    d = math.sqrt(3) + eps

    u = np.array([d, 0.0])
    v = np.array([d/2.0, d*math.sqrt(3)/2.0])

    # Build a cluster of 12 hexagons using ring-based construction
    centers = [np.array([0.0, 0.0])]

    # Ring 1: 6 neighbors of center
    basis = [u, v, v - u, -u, -v, u - v]
    ring1_vectors = [vec.copy() for vec in basis]
    centers.extend(ring1_vectors)

    # Ring 2: find neighbors of ring1 that are not in ring0 or ring1
    ring2_candidates = []
    for i in range(6):
        p1 = ring1_vectors[i]
        for b in basis:
            p2 = p1 + b
            if np.allclose(p2, np.array([0.0, 0.0]), atol=eps*0.5):
                continue
            is_r1 = False
            for p1j in ring1_vectors:
                if np.allclose(p2, p1j, atol=eps*0.5):
                    is_r1 = True
                    break
            if is_r1:
                continue
            duplicate = False
            for pc in ring2_candidates:
                if np.allclose(p2, pc, atol=eps*0.5):
                    duplicate = True
                    break
            if not duplicate:
                ring2_candidates.append(p2.copy())

    ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
    for i in range(min(5, len(ring2_candidates))):
        centers.append(ring2_candidates[i].copy())

    while len(centers) < N:
        if len(ring2_candidates) > len(centers) - 7:
            idx = len(centers) - 7
            if idx < len(ring2_candidates):
                centers.append(ring2_candidates[idx].copy())
            else:
                break
        else:
            angle = random.uniform(0, 2*math.pi)
            dist = d * 2.5
            new_c = np.array([dist*math.cos(angle), dist*math.sin(angle)])
            centers.append(new_c)
        if len(centers) > 50:
            break

    centers = centers[:N]

    # Initial configuration: all hexagons at 30 degrees (flat-top)
    init_positions = np.array([[float(c[0]), float(c[1])] for c in centers])
    init_angles = np.array([30.0] * N)

    def get_hex_vertices(cx, cy, side, angle_deg):
        """Get vertices of a regular hexagon."""
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def get_hex_vertices_vec(positions, angles, side=1.0):
        """Vectorized: get all vertices for all hexagons. Returns (N*6, 2) array."""
        n = len(positions)
        verts = np.zeros((n * 6, 2))
        for i in range(n):
            cx, cy = positions[i]
            angle = math.radians(angles[i])
            for k in range(6):
                theta = angle + k * math.pi / 3
                verts[i * 6 + k, 0] = cx + side * math.cos(theta)
                verts[i * 6 + k, 1] = cy + side * math.sin(theta)
        return verts

    def compute_outer_side_fast(positions, angles, side=1.0):
        """Compute minimum outer hexagon side length for given inner hexagons."""
        all_verts = get_hex_vertices_vec(positions, angles, side)
        
        # Use centroid of all vertices as center
        cx = all_verts[:, 0].mean()
        cy = all_verts[:, 1].mean()
        
        best_s = float('inf')
        
        # Try outer hexagon rotations from 0 to 60 degrees
        for theta_deg in range(0, 61, 3):
            theta = math.radians(theta_deg)
            max_proj = 0.0
            for k in range(6):
                normal_angle = theta + math.pi/6 + k * math.pi/3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                projs = (all_verts[:, 0] - cx) * nx + (all_verts[:, 1] - cy) * ny
                mp = projs.max()
                if mp > max_proj:
                    max_proj = mp
            s = 2.0 * max_proj / math.sqrt(3)
            if s < best_s:
                best_s = s
        
        return best_s, (cx, cy), theta_deg

    def check_disjoint_fast(positions, angles, tol=1e-4):
        """Check disjointness using vertex-in-hexagon test."""
        n = len(positions)
        all_verts = get_hex_vertices_vec(positions, angles, 1.0)
        sqrt3 = math.sqrt(3)
        
        for i in range(n):
            xi, yi = positions[i]
            ai_rad = math.radians(angles[i])
            cos_ai = math.cos(ai_rad)
            sin_ai = math.sin(ai_rad)
            vi = all_verts[i*6:(i+1)*6]
            
            for j in range(i + 1, n):
                xj, yj = positions[j]
                aj_rad = math.radians(angles[j])
                cos_aj = math.cos(aj_rad)
                sin_aj = math.sin(aj_rad)
                vj = all_verts[j*6:(j+1)*6]
                
                # Check vertices of i inside j
                for vx, vy in vi:
                    dx = vx - xj
                    dy = vy - yj
                    lx = dx * cos_aj + dy * sin_aj
                    ly = -dx * sin_aj + dy * cos_aj
                    for k in range(6):
                        normal_angle = math.radians(30 + 60 * k)
                        nx = math.cos(normal_angle)
                        ny = math.sin(normal_angle)
                        proj = lx * nx + ly * ny
                        if proj > sqrt3 / 2 + tol:
                            return False
                
                # Check vertices of j inside i
                for vx, vy in vj:
                    dx = vx - xi
                    dy = vy - yi
                    lx = dx * cos_ai + dy * sin_ai
                    ly = -dx * sin_ai + dy * cos_ai
                    for k in range(6):
                        normal_angle = math.radians(30 + 60 * k)
                        nx = math.cos(normal_angle)
                        ny = math.sin(normal_angle)
                        proj = lx * nx + ly * ny
                        if proj > sqrt3 / 2 + tol:
                            return False
        return True

    # Initial evaluation
    if check_disjoint_fast(init_positions, init_angles):
        outer_side, outer_center, outer_angle = compute_outer_side_fast(init_positions, init_angles)
    else:
        # Fallback
        init_positions = np.array([[4.0 * k, 0.0] for k in range(-(N // 2), N - N // 2)])
        init_angles = np.array([0.0] * N)
        outer_side, outer_center, outer_angle = compute_outer_side_fast(init_positions, init_angles)

    best_side = outer_side
    best_positions = init_positions.copy()
    best_angles = init_angles.copy()

    # Nelder-Mead optimization on (2N positions + N angles) = 3N parameters
    # But we fix angles to reduce dimensionality and improve convergence
    # Actually, let's optimize positions and angles together
    
    def objective(params):
        """Objective: outer hexagon side length + penalty for overlap."""
        if time.time() > deadline - 5:
            return 1e10
        
        n = N
        positions = params[:2*n].reshape(n, 2)
        angles = params[2*n:]
        
        # Check disjointness with soft penalty
        if not check_disjoint_fast(positions, angles, tol=1e-5):
            return 1e6
        
        side, _, _ = compute_outer_side_fast(positions, angles)
        return side

    # Initial parameter vector
    x0 = np.concatenate([init_positions.flatten(), init_angles])
    
    # Run Nelder-Mead
    if time.time() < deadline - 20:
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-4,
                    'fatol': 1e-4,
                    'adaptive': True
                }
            )
            
            if result.fun < best_side:
                best_params = result.x
                best_positions = best_params[:2*N].reshape(N, 2)
                best_angles = best_params[2*N:]
                best_side, outer_center, outer_angle = compute_outer_side_fast(best_positions, best_angles)
        except Exception:
            pass

    # Additional local refinement with smaller steps
    if time.time() < deadline - 10:
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            
            new_pos = best_positions.copy()
            new_ang = best_angles.copy()
            
            idx = random.randint(0, N - 1)
            new_pos[idx, 0] += random.uniform(-0.1, 0.1)
            new_pos[idx, 1] += random.uniform(-0.1, 0.1)
            new_ang[idx] += random.uniform(-5, 5)
            
            if check_disjoint_fast(new_pos, new_ang, tol=1e-5):
                s, c, a = compute_outer_side_fast(new_pos, new_ang)
                if s < best_side - 1e-6:
                    best_side = s
                    best_positions = new_pos
                    best_angles = new_ang
                    outer_center = c
                    outer_angle = a

    # Build final inner list
    inner = [[float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])] for i in range(N)]

    return inner, outer_center, best_side, outer_angle
# EVOLVE-BLOCK-END
