# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    angs = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angs), cy + s * np.sin(angs)])

def _hex_edge_normals(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    angs = a + np.deg2rad(60 * np.arange(6))
    # Edge normals: perpendicular to edges, pointing outward
    # Edge i is between vertex i and vertex i+1
    # Normal direction is at angle a + 30 + 60*i
    normal_angs = a + np.deg2rad(30 + 60 * np.arange(6))
    return np.column_stack([np.cos(normal_angs), np.sin(normal_angs)])

def _hex_separation(A, B):
    """Compute minimum separation between two hexagons using edge normal method.
    Returns positive if separated, negative if overlapping."""
    # A and B are (6,2) vertex arrays
    # Check all 12 edge normals (6 from A, 6 from B)
    min_sep = np.inf
    for i in range(6):
        # Edge of A: vertices A[i], A[(i+1)%6]
        # Normal pointing outward from A
        e = A[(i+1)%6] - A[i]
        n = np.array([-e[1], e[0]])
        n = n / (np.linalg.norm(n) + 1e-15)
        # Project both hexagons onto this normal
        proj_A = A @ n
        proj_B = B @ n
        # For separation, we need max(proj_A) < min(proj_B) or vice versa
        # The gap is min(proj_B) - max(proj_A) if A is "below" B
        sep1 = np.min(proj_B) - np.max(proj_A)
        sep2 = np.min(proj_A) - np.max(proj_B)
        # The separation along this normal is the max of the two (positive means separated)
        min_sep = min(min_sep, max(sep1, sep2))
    for i in range(6):
        # Edge of B
        e = B[(i+1)%6] - B[i]
        n = np.array([-e[1], e[0]])
        n = n / (np.linalg.norm(n) + 1e-15)
        proj_A = A @ n
        proj_B = B @ n
        sep1 = np.min(proj_B) - np.max(proj_A)
        sep2 = np.min(proj_A) - np.max(proj_B)
        min_sep = min(min_sep, max(sep1, sep2))
    return min_sep

def _hexagon_in_outer(outer_verts, inner_verts):
    """Check if inner hexagon is inside outer hexagon using cross product test."""
    # outer_verts: (6,2), inner_verts: (6,2)
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i+1) % 6]
        edge = p2 - p1
        for p in inner_verts:
            cross = edge[0] * (p[1] - p1[1]) - edge[1] * (p[0] - p1[0])
            if cross < -1e-6:
                return False
    return True

def _compute_outer_side(inner_hexes):
    """Given inner hexagons (11,3), compute minimum outer hexagon side length
    for a hexagon centered at origin with 0 degree angle."""
    # For outer hexagon with side s centered at origin, angle 0:
    # Vertices at s*cos(60*i), s*sin(60*i)
    # The constraint is that all inner vertices must be inside.
    # For a regular hexagon, the inradius (distance from center to edge midpoint) = s * sqrt(3)/2
    # The apothem direction for edge i is at angle 30+60*i degrees
    # A point (x,y) is inside if for all edges i: x*cos(30+60*i) + y*sin(30+60*i) <= s*sqrt(3)/2
    # So s >= max over all inner vertices and all 6 edge directions of (dot / (sqrt(3)/2))
    
    # Get all vertices of all inner hexagons
    all_verts = []
    for i in range(len(inner_hexes)):
        cx, cy, ang = inner_hexes[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)  # (66, 2)
    
    # 6 edge normal directions
    edge_angs = np.deg2rad(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(edge_angs), np.sin(edge_angs)])  # (6,2)
    
    # For each vertex and each normal, compute the required s
    # dot = all_verts @ normals.T  -> (66, 6)
    dots = all_verts @ normals.T  # (66, 6)
    max_dot = np.max(dots)
    s = max_dot / (np.sqrt(3) / 2)
    return s

def _objective(x, time_start, deadline):
    """Objective: outer side length. x = [cx0,cy0,a0, cx1,cy1,a1, ...] (33 values)"""
    if time.time() - time_start > deadline:
        return 1e6
    inner = x.reshape(11, 3)
    # Compute outer side
    s = _compute_outer_side(inner)
    # Penalty for overlaps
    penalty = 0.0
    for i in range(11):
        for j in range(i+1, 11):
            vi = _hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2])
            vj = _hex_vertices(inner[j,0], inner[j,1], 1.0, inner[j,2])
            sep = _hex_separation(vi, vj)
            if sep < 0:
                penalty += (-sep) ** 2
    # Penalty for going outside
    outer_verts = _hex_vertices(0, 0, s, 0)
    for i in range(11):
        vi = _hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2])
        if not _hexagon_in_outer(outer_verts, vi):
            penalty += 1.0
    return s + 100.0 * penalty

def _physics_simulate(n_iter=20000, dt=0.01, repulse_k=50.0, attract_k=0.5):
    """Rigid-body dynamics simulation."""
    # Initial positions: triangular cluster
    h = np.sqrt(3) / 2 * 2  # distance between centers in hex grid = 2*h = sqrt(3)*... 
    # Actually for unit hexagons, center-to-center distance in a hex grid:
    # The inradius of unit hex = sqrt(3)/2, so two adjacent hexagons have centers 2*sqrt(3)/2 = sqrt(3) apart? No.
    # For two unit hexagons to be just touching, center distance = 2 * inradius = 2 * sqrt(3)/2 = sqrt(3)
    # Wait, inradius of regular hexagon with side 1 is sqrt(3)/2. Two hexagons touching share an edge,
    # so center distance = 2 * inradius = sqrt(3).
    
    d = np.sqrt(3)  # minimum center distance for touching unit hexagons
    
    # Triangular arrangement: rows of 1,3,3,3,1
    positions = []
    # Row 0: 1 hex
    positions.append([0, 0, 0.0])
    # Row 1: 3 hexes
    positions.append([-d, d*1.5, 0.0])
    positions.append([0, d*1.5, 0.0])
    positions.append([d, d*1.5, 0.0])
    # Row 2: 3 hexes
    positions.append([-d, -d*1.5, 0.0])
    positions.append([0, -d*1.5, 0.0])
    positions.append([d, -d*1.5, 0.0])
    # Row 3: 3 hexes (offset)
    positions.append([-1.5*d, d*3.0, 0.0])
    positions.append([-0.5*d, d*3.0, 0.0])
    positions.append([0.5*d, d*3.0, 0.0])
    # Row 4: 3 hexes
    positions.append([-1.5*d, -d*3.0, 0.0])
    positions.append([-0.5*d, -d*3.0, 0.0])
    positions.append([0.5*d, -d*3.0, 0.0])
    
    # That's 13, too many. Let me use the seed arrangement which has 11.
    # Seed: center, left, right, top-left, top-right, bottom-left, bottom-right, 
    # far top-left, far top-right, far bottom-left, far bottom-right
    positions = [
        [0, 0, 0.0],
        [-2.5, 0, 0.0],
        [2.5, 0, 0.0],
        [-1.25, 2.17, 0.0],
        [1.25, 2.17, 0.0],
        [-1.25, -2.17, 0.0],
        [1.25, -2.17, 0.0],
        [-3.75, 2.17, 0.0],
        [3.75, 2.17, 0.0],
        [-3.75, -2.17, 0.0],
        [3.75, -2.17, 0.0],
    ]
    
    pos = np.array(positions, dtype=float)  # (11, 3)
    vel = np.zeros_like(pos)
    
    best_pos = pos.copy()
    best_side = _compute_outer_side(pos)
    
    for it in range(n_iter):
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Repulsive forces between overlapping hexagons
        for i in range(11):
            vi = _hex_vertices(pos[i,0], pos[i,1], 1.0, pos[i,2])
            for j in range(i+1, 11):
                vj = _hex_vertices(pos[j,0], pos[j,1], 1.0, pos[j,2])
                sep = _hex_separation(vi, vj)
                if sep < 0.05:  # Apply force when close
                    # Force direction: from j to i (push apart)
                    rij = pos[i, :2] - pos[j, :2]
                    dist = np.linalg.norm(rij)
                    if dist < 1e-10:
                        rij = np.array([1.0, 0.0])
                        dist = 1.0
                    else:
                        rij = rij / dist
                    # Strength proportional to overlap
                    strength = repulse_k * (0.05 - sep)
                    forces[i, :2] += rij * strength
                    forces[j, :2] -= rij * strength
        
        # Attractive force toward center
        for i in range(11):
            r = pos[i, :2]
            dist = np.linalg.norm(r)
            if dist > 0.1:
                forces[i, :2] -= r / dist * attract_k * dist
        
        # Angular forces: try to align hexagons (reduce angle differences)
        # This helps with packing efficiency
        for i in range(11):
            forces[i, 2] = -0.1 * pos[i, 2]
        
        # Damping
        vel = 0.9 * vel + forces * dt
        pos = pos + vel * dt
        
        # Clamp angles to [0, 60)
        pos[:, 2] = pos[:, 2] % 60.0
        
        # Check current side
        side = _compute_outer_side(pos)
        if side < best_side:
            best_side = side
            best_pos = pos.copy()
    
    return best_pos, best_side

def _optimize_from_start(start_pos, max_iter=500):
    """Use scipy minimize to polish the solution."""
    x0 = start_pos.flatten()
    
    def obj(x):
        inner = x.reshape(11, 3)
        s = _compute_outer_side(inner)
        # Penalty for overlaps
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                vi = _hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2])
                vj = _hex_vertices(inner[j,0], inner[j,1], 1.0, inner[j,2])
                sep = _hex_separation(vi, vj)
                if sep < 0:
                    penalty += (-sep) ** 2
        # Penalty for being outside
        outer_verts = _hex_vertices(0, 0, s, 0)
        for i in range(11):
            vi = _hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2])
            if not _hexagon_in_outer(outer_verts, vi):
                penalty += 1.0
        return s + 50.0 * penalty
    
    result = minimize(obj, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
    return result.x.reshape(11, 3), result.fun

def _validate(inner, outer, side):
    """Validate the solution."""
    # Check finiteness
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if not np.isfinite(side) or side < np.sqrt(11) - 1e-9:
        return False
    
    # Check all pairs separated
    for i in range(11):
        vi = _hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2])
        for j in range(i+1, 11):
            vj = _hex_vertices(inner[j,0], inner[j,1], 1.0, inner[j,2])
            sep = _hex_separation(vi, vj)
            if sep <= 1e-6:
                return False
    
    # Check all inside outer
    outer_verts = _hex_vertices(outer[0], outer[1], side, outer[2])
    for i in range(11):
        vi = _hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2])
        if not _hexagon_in_outer(outer_verts, vi):
            return False
    
    # Check edge lengths
    for i in range(11):
        vi = _hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2])
        for k in range(6):
            edge_len = np.linalg.norm(vi[(k+1)%6] - vi[k])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    return True

def hexagon_packing_11():
    time_start = time.time()
    deadline = 600.0  # safety margin
    
    # Run physics simulation
    best_pos, best_side = _physics_simulate(n_iter=30000, dt=0.005, repulse_k=80.0, attract_k=0.3)
    
    # Try multiple restarts with different initial conditions
    best_overall = best_pos.copy()
    best_overall_side = best_side
    
    # Try a few more physics runs with perturbed starts
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() - time_start > deadline - 120:
            break
        noise = np.random.randn(11, 3) * 0.5
        noise[:, 2] *= 10  # more angle noise
        perturbed = best_pos + noise
        perturbed[:, 2] = perturbed[:, 2] % 60
        pos, side = _physics_simulate(n_iter=15000, dt=0.005, repulse_k=80.0, attract_k=0.3)
        # Use perturbed as start - actually let's just do fresh with random
        if side < best_overall_side:
            best_overall_side = side
            best_overall = pos.copy()
    
    # Polish with scipy
    if time.time() - time_start < deadline - 60:
        polished, pol_fun = _optimize_from_start(best_overall, max_iter=300)
        pol_side = _compute_outer_side(polished)
        if pol_side < best_overall_side:
            best_overall = polished.copy()
            best_overall_side = pol_side
    
    # Final validation and fix
    inner = best_overall.copy()
    outer = np.array([0.0, 0.0, 0.0])
    side = _compute_outer_side(inner)
    
    # Ensure validity - if invalid, fall back to seed
    if not _validate(inner, outer, side):
        # Try to fix by increasing side slightly
        for _ in range(100):
            side += 0.01
            if _validate(inner, outer, side):
                break
        else:
            # Fall back to seed
            inner = np.array([
                [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
                [-1.25, 2.17, 0], [1.25, 2.17, 0],
                [-1.25, -2.17, 0], [1.25, -2.17, 0],
                [-3.75, 2.17, 0], [3.75, 2.17, 0],
                [-3.75, -2.17, 0], [3.75, -2.17, 0],
            ], dtype=float)
            outer = np.array([0.0, 0.0, 0.0])
            side = 8.0
    
    return inner, outer, float(side)

# EVOLVE-BLOCK-END
