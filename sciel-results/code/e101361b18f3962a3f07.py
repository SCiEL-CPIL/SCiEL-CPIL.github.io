"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, side, angle_rad):
    """Return the 6 vertices of a regular hexagon centered at (cx, cy)
    with side length `side` and rotation angle `angle_rad` (radians)."""
    angles = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])


def _hex_support(points, direction):
    """Maximum projection of points onto a unit direction vector."""
    return np.max(points @ direction)


def _outer_side_length(points, outer_angle):
    """Compute the minimum side length of a regular hexagon (with given rotation)
    that contains all `points`. A regular hexagon of side s centered at origin
    with rotation theta has support s * sqrt(3)/2 in each of 6 normal directions.
    We find the minimum s such that all points are inside."""
    # The 6 outward normals of a hexagon with rotation theta
    normals = np.column_stack([
        np.cos(outer_angle + np.arange(6) * np.pi / 3.0),
        np.sin(outer_angle + np.arange(6) * np.pi / 3.0)
    ])
    # For each normal direction, find max support of points
    supports = np.array([_hex_support(points, normals[i]) for i in range(6)])
    # The hexagon of side s has support s*sqrt(3)/2 in each normal direction
    # So we need s >= max(supports) / (sqrt(3)/2)
    max_support = np.max(supports)
    return max_support / (np.sqrt(3) / 2.0)


def _min_enclosing_hexagon_side(positions, angles_rad):
    """Find the minimum side length of a regular hexagon that encloses all
    unit hexagons at given positions and rotations. We optimize over the
    outer hexagon's rotation and center."""
    # First, get all vertices of all inner hexagons
    all_points = []
    for pos, ang in zip(positions, angles_rad):
        verts = _hex_vertices(pos[0], pos[1], 1.0, ang)
        all_points.append(verts)
    all_points = np.vstack(all_points)

    # Optimize outer hexagon rotation (center will be at centroid for best fit)
    # The center that minimizes the max distance to boundary is the centroid
    # of the support points, but for a regular hexagon we can center at the
    # mean of all vertices for a good initial guess.
    center = np.mean(all_points, axis=0)
    centered_points = all_points - center

    def objective(theta):
        normals = np.column_stack([
            np.cos(theta + np.arange(6) * np.pi / 3.0),
            np.sin(theta + np.arange(6) * np.pi / 3.0)
        ])
        supports = np.array([np.max(centered_points @ normals[i]) for i in range(6)])
        max_support = np.max(supports)
        return max_support / (np.sqrt(3) / 2.0)

    # Try several initial angles
    best_side = np.inf
    for theta0 in np.linspace(0, np.pi/3, 7):
        result = minimize(objective, [theta0], method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        if result.fun < best_side:
            best_side = result.fun

    return best_side


def _check_validity(positions, angles_rad, outer_side, outer_center, outer_angle):
    """Check if the packing is valid: no overlaps between inner hexagons,
    and all inner hexagons are inside the outer hexagon."""
    # Check containment in outer hexagon
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    for i, (pos, ang) in enumerate(zip(positions, angles_rad)):
        inner_verts = _hex_vertices(pos[0], pos[1], 1.0, ang)
        # Check each vertex of inner hexagon is inside outer hexagon
        for v in inner_verts:
            # Point-in-convex-polygon test using cross products
            inside = True
            n = len(outer_verts)
            for j in range(n):
                edge = outer_verts[(j+1) % n] - outer_verts[j]
                cross = np.cross(edge, v - outer_verts[j])
                if cross < -1e-9:  # Allow small tolerance
                    inside = False
                    break
            if not inside:
                return False

    # Check pairwise non-overlap between inner hexagons
    # Two regular hexagons of side 1 don't overlap if their centers are
    # far enough apart considering orientations.
    # Simple check: distance between centers > 2 * inradius = sqrt(3)
    # is sufficient but not necessary. For robustness, check vertex containment.
    for i in range(N):
        for j in range(i+1, N):
            # Quick distance check
            d = np.linalg.norm(positions[i] - positions[j])
            if d < 2.0 * np.sqrt(3) / 2.0 - 0.01:  # inradius * 2
                # More detailed check: see if any vertex of one is inside the other
                verts_i = _hex_vertices(positions[i][0], positions[i][1], 1.0, angles_rad[i])
                verts_j = _hex_vertices(positions[j][0], positions[j][1], 1.0, angles_rad[j])
                # Check if any vertex of i is inside j
                for v in verts_i:
                    inside_j = True
                    n = len(verts_j)
                    for k in range(n):
                        edge = verts_j[(k+1) % n] - verts_j[k]
                        cross = np.cross(edge, v - verts_j[k])
                        if cross < -1e-9:
                            inside_j = False
                            break
                    if inside_j:
                        return False
                # Check if any vertex of j is inside i
                for v in verts_j:
                    inside_i = True
                    n = len(verts_i)
                    for k in range(n):
                        edge = verts_i[(k+1) % n] - verts_i[k]
                        cross = np.cross(edge, v - verts_i[k])
                        if cross < -1e-9:
                            inside_i = False
                            break
                    if inside_i:
                        return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Start with a hexagonal lattice arrangement of 12 hexagons
    # Using axial coordinates in a honeycomb pattern
    # The 12 hexagons form a compact cluster:
    # Layer 0: (0,0)
    # Layer 1: 6 neighbors
    # Layer 2: 5 of the 12 outer ring (to make 12 total)
    
    # Axial to cartesian for hex grid with side 1
    # x = 1.5 * q, y = sqrt(3)/2 * q + sqrt(3) * r
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-1, 2), (-2, 1), (-1, -1)
    ]
    
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = np.sqrt(3) / 2.0 * q + np.sqrt(3) * r
        return np.array([x, y])
    
    # Initial positions on hex lattice, all aligned
    initial_positions = [axial_to_cartesian(q, r) for q, r in axial_coords]
    initial_angles = [0.0] * N
    
    # Compute initial side length
    all_points = []
    for pos, ang in zip(initial_positions, initial_angles):
        verts = _hex_vertices(pos[0], pos[1], 1.0, ang)
        all_points.append(verts)
    all_points = np.vstack(all_points)
    
    # Center the cluster
    centroid = np.mean(all_points, axis=0)
    
    initial_side = _min_enclosing_hexagon_side(
        [p - centroid for p in initial_positions], initial_angles)
    
    best_side = initial_side
    best_positions = [p - centroid for p in initial_positions]
    best_angles = initial_angles[:]
    best_center = centroid
    
    # Try to improve by optimizing positions and angles
    # Represent state as: all x,y positions + all angles (flattened)
    # But that's 36 parameters. Let's be smarter.
    
    # Strategy: optimize the outer hexagon rotation first, then try 
    # small perturbations of inner hexagon positions to reduce the required side.
    
    # Let's define a cost function that computes the minimum enclosing hexagon side
    # given inner hexagon configurations.
    
    def compute_side(positions_flat):
        """positions_flat: array of shape (3N,) = [x0, y0, a0, x1, y1, a1, ...]"""
        positions = []
        angles = []
        for i in range(N):
            positions.append(np.array([positions_flat[3*i], positions_flat[3*i+1]]))
            angles.append(positions_flat[3*i+2])
        return _min_enclosing_hexagon_side(positions, angles)
    
    # Initial flat representation
    init_flat = []
    for pos, ang in zip(best_positions, best_angles):
        init_flat.extend([pos[0], pos[1], ang])
    init_flat = np.array(init_flat)
    
    # Try Nelder-Mead optimization on the full configuration
    # But 36D is a lot. Let's try a simpler approach:
    # Optimize outer hexagon angle and center translation.
    
    def outer_only_cost(params):
        """params = [outer_angle, center_x, center_y]"""
        outer_angle = params[0]
        cx, cy = params[1], params[2]
        
        all_pts = []
        for pos, ang in zip(best_positions, best_angles):
            verts = _hex_vertices(pos[0] - cx, pos[1] - cy, 1.0, ang)
            all_pts.append(verts)
        all_pts = np.vstack(all_pts)
        
        normals = np.column_stack([
            np.cos(outer_angle + np.arange(6) * np.pi / 3.0),
            np.sin(outer_angle + np.arange(6) * np.pi / 3.0)
        ])
        supports = np.array([np.max(all_pts @ normals[i]) for i in range(6)])
        max_support = np.max(supports)
        return max_support / (np.sqrt(3) / 2.0)
    
    # Optimize outer angle and center
    best_outer_params = None
    best_outer_cost = np.inf
    for theta0 in np.linspace(0, np.pi/3, 7):
        res = minimize(outer_only_cost, [theta0, 0.0, 0.0], method='Nelder-Mead',
                      options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < best_outer_cost:
            best_outer_cost = res.fun
            best_outer_params = res.x
    
    if best_outer_params is not None:
        best_side = best_outer_cost
        best_center = np.array([best_outer_params[1], best_outer_params[2]])
        outer_angle = best_outer_params[0]
    else:
        best_center = np.array([0.0, 0.0])
        outer_angle = 0.0
    
    # Now try to improve by moving inner hexagons
    # Use a simple iterative approach: try small random perturbations
    # and keep improvements.
    
    np.random.seed(42)
    current_positions = best_positions[:]
    current_angles = best_angles[:]
    current_side = best_side
    current_center = best_center
    current_outer_angle = outer_angle
    
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Randomly perturb one hexagon
        idx = np.random.randint(N)
        new_pos = current_positions[idx].copy()
        new_ang = current_angles[idx]
        
        if np.random.rand() < 0.7:
            # Perturb position
            new_pos += step_size * np.random.randn(2)
        else:
            # Perturb angle
            new_ang += step_size * np.random.randn()
        
        test_positions = current_positions[:]
        test_angles = current_angles[:]
        test_positions[idx] = new_pos
        test_angles[idx] = new_ang
        
        # Compute new side length
        new_side = _min_enclosing_hexagon_side(test_positions, test_angles)
        
        # Check validity
        if new_side < current_side - 1e-8:
            # Check if valid (no overlaps, all inside)
            if _check_validity(test_positions, test_angles, new_side, 
                              np.mean(np.vstack([_hex_vertices(p[0], p[1], 1.0, a) 
                                                 for p, a in zip(test_positions, test_angles)]), axis=0), 
                              0.0):
                current_positions = test_positions
                current_angles = test_angles
                current_side = new_side
                no_improve_count = 0
                step_size = max(step_size * 0.95, 0.01)
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
    
    # Reconstruct final answer
    # The inner hexagons are at current_positions with current_angles
    # The outer hexagon is centered at origin (we centered the configuration)
    # Side length is current_side, angle is 0 (or optimized)
    
    # Return in the required format
    inner = [[float(p[0]), float(p[1]), float(np.degrees(a))] for p, a in zip(current_positions, current_angles)]
    center = (0.0, 0.0)
    side = float(current_side)
    angle = 0.0
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END

