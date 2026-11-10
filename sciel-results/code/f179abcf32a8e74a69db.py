# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side d and angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _all_inner_vertices(positions):
    """Return (72, 2) array of all vertices of 12 unit hexagons at given positions."""
    n = len(positions)
    all_verts = np.empty((n * 6, 2))
    for i in range(n):
        cx, cy = positions[i]
        verts = _hex_vertices(cx, cy, 1.0, 0)
        all_verts[i*6:(i+1)*6] = verts
    return all_verts

def _min_containing_hex_side(positions, angle_deg=0):
    """
    Compute the minimum side length s of a regular hexagon (centered at origin, angle angle_deg)
    that contains all unit hexagons at given positions (with angle 0).
    """
    all_verts = _all_inner_vertices(positions)
    a_rad = np.radians(angle_deg)
    
    def _is_inside(s):
        """Check if all vertices are inside the outer hexagon with side s."""
        outer_verts = _hex_vertices(0, 0, s, angle_deg)
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i+1) % 6]
            edge = p2 - p1
            diffs = all_verts - p1
            crosses = edge[0] * diffs[:, 1] - edge[1] * diffs[:, 0]
            if np.any(crosses < -1e-6):
                return False
        return True
    
    lo = np.sqrt(12)
    hi = 20.0
    
    for _ in range(50):
        mid = (lo + hi) / 2
        if _is_inside(mid):
            hi = mid
        else:
            lo = mid
    
    return hi

def _check_separation(positions):
    """Check if all pairs of unit hexagons at given positions are separated (non-overlapping)."""
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            vi = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, 0)
            vj = _hex_vertices(positions[j, 0], positions[j, 1], 1.0, 0)
            # Separating axis test: check all 12 edge normals
            for verts in [vi, vj]:
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k+1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    proj_i = vi @ normal
                    proj_j = vj @ normal
                    min_i = proj_i.min()
                    max_i = proj_i.max()
                    min_j = proj_j.min()
                    max_j = proj_j.max()
                    if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                        break
                else:
                    continue
                break
            else:
                return False  # Found overlap
    return True

def _objective(positions):
    """Objective function: return s (the outer hexagon side length) or penalty if invalid."""
    if not _check_separation(positions):
        return 1e10
    return _min_containing_hex_side(positions)

def _hex_lattice_positions(spacing_scale=1.0):
    """Generate a hexagonal lattice arrangement of 12 unit hexagons."""
    # For unit hexagons in a hexagonal lattice:
    # Horizontal spacing: 2.5 * scale
    # Vertical spacing: sqrt(3) * 1.25 * scale ≈ 2.165 * scale
    s = spacing_scale
    h = 2.5 * s
    v = 2.165 * s  # sqrt(3)/2 * 2.5 = 2.165
    
    positions = np.array([
        [0.0, 0.0],
        [-h, 0.0],
        [h, 0.0],
        [-h/2, v],
        [h/2, v],
        [-h/2, -v],
        [h/2, -v],
        [-3*h/2, v],
        [3*h/2, v],
        [-3*h/2, -v],
        [3*h/2, -v],
        [0.0, -2*v],
    ])
    return positions

def _circular_positions(radius=2.0):
    """Generate positions on a circle."""
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    positions = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    return positions

def _optimize_positions(positions, max_iter=2000, deadline=None):
    """
    Optimize positions using a combination of gradient descent and random perturbations.
    Returns optimized positions and their objective value.
    """
    current_s = _objective(positions)
    best_positions = positions.copy()
    best_s = current_s
    
    step_size = 0.1
    n = len(positions)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try random perturbation
        delta = np.random.randn(n, 2) * step_size
        new_positions = positions + delta
        
        new_s = _objective(new_positions)
        
        if new_s < current_s:
            positions = new_positions
            current_s = new_s
            if current_s < best_s:
                best_s = current_s
                best_positions = positions.copy()
            # Slightly increase step size on success
            step_size = min(step_size * 1.05, 0.5)
        else:
            # Decrease step size
            step_size *= 0.95
        
        # Occasionally try a larger jump
        if it % 100 == 0 and step_size > 0.01:
            jump = np.random.randn(n, 2) * step_size * 3
            jump_positions = positions + jump
            jump_s = _objective(jump_positions)
            if jump_s < current_s:
                positions = jump_positions
                current_s = jump_s
                if current_s < best_s:
                    best_s = current_s
                    best_positions = positions.copy()
    
    return best_positions, best_s

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 580  # Leave 20s safety margin
    
    np.random.seed(42)
    
    # Try multiple initial configurations
    initial_configs = []
    
    # Hexagonal lattice with different scales
    for scale in [0.95, 1.0, 1.05, 1.1]:
        initial_configs.append(_hex_lattice_positions(scale))
    
    # Circular arrangements
    for radius in [1.5, 2.0, 2.5, 3.0]:
        initial_configs.append(_circular_positions(radius))
    
    # Perturbed hexagonal lattice
    base = _hex_lattice_positions(1.0)
    for _ in range(5):
        perturbed = base + np.random.randn(12, 2) * 0.3
        initial_configs.append(perturbed)
    
    best_positions = None
    best_s = float('inf')
    
    for i, init_pos in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        # First check if this initial config is valid
        if not _check_separation(init_pos):
            continue
        
        opt_pos, opt_s = _optimize_positions(init_pos, max_iter=1000, deadline=deadline - 10)
        
        if opt_s < best_s:
            best_s = opt_s
            best_positions = opt_pos.copy()
    
    # If no valid solution found, use the default hexagonal lattice
    if best_positions is None:
        best_positions = _hex_lattice_positions(1.0)
        best_s = _min_containing_hex_side(best_positions)
    
    # Construct final result
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = best_positions[:, 0]
    inner_hex_data[:, 1] = best_positions[:, 1]
    inner_hex_data[:, 2] = 0.0  # angle in degrees
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = best_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
