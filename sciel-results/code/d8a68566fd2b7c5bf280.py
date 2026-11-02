"""Packing 12 unit hexagons in a minimal regular hexagon via SAT + simulated annealing."""
import numpy as np
import time
import random

N = 12
DEADLINE = 145.0  # safety margin under 150s wall-clock

# Precompute unit hexagon vertices (flat-top, side=1)
def _unit_hex_vertices():
    verts = []
    for k in range(6):
        a = np.radians(60.0 * k)
        verts.append((np.cos(a), np.sin(a)))
    return np.array(verts)

UNIT_VERTS = _unit_hex_vertices()

def _hex_vertices(cx, cy, angle_deg, scale=1.0):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s], [s, c]])
    return np.array([[cx + scale * (R @ v)[0], cy + scale * (R @ v)[1]] for v in UNIT_VERTS])

def _edge_normals(verts):
    n = len(verts)
    norms = np.zeros((n, 2))
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i + 1) % n]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            norms[i] = normal / norm
    return norms

def _sat_overlap(v1, v2):
    """Check if two convex polygons overlap using SAT. Returns True if they overlap."""
    axes = np.vstack([_edge_normals(v1), _edge_normals(v2)])
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        if np.max(proj1) < np.min(proj2) - 1e-9 or np.max(proj2) < np.min(proj1) - 1e-9:
            return False
    return True

def _min_separation(v1, v2):
    """Compute the minimum distance between two convex polygons. Returns negative if overlapping."""
    # If they overlap, return negative overlap depth (approximate)
    if _sat_overlap(v1, v2):
        # Estimate overlap depth using center distance and size
        c1 = np.mean(v1, axis=0)
        c2 = np.mean(v2, axis=0)
        dist = np.linalg.norm(c1 - c2)
        # Approximate: if centers are closer than expected, they overlap
        # For hexagons with side 1, minimum center distance for non-overlap is ~2.0
        return -(2.0 - dist) if dist < 2.0 else 0.0
    else:
        # Compute minimum distance between edges
        min_dist = float('inf')
        for i in range(len(v1)):
            p1 = v1[i]
            p2 = v1[(i + 1) % len(v1)]
            for j in range(len(v2)):
                q1 = v2[j]
                q2 = v2[(j + 1) % len(v2)]
                # Point-to-segment distance
                d1 = _point_to_segment_dist(p1, q1, q2)
                d2 = _point_to_segment_dist(p2, q1, q2)
                d3 = _point_to_segment_dist(q1, p1, p2)
                d4 = _point_to_segment_dist(q2, p1, p2)
                min_dist = min(min_dist, d1, d2, d3, d4)
        return min_dist

def _point_to_segment_dist(p, a, b):
    """Distance from point p to segment ab."""
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)
    closest = a + t * ab
    return np.linalg.norm(p - closest)

def _required_outer_side(positions, outer_angle):
    """Compute the minimum outer hexagon side length that contains all inner hexagons."""
    all_verts = []
    for cx, cy, angle in positions:
        verts = _hex_vertices(cx, cy, angle, 1.0)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    centroid = np.mean(all_verts, axis=0)
    
    # Try a few centers around centroid
    best_side = float('inf')
    for dx in np.linspace(-1, 1, 3):
        for dy in np.linspace(-1, 1, 3):
            ox, oy = centroid[0] + dx, centroid[1] + dy
            max_proj = 0
            for k in range(6):
                na = np.radians(30 + 60 * k + outer_angle)
                nx, ny = np.cos(na), np.sin(na)
                projs = (all_verts[:, 0] - ox) * nx + (all_verts[:, 1] - oy) * ny
                max_proj = max(max_proj, np.max(projs))
            side = max_proj / (np.sqrt(3) / 2)
            if side < best_side:
                best_side = side
    return best_side

def _energy(positions):
    """Compute the energy of a configuration: outer side length + penalty for overlaps."""
    # Compute outer side length (use angle 0 for simplicity)
    outer_side = _required_outer_side(positions, 0.0)
    
    # Compute overlap penalty
    penalty = 0.0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            v1 = _hex_vertices(positions[i][0], positions[i][1], positions[i][2], 1.0)
            v2 = _hex_vertices(positions[j][0], positions[j][1], positions[j][2], 1.0)
            sep = _min_separation(v1, v2)
            if sep < 0:
                penalty += (-sep) ** 2 * 10.0
    return outer_side + penalty

def _initial_config_hexagonal():
    """Create a hexagonal lattice initial configuration."""
    positions = []
    # 4-4-4 grid with hexagonal offset
    for row in range(3):
        for col in range(4):
            x = (col - 1.5) * 2.1 + (0.0 if row % 2 == 0 else 1.05)
            y = row * 1.82
            positions.append([x, y, 0.0])
    return positions

def _initial_config_circular():
    """Create a circular initial configuration."""
    positions = []
    for i in range(N):
        angle = 2 * np.pi * i / N
        r = 2.5
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        positions.append([x, y, 0.0])
    return positions

def _initial_config_random():
    """Create a random initial configuration."""
    positions = []
    for _ in range(N):
        x = random.uniform(-3, 3)
        y = random.uniform(-3, 3)
        positions.append([x, y, 0.0])
    return positions

def _simulated_annealing(positions, num_iterations=50000, start_temp=1.0, end_temp=0.01):
    """Run simulated annealing to optimize the configuration."""
    start_time = time.time()
    best_positions = [p[:] for p in positions]
    best_energy = _energy(positions)
    current_energy = best_energy
    
    for it in range(num_iterations):
        if time.time() > DEADLINE - 5:
            break
        
        # Propose a move
        idx = random.randint(0, N - 1)
        move_type = random.random()
        
        new_positions = [p[:] for p in positions]
        if move_type < 0.4:
            # Move position
            new_positions[idx][0] += random.gauss(0, 0.1)
            new_positions[idx][1] += random.gauss(0, 0.1)
        elif move_type < 0.8:
            # Rotate
            new_positions[idx][2] += random.gauss(0, 5.0)
        else:
            # Swap two positions
            idx2 = random.randint(0, N - 1)
            new_positions[idx], new_positions[idx2] = new_positions[idx2], new_positions[idx]
        
        new_energy = _energy(new_positions)
        delta = new_energy - current_energy
        
        # Acceptance probability
        temp = start_temp * (end_temp / start_temp) ** (it / num_iterations)
        if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-9)):
            positions = new_positions
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = [p[:] for p in positions]
    
    return best_positions, best_energy

def construct():
    start_time = time.time()
    
    # Try multiple initial configurations
    best_positions = None
    best_score = float('inf')
    
    initial_configs = [
        _initial_config_hexagonal(),
        _initial_config_circular(),
        _initial_config_random()
    ]
    
    for init_pos in initial_configs:
        if time.time() > DEADLINE - 10:
            break
        
        # Run simulated annealing
        positions, energy = _simulated_annealing(init_pos, num_iterations=30000)
        
        if energy < best_score:
            best_score = energy
            best_positions = positions
    
    if best_positions is None:
        best_positions = initial_configs[0]
        best_score = _energy(best_positions)
    
    # Compute final outer hexagon
    outer_side = _required_outer_side(best_positions, 0.0)
    all_verts = []
    for cx, cy, angle in best_positions:
        verts = _hex_vertices(cx, cy, angle, 1.0)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    centroid = np.mean(all_verts, axis=0)
    
    inner = [[p[0], p[1], p[2]] for p in best_positions]
    return inner, [centroid[0], centroid[1]], outer_side, 0.0
