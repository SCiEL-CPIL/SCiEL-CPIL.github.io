import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 118.0  # Safety margin below 150s

# For two unit regular hexagons with arbitrary rotations,
# the minimum center distance for disjointness is 2*sqrt(3) (when aligned).
# For arbitrary relative rotations, the minimum distance is still 2*sqrt(3)
# because the width of a unit hexagon is 2 (vertex-to-vertex) and the
# minimum distance between parallel supporting lines is 2*sqrt(3)/... 
# Actually, let's think more carefully.
# A unit regular hexagon has inradius (apothem) = sqrt(3)/2, circumradius = 1.
# The minimum distance between two non-overlapping unit hexagons depends on rotation.
# For aligned hexagons, min center distance = 2*sqrt(3) (touching at flat sides).
# For 30-degree rotated hexagons, min center distance = 2 (touching at vertices).
# The minimum over all relative rotations is 2 (vertex-to-vertex contact).
# But we need a conservative bound: if distance > 2, they MIGHT still overlap.
# The safest approach: use 2*sqrt(3) as the threshold for "definitely disjoint"
# but that's too conservative. 
#
# Actually, for two congruent regular hexagons of side 1, the minimum center
# distance for non-overlap is 2 (when one is rotated 30 degrees relative to the other,
# they can touch vertex-to-vertex). But for aligned hexagons, it's 2*sqrt(3).
#
# The correct minimum distance for non-overlap of two unit regular hexagons
# over all relative rotations is 2.0 (achieved at 30-degree relative rotation).
# So if center distance > 2.0, they are NOT guaranteed to be disjoint.
# If center distance > 2*sqrt(3) ≈ 3.464, they ARE guaranteed to be disjoint.
#
# For a fast but slightly conservative check, we can use:
# - If dist > 2*sqrt(3): definitely disjoint (no overlap)
# - If dist < 2: definitely overlapping (must be disjoint, so invalid)
# - Otherwise: need SAT check (rare case)
#
# But for speed, let's use a slightly tighter bound. The actual minimum
# non-overlap distance for two unit hexagons is 2.0 (at 30° relative rotation).
# So dist > 2.0 does NOT guarantee disjointness.
#
# Let me use the exact minimum: for two unit regular hexagons, the minimum
# center distance for non-intersection is 2.0 (vertex-to-vertex, 30° rotation).
# So we need dist >= 2.0 as a necessary condition, but it's not sufficient.
#
# For a fast approximate check that's slightly conservative:
# Use dist > 2 * sqrt(3) as "definitely disjoint" and dist < 2 as "definitely overlapping".
# For the intermediate range, we'll do a quick SAT check.

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _get_edge_normals(angle_deg):
    """Get the 6 edge normals for a unit hexagon at given rotation."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    normals = []
    for k in range(6):
        # Edge from vertex k to vertex k+1
        theta1 = math.radians(60 * k)
        theta2 = math.radians(60 * (k + 1))
        x1, y1 = math.cos(theta1), math.sin(theta1)
        x2, y2 = math.cos(theta2), math.sin(theta2)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        nx, ny = -dy / length, dx / length
        # Rotate normal by angle
        rx = ca * nx - sa * ny
        ry = sa * nx + ca * ny
        normals.append((rx, ry))
    return normals

def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    """
    Fast validity check using vectorized distance pre-filter + SAT fallback.
    cx_arr, cy_arr, angle_arr: numpy arrays of shape (N,)
    Returns True if all hexagons are pairwise disjoint.
    """
    n = len(cx_arr)
    
    # Precompute vertices and normals for all hexagons
    verts_list = []
    normals_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
        normals_list.append(_get_edge_normals(angle_arr[i]))
    
    # Vectorized distance pre-filter
    for i in range(n):
        xi, yi = cx_arr[i], cy_arr[i]
        vi = verts_list[i]
        ni = normals_list[i]
        for j in range(i + 1, n):
            xj, yj = cx_arr[j], cy_arr[j]
            vj = verts_list[j]
            nj = normals_list[j]
            
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy
            
            # If distance is large enough, definitely disjoint
            if dist_sq > (2 * math.sqrt(3)) ** 2:
                continue
            
            # If distance is very small, definitely overlapping
            if dist_sq < 4.0 - 1e-9:
                return False
            
            # Intermediate case: do SAT check
            # Check all axes from both polygons
            sep_found = False
            for axis in ni:
                ax, ay = axis
                # Project polygon i
                proj_i = np.array([vx * ax + vy * ay for vx, vy in vi])
                minA, maxA = proj_i.min(), proj_i.max()
                # Project polygon j
                proj_j = np.array([vx * ax + vy * ay for vx, vy in vj])
                minB, maxB = proj_j.min(), proj_j.max()
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    sep_found = True
                    break
            
            if not sep_found:
                for axis in nj:
                    ax, ay = axis
                    proj_i = np.array([vx * ax + vy * ay for vx, vy in vi])
                    minA, maxA = proj_i.min(), proj_i.max()
                    proj_j = np.array([vx * ax + vy * ay for vx, vy in vj])
                    minB, maxB = proj_j.min(), proj_j.max()
                    if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                        sep_found = True
                        break
            
            if not sep_found:
                return False
    
    return True

def _outer_hex_side(cx_arr, cy_arr, angle_arr, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    # 6 outward normals of the outer hexagon
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for i in range(len(cx_arr)):
        verts = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _score_config(cx_arr, cy_arr, angle_arr, center=(0.0, 0.0), angle_deg=0.0):
    """Return (valid, side_length)."""
    if not _is_valid_fast(cx_arr, cy_arr, angle_arr):
        return False, float('inf')
    side = _outer_hex_side(cx_arr, cy_arr, angle_arr, center, angle_deg)
    return True, side

def construct():
    t0 = time.time()
    random.seed(42)
    
    def make_flower_initial():
        """
        Create a 'flower' like arrangement.
        1 center, 6 around, 5 in gaps.
        """
        pts = []
        # Center
        pts.append([0.0, 0.0, 0.0])
        # Ring 1: 6 hexagons touching the center
        r1 = 2 * math.sqrt(3)  # For aligned hexagons, center-to-center distance
        for i in range(6):
            angle = math.radians(60 * i)
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            pts.append([x, y, 0.0])
        
        # We have 7. Need 5 more.
        # Place them in the 'valleys' of the first ring, further out.
        r2 = r1 * 1.8  # Slightly less than 2x for tighter packing
        angles = [30, 90, 150, 210, 270]  # deg
        for deg in angles:
            rad = math.radians(deg)
            x = r2 * math.cos(rad)
            y = r2 * math.sin(rad)
            pts.append([x, y, 0.0])
            
        return pts[:12]

    def make_lattice_initial():
        """
        Hexagonal lattice arrangement.
        """
        pts = []
        row_height = math.sqrt(3)
        col_width = 2.0
        # Rows: 3, 4, 3 = 10 hexes
        for c in range(3):
            x = c * col_width - col_width
            y = -row_height
            pts.append([x, y, 0.0])
        for c in range(4):
            x = c * col_width - col_width - col_width/2
            y = 0.0
            pts.append([x, y, 0.0])
        for c in range(3):
            x = c * col_width - col_width
            y = row_height
            pts.append([x, y, 0.0])
        
        # Add 2 more
        pts.append([-col_width*2 - col_width/2 - col_width, 0.0, 0.0])
        pts.append([col_width*2 - col_width/2 + col_width, 0.0, 0.0])
        
        return pts

    def make_random_initial():
        """Random scatter."""
        pts = []
        for _ in range(N):
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            a = random.uniform(0, 60)
            pts.append([x, y, a])
        return pts

    # Generate multiple initial configurations
    initializers = [
        make_flower_initial,
        make_lattice_initial,
        make_random_initial,
        make_random_initial,
        make_random_initial,
        make_random_initial,
        make_random_initial,
        make_random_initial,
    ]
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Try each initializer and find the best valid starting point
    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 30:
            break
            
        cfg = init_func()
        if len(cfg) != N:
            continue
        
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        valid, side = _score_config(cx_arr, cy_arr, angle_arr)
        if valid and side < best_side:
            best_side = side
            best_cfg = cfg[:]
            best_center = (0.0, 0.0)
            best_angle = 0.0
    
    # If no valid configuration found, use a loose one
    if best_cfg is None:
        cfg = make_lattice_initial()
        # Scale up to ensure validity
        scale = 1.5
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_cfg = cfg
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        valid, side = _score_config(cx_arr, cy_arr, angle_arr)
        best_side = side if valid else 10.0
        best_center = (0.0, 0.0)
        best_angle = 0.0

    # Phase 1: Differential Evolution style global search
    def _quick_score(cx_arr, cy_arr, angle_arr):
        """Fast score computation assuming center (0,0) and angle 0."""
        apothem_unit = math.sqrt(3) / 2.0
        normals = []
        for k in range(6):
            na = math.radians(90 + 60 * k)
            normals.append((math.cos(na), math.sin(na)))
        
        max_need = 0.0
        for i in range(N):
            a = math.radians(angle_arr[i])
            ca, sa = math.cos(a), math.sin(a)
            for k in range(6):
                theta = math.radians(60 * k)
                vx = cx_arr[i] + ca * math.cos(theta) - sa * math.sin(theta)
                vy = cy_arr[i] + sa * math.cos(theta) + ca * math.sin(theta)
                for (nx, ny) in normals:
                    val = (nx * vx + ny * vy) / apothem_unit
                    if val > max_need:
                        max_need = val
        return max_need

    # DE-style search with population
    pop_size = 8
    # Initialize population
    population = []
    for _ in range(pop_size):
        cfg = make_random_initial()
        # Perturb slightly
        for p in cfg:
            p[0] += random.gauss(0, 0.5)
            p[1] += random.gauss(0, 0.5)
        population.append(cfg)
    
    # Add best initial to population
    population[0] = best_cfg[:]
    
    de_generations = 30
    de_f = 0.8  # Scaling factor
    de_cr = 0.9  # Crossover rate
    
    for gen in range(de_generations):
        if time.time() - t0 > TIME_LIMIT - 20:
            break
        
        # Sort population by score
        scored_pop = []
        for cfg in population:
            cx_arr = np.array([p[0] for p in cfg])
            cy_arr = np.array([p[1] for p in cfg])
            angle_arr = np.array([p[2] for p in cfg])
            valid = _is_valid_fast(cx_arr, cy_arr, angle_arr)
            if valid:
                side = _quick_score(cx_arr, cy_arr, angle_arr)
            else:
                side = float('inf')
            scored_pop.append((side, cfg))
        
        scored_pop.sort(key=lambda x: x[0])
        
        # Update best
        if scored_pop[0][0] < best_side:
            best_side = scored_pop[0][0]
            best_cfg = scored_pop[0][1][:]
        
        # Generate new population
        new_population = [scored_pop[0][1][:]]  # Keep best
        
        for i in range(1, pop_size):
            # Select three distinct random individuals
            a_idx, b_idx, c_idx = random.sample(range(pop_size), 3)
            a = scored_pop[a_idx][1]
            b = scored_pop[b_idx][1]
            c = scored_pop[c_idx][1]
            
            # Create mutant
            mutant = []
            for d in range(N):
                mx = a[d][0] + de_f * (b[d][0] - c[d][0]) + random.gauss(0, 0.1)
                my = a[d][1] + de_f * (b[d][1] - c[d][1]) + random.gauss(0, 0.1)
                ma = a[d][2] + de_f * (b[d][2] - c[d][2]) + random.gauss(0, 2.0)
                mutant.append([mx, my, ma])
            
            # Crossover with current
            current = scored_pop[i][1]
            trial = []
            for d in range(N):
                if random.random() < de_cr:
                    trial.append(mutant[d][:])
                else:
                    trial.append(current[d][:])
            
            new_population.append(trial)
        
        population = new_population
    
    # Phase 2: SA polish on best configuration
    cfg = [row[:] for row in best_cfg]
    cx_arr = np.array([p[0] for p in cfg])
    cy_arr = np.array([p[1] for p in cfg])
    angle_arr = np.array([p[2] for p in cfg])
    
    cur_score = best_side
    T = 0.2
    T_end = 0.003
    alpha = 0.997
    
    # Precompute normals for outer hexagon (angle=0, center=(0,0))
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    # Main SA loop
    while time.time() - t0 < TIME_LIMIT - 3:
        r = random.random()
        idx = random.randrange(N)
        
        # Save old values
        old_x, old_y, old_a = cx_arr[idx], cy_arr[idx], angle_arr[idx]
        
        if r < 0.6:
            # Move one hexagon
            cx_arr[idx] += random.gauss(0, 0.1)
            cy_arr[idx] += random.gauss(0, 0.1)
            angle_arr[idx] += random.gauss(0, 2.0)
        elif r < 0.8:
            # Rotate one hexagon
            angle_arr[idx] += random.gauss(0, 5.0)
        else:
            # Move two hexagons
            idx2 = random.randrange(N)
            if idx2 == idx:
                idx2 = (idx + 1) % N
            old_x2, old_y2, old_a2 = cx_arr[idx2], cy_arr[idx2], angle_arr[idx2]
            cx_arr[idx] += random.gauss(0, 0.08)
            cy_arr[idx] += random.gauss(0, 0.08)
            cx_arr[idx2] += random.gauss(0, 0.08)
            cy_arr[idx2] += random.gauss(0, 0.08)
            angle_arr[idx] += random.gauss(0, 1.5)
            angle_arr[idx2] += random.gauss(0, 1.5)
        
        # Check validity
        valid = _is_valid_fast(cx_arr, cy_arr, angle_arr)
        
        if valid:
            new_score = _quick_score(cx_arr, cy_arr, angle_arr)
            
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_side:
                    best_side = new_score
                    best_cfg = [[cx_arr[i], cy_arr[i], angle_arr[i]] for i in range(N)]
                    best_center = (0.0, 0.0)
                    best_angle = 0.0
            else:
                # Revert
                cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
                if r >= 0.8:
                    cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
        else:
            # Revert
            cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
            if r >= 0.8:
                cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
        
        T = max(T_end, T * alpha)
    
    # Final verification
    cx_final = np.array([p[0] for p in best_cfg])
    cy_final = np.array([p[1] for p in best_cfg])
    angle_final = np.array([p[2] for p in best_cfg])
    
    if not _is_valid_fast(cx_final, cy_final, angle_final):
        # Fallback: use a known valid loose configuration
        cfg = make_lattice_initial()
        scale = 1.3
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_cfg = cfg
        cx_final = np.array([p[0] for p in cfg])
        cy_final = np.array([p[1] for p in cfg])
        angle_final = np.array([p[2] for p in cfg])
        best_side = _outer_hex_side(cx_final, cy_final, angle_final, (0.0, 0.0), 0.0)
    else:
        best_side = _outer_hex_side(cx_final, cy_final, angle_final, (0.0, 0.0), 0.0)
    
    inner = [[p[0], p[1], p[2]] for p in best_cfg]
    return inner, best_center, best_side, best_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
