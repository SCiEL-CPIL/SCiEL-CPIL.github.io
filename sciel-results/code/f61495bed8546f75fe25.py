# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt, cos, sin, pi

def hexagon_vertices(cx, cy, d, a_deg):
    a_rad = np.radians(a_deg)
    angles = a_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def compute_min_s(inner_hex_data):
    """Compute minimum s such that all inner vertices fit inside outer hexagon centered at origin, angle 0."""
    # Outer hexagon centered at origin with angle 0: vertices at (s*cos(60k), s*sin(60k))
    # For a point v to be inside, for each edge k: cross(edge_k, v - v_k) >= 0
    # Edge k goes from v_k to v_{k+1}
    # We need: s * A_k - s^2 * B_k >= 0 for each edge and each vertex
    # where A_k = dx_k * vy - dy_k * vx, B_k = dx_k * sin(ak) - dy_k * cos(ak)
    # with dx_k = cos(a_{k+1}) - cos(a_k), dy_k = sin(a_{k+1}) - sin(a_k)
    
    all_verts = []
    for i in range(12):
        ix, iy, ia = inner_hex_data[i]
        verts = hexagon_vertices(ix, iy, 1, ia)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)  # (72, 2)
    
    min_s = sqrt(12)
    
    # Precompute edge vectors for angle 0 outer hexagon
    for k in range(6):
        a1 = k * pi / 3
        a2 = (k + 1) * pi / 3
        dx = cos(a2) - cos(a1)
        dy = sin(a2) - sin(a1)
        
        # For all vertices: A = dx*vy - dy*vx, B = dx*sin(a1) - dy*cos(a1)
        A = dx * all_verts[:, 1] - dy * all_verts[:, 0]  # (72,)
        B = dx * sin(a1) - dy * cos(a1)  # scalar
        
        if abs(B) < 1e-12:
            if np.any(A < -1e-9):
                return np.inf
        else:
            ratios = A / B
            if B > 1e-12:
                # Need s <= min(ratios)
                max_r = np.max(ratios)
                if max_r < min_s:
                    min_s = max_r
            else:
                # Need s >= max(ratios) where ratios > 0
                pos_ratios = ratios[ratios > 0]
                if len(pos_ratios) > 0:
                    min_r = np.min(pos_ratios)
                    if min_r > min_s:
                        min_s = min_r
    
    return min_s

def check_separation(inner_hex_data, tol=1e-6):
    """Check that no two inner hexagons overlap using SAT."""
    n = 12
    # Compute all vertices
    all_verts = []
    for i in range(n):
        ix, iy, ia = inner_hex_data[i]
        verts = hexagon_vertices(ix, iy, 1, ia)
        all_verts.append(verts)
    
    # Get all edge normals (axes)
    axes = []
    for i in range(n):
        verts = all_verts[i]
        for j in range(6):
            p1 = verts[j]
            p2 = verts[(j + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)
    axes = np.array(axes)
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]
            sep = False
            for axis in axes:
                proj_i = vi @ axis
                proj_j = vj @ axis
                if np.max(proj_i) + tol < np.min(proj_j) or np.max(proj_j) + tol < np.min(proj_i):
                    sep = True
                    break
            if not sep:
                return False
    return True

def check_validity(inner_hex_data, outer_hex_data, s):
    """Full validity check."""
    n = 12
    if inner_hex_data.shape != (n, 3) or outer_hex_data.shape != (3,):
        return False
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if s < sqrt(12):
        return False
    
    # Check separation
    if not check_separation(inner_hex_data):
        return False
    
    # Check containment with the actual outer hexagon
    ox, oy, oa = outer_hex_data
    oa_rad = np.radians(oa)
    
    # Outer vertices
    outer_angles = oa_rad + np.arange(6) * pi / 3
    outer_verts = np.column_stack([ox + s * np.cos(outer_angles), oy + s * np.sin(outer_angles)])
    
    # Check distinct outer vertices
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    
    # Check all inner vertices inside outer
    for i in range(n):
        ix, iy, ia = inner_hex_data[i]
        verts = hexagon_vertices(ix, iy, 1, ia)
        for v in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                e = p2 - p1
                pv = v - p1
                cross = e[0] * pv[1] - e[1] * pv[0]
                if cross < -1e-6:
                    return False
    return True

def initial_packing():
    """Create initial honeycomb-like packing of 12 hexagons."""
    centers = []
    # Center
    centers.append([0, 0])
    # Ring 1: 6 hexagons at distance sqrt(3)
    d = sqrt(3)
    for k in range(6):
        a = pi / 3 * k
        centers.append([d * cos(a), d * sin(a)])
    # Ring 2: 5 more hexagons at distance 2*sqrt(3), offset by 30 degrees
    r2 = 2 * sqrt(3)
    for k in range(5):
        a = pi / 6 + pi / 3 * k
        centers.append([r2 * cos(a), r2 * sin(a)])
    
    inner = np.column_stack([np.array(centers), np.zeros(12)])
    return inner

def optimize_packing(seed=0, time_limit=10.0):
    """Adaptive local search to minimize s."""
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    best_inner = initial_packing()
    best_s = compute_min_s(best_inner)
    
    if not np.isfinite(best_s) or best_s > 10:
        # Try a looser initial packing
        for attempt in range(10):
            candidate = initial_packing()
            # Add some random perturbation
            candidate[:, 0] += rng.normal(0, 0.1, 12)
            candidate[:, 1] += rng.normal(0, 0.1, 12)
            candidate[:, 2] += rng.uniform(-5, 5, 12)
            s_cand = compute_min_s(candidate)
            if np.isfinite(s_cand) and s_cand < best_s:
                best_inner = candidate
                best_s = s_cand
    
    step_size = 0.3
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit:
        # Propose move
        new_inner = best_inner.copy()
        idx = rng.randint(12)
        new_inner[idx, 0] += rng.normal(0, step_size)
        new_inner[idx, 1] += rng.normal(0, step_size)
        new_inner[idx, 2] += rng.uniform(-15, 15)
        
        # Check separation (fast)
        if not check_separation(new_inner):
            no_improve_count += 1
            if no_improve_count % 50 == 0:
                step_size *= 0.95
            continue
        
        # Compute new s
        new_s = compute_min_s(new_inner)
        
        if np.isfinite(new_s) and new_s < best_s:
            best_inner = new_inner
            best_s = new_s
            no_improve_count = 0
            # Slightly increase step size on success
            step_size = min(step_size * 1.02, 0.5)
        else:
            no_improve_count += 1
            if no_improve_count % 50 == 0:
                step_size *= 0.95
        
        if no_improve_count > max_no_improve:
            # Reset with a new random perturbation
            best_inner[:, 0] += rng.normal(0, 0.2, 12)
            best_inner[:, 1] += rng.normal(0, 0.2, 12)
            best_inner[:, 2] += rng.uniform(-10, 10, 12)
            best_s = compute_min_s(best_inner)
            if not np.isfinite(best_s) or best_s > 8:
                best_inner = initial_packing()
                best_s = compute_min_s(best_inner)
            step_size = 0.3
            no_improve_count = 0
    
    # Final: center the configuration at origin for the outer hexagon
    # Compute centroid of all inner vertices
    all_verts = []
    for i in range(12):
        ix, iy, ia = best_inner[i]
        verts = hexagon_vertices(ix, iy, 1, ia)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    centroid = np.mean(all_verts, axis=0)
    
    # Shift all inner hexagons so centroid is at origin
    best_inner[:, 0] -= centroid[0]
    best_inner[:, 1] -= centroid[1]
    
    # Recompute s
    final_s = compute_min_s(best_inner)
    if not np.isfinite(final_s) or final_s > best_s:
        # Revert shift
        best_inner[:, 0] += centroid[0]
        best_inner[:, 1] += centroid[1]
        final_s = best_s
    
    # Try different outer angles to see if we can reduce s further
    best_final_s = final_s
    best_outer_angle = 0.0
    for angle_deg in [0, 30, 15, 45]:
        # Rotate all inner hexagons by this angle
        rot_inner = best_inner.copy()
        for i in range(12):
            rot_inner[i, 2] = best_inner[i, 2] + angle_deg
        # Shift to centroid
        all_v = []
        for i in range(12):
            ix, iy, ia = rot_inner[i]
            v = hexagon_vertices(ix, iy, 1, ia)
            all_v.append(v)
        all_v = np.vstack(all_v)
        c = np.mean(all_v, axis=0)
        rot_inner[:, 0] -= c[0]
        rot_inner[:, 1] -= c[1]
        s_test = compute_min_s(rot_inner)
        if np.isfinite(s_test) and s_test < best_final_s:
            best_final_s = s_test
            best_outer_angle = angle_deg
            best_inner = rot_inner
    
    outer = np.array([0.0, 0.0, best_outer_angle])
    return best_inner, outer, best_final_s

def hexagon_packing_12():
    deadline = time.time() + 590.0
    best_result = None
    best_score = 0
    
    for seed in range(20):
        if time.time() > deadline:
            break
        try:
            remaining = deadline - time.time()
            if remaining < 2:
                break
            time_limit = min(remaining * 0.4, 15.0)
            inner, outer, s = optimize_packing(seed, time_limit=time_limit)
            if np.isfinite(s) and s > 0:
                score = 1.0 / s
                if score > best_score:
                    best_score = score
                    best_result = (inner, outer, s)
        except Exception:
            continue
    
    if best_result is None:
        # Fallback: very loose packing
        inner = np.array(
            [
                [0, 0, 0],
                [-3, 0, 0], [3, 0, 0],
                [-1.5, 2.6, 0], [1.5, 2.6, 0],
                [-1.5, -2.6, 0], [1.5, -2.6, 0],
                [-4.5, 2.6, 0], [4.5, 2.6, 0],
                [-4.5, -2.6, 0], [4.5, -2.6, 0],
                [0, -5.2, 0],
            ]
        )
        outer = np.array([0.0, 0.0, 0.0])
        s = 10.0
        best_result = (inner, outer, s)
    
    return best_result

# EVOLVE-BLOCK-END
