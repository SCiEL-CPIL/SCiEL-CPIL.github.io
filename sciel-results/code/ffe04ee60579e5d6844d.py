# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _edge_normals(verts):
    n = len(verts)
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i+1) % n]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normals[i] = normal / norm
    return normals

def _sat_overlap(verts1, normals1, verts2, normals2):
    all_normals = np.vstack([normals1, normals2])
    min_overlap = float('inf')
    best_axis = None
    best_sign = 0
    for normal in all_normals:
        proj1 = np.dot(verts1, normal)
        proj2 = np.dot(verts2, normal)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 - 1e-6:
            return 0.0, None, 0
        elif max2 < min1 - 1e-6:
            return 0.0, None, 0
        if max1 < min2:
            overlap = min2 - max1
            sign = 1
        elif max2 < min1:
            overlap = min1 - max2
            sign = -1
        else:
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < 0:
                return 0.0, None, 0
            c1 = (min1 + max1) / 2
            c2 = (min2 + max2) / 2
            sign = 1 if c1 > c2 else -1
        if overlap < min_overlap:
            min_overlap = overlap
            best_axis = normal
            best_sign = sign
    return min_overlap, best_axis, best_sign

def _compute_all_verts_normals(inner):
    centers = inner[:, :2]
    angles = inner[:, 2]
    all_verts = np.zeros((12, 6, 2))
    all_normals = np.zeros((12, 6, 2))
    for i in range(12):
        verts = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        all_verts[i] = verts
        all_normals[i] = _edge_normals(verts)
    return all_verts, all_normals

def _compute_s_and_overlap_fast(inner, outer_angle):
    centers = inner[:, :2]
    all_verts, all_normals = _compute_all_verts_normals(inner)
    a_rad = np.radians(outer_angle)
    outer_unit_verts = np.column_stack([
        np.cos(a_rad + np.radians(60) * np.arange(6)),
        np.sin(a_rad + np.radians(60) * np.arange(6))
    ])
    outer_normals = _edge_normals(outer_unit_verts)
    all_inner_verts_flat = all_verts.reshape(72, 2)
    max_proj = 0.0
    for normal in outer_normals:
        projs = np.dot(all_inner_verts_flat, normal)
        max_k = np.max(projs)
        if max_k > max_proj:
            max_proj = max_k
    s = max_proj * 2.0 / np.sqrt(3)
    s = max(s, np.sqrt(12))
    is_valid = True
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist > 2.0 * np.sqrt(3) + 0.1:
                continue
            overlap, _, _ = _sat_overlap(all_verts[i], all_normals[i], all_verts[j], all_normals[j])
            if overlap > 0:
                is_valid = False
                break
        if not is_valid:
            break
    return s, is_valid

def _resolve_collisions(inner, outer_angle, max_iters=50, push_factor=0.6):
    inner = inner.copy()
    for _ in range(max_iters):
        all_verts, all_normals = _compute_all_verts_normals(inner)
        centers = inner[:, :2]
        any_overlap = False
        for i in range(12):
            for j in range(i+1, 12):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist > 2.0 * np.sqrt(3) + 0.1:
                    continue
                overlap, axis, sign = _sat_overlap(all_verts[i], all_normals[i], all_verts[j], all_normals[j])
                if overlap > 1e-8 and axis is not None:
                    any_overlap = True
                    push = sign * axis * overlap * push_factor
                    inner[i, 0] += push[0]
                    inner[i, 1] += push[1]
                    inner[j, 0] -= push[0]
                    inner[j, 1] -= push[1]
        if not any_overlap:
            break
    return inner

def _generate_symmetric_seeds():
    """Generate seeds based on the known optimal hexagonal packing structure."""
    seeds = []
    L = 2 * np.sqrt(3)  # exact minimum center-to-center distance for touching hexagons
    
    # Template 1: Center + 6 ring + 5 outer (classic hex cluster)
    # This is the most natural packing
    points = [[0.0, 0.0, 0.0]]
    for k in range(6):
        angle = np.radians(60 * k)
        points.append([L * np.cos(angle), L * np.sin(angle), 0.0])
    # Second ring: 5 hexagons at the "corners" of the hexagonal arrangement
    # In a full hexagonal lattice, the second ring has 12 positions, but we only need 5
    # Place them at alternating positions for symmetry
    second_ring_angles = [0, 120, 240, 60, 300]  # 5 positions with 3-fold symmetry
    for ang in second_ring_angles:
        a = np.radians(ang)
        # Distance for second ring in hex lattice
        r = L * np.sqrt(3)
        points.append([r * np.cos(a), r * np.sin(a), 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts.copy())
    
    # Template 2: 3 rows of 4 (triangular lattice)
    points = []
    for row in range(3):
        for col in range(4):
            x = (col - 1.5 + row/2.0) * L
            y = (row - 1) * L * np.sqrt(3)/2
            points.append([x, y, 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts.copy())
    
    # Template 3: 4-4-4 with offset
    points = []
    for row in range(3):
        for col in range(4):
            x = (col - 1.5) * L
            y = (row - 1) * L * np.sqrt(3)/2
            if row % 2 == 1:
                x += L/2
            points.append([x, y, 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts.copy())
    
    # Template 4: Compact circular with center
    points = [[0.0, 0.0, 0.0]]
    for k in range(6):
        angle = np.radians(60 * k)
        points.append([L * np.cos(angle), L * np.sin(angle), 0.0])
    for k in range(5):
        angle = np.radians(72 * k + 30)
        r = L * 1.7
        points.append([r * np.cos(angle), r * np.sin(angle), 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts.copy())
    
    # Template 5: Two rings of 6 (no center)
    points = []
    for k in range(6):
        angle = np.radians(60 * k)
        points.append([L * 0.8 * np.cos(angle), L * 0.8 * np.sin(angle), 0.0])
    for k in range(6):
        angle = np.radians(60 * k + 30)
        r = L * 1.6
        points.append([r * np.cos(angle), r * np.sin(angle), 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts.copy())
    
    # Template 6: Rotated hex cluster
    angle_rot = np.radians(15)
    points = [[0.0, 0.0, 0.0]]
    for k in range(6):
        angle = np.radians(60 * k)
        points.append([L * np.cos(angle), L * np.sin(angle), 0.0])
    second_ring_angles = [0, 120, 240, 60, 300]
    for ang in second_ring_angles:
        a = np.radians(ang)
        r = L * np.sqrt(3)
        x = r * np.cos(a)
        y = r * np.sin(a)
        xr = x * np.cos(angle_rot) - y * np.sin(angle_rot)
        yr = x * np.sin(angle_rot) + y * np.cos(angle_rot)
        points.append([xr, yr, 15.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts.copy())
    
    # Template 7: Diamond/rhombus arrangement
    points = []
    for i in range(3):
        for j in range(4):
            x = (i + j*0.5) * L
            y = j * L * np.sqrt(3)/2
            points.append([x, y, 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts.copy())
    
    return seeds

def _softmax_s(inner, outer_angle, beta=50.0):
    """Smooth approximation of the outer hexagon size using log-sum-exp."""
    centers = inner[:, :2]
    all_verts, _ = _compute_all_verts_normals(inner)
    a_rad = np.radians(outer_angle)
    outer_unit_verts = np.column_stack([
        np.cos(a_rad + np.radians(60) * np.arange(6)),
        np.sin(a_rad + np.radians(60) * np.arange(6))
    ])
    outer_normals = _edge_normals(outer_unit_verts)
    all_inner_verts_flat = all_verts.reshape(72, 2)
    
    # For each outer edge normal, compute the max projection (soft)
    max_projs = []
    for normal in outer_normals:
        projs = np.dot(all_inner_verts_flat, normal)
        # Soft max
        max_p = np.max(projs)
        log_sum_exp = max_p + np.log(np.sum(np.exp(projs - max_p)))
        # Soften: use beta to control smoothness
        soft_max = (1.0/beta) * np.log(np.sum(np.exp(beta * projs)))
        max_projs.append(soft_max)
    
    # The outer hexagon size is determined by the max of these
    max_p = np.max(max_projs)
    log_sum_exp = max_p + np.log(np.sum(np.exp(np.array(max_projs) - max_p)))
    s = log_sum_exp * 2.0 / np.sqrt(3)
    s = max(s, np.sqrt(12))
    return s

def _smooth_objective(params, beta=50.0):
    """Smooth objective for gradient-based optimization."""
    inner = params[:36].reshape(12, 3).copy()
    outer_angle = params[36]
    
    s = _softmax_s(inner, outer_angle, beta)
    
    # Add penalty for overlaps (smooth)
    all_verts, all_normals = _compute_all_verts_normals(inner)
    penalty = 0.0
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(inner[i, :2] - inner[j, :2])
            if dist > 2.0 * np.sqrt(3) + 0.2:
                continue
            overlap, _, _ = _sat_overlap(all_verts[i], all_normals[i], all_verts[j], all_normals[j])
            if overlap > 0:
                penalty += overlap * overlap  # Quadratic penalty for smoothness
    
    return s + 10.0 * penalty

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 620.0
    
    seeds = _generate_symmetric_seeds()
    
    best_inner = None
    best_s = float('inf')
    best_outer_angle = 0.0
    
    # Evaluate all seeds with multiple outer angles
    outer_angles = [0, 30, 15, 45, 60, 10, 20, 50, 25, 55, 75, 120]
    for seed in seeds:
        for angle_deg in outer_angles:
            s, is_valid = _compute_s_and_overlap_fast(seed, angle_deg)
            if is_valid and s < best_s:
                best_s = s
                best_inner = seed.copy()
                best_outer_angle = angle_deg
    
    if best_inner is None:
        best_inner = seeds[0].copy()
        best_s, _ = _compute_s_and_overlap_fast(best_inner, 0)
        best_outer_angle = 0
    
    # Phase 1: Gradient-based optimization with smooth surrogate
    if time.time() < deadline - 200.0:
        x0 = np.concatenate([best_inner.flatten(), [best_outer_angle]])
        
        for restart in range(10):
            if time.time() > deadline - 200.0:
                break
            
            if restart == 0:
                x_start = x0.copy()
            else:
                # Perturb the best solution
                x_start = x0 + np.random.randn(37) * 0.01
            
            try:
                result = minimize(_smooth_objective, x_start, method='BFGS',
                                 options={'maxiter': 500, 'gtol': 1e-10})
                candidate_inner = result.x[:36].reshape(12, 3).copy()
                candidate_angle = result.x[36]
                
                # Resolve any remaining collisions
                candidate_inner = _resolve_collisions(candidate_inner, candidate_angle, max_iters=30, push_factor=0.5)
                
                candidate_s, candidate_valid = _compute_s_and_overlap_fast(candidate_inner, candidate_angle)
                if candidate_valid and candidate_s < best_s:
                    best_inner = candidate_inner
                    best_s = candidate_s
                    best_outer_angle = candidate_angle
                    x0 = np.concatenate([best_inner.flatten(), [best_outer_angle]])
            except Exception:
                pass
    
    # Phase 2: Simulated Annealing with collision resolution
    np.random.seed(42)
    current_inner = best_inner.copy()
    current_s = best_s
    current_outer_angle = best_outer_angle
    
    T_initial = 0.02
    T_min = 0.00002
    n_sa_iterations = 15000
    
    for it in range(n_sa_iterations):
        if time.time() > deadline - 120.0:
            break
        
        T = T_initial * (T_min / T_initial) ** (it / n_sa_iterations)
        new_inner = current_inner.copy()
        new_outer_angle = current_outer_angle
        
        move_type = np.random.rand()
        
        if move_type < 0.5:
            # Perturb 1-3 hexagons
            n_perturb = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            indices = np.random.choice(12, size=n_perturb, replace=False)
            for idx in indices:
                new_inner[idx, 0] += np.random.randn() * 0.06
                new_inner[idx, 1] += np.random.randn() * 0.06
                new_inner[idx, 2] += np.random.randn() * 0.03
            new_inner = _resolve_collisions(new_inner, new_outer_angle, max_iters=30, push_factor=0.5)
        elif move_type < 0.8:
            # Perturb 1 hexagon more aggressively
            idx = np.random.randint(0, 12)
            new_inner[idx, 0] += np.random.randn() * 0.1
            new_inner[idx, 1] += np.random.randn() * 0.1
            new_inner[idx, 2] += np.random.randn() * 0.05
            new_inner = _resolve_collisions(new_inner, new_outer_angle, max_iters=40, push_factor=0.6)
        else:
            # Perturb outer angle and 1-2 hexagons
            new_outer_angle += np.random.randn() * 0.03
            n_perturb = np.random.choice([1, 2], p=[0.7, 0.3])
            indices = np.random.choice(12, size=n_perturb, replace=False)
            for idx in indices:
                new_inner[idx, 0] += np.random.randn() * 0.05
                new_inner[idx, 1] += np.random.randn() * 0.05
                new_inner[idx, 2] += np.random.randn() * 0.02
            new_inner = _resolve_collisions(new_inner, new_outer_angle, max_iters=30, push_factor=0.5)
        
        new_s, new_valid = _compute_s_and_overlap_fast(new_inner, new_outer_angle)
        
        if new_valid:
            if new_s < current_s:
                current_inner = new_inner
                current_s = new_s
                current_outer_angle = new_outer_angle
                if current_s < best_s:
                    best_inner = current_inner.copy()
                    best_s = current_s
                    best_outer_angle = current_outer_angle
            else:
                delta = new_s - current_s
                if np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                    current_inner = new_inner
                    current_s = new_s
                    current_outer_angle = new_outer_angle
        else:
            # Try to resolve collisions
            resolved = _resolve_collisions(new_inner, new_outer_angle, max_iters=50, push_factor=0.7)
            resolved_s, resolved_valid = _compute_s_and_overlap_fast(resolved, new_outer_angle)
            if resolved_valid:
                if resolved_s < current_s:
                    current_inner = resolved
                    current_s = resolved_s
                    current_outer_angle = new_outer_angle
                    if current_s < best_s:
                        best_inner = current_inner.copy()
                        best_s = current_s
                        best_outer_angle = current_outer_angle
                else:
                    delta = resolved_s - current_s
                    if np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                        current_inner = resolved
                        current_s = resolved_s
                        current_outer_angle = new_outer_angle
    
    # Phase 3: Nelder-Mead local refinement
    if time.time() < deadline - 60.0:
        def objective(params):
            if time.time() > deadline - 50.0:
                return 1e10
            inner = params[:36].reshape(12, 3).copy()
            outer_angle = params[36]
            s, is_valid = _compute_s_and_overlap_fast(inner, outer_angle)
            if is_valid:
                return s
            else:
                all_verts, all_normals = _compute_all_verts_normals(inner)
                penalty = 0.0
                for i in range(12):
                    for j in range(i+1, 12):
                        dist = np.linalg.norm(inner[i, :2] - inner[j, :2])
                        if dist > 2.0 * np.sqrt(3) + 0.1:
                            continue
                        overlap, _, _ = _sat_overlap(all_verts[i], all_normals[i], all_verts[j], all_normals[j])
                        penalty += overlap
                return s + 5.0 * penalty + 2.0
        
        x0 = np.concatenate([best_inner.flatten(), [best_outer_angle]])
        
        for restart in range(6):
            if time.time() > deadline - 60.0:
                break
            if restart == 0:
                x_start = x0.copy()
            else:
                x_start = x0 + np.random.randn(37) * 0.005
            
            try:
                result = minimize(objective, x_start, method='Nelder-Mead',
                                 options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
                candidate_inner = result.x[:36].reshape(12, 3).copy()
                candidate_angle = result.x[36]
                candidate_s, candidate_valid = _compute_s_and_overlap_fast(candidate_inner, candidate_angle)
                if candidate_valid and candidate_s < best_s:
                    best_inner = candidate_inner
                    best_s = candidate_s
                    best_outer_angle = candidate_angle
                    x0 = np.concatenate([best_inner.flatten(), [best_outer_angle]])
            except Exception:
                pass
    
    # Phase 4: Fine-tuning with small steps
    if time.time() < deadline - 30.0:
        np.random.seed(123)
        current_inner = best_inner.copy()
        current_s = best_s
        current_outer_angle = best_outer_angle
        
        for it in range(8000):
            if time.time() > deadline - 30.0:
                break
            new_inner = current_inner.copy()
            new_outer_angle = current_outer_angle
            idx = np.random.randint(0, 12)
            new_inner[idx, 0] += np.random.randn() * 0.002
            new_inner[idx, 1] += np.random.randn() * 0.002
            new_inner[idx, 2] += np.random.randn() * 0.001
            if np.random.rand() < 0.3:
                new_outer_angle += np.random.randn() * 0.0015
            
            new_s, new_valid = _compute_s_and_overlap_fast(new_inner, new_outer_angle)
            if not new_valid:
                new_inner = _resolve_collisions(new_inner, new_outer_angle, max_iters=20, push_factor=0.4)
                new_s, new_valid = _compute_s_and_overlap_fast(new_inner, new_outer_angle)
            
            if new_valid:
                if new_s < current_s:
                    current_inner = new_inner
                    current_s = new_s
                    current_outer_angle = new_outer_angle
                    if current_s < best_s:
                        best_inner = current_inner.copy()
                        best_s = current_s
                        best_outer_angle = current_outer_angle
                else:
                    delta = new_s - current_s
                    if np.random.rand() < np.exp(-delta / max(0.0015, 1e-10)):
                        current_inner = new_inner
                        current_s = new_s
                        current_outer_angle = new_outer_angle
    
    # Phase 5: Ultra-fine polish
    if time.time() < deadline - 10.0:
        np.random.seed(456)
        current_inner = best_inner.copy()
        current_s = best_s
        current_outer_angle = best_outer_angle
        
        for it in range(3000):
            if time.time() > deadline - 10.0:
                break
            new_inner = current_inner.copy()
            new_outer_angle = current_outer_angle
            idx = np.random.randint(0, 12)
            new_inner[idx, 0] += np.random.randn() * 0.0005
            new_inner[idx, 1] += np.random.randn() * 0.0005
            new_inner[idx, 2] += np.random.randn() * 0.00025
            if np.random.rand() < 0.4:
                new_outer_angle += np.random.randn() * 0.0004
            
            new_s, new_valid = _compute_s_and_overlap_fast(new_inner, new_outer_angle)
            if new_valid:
                if new_s < current_s:
                    current_inner = new_inner
                    current_s = new_s
                    current_outer_angle = new_outer_angle
                    if current_s < best_s:
                        best_inner = current_inner.copy()
                        best_s = current_s
                        best_outer_angle = current_outer_angle
    
    # Final validity check
    s_final, valid_final = _compute_s_and_overlap_fast(best_inner, best_outer_angle)
    if not valid_final:
        best_inner = seeds[0].copy()
        best_s, _ = _compute_s_and_overlap_fast(best_inner, 0)
        best_outer_angle = 0
    
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    return best_inner, outer_hex_data, float(best_s)

# EVOLVE-BLOCK-END
