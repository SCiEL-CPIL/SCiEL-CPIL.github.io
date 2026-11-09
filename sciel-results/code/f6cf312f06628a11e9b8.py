"""Direct geometric construction for 300 equal circles in a unit circle."""
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START


def _score(points):
    """Feasible common radius using KDTree for nearest-neighbor distance."""
    if points is None or len(points) < 2:
        return 0.0
    norms = np.linalg.norm(points, axis=1)
    min_boundary = float(1.0 - np.max(norms))
    if min_boundary <= 0:
        return 0.0
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    min_pair = float(np.min(dists[:, 1])) / 2.0
    return min(min_boundary, min_pair)


def _build_hex_lattice(spacing):
    """Hex lattice points inside unit disk, pick N closest to center."""
    k = int(1.0 / spacing) + 3
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing * 0.5:
                pts.append((x, y))
    if len(pts) < N:
        return None
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]


def _build_hex_rings(spacing, ring_radii=None):
    """
    Build N points in concentric hexagonal rings with given spacing.
    """
    pts = []
    if ring_radii is None:
        n_rings = int(1.0 / spacing) + 1
        ring_radii = np.linspace(spacing * 0.8, 1.0 - spacing * 0.5, n_rings)
    counts = []
    total = 0
    for k, r in enumerate(ring_radii):
        if k == 0 and r < 0.1:
            c = 1
        else:
            c = max(6, int(2 * np.pi * r / spacing))
        counts.append(c)
        total += c
    if total < N:
        deficit = N - total
        for i in range(len(counts) - 1, -1, -1):
            add = min(deficit, max(1, counts[i] // 2))
            counts[i] += add
            deficit -= add
            if deficit <= 0:
                break
    elif total > N:
        excess = total - N
        for i in range(len(counts) - 1, -1, -1):
            rem = min(excess, counts[i] - (1 if i == 0 and ring_radii[i] < 0.1 else 6))
            counts[i] -= rem
            excess -= rem
            if excess <= 0:
                break
    for k, r in enumerate(ring_radii):
        c = counts[k]
        if c <= 0:
            continue
        if c == 1 and r < 0.1:
            pts.append([0.0, 0.0])
        else:
            angles = 2 * np.pi * np.arange(c) / c + (np.pi / c) * (k % 2)
            pts.append(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
    pts = np.vstack(pts)
    if len(pts) != N:
        if len(pts) < N:
            extra = N - len(pts)
            angles = 2 * np.pi * np.arange(extra) / extra
            rr = 0.9
            pts = np.vstack([pts, np.column_stack([rr * np.cos(angles), rr * np.sin(angles)])])
        else:
            pts = pts[:N]
    return pts


def _build_concentric_rings(n_rings=None):
    """Build N points in concentric rings with optimal ring radii."""
    if n_rings is None:
        n_rings = 10
    # Distribute N points across rings: outer rings get more points
    # Use area-proportional distribution
    radii = np.linspace(0.05, 0.95, n_rings)
    areas = np.diff(np.concatenate([[0], radii**2]))
    counts = np.maximum(1, np.round(areas / areas.sum() * N)).astype(int)
    # Adjust to sum to N
    diff = N - counts.sum()
    if diff > 0:
        for i in range(n_rings - 1, -1, -1):
            if diff <= 0:
                break
            counts[i] += 1
            diff -= 1
    elif diff < 0:
        for i in range(n_rings - 1, -1, -1):
            if diff >= 0:
                break
            counts[i] = max(1, counts[i] + 1)
            diff += 1
    
    pts = []
    for k, r in enumerate(radii):
        c = counts[k]
        if c <= 0:
            continue
        if k == 0 and r < 0.1:
            pts.append([0.0, 0.0])
        else:
            angles = 2 * np.pi * np.arange(c) / c
            if k % 2 == 1:
                angles += np.pi / c
            pts.append(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
    pts = np.vstack(pts)
    if len(pts) != N:
        if len(pts) < N:
            extra = N - len(pts)
            angles = 2 * np.pi * np.arange(extra) / extra
            rr = 0.9
            pts = np.vstack([pts, np.column_stack([rr * np.cos(angles), rr * np.sin(angles)])])
        else:
            pts = pts[:N]
    return pts


def _soft_min_obj(flat_pts, beta=80.0):
    """
    Soft-min objective: minimize -r where r is approximated by soft-min of
    all pairwise half-distances and boundary distances.
    Returns (value, gradient) for L-BFGS-B.
    """
    pts = flat_pts.reshape(N, 2)
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms
    
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    pair_half = dists[:, 1] / 2.0
    
    all_vals = np.concatenate([pair_half, boundary])
    all_vals = np.maximum(all_vals, 1e-12)
    
    scaled = -beta * all_vals
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum / beta
    
    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)
    
    grad_flat = np.zeros(2 * N)
    
    for i in range(N):
        w = weights[N + i]
        if norms[i] > 1e-10:
            grad_flat[2*i] += -w * (-pts[i, 0] / norms[i])
            grad_flat[2*i+1] += -w * (-pts[i, 1] / norms[i])
    
    _, nn_idx = tree.query(pts, k=2)
    for i in range(N):
        w = weights[i]
        j = nn_idx[i, 1]
        d = dists[i, 1]
        if d > 1e-10:
            grad_flat[2*i] += -w * (pts[i, 0] - pts[j, 0]) / (2.0 * d)
            grad_flat[2*i+1] += -w * (pts[i, 1] - pts[j, 1]) / (2.0 * d)
            grad_flat[2*j] += -w * (pts[j, 0] - pts[i, 0]) / (2.0 * d)
            grad_flat[2*j+1] += -w * (pts[j, 1] - pts[i, 1]) / (2.0 * d)
    
    return -soft_min_val, grad_flat


def _soft_min_obj_multi_nn(flat_pts, beta=80.0, k_nn=3):
    """
    Soft-min objective using multiple nearest neighbors for smoother gradients.
    This helps escape local minima by considering more constraints simultaneously.
    """
    pts = flat_pts.reshape(N, 2)
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms
    
    tree = cKDTree(pts)
    dists, nn_idx = tree.query(pts, k=k_nn + 1)
    
    # Collect all pairwise half-distances for k_nn nearest neighbors
    pair_half_list = []
    for i in range(N):
        for k in range(1, k_nn + 1):
            j = nn_idx[i, k]
            if j > i:  # Avoid double counting
                pair_half_list.append(dists[i, k] / 2.0)
    
    pair_half = np.array(pair_half_list) if pair_half_list else np.array([1.0])
    
    all_vals = np.concatenate([pair_half, boundary])
    all_vals = np.maximum(all_vals, 1e-12)
    
    scaled = -beta * all_vals
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum / beta
    
    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)
    
    grad_flat = np.zeros(2 * N)
    
    # Boundary gradients
    for i in range(N):
        w = weights[len(pair_half) + i]
        if norms[i] > 1e-10:
            grad_flat[2*i] += -w * (-pts[i, 0] / norms[i])
            grad_flat[2*i+1] += -w * (-pts[i, 1] / norms[i])
    
    # Pair gradients
    pair_idx = 0
    for i in range(N):
        for k in range(1, k_nn + 1):
            j = nn_idx[i, k]
            if j > i:
                w = weights[pair_idx]
                d = dists[i, k]
                if d > 1e-10:
                    grad_flat[2*i] += -w * (pts[i, 0] - pts[j, 0]) / (2.0 * d)
                    grad_flat[2*i+1] += -w * (pts[i, 1] - pts[j, 1]) / (2.0 * d)
                    grad_flat[2*j] += -w * (pts[j, 0] - pts[i, 0]) / (2.0 * d)
                    grad_flat[2*j+1] += -w * (pts[j, 1] - pts[i, 1]) / (2.0 * d)
                pair_idx += 1
    
    return -soft_min_val, grad_flat


def _lbfgs_optimize(points, deadline, max_iters=1000, betas=None):
    """
    Multi-stage L-BFGS-B optimization with increasing beta for better convergence.
    Starts with coarse soft-min (low beta) for global movement, then refines
    with higher beta for precise local optimization.
    """
    if betas is None:
        betas = [40.0, 80.0, 150.0, 300.0]
    
    best_flat = points.flatten().copy()
    best_r = _score(points)
    
    for beta in betas:
        if time.time() > deadline - 5.0:
            break
        
        current_flat = best_flat.copy()
        try:
            res = minimize(
                _soft_min_obj,
                current_flat,
                args=(beta,),
                jac=True,
                method='L-BFGS-B',
                options={
                    'maxiter': max_iters,
                    'maxfun': max_iters * 10,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxls': 50
                }
            )
            if res.x is not None:
                cand = res.x.reshape(N, 2)
                r = _score(cand)
                if r > best_r:
                    best_r = r
                    best_flat = res.x.copy()
        except Exception:
            pass
    
    return best_flat.reshape(N, 2), best_r


def _local_refine_fast(points, deadline, max_iters=8000, base_step=0.02):
    """
    Fast local refinement using only neighbor checks for moved point.
    Uses cached KDTree and only recomputes distances involving the moved point.
    """
    points = points.copy()
    best_points = points.copy()
    best_r = _score(points)
    rng = np.random.default_rng(42)
    step = base_step
    
    # Precompute KDTree
    tree = cKDTree(points)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        idx = rng.integers(0, N)
        original = points[idx].copy()
        best_local = original
        best_local_r = best_r
        
        # Find current neighbors of idx (k=8 to be safe)
        _, nn_idx = tree.query(original, k=8)
        
        for attempt in range(6):
            if time.time() > deadline:
                break
            s = step * rng.uniform(0.3, 1.0)
            ang = rng.uniform(0, 2 * np.pi)
            cand = original + s * np.array([np.cos(ang), np.sin(ang)])
            
            # Check boundary
            cnorm = np.linalg.norm(cand)
            if cnorm >= 1.0:
                continue
            
            # Check against neighbors only
            min_dist = 1.0 - cnorm  # boundary constraint
            for j in nn_idx:
                if j == idx:
                    continue
                d = np.linalg.norm(cand - points[j])
                if d < min_dist:
                    min_dist = d
            # Also check a few more random points for safety
            extra_idx = rng.integers(0, N, size=5)
            for j in extra_idx:
                if j == idx:
                    continue
                d = np.linalg.norm(cand - points[j])
                if d < min_dist:
                    min_dist = d
            
            cand_r = min_dist / 2.0
            if cand_r > best_local_r:
                best_local_r = cand_r
                best_local = cand.copy()
        
        points[idx] = best_local
        if best_local_r > best_r:
            best_r = best_local_r
            best_points = points.copy()
            # Update tree periodically
            if it % 50 == 0:
                tree = cKDTree(points)
            step = max(0.003, step * 0.997)
    
    # Final exact score
    final_r = _score(best_points)
    return best_points, final_r


def _polish_min_constraint(points, deadline, n_iters=5000):
    """
    High-precision polish: identify the minimum constraint (pair or boundary)
    and perform targeted moves to increase it. Uses line search along the
    gradient direction of the active constraint.
    """
    points = points.copy()
    best_points = points.copy()
    best_r = _score(points)
    rng = np.random.default_rng(777)
    
    tree = cKDTree(points)
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        # Compute current radius and identify active constraint
        norms = np.linalg.norm(points, axis=1)
        boundary_dists = 1.0 - norms
        min_boundary_idx = np.argmin(boundary_dists)
        min_boundary_val = boundary_dists[min_boundary_idx]
        
        dists, nn_idx = tree.query(points, k=2)
        pair_dists = dists[:, 1] / 2.0
        min_pair_idx = np.argmin(pair_dists)
        min_pair_val = pair_dists[min_pair_idx]
        
        # Determine active constraint
        if min_pair_val < min_boundary_val:
            # Pair constraint active
            i = min_pair_idx
            j = nn_idx[i, 1]
            d = dists[i, 1]
            if d > 1e-10:
                # Try to move i away from j with line search
                direction = (points[i] - points[j]) / d
                base_step = 0.005
                for ls in range(8):
                    s = base_step * (0.5 ** ls)
                    new_i = points[i] + s * direction
                    if np.linalg.norm(new_i) >= 1.0:
                        continue
                    # Quick check: does this improve min distance?
                    # Check against i's neighbors
                    _, i_nns = tree.query(new_i, k=5)
                    min_d = 1.0 - np.linalg.norm(new_i)
                    for jj in i_nns:
                        if jj == i:
                            continue
                        dd = np.linalg.norm(new_i - points[jj])
                        if dd < min_d:
                            min_d = dd
                    if min_d / 2.0 > best_r:
                        points[i] = new_i
                        best_r = min_d / 2.0
                        best_points = points.copy()
                        tree = cKDTree(points)
                        break
        else:
            # Boundary constraint active
            i = min_boundary_idx
            norm_i = norms[i]
            if norm_i > 1e-10:
                direction = -points[i] / norm_i
                base_step = 0.005
                for ls in range(8):
                    s = base_step * (0.5 ** ls)
                    new_i = points[i] + s * direction
                    if np.linalg.norm(new_i) >= 1.0:
                        continue
                    # Check if this improves
                    _, i_nns = tree.query(new_i, k=5)
                    min_d = 1.0 - np.linalg.norm(new_i)
                    for jj in i_nns:
                        if jj == i:
                            continue
                        dd = np.linalg.norm(new_i - points[jj])
                        if dd < min_d:
                            min_d = dd
                    if min_d / 2.0 > best_r:
                        points[i] = new_i
                        best_r = min_d / 2.0
                        best_points = points.copy()
                        tree = cKDTree(points)
                        break
        
        # Periodic full score check
        if it % 200 == 0:
            r = _score(points)
            if r > best_r:
                best_r = r
                best_points = points.copy()
    
    final_r = _score(best_points)
    if final_r > best_r:
        best_r = final_r
        best_points = points.copy()
    
    return best_points, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s

    candidates = []

    # Generate diverse initial candidates
    # 1. Hex lattice over a range of spacings
    for trial in range(25):
        spacing = 0.055 + trial * 0.004
        pts = _build_hex_lattice(spacing)
        if pts is not None:
            r = _score(pts)
            if r > 0:
                candidates.append((pts, r))

    # 2. Hex rings with tuned radii
    for trial in range(10):
        spacing = 0.058 + trial * 0.004
        pts = _build_hex_rings(spacing)
        if pts is not None:
            r = _score(pts)
            if r > 0:
                candidates.append((pts, r))

    # 3. Concentric rings with area-proportional distribution
    for n_rings in [8, 10, 12, 14]:
        pts = _build_concentric_rings(n_rings)
        r = _score(pts)
        if r > 0:
            candidates.append((pts, r))

    if not candidates:
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_points, best_r = candidates[0]

    # Phase 1: Quick local refinement on top candidates to get good starting points
    for cand_idx in range(min(3, len(candidates))):
        if time.time() > deadline - 100.0:
            break
        cand_pts, cand_r = candidates[cand_idx]
        refined_pts, refined_r = _local_refine_fast(cand_pts, deadline - 100.0,
                                                     max_iters=2000, base_step=0.02)
        if refined_pts is not None and refined_r > best_r:
            best_points, best_r = refined_pts, refined_r

    # Phase 2: Primary L-BFGS-B optimization (the key improvement path)
    # Allocate ~80% of remaining time to gradient-based optimization
    if time.time() < deadline - 30.0:
        lbfgs_deadline = deadline - 25.0
        lbfgs_pts, lbfgs_r = _lbfgs_optimize(best_points, lbfgs_deadline, max_iters=800)
        if lbfgs_pts is not None and lbfgs_r > best_r:
            best_points, best_r = lbfgs_pts, lbfgs_r

    # Phase 3: Additional L-BFGS-B from a different starting point if time allows
    if time.time() < deadline - 20.0 and len(candidates) > 1:
        alt_pts, alt_r = candidates[1]
        # Quick local refine first
        alt_refined, alt_refined_r = _local_refine_fast(alt_pts, deadline - 20.0,
                                                         max_iters=1000, base_step=0.02)
        if alt_refined_r > best_r:
            best_points, best_r = alt_refined, alt_refined_r
        if time.time() < deadline - 15.0:
            lbfgs_pts2, lbfgs_r2 = _lbfgs_optimize(alt_refined, deadline - 15.0, max_iters=500)
            if lbfgs_pts2 is not None and lbfgs_r2 > best_r:
                best_points, best_r = lbfgs_pts2, lbfgs_r2

    # Phase 4: High-precision polish on the minimum constraint
    if time.time() < deadline - 5.0:
        polished_pts, polished_r = _polish_min_constraint(best_points, deadline - 3.0, n_iters=3000)
        if polished_pts is not None and polished_r > best_r:
            best_points, best_r = polished_pts, polished_r

    # Phase 5: Final fast local polish
    if time.time() < deadline - 2.0:
        final_pts, final_r = _local_refine_fast(best_points, deadline - 1.0,
                                                 max_iters=1000, base_step=0.005)
        if final_pts is not None and final_r > best_r:
            best_points, best_r = final_pts, final_r

    return best_points


# EVOLVE-BLOCK-END
