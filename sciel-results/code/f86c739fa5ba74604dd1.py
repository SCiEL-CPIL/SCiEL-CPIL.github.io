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


def _local_score_fast(points, idx, neighbor_indices, neighbor_dists):
    """Fast local score: only check moved point against its precomputed neighbors + boundary."""
    p = points[idx]
    # Boundary constraint
    boundary_margin = 1.0 - np.linalg.norm(p)
    if boundary_margin <= 0:
        return 0.0
    # Check distances to precomputed neighbors
    nbs = neighbor_indices[idx]
    if len(nbs) > 0:
        diffs = points[nbs] - p
        d2 = np.sum(diffs * diffs, axis=1)
        min_pair_d2 = np.min(d2)
        min_pair = np.sqrt(min_pair_d2) / 2.0
    else:
        min_pair = boundary_margin
    return min(boundary_margin, min_pair)


def _build_hex_rings(spacing, ring_radii=None):
    """
    Build N points in concentric hexagonal rings with given spacing.
    Each ring holds points at a given radius, placed to maximize min pair distance
    (uniform angles). The lattice spacing determines ring radii.
    """
    pts = []
    if ring_radii is None:
        # Determine ring radii from hex geometry: rings at sqrt(3)*spacing * k/2
        # Actually standard hex rings: distance from center = spacing * k for ring k
        # For a hex lattice, points in ring k are at distance k*spacing (for axial)
        # We'll just use evenly spaced radii tuned to spacing.
        n_rings = int(1.0 / spacing) + 1
        ring_radii = np.linspace(spacing * 0.8, 1.0 - spacing * 0.5, n_rings)
    # Distribute N points across rings: outer rings get more (circumference)
    counts = []
    total = 0
    for k, r in enumerate(ring_radii):
        if k == 0 and r < 0.1:
            c = 1
        else:
            c = max(6, int(2 * np.pi * r / spacing))
        counts.append(c)
        total += c
    # Adjust to match N
    if total < N:
        deficit = N - total
        # Add to outer rings
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
    # Build points
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
        # Fallback fill
        if len(pts) < N:
            extra = N - len(pts)
            angles = 2 * np.pi * np.arange(extra) / extra
            rr = 0.9
            pts = np.vstack([pts, np.column_stack([rr * np.cos(angles), rr * np.sin(angles)])])
        else:
            pts = pts[:N]
    return pts


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


def _refine(points, deadline, max_iters=4000, base_step=0.015):
    """Local refinement: move one point toward its best feasible displacement using fast local scoring."""
    if points is None:
        return None, 0.0
    points = points.copy()
    best_points = points.copy()
    best_r = _score(points)
    rng = np.random.default_rng(7)
    step = base_step
    
    # Precompute neighbor indices for fast local scoring
    tree = cKDTree(points)
    neighbor_indices = tree.query_ball_point(points, r=base_step * 3)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        idx = rng.integers(0, N)
        original = points[idx].copy()
        best_local = original
        best_local_r = best_r
        
        # Try a set of candidate moves: random perturbations at decreasing step
        for _ in range(12):
            if time.time() > deadline:
                break
            s = step * rng.uniform(0.2, 1.0)
            ang = rng.uniform(0, 2 * np.pi)
            cand = original + s * np.array([np.cos(ang), np.sin(ang)])
            points[idx] = cand
            # Fast local score
            r_local = _local_score_fast(points, idx, neighbor_indices, None)
            if r_local > best_local_r:
                # Verify with full score occasionally
                r_full = _score(points)
                if r_full > best_local_r:
                    best_local_r = r_full
                    best_local = cand.copy()
                else:
                    points[idx] = original
        points[idx] = best_local
        if best_local_r > best_r:
            best_r = best_local_r
            best_points = points.copy()
            step = max(0.002, step * 0.997)  # slower decay
            # Update neighbors occasionally
            if it % 200 == 0:
                tree = cKDTree(points)
                neighbor_indices = tree.query_ball_point(points, r=step * 3)
    
    return best_points, best_r


def _soft_min_objective(x, n_points, beta=50.0):
    """
    Smooth surrogate for the min-pair-distance / boundary-margin objective.
    x is flattened (N,2) array. Returns negative of soft-min radius (to minimize).
    """
    pts = x.reshape(n_points, 2)
    # Boundary margins
    norms = np.linalg.norm(pts, axis=1)
    boundary_margins = 1.0 - norms
    
    # Pairwise distances - use only a subset for speed
    # Compute all pairwise distances efficiently
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1e10)
    d = np.sqrt(d2)
    # Soft min of pair distances / 2
    pair_dists = d / 2.0
    # Use log-sum-exp for soft min
    min_pair_d2 = np.min(d2)
    # Soft min via exp(-beta * x)
    # For stability, work with squared distances
    exp_terms = np.exp(-beta * (pair_dists - min_pair_d2 / 2.0))
    soft_min_pair = min_pair_d2 / 2.0 - np.log(np.sum(exp_terms)) / beta
    
    # Also need soft min of boundary margins
    min_boundary = np.min(boundary_margins)
    exp_terms_b = np.exp(-beta * (boundary_margins - min_boundary))
    soft_min_boundary = min_boundary - np.log(np.sum(exp_terms_b)) / beta
    
    # The radius is min(soft_min_pair, soft_min_boundary)
    # Use soft min of the two
    m = min(soft_min_pair, soft_min_boundary)
    exp_p = np.exp(-beta * (soft_min_pair - m))
    exp_b = np.exp(-beta * (soft_min_boundary - m))
    soft_min_radius = m - np.log(exp_p + exp_b) / beta
    
    return -soft_min_radius


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s

    candidates = []

    # Hex lattice over a range of spacings
    for trial in range(40):
        spacing = 0.052 + trial * 0.003
        pts = _build_hex_lattice(spacing)
        if pts is not None:
            r = _score(pts)
            if r > 0:
                candidates.append((pts, r))

    # Hex rings with tuned radii
    for trial in range(25):
        spacing = 0.055 + trial * 0.003
        pts = _build_hex_rings(spacing)
        if pts is not None:
            r = _score(pts)
            if r > 0:
                candidates.append((pts, r))

    if not candidates:
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_points, best_r = candidates[0]

    # Refine top candidates
    for cand_idx in range(min(5, len(candidates))):
        if time.time() > deadline - 15.0:
            break
        cand_pts, cand_r = candidates[cand_idx]
        refined_pts, refined_r = _refine(cand_pts, deadline, max_iters=5000,
                                         base_step=0.012)
        if refined_pts is not None and refined_r > best_r:
            best_points, best_r = refined_pts, refined_r

    # Final polish with smooth surrogate optimization
    if time.time() < deadline - 10.0:
        try:
            x0 = best_points.flatten()
            # Use L-BFGS-B with soft-min objective
            res = minimize(
                _soft_min_objective,
                x0,
                args=(N, 30.0),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.x is not None:
                polished = res.x.reshape(N, 2)
                # Verify feasibility
                r_polished = _score(polished)
                if r_polished > best_r:
                    best_points = polished
                    best_r = r_polished
        except Exception:
            pass

    return best_points


# EVOLVE-BLOCK-END
