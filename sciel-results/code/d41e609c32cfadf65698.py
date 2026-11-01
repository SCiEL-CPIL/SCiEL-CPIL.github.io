"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_score(points):
    """Compute the score: min of half-min-pair-distance and min distance to boundary."""
    n = len(points)
    if n == 0:
        return 0.0

    # Min distance to boundary
    dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = float(np.min(dists_to_boundary))

    # Min pair distance via vectorized computation
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = float(np.min(dist_matrix))

    return min(min_pair / 2.0, min_boundary)


def _soft_min_gradient(pos, t=80.0):
    """
    Compute gradient of the soft-min of all pairwise and boundary distances.
    We want to maximize the minimum distance, so we minimize the soft-min.
    Returns (gradient, soft_min_value, actual_min).
    """
    n = len(pos)
    pos = pos.reshape(n, 2)

    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)

    # Boundary distances
    boundary_dists = 1.0 - np.linalg.norm(pos, axis=1)

    # Flatten pairwise distances (upper triangle)
    iu = np.triu_indices(n, k=1)
    pair_dists = dist_matrix[iu]

    # All distances
    all_dists = np.concatenate([pair_dists, boundary_dists])

    # Compute weights using log-sum-exp for numerical stability
    d_min = np.min(all_dists)
    shifted = all_dists - d_min
    exp_terms = np.exp(-t * shifted)
    S = np.sum(exp_terms)

    # Gradient of soft-min: f = (1/t)*log(S) + d_min
    # df/dx = -(1/S) * sum_k exp(-t*(d_k - d_min)) * d(d_k)/dx
    #        = -sum_k w_k * d(d_k)/dx  where w_k = exp(-t*(d_k-d_min))/S

    grad = np.zeros_like(pos)

    # Pairwise gradient
    pair_weights = exp_terms[:len(pair_dists)] / S
    diff_pairs = diff[iu]
    norm_pairs = pair_dists[:, np.newaxis]
    norm_pairs[norm_pairs < 1e-12] = 1e-12
    unit_vecs = diff_pairs / norm_pairs

    i_idx, j_idx = iu
    np.add.at(grad, i_idx, -pair_weights[:, np.newaxis] * unit_vecs)
    np.add.at(grad, j_idx, pair_weights[:, np.newaxis] * unit_vecs)

    # Boundary gradient
    boundary_weights = exp_terms[len(pair_dists):] / S
    norms = np.linalg.norm(pos, axis=1)[:, np.newaxis]
    norms[norms < 1e-12] = 1e-12
    unit_pos = pos / norms
    # d(1-|x|)/dx = -x/|x|, so contribution is -w * (-unit_pos) = w * unit_pos
    grad += boundary_weights[:, np.newaxis] * unit_pos

    soft_min_val = (1.0 / t) * np.log(S) + d_min
    return grad, soft_min_val, d_min


def _local_optimize(points, max_iter=800, lr=0.005, deadline=None):
    """
    Local optimization using gradient descent on the soft-min of distances.
    Maximizes the minimum pairwise and boundary distance.
    """
    n = len(points)
    best_points = points.copy()
    best_score = _compute_score(best_points)

    t = 80.0  # Temperature for soft-min
    current_lr = lr
    patience = 0
    max_patience = 50

    start_time = time.time()

    for iteration in range(max_iter):
        # Deadline check
        if deadline is not None and time.time() > deadline:
            break

        # Compute gradient
        grad, soft_min_val, actual_min = _soft_min_gradient(best_points, t)

        # Gradient ascent (we want to maximize min distance)
        new_points = best_points + current_lr * grad

        # Keep points inside unit circle
        norms = np.linalg.norm(new_points, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            new_points[mask] = new_points[mask] / norms[mask, np.newaxis]

        # Compute actual score
        score = _compute_score(new_points)

        if score > best_score:
            best_score = score
            best_points = new_points.copy()
            patience = 0
            # Slightly increase learning rate if making progress
            current_lr = min(current_lr * 1.02, 0.02)
        else:
            patience += 1
            # Decay learning rate
            current_lr *= 0.97

        if patience >= max_patience:
            break

    return best_points


def _hex_lattice_init(spacing, angle_deg=0):
    """Generate hexagonal lattice points within the unit disk."""
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    k = int(1.0 / spacing) + 3
    rows = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            # Rotate
            xr = x * cos_a - y * sin_a
            yr = x * sin_a + y * cos_a
            rows.append((xr, yr))

    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]

    if len(points) < N:
        return None

    # Sort by distance from center and take N closest
    radius = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(radius, kind="stable")[:N]
    return points[idx]


def _radial_ring_init_optimal():
    """
    Deterministic radial-layer construction with analytically optimized radii.
    Circles are arranged in concentric rings. For each ring with m circles at
    radius r, the angular spacing constraint gives: r * sin(pi/m) >= r_c (circle radius).
    The radial constraint between adjacent rings: r_{i+1} - r_i >= 2*r_c.
    
    We solve for the maximum r_c by trying different ring configurations.
    """
    best_points = None
    best_score = -np.inf

    # Try different numbers of rings
    for n_rings in range(3, 20):
        # We need to distribute N circles among n_rings rings
        # Ring 0 is the center (1 circle), rings 1..n_rings-1 have m_i circles
        
        # Start with center circle
        counts = [1]
        remaining = N - 1
        
        # Distribute remaining circles proportionally to circumference
        # Ring i (i>=1) has radius roughly proportional to i
        weights = [i for i in range(1, n_rings)]
        total_w = sum(weights)
        
        for i in range(1, n_rings):
            c_i = max(1, int(round(remaining * weights[i-1] / total_w)))
            c_i = min(c_i, remaining - (n_rings - i))
            counts.append(c_i)
            remaining -= c_i
            total_w -= weights[i-1]
        
        if remaining > 0:
            counts[-1] += remaining
        
        if len(counts) != n_rings:
            continue
        
        # Now solve for optimal radii
        # r_c is the circle radius
        # Ring 0: center, r_0 = 0
        # Ring i: m_i circles at radius r_i
        # Constraints:
        #   r_i * sin(pi / m_i) >= r_c  (angular spacing)
        #   r_{i+1} - r_i >= 2 * r_c    (radial spacing)
        #   r_{n_rings-1} + r_c <= 1.0  (boundary)
        
        # From radial constraint: r_i >= 2*i*r_c
        # From angular constraint: r_i >= r_c / sin(pi / m_i)
        # So r_i >= max(2*i*r_c, r_c / sin(pi / m_i))
        
        # To maximize r_c, set r_i = max(2*i*r_c, r_c / sin(pi / m_i))
        # The binding constraint is r_{n_rings-1} + r_c <= 1.0
        
        # r_{n_rings-1} = max(2*(n_rings-1)*r_c, r_c / sin(pi / m_{n_rings-1}))
        # So r_c * max(2*(n_rings-1), 1/sin(pi / m_{n_rings-1})) + r_c <= 1.0
        # r_c * (max(2*(n_rings-1), 1/sin(pi / m_{n_rings-1})) + 1) <= 1.0
        
        # But we also need to check intermediate rings don't violate
        # Let's just compute r_c from the outermost constraint and verify
        
        m_outer = counts[-1]
        angular_factor = 1.0 / np.sin(np.pi / m_outer) if m_outer > 1 else 0.0
        radial_factor = 2.0 * (n_rings - 1)
        outer_factor = max(radial_factor, angular_factor)
        
        r_c = 1.0 / (outer_factor + 1.0)
        
        # Compute ring radii
        radii = [0.0]
        for i in range(1, n_rings):
            m_i = counts[i]
            ang_factor_i = 1.0 / np.sin(np.pi / m_i) if m_i > 1 else 0.0
            rad_factor_i = 2.0 * i
            r_i = r_c * max(rad_factor_i, ang_factor_i)
            radii.append(r_i)
        
        # Check boundary constraint
        if radii[-1] + r_c > 1.0 + 1e-9:
            continue
        
        # Check all constraints
        valid = True
        for i in range(n_rings):
            m_i = counts[i]
            if m_i > 1:
                if radii[i] * np.sin(np.pi / m_i) < r_c - 1e-9:
                    valid = False
                    break
        for i in range(1, n_rings):
            if radii[i] - radii[i-1] < 2.0 * r_c - 1e-9:
                valid = False
                break
        if radii[-1] + r_c > 1.0 + 1e-9:
            valid = False
        
        if not valid:
            continue
        
        # Generate points
        points = [(0.0, 0.0)]
        for i in range(1, n_rings):
            m_i = counts[i]
            r_i = radii[i]
            # Stagger odd rings
            offset = np.pi / m_i if i % 2 == 1 else 0.0
            angles = np.linspace(0, 2 * np.pi, m_i, endpoint=False) + offset
            for a in angles:
                points.append((r_i * np.cos(a), r_i * np.sin(a)))
        
        if len(points) != N:
            continue
        
        points = np.array(points)
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    return best_points


def _radial_ring_init():
    """
    Place circles in concentric rings. Each ring has a radial distance and
    an angular offset. We solve for ring radii and circle counts that maximize
    the minimum spacing.
    """
    best_points = None
    best_score = -np.inf

    # Try different ring structures
    for n_rings in range(1, 12):
        # Distribute N circles among rings
        # Outer rings can hold more circles
        # Use a heuristic: ring i (0-indexed) gets proportional to its circumference
        weights = []
        for i in range(n_rings):
            # Radius of ring i
            r_i = (i + 1) / n_rings * 0.95
            weights.append(2 * np.pi * r_i)

        total_w = sum(weights)
        counts = []
        remaining = N
        for i in range(n_rings - 1):
            c_i = max(1, int(round(N * weights[i] / total_w)))
            c_i = min(c_i, remaining - (n_rings - 1 - i))  # Leave at least 1 for each remaining ring
            counts.append(c_i)
            remaining -= c_i
        counts.append(remaining)

        # Now compute positions
        points = []
        for i, count in enumerate(counts):
            if count <= 0:
                continue
            r_i = (i + 1) / n_rings * 0.95
            # Angular offset to stagger rings
            offset = (i * np.pi) / max(1, count)
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + offset
            for a in angles:
                points.append((r_i * np.cos(a), r_i * np.sin(a)))

        if len(points) != N:
            continue

        points = np.array(points)
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

    return best_points


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin

    best_points = None
    best_score = -np.inf

    # Generate multiple initializations
    inits = []

    # 1. Hexagonal lattice with various spacings and angles
    for angle_deg in [0, 30, 60, 90]:
        for spacing_factor in [0.96, 0.98, 1.0, 1.02, 1.04]:
            spacing = 2.0 * spacing_factor / (np.sqrt(N) + 1.0)
            pts = _hex_lattice_init(spacing, angle_deg)
            if pts is not None:
                score = _compute_score(pts)
                inits.append((score, pts))

    # 2. Analytically optimized radial ring initialization
    ring_pts_opt = _radial_ring_init_optimal()
    if ring_pts_opt is not None:
        score = _compute_score(ring_pts_opt)
        inits.append((score, ring_pts_opt))

    # 3. Heuristic radial ring initialization
    ring_pts = _radial_ring_init()
    if ring_pts is not None:
        score = _compute_score(ring_pts)
        inits.append((score, ring_pts))

    # Sort by score, keep top few
    inits.sort(key=lambda x: -x[0])
    top_inits = inits[:5] if len(inits) >= 5 else inits

    # Local optimization on the best initializations
    for score, pts in top_inits:
        if time.time() > deadline - 30:
            break

        optimized = _local_optimize(pts, max_iter=500, lr=0.005, deadline=deadline - 20)
        opt_score = _compute_score(optimized)

        if opt_score > best_score:
            best_score = opt_score
            best_points = optimized.copy()

    # If we have time, do another round with more iterations on the best
    if best_points is not None and time.time() < deadline - 25:
        optimized = _local_optimize(best_points, max_iter=400, lr=0.003, deadline=deadline - 10)
        opt_score = _compute_score(optimized)
        if opt_score > best_score:
            best_score = opt_score
            best_points = optimized.copy()

    # Final validation and fallback
    if best_points is None:
        # Emergency fallback: hexagonal lattice
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        pts = _hex_lattice_init(spacing, 0)
        if pts is not None:
            best_points = pts
        else:
            # Last resort: spiral
            angles = np.linspace(0, 2 * np.pi * N, N)
            radii = np.linspace(0, 0.95, N)
            best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    # Ensure all points are strictly inside the unit circle
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] = best_points[mask] / norms[mask, np.newaxis] * 0.9999

    return best_points
# EVOLVE-BLOCK-END
