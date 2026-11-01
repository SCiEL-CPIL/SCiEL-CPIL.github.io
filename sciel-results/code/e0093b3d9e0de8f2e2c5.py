"""Improved packer: physics-based repulsive force simulation + gradient polish for 300 circles."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 300
SEED = 12345
DEADLINE = 145.0  # safety margin before 150s wall-clock limit


def _score(points):
    """Return the feasible common radius (0.0 if invalid)."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - norm
    if np.min(r_bound) < -1e-12:
        return 0.0
    # Clip tiny negative margins to zero for robustness
    r_bound = np.maximum(r_bound, 0.0)
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    r_pair = 0.5 * np.min(dists[:, 1])
    r = min(r_bound.min(), r_pair)
    return float(r) if r > 0.0 else 0.0


def _min_dist(points):
    """Efficiently compute the minimum pairwise distance."""
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    return float(np.min(dists[:, 1]))


def _hex_lattice_seed(spacing):
    """Generate a hexagonal lattice and pick N points inside the disk."""
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        keep = rad <= 1.0 - spacing / 2.0
        pts = pts[keep]
        if len(pts) >= N:
            order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
            return pts[order[:N]].copy()
        spacing *= 0.97


def _physics_simulate(points, t_budget, rng):
    """
    Physics-based simulation: circles repel each other and the wall.
    Uses a temperature-annealed velocity Verlet-like scheme.
    """
    start = time.time()
    pts = points.copy()
    vel = np.zeros_like(pts)
    best = pts.copy()
    best_r = _score(best)

    # Initial parameters
    d0 = _min_dist(pts)
    if d0 <= 0:
        d0 = 0.01
    T = 0.5 * d0  # temperature (max step scale)
    T_min = 1e-6 * d0
    damping = 0.95  # velocity damping per step

    # Force parameters
    repel_range = 3.0 * d0  # range for pair repulsion
    wall_force_k = 100.0
    pair_force_k = 50.0

    while time.time() - start < t_budget and T > T_min:
        # Compute forces
        forces = np.zeros_like(pts)

        # Wall forces: push inward if close to boundary
        norms = np.linalg.norm(pts, axis=1)
        wall_dist = 1.0 - norms
        wall_mask = wall_dist < d0
        if np.any(wall_mask):
            # Direction is inward (toward origin)
            if np.any(norms[wall_mask] > 1e-12):
                wall_dir = -pts[wall_mask] / norms[wall_mask, None]
                # Force magnitude proportional to how close to wall
                wall_strength = np.clip(1.0 - wall_dist[wall_mask] / d0, 0, 1) ** 2
                forces[wall_mask] += wall_force_k * wall_strength[:, None] * wall_dir

        # Pair repulsion forces using KD-tree for neighbors
        tree = cKDTree(pts)
        # Query neighbors within repel_range
        pairs = tree.query_pairs(r=repel_range, output_type='ndarray')
        if len(pairs) > 0:
            # Vectorize pair force computation
            p1 = pts[pairs[:, 0]]
            p2 = pts[pairs[:, 1]]
            diff = p1 - p2  # vector from j to i
            dists = np.linalg.norm(diff, axis=1)
            # Avoid division by zero
            mask = dists > 1e-12
            if np.any(mask):
                # Repulsive force: stronger when closer
                # Force magnitude ~ k * (1 - d/range)^2 for d < range
                strength = np.zeros(len(pairs))
                strength[mask] = pair_force_k * np.clip(1.0 - dists[mask] / repel_range, 0, 1) ** 2
                # Direction: away from neighbor (diff/dist)
                dir_vec = diff[mask] / dists[mask, None]
                f_vec = strength[mask, None] * dir_vec
                # Accumulate forces
                for idx in range(len(f_vec)):
                    forces[pairs[idx, 0]] += f_vec[idx]
                    forces[pairs[idx, 1]] -= f_vec[idx]

        # Update velocities and positions
        # Scale forces to reasonable step size
        max_force = np.max(np.linalg.norm(forces, axis=1)) if np.any(forces != 0) else 1.0
        if max_force > 0:
            forces_scaled = forces / max_force * T
        else:
            forces_scaled = forces

        # Add small random jitter to break symmetry
        jitter = rng.standard_normal((N, 2)) * T * 0.1
        vel = damping * vel + forces_scaled + jitter
        pts += vel

        # Enforce boundary constraint: project back inside
        norms = np.linalg.norm(pts, axis=1)
        out_mask = norms > 1.0 - 1e-9
        if np.any(out_mask):
            pts[out_mask] = pts[out_mask] / norms[out_mask, None] * (1.0 - 1e-9)
            vel[out_mask] *= 0.5  # damp velocity of boundary-hitting points

        # Evaluate and track best
        cur_r = _score(pts)
        if cur_r > best_r:
            best_r = cur_r
            best = pts.copy()

        # Cool down
        T *= 0.9995
        damping = min(damping * 1.0005, 0.99)  # gradually increase damping

    return best, best_r


def _gradient_polish(points, t_budget, rng):
    """
    Gradient-based polish: iteratively move each point in the direction
    that increases the minimum distance (either to neighbors or to wall).
    """
    start = time.time()
    pts = points.copy()
    best = pts.copy()
    best_r = _score(best)

    d0 = _min_dist(pts)
    if d0 <= 0:
        d0 = 0.01
    step = 0.05 * d0
    min_step = 1e-7 * d0

    while time.time() - start < t_budget:
        # Find the bottleneck: which constraint is tightest?
        norms = np.linalg.norm(pts, axis=1)
        wall_margins = 1.0 - norms
        tree = cKDTree(pts)
        dists, idxs = tree.query(pts, k=2)
        min_neighbor_dists = dists[:, 1]
        min_neighbor_idxs = idxs[:, 1]

        # Find the point with the smallest margin
        # Combine wall and pair constraints
        min_margin = min(wall_margins.min(), 0.5 * min_neighbor_dists.min())

        # Identify tight points
        tight_thresh = 0.05 * d0
        wall_tight = np.where(wall_margins < tight_thresh)[0]
        pair_tight = np.where(min_neighbor_dists < 2.0 * tight_thresh)[0]
        tight_idx = np.unique(np.concatenate([wall_tight, pair_tight]))

        if len(tight_idx) == 0:
            idx = rng.integers(N)
        else:
            idx = tight_idx[rng.integers(len(tight_idx))]

        # Compute gradient direction for point idx
        # Try to increase the minimum constraint involving idx
        force = np.zeros(2)

        # Wall constraint
        norm_i = np.linalg.norm(pts[idx])
        if norm_i > 1e-12:
            wall_dir = -pts[idx] / norm_i  # inward
            wall_margin = 1.0 - norm_i
            if wall_margin < tight_thresh:
                force += wall_dir * (tight_thresh - wall_margin) / tight_thresh

        # Pair constraints
        # Find neighbors within 2*tight_thresh
        neighbor_mask = min_neighbor_dists < 2.0 * tight_thresh
        for j in range(N):
            if j == idx:
                continue
            d = np.linalg.norm(pts[idx] - pts[j])
            if d < 2.0 * tight_thresh and d > 1e-12:
                # Push away from j
                dir_away = (pts[idx] - pts[j]) / d
                force += dir_away * (2.0 * tight_thresh - d) / (2.0 * tight_thresh)

        # Normalize force
        force_norm = np.linalg.norm(force)
        if force_norm > 1e-12:
            force = force / force_norm
        else:
            # Random direction if no clear gradient
            ang = rng.random() * 2.0 * np.pi
            force = np.array([np.cos(ang), np.sin(ang)])

        # Propose move
        new_pt = pts[idx].copy()
        new_pt += force * step

        # Check boundary
        if np.hypot(new_pt[0], new_pt[1]) > 1.0 - 1e-9:
            # Project back
            n = np.linalg.norm(new_pt)
            new_pt = new_pt / n * (1.0 - 1e-9)

        old_pt = pts[idx].copy()
        pts[idx] = new_pt
        new_r = _score(pts)

        if new_r >= best_r:
            if new_r > best_r:
                best_r = new_r
                best = pts.copy()
            step *= 0.999
        else:
            pts[idx] = old_pt
            step *= 1.01

        if step < min_step:
            step = min_step
        if step > 0.2 * d0:
            step = 0.2 * d0

    return best, best_r


def _local_optimize(points, t_budget, rng):
    """
    Local optimization: for each point, try to find a better position
    by moving it in small steps along the direction that increases
    its minimum distance to neighbors and wall.
    """
    start = time.time()
    pts = points.copy()
    best = pts.copy()
    best_r = _score(best)

    d0 = _min_dist(pts)
    if d0 <= 0:
        d0 = 0.01
    base_step = 0.02 * d0

    # Iteratively improve each point
    max_iter = 10000
    for it in range(max_iter):
        if time.time() - start > t_budget:
            break

        for idx in range(N):
            if time.time() - start > t_budget:
                break

            # Current minimum distance for this point
            tree = cKDTree(pts)
            dists, _ = tree.query(pts[idx:idx+1], k=2)
            min_d = dists[0, 1]
            norm_i = np.linalg.norm(pts[idx])
            wall_d = 1.0 - norm_i
            min_constraint = min(min_d / 2.0, wall_d)

            # Try 8 directions
            best_move = None
            best_new_r = _score(pts)

            for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                dx = base_step * np.cos(ang)
                dy = base_step * np.sin(ang)
                new_pt = pts[idx].copy()
                new_pt[0] += dx
                new_pt[1] += dy

                if np.hypot(new_pt[0], new_pt[1]) > 1.0 - 1e-9:
                    continue

                old_pt = pts[idx].copy()
                pts[idx] = new_pt
                new_r = _score(pts)

                if new_r > best_new_r:
                    best_new_r = new_r
                    best_move = new_pt.copy()
                else:
                    pts[idx] = old_pt

            if best_move is not None:
                pts[idx] = best_move
                if best_new_r > best_r:
                    best_r = best_new_r
                    best = pts.copy()

    return best, best_r


def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # Initial hex-lattice seed
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    pts = _hex_lattice_seed(spacing)
    # Add random perturbation to break symmetry
    jitter = 0.02 * rng.standard_normal((N, 2))
    pts = pts + jitter
    # Clip inside container
    rad = np.linalg.norm(pts, axis=1)
    mask = rad > 1.0 - 1e-6
    if np.any(mask):
        pts[mask] = pts[mask] / rad[mask, None] * (1.0 - 1e-6)

    best = pts.copy()
    best_r = _score(best)

    # Phase 1: Physics-based simulation
    t_left = DEADLINE - (time.time() - t0)
    if t_left > 30.0:
        t_alloc = min(t_left - 25.0, 60.0)
        best, best_r = _physics_simulate(best, t_alloc, rng)

    # Phase 2: Gradient polish
    t_left = DEADLINE - (time.time() - t0)
    if t_left > 15.0:
        t_alloc = min(t_left - 10.0, 30.0)
        best, best_r = _gradient_polish(best, t_alloc, rng)

    # Phase 3: Local optimization
    t_left = DEADLINE - (time.time() - t0)
    if t_left > 5.0:
        t_alloc = min(t_left - 2.0, 15.0)
        best, best_r = _local_optimize(best, t_alloc, rng)

    # Final safety check
    final_r = _score(best)
    if final_r <= 0.0:
        best = _hex_lattice_seed(2.0 / (np.sqrt(N) + 1.0))

    return best
