"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 150.0
_START = None


def _score(pts):
    """Return the common radius r for a set of centres (higher is better)."""
    # pts: (N,2)
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # edge clearance: distance from each centre to the nearest wall
    edge = 0.5 - np.max(np.abs(pts), axis=1)
    min_edge = float(np.min(edge))
    if min_edge <= 0.0:
        return 0.0
    # pairwise distances
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = float(np.min(dist))
    return min(min_pair / 2.0, min_edge)


def _make_hex_initial(n):
    """Hexagonal lattice packed into [-0.5,0.5]^2, return first n points."""
    # target: we want min distance as large as possible. Hex packing density ~ pi/(2*sqrt(3)).
    # For N points in area 1, hex nearest-neighbour spacing s ~ sqrt(2*sqrt(3)/(pi*N)) * ... 
    # Actually area per point in hex = s^2 * sqrt(3)/2. So s ~ sqrt(2/(sqrt(3)*N)).
    s = np.sqrt(2.0 / (np.sqrt(3.0) * n))
    # generate hex lattice
    pts = []
    # spacing in x = s, in y = s*sqrt(3)/2
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    # rows
    max_row = int(1.0 / dy) + 3
    for row in range(-max_row, max_row + 1):
        y = row * dy
        if y > 0.5 or y < -0.5:
            continue
        # x offset for stagger
        xoff = (dx / 2.0) if (row % 2 != 0) else 0.0
        max_col = int(1.0 / dx) + 3
        for col in range(-max_col, max_col + 1):
            x = col * dx + xoff
            if x > 0.5 or x < -0.5:
                continue
            pts.append((x, y))
    pts = np.array(pts)
    if len(pts) == 0:
        # fallback grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]
    # sort by distance from centre to keep a balanced core
    order = np.argsort(np.sum(pts * pts, axis=1))
    pts = pts[order]
    if len(pts) < n:
        # fill with grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        pts = np.vstack([pts, grid[:n - len(pts)]])
    return pts[:n].copy()


def _local_search(pts, budget):
    """Repulsion-based local search to maximise min distance."""
    global _START
    best = pts.copy()
    best_score = _score(best)
    n = len(pts)

    # cooling schedule: start with a step that's a fraction of the current min distance
    # We'll iterate, moving each point by a small fraction of the repulsion.
    # Use a simple iterative relaxation with decreasing step.

    iters = 0
    # Determine a reasonable number of iterations based on budget
    # Each iteration is O(N^2). For N=200, that's 40000 ops, very fast.
    # We can afford many iterations.
    max_iters = 50000
    # We'll do a geometric cooling of step size.
    step0 = 0.01  # initial step as fraction
    # Actually, let's use a different approach: for each point, compute the vector
    # that would increase its distance to the closest neighbor, and move a bit.

    # Simpler and effective: "repulsion" where each pair pushes apart if too close,
    # and walls push inward. Use a global step that cools.

    # Let's use the approach: at each iteration, for each point, compute
    #   force = sum over j of (p_i - p_j)/|p_i-p_j|  (for all j, weighted by 1/dist)
    # plus wall force. Then move p_i += step * force / |force| (normalize to unit)
    # This tends to equalize distances.

    # Better: use the "max-min" gradient. The objective is min over pairs of dist.
    # Subgradient: for the closest pair, push apart. But that's unstable.
    # Use soft repulsion: force_i = sum_j (p_i - p_j) / max(dist_ij, eps)^2
    # This gives stronger push for closer pairs.

    eps = 1e-6
    step = 0.005
    cool = 0.9995
    stall = 0

    for it in range(max_iters):
        # deadline check
        if time.time() - _START > _DEADLINE - 5.0:
            break

        p = best.copy()
        # compute forces
        forces = np.zeros_like(p)
        # pairwise
        d = p[:, None, :] - p[None, :, :]  # (N,N,2): d[i,j] = p_i - p_j
        dist = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dist, np.inf)
        # repulsion: d_ij / dist_ij^2  (unit direction scaled by 1/dist)
        # force_i = sum_j d_ij / dist_ij^2
        inv_dist2 = np.zeros_like(dist)
        mask = dist > eps
        inv_dist2[mask] = 1.0 / (dist[mask] * dist[mask])
        forces = np.sum(d * inv_dist2[:, :, None], axis=1)

        # wall force: push inward if close to wall
        for k in range(2):
            close_hi = p[:, k] > 0.5 - 0.05
            close_lo = p[:, k] < -0.5 + 0.05
            forces[close_hi, k] -= 1.0
            forces[close_lo, k] += 1.0

        # normalize forces to unit magnitude, then scale by step
        fmag = np.sqrt(np.sum(forces * forces, axis=1))
        fmag[fmag < 1e-12] = 1.0
        move = forces / fmag[:, None] * step

        # apply
        new_p = p + move
        # clamp to square (with tiny margin)
        new_p = np.clip(new_p, -0.5, 0.5)

        sc = _score(new_p)
        if sc > best_score:
            best = new_p
            best_score = sc
            stall = 0
        else:
            stall += 1
            # if stalled, reduce step
            if stall > 200:
                step *= 0.9
                stall = 0
                if step < 1e-5:
                    break

        iters += 1
        # periodic full recompute of best (already done)
        if it % 500 == 0 and it > 0:
            # occasionally try a small random perturbation to escape local minima
            if np.random.random() < 0.1:
                pert = best + np.random.normal(0, 0.002, best.shape)
                pert = np.clip(pert, -0.5, 0.5)
                sc2 = _score(pert)
                if sc2 > best_score:
                    best = pert
                    best_score = sc2

    return best, best_score


def construct():
    global _START
    if _START is None:
        _START = time.time()

    # initial hex packing
    pts = _make_hex_initial(N)
    base_score = _score(pts)

    # local search
    pts, sc = _local_search(pts, budget=_DEADLINE - 10.0)

    # final safety: ensure validity
    if not np.all(np.isfinite(pts)) or len(pts) != N:
        # fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    # clamp to be strictly inside
    pts = np.clip(pts, -0.5, 0.5)
    return pts
# EVOLVE-BLOCK-END
