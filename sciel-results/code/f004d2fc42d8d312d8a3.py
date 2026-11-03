"""Fast force-based dynamics + local polish for packing 100 equal spheres
in the unit cube [-1/2, 1/2]^3.

Strategy:
  1. Start from structured (BCC) or random initializations.
  2. Run fast gradient-descent dynamics with a repulsive force model that only
     considers pairs closer than a cutoff (much faster than full O(N^2) exp).
  3. Polish with a "shrink-and-relax" local search that directly maximizes the
     minimum inter-particle / wall distance.
  4. Track the best exact radius seen and return that configuration.
"""
import time
import numpy as np

N = 100
HALF = 0.5
DEADLINE_MARGIN = 8.0   # safety margin before the 150 s kill


# ---------------- exact objective (the thing we actually score) ----------------
def _radius(pts):
    """Exact feasible common radius for a set of centres (vectorized)."""
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2))
    wall = HALF - np.max(np.abs(pts), axis=1)
    min_wall = np.min(wall)
    r = 0.5 * min(min_pair, 2.0 * min_wall)
    if not np.isfinite(r) or r < 0:
        return 0.0
    return float(r)


def _min_dist_and_wall(pts):
    """Return (min_pair_dist, min_wall_dist) without computing radius."""
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2))
    wall = HALF - np.max(np.abs(pts), axis=1)
    min_wall = np.min(wall)
    return min_pair, min_wall


# ---------------- fast force computation with cutoff ----------------
def _forces(pts, cutoff=0.35):
    """Compute repulsive forces on each particle from neighbors within cutoff.
    Force on i from j: (pts_i - pts_j) / d_ij^2  for d_ij < cutoff (soft repulsion).
    Also add wall forces pushing inward.
    """
    n = pts.shape[0]
    diff = pts[:, None, :] - pts[None, :, :]          # (n,n,3)
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    
    # Only interact within cutoff
    mask = d2 < cutoff * cutoff
    # Force magnitude: 1/d^2 for close pairs, 0 otherwise
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_d2 = np.where(mask, 1.0 / d2, 0.0)
    # Force vector: (pts_i - pts_j) / d_ij^2
    forces = np.einsum('ij,ijk->ik', inv_d2, diff)
    
    # Wall forces: push inward if close to wall
    wall_dist = HALF - np.abs(pts)  # (n,3), positive if inside
    # Strong force when very close to wall
    wall_forces = np.where(wall_dist < 0.05, 
                          np.sign(pts) * np.maximum(0, 0.05 - wall_dist) * 10.0, 
                          0.0)
    
    return forces + wall_forces


# ---------------- initializers ----------------
def _init_grid():
    """Simple cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].astype(np.float64)


def _init_bcc():
    """BCC lattice initialization - known good starting point for sphere packing."""
    # BCC: points at corners and body centers of small cubes
    # In a unit cube [-0.5, 0.5]^3, place BCC points
    m = 5  # 5x5x5 grid of BCC unit cells gives 250 points, we take 100
    pts = []
    s = 1.0 / m
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                cx = (ix + 0.5) * s - 0.5
                cy = (iy + 0.5) * s - 0.5
                cz = (iz + 0.5) * s - 0.5
                # Corner
                pts.append([cx - s/2, cy - s/2, cz - s/2])
                # Body center
                pts.append([cx + s/2, cy + s/2, cz + s/2])
    pts = np.array(pts[:N], dtype=np.float64)
    return pts


def _init_random(rng):
    return (rng.random((N, 3)) - 0.5) * 0.85


def _clamp(pts):
    return np.clip(pts, -HALF, HALF)


# ---------------- fast gradient descent with annealed noise ----------------
def _fast_dynamics(pts, rng, deadline, n_steps=3000):
    """Run fast force-based dynamics with annealed noise.
    Much faster than SGLD because forces use cutoff and simpler computation.
    """
    pts = _clamp(pts.copy())
    best_pts = pts.copy()
    best_r = _radius(best_pts)

    lr0, lr1 = 0.005, 1e-5
    T0, T1 = 0.01, 1e-5
    cutoff0, cutoff1 = 0.4, 0.15

    for it in range(n_steps):
        if time.time() > deadline:
            break
        frac = it / max(1, n_steps - 1)
        lr = lr0 * (lr1 / lr0) ** frac
        T = T0 * (T1 / T0) ** frac
        cutoff = cutoff0 * (cutoff1 / cutoff0) ** frac

        f = _forces(pts, cutoff)
        noise = np.sqrt(2.0 * lr * T) * rng.standard_normal((N, 3))
        pts = _clamp(pts + lr * f + noise)

        # Periodic exact evaluation
        if it % 200 == 0 or it == n_steps - 1:
            r = _radius(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()

    # final exact eval
    r = _radius(pts)
    if r > best_r:
        best_r = r
        best_pts = pts.copy()

    return best_pts, best_r


# ---------------- local shrink-and-relax polish ----------------
def _local_polish(pts, deadline, max_iter=200):
    """Local search: repeatedly find the closest pair/wall, push them apart.
    This directly optimizes the minimum distance.
    """
    pts = _clamp(pts.copy())
    best_pts = pts.copy()
    best_r = _radius(best_pts)

    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Find the minimum constraint
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, np.inf)
        
        wall = HALF - np.max(np.abs(pts), axis=1)
        
        min_pair_d2 = np.min(d2)
        min_wall = np.min(wall)
        
        # Current radius
        r = 0.5 * min(np.sqrt(min_pair_d2), 2.0 * min_wall)
        
        # If we improved, update best
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
        
        # Find which constraint is active
        if np.sqrt(min_pair_d2) < 2.0 * min_wall:
            # Pair constraint is active
            idx = np.unravel_index(np.argmin(d2), d2.shape)
            i, j = idx[0], idx[1]
            if i == j:
                continue
            d_vec = pts[i] - pts[j]
            d_mag = np.sqrt(min_pair_d2)
            if d_mag < 1e-10:
                continue
            # Push apart
            push = 0.002
            pts[i] += push * d_vec / d_mag
            pts[j] -= push * d_vec / d_mag
        else:
            # Wall constraint is active
            wi = np.argmin(wall)
            # Find which coordinate is closest to wall
            coords = np.abs(pts[wi])
            k = np.argmax(coords)
            sign = np.sign(pts[wi, k])
            pts[wi, k] -= 0.002 * sign
        
        pts = _clamp(pts)
    
    # Final eval
    r = _radius(pts)
    if r > best_r:
        best_r = r
        best_pts = pts.copy()
    
    return best_pts, best_r


def _polish(pts, deadline):
    """Combined polish: local search + small perturbations."""
    pts = _clamp(pts.copy())
    best_r = _radius(pts)
    best_pts = pts.copy()
    
    # First, local polish
    p1, r1 = _local_polish(pts, deadline, max_iter=150)
    if r1 > best_r:
        best_r = r1
        best_pts = p1
    
    # Try a few random perturbations and re-polish
    rng = np.random.default_rng(42)
    for _ in range(5):
        if time.time() > deadline:
            break
        perturbed = best_pts + 0.005 * rng.standard_normal((N, 3))
        perturbed = _clamp(perturbed)
        p2, r2 = _local_polish(perturbed, deadline, max_iter=80)
        if r2 > best_r:
            best_r = r2
            best_pts = p2
    
    return best_pts, best_r


# ---------------- top level ----------------
def construct():
    t0 = time.time()
    deadline = 150.0 - DEADLINE_MARGIN

    rng = np.random.default_rng(20240517)

    # Global best
    best_pts = _init_grid()
    best_r = _radius(best_pts)

    # Try BCC initialization first - often gives good starting point
    bcc = _init_bcc()
    bcc_r = _radius(bcc)
    if bcc_r > best_r:
        best_r = bcc_r
        best_pts = bcc.copy()

    # List of initializations to process
    inits = [best_pts.copy()]
    for _ in range(4):
        inits.append(_init_random(rng))

    for init in inits:
        if time.time() > deadline:
            break
        
        # Fast dynamics to explore the landscape
        p1, r1 = _fast_dynamics(init, rng, deadline, n_steps=2500)
        if r1 > best_r:
            best_r = r1
            best_pts = p1.copy()
        
        # Local polish to sharpen
        p2, r2 = _polish(p1, deadline)
        if r2 > best_r:
            best_r = r2
            best_pts = p2.copy()

    # Use spare time for extra restarts
    guard = 0
    while time.time() < deadline - 2.0 and guard < 6:
        guard += 1
        start = _init_random(rng)
        p1, r1 = _fast_dynamics(start, rng, deadline, n_steps=1500)
        if r1 > best_r:
            best_r = r1
            best_pts = p1.copy()
        p2, r2 = _polish(p1, deadline)
        if r2 > best_r:
            best_r = r2
            best_pts = p2.copy()

    # Final safety: ensure the returned point is feasible and finite
    out = _clamp(np.asarray(best_pts, dtype=np.float64))
    if not np.all(np.isfinite(out)):
        out = _init_grid()
    return out
