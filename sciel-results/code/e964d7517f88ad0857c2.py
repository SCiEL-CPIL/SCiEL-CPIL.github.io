"""Packing 50 equal circles in a unit circle - force-based optimization."""
import time
import numpy as np

N = 50
_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 0.0
    if N < 2:
        return r_cont
    # Pairwise distances using broadcasting
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dmat = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(dmat, np.inf)
    min_pair = np.min(dmat)
    r_pair = min_pair / 2.0
    return min(r_cont, r_pair)


def _hex_lattice_init(n, jitter=0.0, seed=42):
    """Generate initial positions from hexagonal lattice, trimmed to n points."""
    rng = np.random.RandomState(seed)
    # Try different spacings until we get at least n points in the unit disk
    for spacing in np.linspace(0.14, 0.35, 40):
        k = int(1.0 / spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    rows.append((x, y))
        pts = np.array(rows)
        if len(pts) >= n:
            # Shuffle and take first n
            idx = rng.permutation(len(pts))[:n]
            pts = pts[idx]
            if jitter > 0:
                pts = pts + rng.randn(n, 2) * jitter
            # Project back inside unit circle
            r = np.hypot(pts[:, 0], pts[:, 1])
            mask = r > 1.0
            if np.any(mask):
                pts[mask] = pts[mask] / r[mask, None] * 0.999
            return pts
    # Fallback: random
    theta = rng.uniform(0, 2 * np.pi, n)
    rho = np.sqrt(rng.uniform(0, 1, n))
    return np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])


def _force_optimize(pts, max_iters=20000, dt_init=0.05, dt_min=1e-6, decay=0.9995):
    """Force-based optimization to maximize minimum distance."""
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    dt = dt_init
    best_score = _score(pts)
    best_pts = pts.copy()
    
    # Precompute indices for pairwise
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5.0:
            break
        
        # Compute all pairwise distances
        dx = pts[i_idx, 0] - pts[j_idx, 0]
        dy = pts[i_idx, 1] - pts[j_idx, 1]
        d = np.sqrt(dx * dx + dy * dy)
        d = np.maximum(d, 1e-12)
        
        # Repulsion force: strong when close, F = 1/d^2 in direction away
        force_mag = 1.0 / (d * d)
        fx = force_mag * dx / d
        fy = force_mag * dy / d
        
        # Accumulate forces on each point
        Fx = np.zeros(n)
        Fy = np.zeros(n)
        np.add.at(Fx, i_idx, -fx)  # i is pushed away from j
        np.add.at(Fx, j_idx, fx)
        np.add.at(Fy, i_idx, -fy)
        np.add.at(Fy, j_idx, fy)
        
        # Attraction toward center to keep points inside
        r = np.hypot(pts[:, 0], pts[:, 1])
        r_safe = np.maximum(r, 1e-12)
        # Stronger attraction near boundary
        center_force = 0.1 * (r_safe > 0.5).astype(float)
        Fx -= center_force * pts[:, 0] / r_safe
        Fy -= center_force * pts[:, 1] / r_safe
        
        # Normalize forces to prevent explosion
        fmag = np.hypot(Fx, Fy)
        fmag_max = np.max(fmag)
        if fmag_max > 10.0:
            scale = 10.0 / fmag_max
            Fx *= scale
            Fy *= scale
        
        # Adaptive step size: reduce if score decreases
        new_pts = pts + dt * np.column_stack([Fx, Fy])
        
        # Keep inside container
        nr = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = nr > 1.0
        if np.any(mask):
            new_pts[mask] = new_pts[mask] / nr[mask, None] * 0.9999
        
        new_score = _score(new_pts)
        
        if new_score >= best_score - 1e-12:
            pts = new_pts
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
            dt = min(dt * 1.01, 0.1)  # Slightly increase
        else:
            dt *= decay
        
        if dt < dt_min:
            break
    
    return best_pts, best_score


def construct():
    """Construct 50 circle centers in unit circle maximizing common radius."""
    best_pts = None
    best_score = 0.0
    
    # Try multiple initializations
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        if restart == 0:
            # First: hexagonal lattice, no jitter
            pts = _hex_lattice_init(N, jitter=0.0, seed=42)
        elif restart == 1:
            # Second: hexagonal with small jitter
            pts = _hex_lattice_init(N, jitter=0.02, seed=123)
        else:
            # Random perturbations of hex lattice
            base = _hex_lattice_init(N, jitter=0.0, seed=42)
            pts = base + np.random.RandomState(42 + restart).randn(N, 2) * 0.05 * (restart - 1)
            r = np.hypot(pts[:, 0], pts[:, 1])
            mask = r > 1.0
            if np.any(mask):
                pts[mask] = pts[mask] / r[mask, None] * 0.999
        
        # Run force optimization
        optimized_pts, score = _force_optimize(
            pts,
            max_iters=15000,
            dt_init=0.03,
            dt_min=1e-7,
            decay=0.999
        )
        
        if score > best_score:
            best_score = score
            best_pts = optimized_pts.copy()
    
    if best_pts is None:
        # Fallback
        best_pts = _hex_lattice_init(N, jitter=0.0, seed=42)
    
    # Final validation and cleanup
    r = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = r > 1.0
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / r[mask, None] * 0.9999
    
    return best_pts
