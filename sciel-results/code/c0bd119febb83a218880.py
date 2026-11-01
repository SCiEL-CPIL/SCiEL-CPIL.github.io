"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 150.0  # wall-clock limit; leave safety margin
_MARGIN = 5.0      # seconds to spare


def _build_rings(n_total):
    """Build a ring-based hexagonal packing. Returns (points, d) where d is the
    target circle diameter (2*r). Rings are placed so adjacent rings touch.
    Ring k: radius r_k, n_k points. r_1 = d/2, r_{k+1} = r_k + d*cos(pi/n_k).
    We pick n_k to roughly match hexagonal growth and stop when we have >= n_total."""
    # Start with ring 0 (center) + rings outward
    # For hex packing, ring k (k>=1) has 6k points, but we adapt.
    # We'll use: ring 0: 1 point at center; ring k>=1: n_k = 6*k (hexagonal)
    # But that gives 1 + 6 + 12 + ... = 1 + 3k(k+1) points for k rings.
    # For 300: 3k(k+1) + 1 >= 300 -> k(k+1) >= 99.67 -> k=10: 110, 1+330=331. Good.
    # But 331 > 300, so we trim.
    #
    # However, the touching condition with varying n_k per ring is what matters.
    # Let's use a simpler approach: fixed number of points per ring, chosen so
    # the total is close to N, and the radii follow the touching recurrence.
    
    # Strategy: use ring sizes that grow ~ linearly (hexagonal: 6, 12, 18, ...)
    # Ring 0: 1 (center)
    # Ring k: 6*k points for k = 1, 2, ...
    # Total after K rings: 1 + 6*(1+2+...+K) = 1 + 3K(K+1)
    # Find K such that total >= N
    K = 1
    while 1 + 3 * K * (K + 1) < n_total:
        K += 1
    
    ring_sizes = [1] + [6 * k for k in range(1, K + 1)]
    total = sum(ring_sizes)
    
    # Trim: remove excess from the outermost rings (least important for packing)
    excess = total - n_total
    while excess > 0:
        # Remove from the last ring
        last = ring_sizes[-1]
        if last > 0:
            remove = min(excess, last)
            ring_sizes[-1] -= remove
            excess -= remove
            if ring_sizes[-1] == 0:
                ring_sizes.pop()
        else:
            ring_sizes.pop()
    
    # Now build radii using touching condition
    # r_0 = 0 (center)
    # For ring k >= 1: r_k = r_{k-1} + d * cos(pi / n_k) where n_k is the number
    # of points in ring k (the angular spacing determines the chord).
    # Actually the touching between ring k-1 and ring k: the distance between
    # adjacent points in neighboring rings should be ~ d.
    # A good approximation: r_k = r_{k-1} + d * cos(pi / max(n_k, 3))
    
    d = 1.0  # will be scaled
    radii = [0.0]
    for k in range(1, len(ring_sizes)):
        n_k = ring_sizes[k]
        if n_k >= 3:
            radii.append(radii[-1] + d * np.cos(np.pi / n_k))
        else:
            radii.append(radii[-1] + d)
    
    # Scale so the outermost point is at radius 1 - d/2 (circle fits inside unit disk)
    r_max = radii[-1]
    if r_max > 0:
        scale = (1.0 - d / 2.0) / r_max
        radii = [r * scale for r in radii]
        # Recompute d after scaling: the actual diameter is the minimum distance
        # But for construction, we just need positions; d will be determined by optimization.
    
    # Build points
    points = []
    for k, (r, n) in enumerate(zip(radii, ring_sizes)):
        if n == 1:
            points.append([0.0, 0.0])
        elif n == 2:
            points.append([r, 0.0])
            points.append([-r, 0.0])
        else:
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / n  # offset
            for a in angles:
                points.append([r * np.cos(a), r * np.sin(a)])
    
    return np.array(points[:n_total]), len(points)


def _score(points):
    """Compute the packing radius: min of pairwise distances/2 and boundary distances."""
    n = len(points)
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    # Boundary
    r_center = np.sqrt(np.sum(points ** 2, axis=1))
    min_bound = np.min(1.0 - r_center)
    return min(min_pair, min_bound)


def _soft_score(points, temp=0.001):
    """Differentiable soft version of the packing radius."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)
    np.fill_diagonal(dist, np.inf)
    # Soft min of pairwise distances / 2
    log_sum_pair = np.logaddexp.reduce(np.log(dist / 2.0) / temp) * temp
    soft_pair = np.exp(log_sum_pair)
    # Boundary
    r_center = np.sqrt(np.sum(points ** 2, axis=1))
    bound = 1.0 - r_center
    log_sum_bound = np.logaddexp.reduce(np.log(bound + 1e-12) / temp) * temp
    soft_bound = np.exp(log_sum_bound)
    # Overall soft min
    return np.minimum(soft_pair, soft_bound)


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE - _MARGIN
    
    # Step 1: Build initial ring-based configuration
    pts, _ = _build_rings(N)
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Step 2: Optimize using L-BFGS-B to maximize the soft packing radius
    # Flatten points to 1D
    x0 = pts.flatten()
    
    def neg_soft_score(x):
        if time.time() > deadline:
            return 1e6  # signal to stop
        p = x.reshape(-1, 2)
        # Keep points inside the unit disk (hard constraint via penalty)
        r = np.sqrt(np.sum(p ** 2, axis=1))
        penalty = np.sum(np.maximum(0, r - 0.999) ** 2)
        return -_soft_score(p, temp=0.001) + penalty
    
    # Try optimization with time limit
    try:
        result = minimize(
            neg_soft_score,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 2000, 'gtol': 1e-6, 'disp': False},
            callback=None
        )
        if result.x is not None:
            new_pts = result.x.reshape(-1, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_pts = new_pts
                best_score = new_score
    except Exception:
        pass
    
    # Step 3: If time remains, do a second optimization pass with lower temperature
    if time.time() < deadline - 10:
        try:
            x1 = best_pts.flatten()
            result2 = minimize(
                neg_soft_score,
                x1,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 1000, 'gtol': 1e-7, 'disp': False},
            )
            if result2.x is not None:
                new_pts = result2.x.reshape(-1, 2)
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_pts = new_pts
                    best_score = new_score
        except Exception:
            pass
    
    # Step 4: Final polish with exact local solve if time remains
    if time.time() < deadline - 5:
        try:
            x2 = best_pts.flatten()
            # Use a tighter temperature for the final polish
            def neg_soft_score_tight(x):
                p = x.reshape(-1, 2)
                r = np.sqrt(np.sum(p ** 2, axis=1))
                penalty = np.sum(np.maximum(0, r - 0.999) ** 2)
                return -_soft_score(p, temp=0.0001) + penalty
            
            result3 = minimize(
                neg_soft_score_tight,
                x2,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 500, 'gtol': 1e-8, 'disp': False},
            )
            if result3.x is not None:
                new_pts = result3.x.reshape(-1, 2)
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_pts = new_pts
                    best_score = new_score
        except Exception:
            pass
    
    # Ensure validity: no points outside unit disk, finite values
    r = np.sqrt(np.sum(best_pts ** 2, axis=1))
    if np.any(~np.isfinite(best_pts)) or np.any(r > 1.0):
        # Fallback to ring construction
        best_pts, _ = _build_rings(N)
    
    return best_pts
# EVOLVE-BLOCK-END
