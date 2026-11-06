"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _score_from_centers(centers):
    """Compute the common radius for a set of centers inside [-0.5, 0.5]^2."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check boundaries: distance from each center to the nearest wall
    # Walls are at x=0.5, x=-0.5, y=0.5, y=-0.5
    dist_x = 0.5 - np.abs(centers[:, 0])
    dist_y = 0.5 - np.abs(centers[:, 1])
    min_boundary = np.min(np.minimum(dist_x, dist_y))
    
    if min_boundary < 0:
        # If any center is outside, the radius is effectively negative/invalid
        # We return a large negative number to penalize heavily in minimization
        return min_boundary - 10.0

    # Check pairwise distances
    # Efficient calculation for N=50
    # We can use pdist or manual broadcasting. Manual is fine for 50.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    if min_pair < 0:
        return min_pair - 10.0

    return min(min_boundary, min_pair)


def _generate_hex_packing(params, row_lengths):
    """
    Generate hexagonal lattice packing.
    params: [sx, sy, x_shift_odd, x_global, y_global]
    row_lengths: list of integers for each row's circle count.
    Returns array of shape (N, 2) or None if invalid.
    """
    sx, sy, x_shift_odd, x_global, y_global = params
    n_rows = len(row_lengths)
    
    centers = []
    for i, length in enumerate(row_lengths):
        if length == 0:
            continue
        # Vertical position: centered around y_global
        y = y_global + (i - (n_rows - 1) / 2.0) * sy
        
        # Horizontal position
        # Center the row horizontally around x_global
        # Standard hex shift: odd rows (0-indexed) shifted by x_shift_odd
        shift = x_shift_odd if i % 2 != 0 else 0.0
        
        x_start = x_global + shift - sx * (length - 1) / 2.0
        for j in range(length):
            x = x_start + j * sx
            centers.append([x, y])
            
    if len(centers) != N:
        return None
        
    return np.array(centers)


def construct():
    """
    Construct 50 circle centers using hexagonal lattice packing with
    various row length patterns, optimized to maximize the common radius.
    """
    
    # Define candidate row patterns summing to 50.
    # We focus on patterns that allow for a hexagonal structure.
    # Common optimal structures for 50 circles often involve 6-8 rows.
    patterns = [
        # 7 rows
        [7, 7, 7, 7, 7, 7, 8],
        [8, 7, 7, 7, 7, 7, 7],
        [7, 7, 7, 7, 7, 8, 7],
        [7, 7, 7, 7, 8, 7, 7],
        [7, 7, 7, 8, 7, 7, 7],
        [7, 7, 8, 7, 7, 7, 7],
        [7, 8, 7, 7, 7, 7, 7],
        [8, 8, 7, 7, 7, 7, 6],
        [6, 8, 7, 7, 7, 8, 7],
        [7, 8, 7, 8, 7, 8, 5],
        [5, 8, 7, 8, 7, 8, 7],
        [7, 8, 7, 8, 7, 7, 8],
        [8, 7, 8, 7, 7, 8, 7],
        
        # 6 rows
        [9, 8, 8, 8, 8, 9],
        [8, 9, 8, 8, 9, 8],
        [9, 9, 8, 8, 8, 8],
        [8, 8, 9, 8, 9, 8],
        [8, 9, 9, 8, 8, 8],
        [9, 8, 9, 8, 8, 8],
        [8, 8, 8, 9, 9, 8],
        [9, 9, 9, 8, 8, 7],
        [7, 8, 9, 9, 9, 8],
        [8, 9, 9, 8, 9, 7],
        [7, 9, 8, 9, 8, 9],
        [9, 8, 9, 8, 9, 7],
        [8, 9, 7, 9, 9, 8],
        
        # 8 rows
        [6, 7, 6, 7, 6, 7, 6, 7],
        [7, 6, 7, 6, 7, 6, 7, 4],
        [6, 7, 6, 7, 6, 7, 7, 4],
        [7, 6, 7, 6, 7, 7, 6, 4],
        [6, 6, 7, 7, 7, 7, 6, 4],
        [4, 7, 7, 7, 7, 7, 7, 4],
        [5, 6, 7, 7, 7, 7, 6, 5],
        [6, 6, 7, 7, 7, 7, 6, 4],
        [7, 7, 7, 7, 7, 7, 7, 1],
        [8, 6, 8, 6, 8, 6, 8, 0], # Sum 50? 8+6+8+6+8+6+8 = 50. Yes.
        [6, 8, 6, 8, 6, 8, 8, 0], # Sum 50? 6+8+6+8+6+8+8 = 50. Yes.
        
        # 5 rows (likely worse but check)
        [10, 10, 10, 10, 10],
        [11, 10, 10, 10, 9],
        [10, 11, 10, 9, 10],
        [10, 10, 11, 9, 10],
        [10, 10, 10, 11, 9],
        [9, 11, 10, 10, 10],
        [11, 11, 9, 10, 9],
        [10, 11, 11, 9, 9],
        [9, 10, 11, 11, 9],
        [9, 11, 10, 11, 9],
        
        # 9 rows
        [5, 6, 5, 6, 5, 6, 5, 6, 6],
        [6, 5, 6, 5, 6, 5, 6, 5, 6],
        [5, 6, 6, 5, 6, 5, 6, 5, 6],
        [6, 5, 6, 6, 5, 6, 5, 6, 5],
        [5, 6, 5, 6, 6, 5, 6, 5, 6],
        [6, 5, 6, 5, 6, 6, 5, 6, 5],
        [5, 5, 6, 6, 6, 6, 6, 6, 5],
        [5, 6, 5, 6, 6, 6, 6, 6, 5],
    ]
    
    valid_patterns = []
    seen = set()
    for p in patterns:
        if sum(p) == N and len(p) > 0:
            # Normalize pattern to avoid duplicates (e.g., reversed)
            # We treat patterns as ordered, but reversing a symmetric pattern is same.
            # For optimization, we just need unique lists.
            key = tuple(p)
            if key not in seen:
                seen.add(key)
                valid_patterns.append(p)
                
    best_score = -np.inf
    best_centers = None
    
    # Pre-calculate some constants
    sqrt3 = np.sqrt(3)
    
    for pattern in valid_patterns:
        n_rows = len(pattern)
        max_len = max(pattern)
        
        # Estimate initial sx based on width constraint
        # Width = sx * (max_len - 1) + 2*r <= 1
        # If r = sx/2 (hexagonal touching), Width = sx * max_len <= 1 => sx <= 1/max_len
        # However, we also have vertical constraints.
        
        # Ideal hexagonal spacing: sy = sx * sqrt(3) / 2
        # Height = sy * (n_rows - 1) + 2*r <= 1
        # Height = sx * sqrt(3) / 2 * (n_rows - 1) + sx <= 1
        # sx * (sqrt(3)/2 * (n_rows - 1) + 1) <= 1
        # sx <= 1 / (sqrt(3)/2 * (n_rows - 1) + 1)
        
        sx_width = 1.0 / max_len
        sx_height = 1.0 / (sqrt3 / 2.0 * (n_rows - 1) + 1.0)
        sx_init = min(sx_width, sx_height) * 0.95 # Slightly smaller to allow optimization
        
        sy_init = sx_init * sqrt3 / 2.0
        
        # If n_rows == 1, sy is irrelevant but we set it to sx
        if n_rows == 1:
            sy_init = sx_init
            
        # Initial shift
        shift_init = sx_init / 2.0
        
        # Try multiple starting points for Nelder-Mead
        starts = [
            [sx_init, sy_init, shift_init, 0.0, 0.0],
            [sx_init, sy_init, -shift_init, 0.0, 0.0],
            [sx_init * 1.05, sy_init * 0.95, shift_init, 0.0, 0.0],
            [sx_init * 0.95, sy_init * 1.05, shift_init, 0.0, 0.0],
            [sx_init, sy_init, 0.0, 0.0, 0.0], # No shift
        ]
        
        def _neg_score(params):
            centers = _generate_hex_packing(params, pattern)
            if centers is None:
                return 100.0 # Large penalty
            score = _score_from_centers(centers)
            return -score
        
        bounds = [
            (0.01, 0.5),   # sx
            (0.01, 0.5),   # sy
            (-0.5, 0.5),   # x_shift_odd
            (-0.5, 0.5),   # x_global
            (-0.5, 0.5)    # y_global
        ]
        
        for x0 in starts:
            try:
                # Use Nelder-Mead for local optimization
                # We don't strictly need bounds if we penalize invalid states,
                # but Nelder-Mead doesn't handle bounds natively well.
                # We rely on the penalty function to keep values reasonable.
                result = minimize(_neg_score, x0, method='Nelder-Mead',
                                  options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_centers = _generate_hex_packing(result.x, pattern)
            except Exception:
                pass
    
    if best_centers is not None and best_score > 0:
        return best_centers
    
    # Fallback: Simple Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]
# EVOLVE-BLOCK-END
