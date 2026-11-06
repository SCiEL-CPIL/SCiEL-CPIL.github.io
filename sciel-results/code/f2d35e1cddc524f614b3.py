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
    # Check boundaries
    dist_to_bound = np.min(np.stack([
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ], axis=1), axis=1)
    min_boundary = np.min(dist_to_bound)
    if min_boundary < -1e-12:
        return 0.0
    # Check pairwise distances
    # Use broadcasting for N=50 (2500 pairs, fine)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    if min_pair < -1e-12:
        return 0.0
    return min(min_boundary, min_pair)


def _hex_packing(params):
    """
    Generate hexagonal lattice packing with alternating row lengths.
    params: [sx, sy, x_offset, y_offset]
    sx: horizontal spacing between centers in a row
    sy: vertical spacing between rows
    x_offset: horizontal shift for odd rows (relative to even rows)
    y_offset: vertical centering offset
    """
    sx, sy, x_off, y_off = params
    
    # Row lengths pattern: 7, 6, 7, 6, 7, 6, 7, 4
    row_lengths = [7, 6, 7, 6, 7, 6, 7, 4]
    n_rows = len(row_lengths)
    
    centers = []
    for i, length in enumerate(row_lengths):
        y = -0.5 + sy * (i + 0.5) + y_off
        # Horizontal positions for this row
        # Center the row horizontally
        if i % 2 == 0:
            x_start = -sx * (length - 1) / 2.0
        else:
            x_start = -sx * (length - 1) / 2.0 + x_off
        
        for j in range(length):
            x = x_start + j * sx
            centers.append([x, y])
    
    centers = np.array(centers)
    return centers


def construct():
    """
    Construct 50 circle centers using hexagonal lattice packing with
    alternating row lengths, optimized to maximize the common radius.
    """
    # Initial guess
    # For hexagonal packing, ideal sy = sx * sqrt(3)/2
    # We want to fit 7 circles in the widest row: 6*sx <= 1.0 => sx <= 1/6 ≈ 0.1667
    # But we also need to fit vertically: 7 rows with spacing sy
    # Total height = 7 * sy, need 7*sy <= 1.0 => sy <= 1/7 ≈ 0.1429
    # For hex: sy = sx * sqrt(3)/2 ≈ 0.866 * sx
    # If sx = 1/6, sy = 0.1443, which is slightly > 1/7. So we need to adjust.
    
    # Let's start with a reasonable guess
    x0 = np.array([1.0/6.0, 1.0/7.0, 0.0, 0.0])
    
    def _neg_score(params):
        centers = _hex_packing(params)
        return -_score_from_centers(centers)
    
    # Bounds: sx, sy positive; x_off small; y_off small
    bounds = [(0.05, 0.5), (0.05, 0.5), (-0.5, 0.5), (-0.5, 0.5)]
    
    best_result = None
    best_score = -np.inf
    
    # Try multiple starting points
    starts = [
        np.array([1.0/6.0, 1.0/7.0, 0.0, 0.0]),
        np.array([0.15, 0.14, 0.0, 0.0]),
        np.array([0.14, 0.13, 0.0, 0.0]),
        np.array([0.16, 0.145, 0.0, 0.0]),
        np.array([0.155, 0.135, 0.0, 0.0]),
    ]
    
    for x0_trial in starts:
        try:
            result = minimize(_neg_score, x0_trial, method='Nelder-Mead',
                              options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12})
            score = -result.fun
            if score > best_score:
                best_score = score
                best_result = result
        except Exception:
            pass
    
    if best_result is not None:
        centers = _hex_packing(best_result.x)
        # Verify validity
        final_score = _score_from_centers(centers)
        if final_score > 0:
            return centers
    
    # Fallback: grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]
# EVOLVE-BLOCK-END
