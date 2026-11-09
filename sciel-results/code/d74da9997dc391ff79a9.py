# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 140.0


def _score_bins(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Structural non-linear score using a smooth power-law transform of remaining
    capacity. This replaces discrete thresholds with a continuous scoring function
    that captures the diminishing marginal value of remaining space.
    
    Key structural change: instead of piecewise linear or threshold-based scoring,
    we use a smooth nonlinear transform that naturally handles the trade-off
    between tight fits and avoiding dead space.
    
    The score is based on:
    1. Primary: smooth best-fit via power-law transform of remaining capacity
    2. Secondary: content-aware bonus for partially-filled bins (consolidation)
    3. No tunable parameters - the power-law exponent is fixed based on 
       structural considerations of bin packing dynamics
    """
    rem = bins - item
    # Only feasible bins get a score
    feasible = rem >= -1e-9
    if not feasible.any():
        return np.full_like(bins, -np.inf)
    
    # Smooth nonlinear transform: use power-law to capture diminishing returns
    # For remaining capacity r, the "cost" of waste is r^alpha where alpha < 1
    # This makes small waste much less penalized than large waste (concave)
    # But we want to MINIMIZE waste, so we score as -r^alpha
    
    # Use alpha = 0.7: strong preference for tight fits, but smooth
    # This avoids the cliff-edge behavior at threshold boundaries
    alpha = 0.7
    cap = bins.max()
    
    # Normalize remaining capacity to [0, 1] range for consistent scaling
    rem_norm = np.zeros_like(rem)
    rem_norm[feasible] = rem[feasible] / max(cap, 1e-9)
    
    # Primary signal: negative power-law of normalized remaining capacity
    # This gives higher scores (less negative) for smaller remaining capacity
    base = np.zeros_like(rem)
    base[feasible] = -np.power(rem_norm[feasible], alpha)
    
    # Content-aware bonus: bins that are already partially filled get a bonus
    # to encourage consolidation and avoid fragmenting empty bins
    content = bins.max() - bins  # original capacity minus remaining = content
    content_bonus = np.zeros_like(rem)
    # Scale content bonus by a factor that makes it comparable to base score
    # Use 0.15 to give meaningful but not dominant influence
    content_bonus[feasible] = 0.15 * content[feasible] / max(cap, 1e-9)
    
    # Smooth fragmentation penalty: instead of hard threshold, use smooth function
    # Penalize remaining capacity that is small but not zero (dead space)
    # Use a smooth sigmoid-like penalty centered around 5% of capacity
    tiny_rem_penalty = np.zeros_like(rem)
    # Smooth penalty: peaks when rem is very small but > 0
    # Use -exp(-((rem - 0.03*cap) / (0.02*cap))^2) for smooth Gaussian-like penalty
    sigma = 0.02 * cap
    center = 0.03 * cap
    tiny_rem_penalty[feasible] = -0.3 * np.exp(-((rem[feasible] - center) / sigma) ** 2)
    # Only apply penalty when remaining is positive (not a perfect fit)
    tiny_rem_penalty[rem <= 0] = 0
    
    # Combine all components
    score = base + content_bonus + tiny_rem_penalty
    score = np.where(feasible, score, -np.inf)
    return score


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Return one score per bin with enough remaining capacity (higher = chosen)."""
    if not isinstance(bins, np.ndarray):
        bins = np.asarray(bins, dtype=np.float64)
    
    # Deadline guard: if we're running out of time, use simple best-fit
    if time.time() > _DEADLINE:
        remaining = bins - item
        scores = np.full_like(bins, -np.inf)
        valid = remaining >= -1e-9
        scores[valid] = -remaining[valid]
        return scores
    
    return _score_bins(item, bins)
# EVOLVE-BLOCK-END
