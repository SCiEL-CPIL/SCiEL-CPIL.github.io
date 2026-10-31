"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for online learning of the item size distribution
_stats = {'sum': 0.0, 'sum_sq': 0.0, 'count': 0}
_start_time = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill
_SAFETY_MARGIN = 5.0


def _weibull_optimal_gap(item_size, mean_item):
    """
    Estimate the theoretical optimal gap for a Weibull-distributed item.
    For Weibull with shape k~2 (typical for bin packing), the optimal
    remaining capacity after packing an item of size s is related to
    the tail behavior. We approximate the "ideal" gap as a function
    of the current mean item size and the specific item.
    
    The idea: if items follow Weibull(k, lambda), the optimal gap to
    maximize the probability of fitting future items is roughly
    proportional to the mean but adjusted for the specific item size.
    """
    # Use a simple heuristic: optimal gap is mean_item * (1 - item_size / (mean_item + 10))
    # This creates a gap that's larger for smaller items (more room for many small items)
    # and smaller for larger items (they're harder to fill)
    if mean_item <= 0:
        return mean_item
    # Clamp to avoid extreme values
    gap = mean_item * max(0.1, 1.0 - item_size / (mean_item + 15.0))
    return gap


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Update global statistics with the current item
    _stats['sum'] += item
    _stats['sum_sq'] += item * item
    _stats['count'] += 1
    
    # Check time - if we're close to the limit, fall back to simple best-fit
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT - _SAFETY_MARGIN:
        # Simple best-fit as fallback
        remaining = bins - item
        mask = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        scores[mask] = -remaining[mask]
        return scores
    
    # Compute remaining capacities
    remaining = bins - item
    mask = remaining >= 0
    
    # If no bins can fit the item, return all -inf (invalid)
    if not np.any(mask):
        return np.full(len(bins), -np.inf)
    
    # Get current mean item size
    n = _stats['count']
    mean_item = _stats['sum'] / n if n > 0 else 10.0  # Default to 10.0
    
    # Compute the "optimal gap" for this item
    optimal_gap = _weibull_optimal_gap(item, mean_item)
    
    # Score: we want the bin whose remaining capacity is closest to optimal_gap
    # Use negative absolute deviation as the score (higher is better)
    deviation = np.abs(remaining - optimal_gap)
    
    # Additional refinement: penalize very large gaps (waste) and very small gaps (fragility)
    # The KL-divergence-inspired approach: score is higher when the gap distribution
    # matches the expected equilibrium distribution
    # Approximate: prefer gaps near optimal, with diminishing returns
    
    # Score = -deviation^2 / (optimal_gap + 1) to normalize by scale
    scores = np.full(len(bins), -np.inf)
    scores[mask] = -(deviation[mask] ** 2) / (optimal_gap + 1.0)
    
    # Add a small preference for bins that already have some items (not empty)
    # to avoid fragmenting bins, but keep it small
    # This is a soft constraint: bins with remaining capacity close to full capacity
    # (i.e., few items) get a slight penalty
    capacity = 100.0  # From the task description
    used_fraction = 1.0 - remaining / capacity
    # Small bonus for bins that are moderately filled (0.3 to 0.8)
    fill_bonus = np.zeros(len(bins))
    moderate = (used_fraction >= 0.3) & (used_fraction <= 0.8) & mask
    fill_bonus[moderate] = 0.01
    
    scores[mask] += fill_bonus[mask]
    
    return scores
# EVOLVE-BLOCK-END
