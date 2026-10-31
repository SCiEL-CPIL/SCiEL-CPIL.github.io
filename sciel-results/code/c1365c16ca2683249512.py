"""Combined strategy: Probabilistic Best-Fit with Polarization and Adaptive Hybrid."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.special

# Module-level time tracking for deadline guard
_start_time = time.time()
_DEADLINE_SECONDS = 135.0  # Safety margin before 150s kill

# Weibull distribution parameters for the generated sequences
# Assuming a standard Weibull with shape k=2.0 and scale lambda=25.0
# This yields a mean of ~22 and median of ~19, which is reasonable for capacity 100.
# We calculate the expected size of the next item.
# For Weibull(k, lambda), E[X] = lambda * Gamma(1 + 1/k)
_k = 2.0
_lambda_scale = 25.0
_EXPECTED_NEXT_ITEM = _lambda_scale * scipy.special.gamma(1 + 1/_k)

CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for placing the given item.
    bins: array of remaining capacities.
    Returns: array of scores, higher is better.
    Only bins with enough capacity are scored; others get -inf.
    
    Strategy: Combined Probabilistic Best-Fit with Polarization and Adaptive Hybrid.
    1. Probabilistic Penalty: Discourages creating very small gaps that cannot
       accommodate future items (which have an expected size E[X]).
    2. Polarization Penalty: Discourages "middle" remainders (around CAPACITY/2)
       because half-full bins are less flexible than nearly-full or nearly-empty bins.
    3. Adaptive Hybrid: Blends best-fit and worst-fit based on system density and gap spread.
    """
    # Deadline guard: if we're close to timeout, use simple best-fit
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE_SECONDS:
        mask = bins >= item
        if not np.any(mask):
            return -np.inf * np.ones_like(bins)
        scores = -np.inf * np.ones_like(bins)
        gaps = bins[mask] - item
        scores[mask] = -gaps  # simple best-fit
        return scores
    
    remaining = bins
    mask = remaining >= item
    
    if not np.any(mask):
        return -np.inf * np.ones_like(remaining)
    
    valid_remaining = remaining[mask]
    gaps = valid_remaining - item
    
    # --- Component 1: Probabilistic Multiplier ---
    # Penalizes gaps that are too small to likely fit the next item.
    # If gap >> E[X], exp term is ~0, multiplier ~ 1.
    # If gap << E[X], exp term is large, multiplier is large -> strong penalty.
    sigma = 5.0
    arg = (gaps - _EXPECTED_NEXT_ITEM) / sigma
    arg = np.clip(arg, -10, 10)
    multiplier = 1.0 + np.exp(-arg)
    
    # Base score from Probabilistic Best-Fit
    base_score = -gaps * multiplier
    
    # --- Component 2: Polarization Penalty ---
    # Penalizes remainders that are close to CAPACITY/2.
    # min(rem_after, CAPACITY - rem_after) measures "how close to the middle".
    # We want to MINIMIZE this, so we subtract it from the score.
    # Note: rem_after is the gap itself in this context (space left after placing item).
    # Wait, polarization usually applies to the *filled* amount or the *remaining* amount relative to capacity.
    # If remaining is ~50, it's "middle". If remaining is ~0 or ~100, it's "polarized".
    # So we penalize when gap is close to CAPACITY/2? No, that doesn't make sense for bin packing.
    # Let's re-evaluate.
    # Inspiration 1 says: "min(rem_after, CAPACITY - rem_after) penalizes middle states".
    # If rem_after = 50, min(50, 50) = 50.
    # If rem_after = 10, min(10, 90) = 10.
    # If rem_after = 90, min(90, 10) = 10.
    # So it penalizes 50 more than 10 or 90. This encourages bins to be either nearly full (small gap) or nearly empty (large gap).
    # This aligns with the idea that half-full bins are less flexible.
    pol_penalty = np.minimum(gaps, CAPACITY - gaps)
    pol_weight = 0.5
    pol_score = -pol_weight * pol_penalty
    
    # --- Component 3: Adaptive Hybrid Adjustment ---
    # Compute average fill level across ALL bins (not just valid ones)
    avg_fill = 1.0 - np.mean(remaining) / CAPACITY
    
    # Gap spread
    gap_spread = np.std(gaps) if len(gaps) > 1 else 0.0
    gap_mean = np.mean(gaps) if len(gaps) > 0 else 0.0
    spread_ratio = gap_spread / (gap_mean + 1e-9) if gap_mean > 0 else 0.0
    
    # Weight for worst-fit component: increases as fill decreases and spread increases
    fill_component = max(0.0, 0.5 - avg_fill)  # 0.5 when empty, 0 when 50%+ full
    spread_component = min(0.5, spread_ratio * 0.3)
    worst_fit_weight = fill_component + spread_component
    worst_fit_weight = min(worst_fit_weight, 0.5)  # cap at 0.5
    
    # The adaptive hybrid in the current program adjusted the sign/magnitude of the gap term.
    # Here, we can modulate the weight of the probabilistic term vs a pure gap term.
    # Or, we can add a term that encourages larger gaps when worst_fit_weight is high.
    # Let's add a term: worst_fit_weight * gaps
    # This counteracts the -gaps term in base_score when worst_fit_weight is high.
    # base_score = -gaps * multiplier
    # hybrid_term = worst_fit_weight * gaps
    # Net gap coefficient = -multiplier + worst_fit_weight
    # If multiplier ~ 1 (large gap), net = -1 + w. If w=0.5, net = -0.5 (still prefers smaller gap, but less strongly).
    # If multiplier is large (small gap), net = -large + w. Still strongly negative.
    
    hybrid_score = worst_fit_weight * gaps
    
    # --- Combine ---
    final_scores = base_score + pol_score + hybrid_score
    
    # Create full score array
    scores = -np.inf * np.ones_like(remaining)
    scores[mask] = final_scores
    
    return scores

# EVOLVE-BLOCK-END
