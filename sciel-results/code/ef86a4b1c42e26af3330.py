"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin with enough remaining capacity using Next-Item Awareness.
    
    Estimates the expected size of the next item from the distribution of
    last-item-sizes in partially filled bins, then scores each candidate bin
    based on how well its post-placement remaining capacity aligns with this
    predicted next item size to minimize expected capacity waste.
    """
    scores = np.full(len(bins), -np.inf, dtype=float)
    
    # 1. Identify valid bins (those with enough capacity for the current item)
    valid_mask = bins >= item
    if not np.any(valid_mask):
        return scores
        
    valid_indices = np.where(valid_mask)[0]
    new_rems = bins[valid_indices] - item
    
    # 2. Estimate the expected size of the next item (S_next)
    # Extract last-item-sizes from partially filled bins: last_item = 100 - bin_capacity
    used_mask = (bins < 100) & (bins > 0)
    if np.any(used_mask):
        last_item_sizes = 100.0 - bins[used_mask]
        # Use median for robustness against outliers
        s_next = np.median(last_item_sizes)
    else:
        # No partially filled bins; assume next item is similar to current
        s_next = item
        
    # Clamp s_next to reasonable range
    s_next = float(np.clip(s_next, 0.1, 100.0))
    
    # 3. Classify bins as "Safe" or "Unsafe"
    # Safe: remaining space >= s_next (can fit next item)
    # Unsafe: remaining space < s_next (dead zone likely)
    is_safe = new_rems >= s_next
    
    # 4. Calculate scores
    # Safe bins: prefer tighter fits (smaller new_rem) -> score = -new_rem
    # Unsafe bins: prefer larger new_rem (closer to s_next) -> score = new_rem
    #   but penalize heavily so they rank below all safe bins
    
    valid_scores = np.empty(len(new_rems), dtype=float)
    
    # Safe bins: best-fit among safe
    safe_mask = is_safe
    valid_scores[safe_mask] = -new_rems[safe_mask]
    
    # Unsafe bins: prefer those closest to s_next from below
    # Add a large negative offset to ensure they rank below safe bins
    unsafe_mask = ~is_safe
    valid_scores[unsafe_mask] = new_rems[unsafe_mask] - 1000.0
    
    # 5. Assign scores back to the output array
    scores[valid_indices] = valid_scores
    
    return scores
# EVOLVE-BLOCK-END
