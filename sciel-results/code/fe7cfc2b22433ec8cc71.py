"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for an online bin packing problem with Weibull-distributed items.
    Strategy: Model bin packing as fragmentation minimization by scoring each bin
    based on the negative variance of the current bin remainder distribution,
    thereby penalizing strategies that create a wide spread of unusable small gaps
    and favoring those that maintain a tight cluster of remainders likely to be
    filled by future items.
    """
    # Deadline guard
    t0 = time.time()
    deadline = 120.0  # seconds

    rem = bins - item
    valid = rem >= 0
    scores = np.full_like(rem, -np.inf, dtype=np.float64)

    if not np.any(valid):
        return scores

    valid_rems = rem[valid]
    n_valid = len(valid_rems)

    if n_valid == 0:
        return scores

    # Compute the variance of the remainder distribution across all valid bins.
    # We want to penalize bins that would increase the variance of remainders.
    # Idea: For each valid bin, compute what the new remainder would be, then
    # estimate the change in variance of the overall remainder distribution.
    #
    # Variance = mean(rem^2) - mean(rem)^2
    #
    # If we place item in bin i, the new remainder for bin i becomes rem[i],
    # and the mean and mean-square change accordingly.
    #
    # Let S1 = sum(rem[valid]), S2 = sum(rem[valid]^2), n = n_valid
    # Old mean = S1/n, Old mean_sq = S2/n
    # Old var = S2/n - (S1/n)^2
    #
    # After placing item in bin i, the remainder for bin i changes from
    # bins[i] to bins[i] - item = rem[i].
    # New S1' = S1 - bins[i] + rem[i] = S1 - item
    # New S2' = S2 - bins[i]^2 + rem[i]^2
    # New mean = S1'/n, New mean_sq = S2'/n
    # New var = S2'/n - (S1'/n)^2
    #
    # We want to MINIMIZE the new variance, so we score = -new_var.

    S1 = np.sum(valid_rems)
    S2 = np.sum(valid_rems ** 2)
    n = float(n_valid)

    # Old mean and mean_sq (not strictly needed for delta, but useful)
    # old_mean = S1 / n
    # old_var = S2 / n - (S1 / n) ** 2

    # New S1 is constant regardless of which bin we pick: S1 - item
    new_S1 = S1 - item
    new_mean = new_S1 / n

    # For each valid bin i, compute new S2
    # S2' = S2 - bins[valid_i]^2 + rem[valid_i]^2
    valid_bins = bins[valid]
    new_S2 = S2 - valid_bins ** 2 + valid_rems ** 2
    new_mean_sq = new_S2 / n
    new_var = new_mean_sq - new_mean ** 2

    # Score: lower variance is better, so score = -new_var
    # But we also need a tiebreaker: prefer bins that fit the item well
    # (i.e., smaller remainder) to reduce waste.
    # Combine: primary score is -new_var, secondary is -rem (best-fit).

    # Normalize -new_var to be comparable with -rem
    # Actually, let's just use a weighted combination.
    # The variance term should dominate, but we add a small best-fit component.

    # Clip new_var to avoid numerical issues
    new_var = np.maximum(new_var, 0.0)

    # Primary: minimize variance -> score_var = -new_var
    # Secondary: minimize remaining -> score_rem = -rem
    #
    # Normalize both to [0, 1] range for blending
    var_min = new_var.min()
    var_max = new_var.max()
    rem_min = valid_rems.min()
    rem_max = valid_rems.max()

    if var_max > var_min:
        norm_var = (new_var - var_min) / (var_max - var_min)
    else:
        norm_var = np.zeros_like(new_var)

    if rem_max > rem_min:
        norm_rem = (valid_rems - rem_min) / (rem_max - rem_min)
    else:
        norm_rem = np.zeros_like(valid_rems)

    # Lower variance is better -> score_var = 1 - norm_var
    # Lower remainder is better -> score_rem = 1 - norm_rem
    score_var = 1.0 - norm_var
    score_rem = 1.0 - norm_rem

    # Weighted blend: 70% variance minimization, 30% best-fit
    blended = 0.7 * score_var + 0.3 * score_rem

    # Check deadline
    if time.time() - t0 > deadline - 2.0:
        # Return best-fit only as a safe fallback
        scores[valid] = -valid_rems
        return scores

    scores[valid] = blended
    return scores
# EVOLVE-BLOCK-END
