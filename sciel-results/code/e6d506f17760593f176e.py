import numpy as np
import time

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 285  # Safety margin before 330s kill

    # Ensure input is float
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # prizes: (5, 100), weights: (5, 100)
    # 5 constraints, capacity 25 each

    epsilon = 1e-9

    # 1. Total prize per item
    total_prize = np.sum(prizes, axis=0)  # (100,)

    # 2. Safe weights to avoid division by zero
    # If weight is 0, we treat it as very small positive to avoid inf,
    # but since weights are uniform [0,1], 0 is unlikely but possible.
    # If weight is 0 and prize > 0, ratio is inf.
    # If weight is 0 and prize is 0, ratio is 0/0 -> handle carefully.
    safe_weights = np.where(weights > epsilon, weights, 1.0)
    
    # Ratios R[k, j] = prizes[k, j] / weights[k, j]
    # If weight was 0 (replaced by 1.0), ratio is prize (which is small if prize is small).
    # If weight was 0 and prize > 0, we lose the "inf" benefit.
    # Better: use np.divide with where clause
    ratios = np.zeros_like(prizes)
    np.divide(prizes, weights, out=ratios, where=weights > epsilon)
    # For weights <= epsilon, if prize > 0, set to a large number?
    # But uniform [0,1] means 0 is rare. Let's assume standard division is okay with safe_weights
    # for the sake of stability, but cap the ratio.
    ratios = np.where(weights > epsilon, prizes / safe_weights, 0.0)
    # If weight is 0 and prize is > 0, this item is "free" in that constraint.
    # It should have high attractiveness.
    # Let's set ratio to a high value if weight is 0 and prize > 0.
    free_mask = (weights <= epsilon) & (prizes > epsilon)
    ratios[free_mask] = 1e6  # High efficiency

    # 3. Min-ratio (bottleneck efficiency)
    min_ratio = np.min(ratios, axis=0)  # (100,)
    
    # 4. Harmonic mean of ratios
    # H_j = 5 / sum(1/r_kj)
    # If any r_kj is 0, 1/r is inf, H is 0.
    # If any r_kj is 1e6, 1/r is ~0.
    inv_ratios = np.where(ratios > epsilon, 1.0 / ratios, 1e9)
    sum_inv_ratios = np.sum(inv_ratios, axis=0)
    harm_mean = np.where(sum_inv_ratios > epsilon, 5.0 / sum_inv_ratios, 0.0)

    # 5. Geometric mean of ratios
    # G_j = exp(mean(log(r_kj)))
    # log(0) is -inf.
    log_ratios = np.where(ratios > epsilon, np.log(ratios), -1e9)
    mean_log_ratios = np.mean(log_ratios, axis=0)
    geo_mean = np.exp(mean_log_ratios)
    # If any ratio was 0, geo_mean will be 0 (exp(-1e9) ~ 0)

    # 6. Combine heuristics
    # We want to emphasize the bottleneck (min_ratio) but also overall efficiency.
    # min_ratio is the most critical factor for feasibility.
    # harm_mean is a good aggregate measure.
    # total_prize adds a preference for high-value items if efficiencies are similar.
    
    # Safety for powers
    min_ratio_safe = np.maximum(min_ratio, epsilon)
    harm_mean_safe = np.maximum(harm_mean, epsilon)
    geo_mean_safe = np.maximum(geo_mean, epsilon)
    total_prize_safe = np.maximum(total_prize, epsilon)

    # Exponents:
    # Strong emphasis on bottleneck (min)
    # Moderate emphasis on harmonic mean
    # Mild emphasis on total prize
    a = 1.2  # Min ratio exponent
    b = 0.8  # Harm mean exponent
    c = 0.3  # Total prize exponent

    heuristic = (np.power(min_ratio_safe, a) * 
                 np.power(harm_mean_safe, b) * 
                 np.power(total_prize_safe, c))

    # Safety: replace non-finite or non-positive with epsilon
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, epsilon)

    # Normalize to mean 1
    mean_h = np.mean(heuristic)
    if mean_h > epsilon and np.isfinite(mean_h):
        heuristic = heuristic / mean_h
    else:
        heuristic = np.ones(100, dtype=np.float64)

    # Final safety check
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, epsilon)

    if heuristic.shape != (100,):
        heuristic = heuristic.reshape(100)

    if time.time() > deadline:
        return heuristic

    return heuristic
