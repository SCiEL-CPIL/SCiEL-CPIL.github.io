"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute one attractiveness score per item for ACO of multi-constraint knapsack.
    
    prizes: (5, 100) or (100,) uniform [0,1]
    weights: (5, 100) uniform [0,1]
    Returns: (100,) array, larger = more attractive.
    
    Strategy:
    1. Compute per-constraint efficiency ratios: p / w_c.
    2. Use soft-min (log-sum-exp) for smooth bottleneck identification.
    3. Use harmonic mean for overall efficiency.
    4. Compute cosine similarity with the efficient frontier direction.
    5. Combine all three signals with tuned exponents.
    6. Normalize to mean 1 for stability.
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle shape of prizes
    if prizes.ndim == 2:
        # Sum across the 5 dimensions to get a single "total prize" per item
        item_prizes = np.sum(prizes, axis=0)
    else:
        item_prizes = prizes
    
    # Ensure weights is 2D
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
        
    n_constraints = weights.shape[0]
    n_items = weights.shape[1]
    
    # Avoid division by zero
    safe_weights = np.where(weights > 1e-9, weights, 1e-9)
    
    # Per-constraint efficiency ratios: (5, 100)
    ratios = item_prizes / safe_weights
    
    # --- Signal 1: Soft-min bottleneck (smooth surrogate for hard min) ---
    # soft_min = -alpha * log(sum(exp(-ratios/alpha)))
    # Small alpha -> closer to true min, larger alpha -> smoother
    alpha = 0.1
    neg_ratios_over_alpha = -ratios / alpha
    # Clip to avoid overflow in exp
    neg_ratios_over_alpha = np.clip(neg_ratios_over_alpha, -500, 500)
    log_sum_exp = alpha * np.log(np.sum(np.exp(neg_ratios_over_alpha), axis=0))
    soft_min = -log_sum_exp
    
    # --- Signal 2: Harmonic mean of ratios ---
    # harm_mean = n / sum(1/r_i)
    inv_ratios = 1.0 / ratios
    harm_mean = float(n_constraints) / np.sum(inv_ratios, axis=0)
    
    # --- Signal 3: Cosine similarity with efficient frontier ---
    # Frontier direction: mean efficiency per constraint across all items
    frontier = np.mean(ratios, axis=1)  # shape (5,)
    frontier_norm = np.linalg.norm(frontier)
    if frontier_norm < 1e-9:
        frontier_norm = 1e-9
    
    # Item efficiency vectors: ratios is (5, 100), so each column is an item's vector
    item_norms = np.linalg.norm(ratios, axis=0)  # shape (100,)
    item_norms_safe = np.where(item_norms > 1e-9, item_norms, 1e-9)
    
    # Cosine similarity: (frontier . item_vec) / (|frontier| * |item_vec|)
    # frontier is (5,), ratios is (5,100), so dot product: frontier @ ratios gives (100,)
    dot_products = np.dot(frontier, ratios)  # shape (100,)
    cosine_sim = dot_products / (frontier_norm * item_norms_safe)
    # Clamp to [0, 1] since we expect positive alignment
    cosine_sim = np.clip(cosine_sim, 0.0, 1.0)
    
    # --- Combine signals ---
    # Use the measured successful exponent structure: 1.5 for bottleneck, 0.5 for overall
    # Add cosine similarity as a multiplicative boost
    h = (soft_min ** 1.5) * (harm_mean ** 0.5) * (cosine_sim ** 0.3)
    
    # Ensure positive values
    h = np.where(h > 0, h, 1e-9)
    
    # Normalize to mean 1 to keep the scale consistent with pheromone initialization
    mean_h = np.mean(h)
    if mean_h > 0:
        h = h / mean_h
        
    return h
# EVOLVE-BLOCK-END
