"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin based on the negative variance of the remaining capacities
    after placing the item in that bin.
    
    This approach treats bin selection as a risk minimization problem where the
    'risk' is the variance of the remaining space distribution. By minimizing
    the variance of the post-placement state, we aim to keep the distribution
    of remaining spaces tight, which is hypothesized to reduce fragmentation
    and improve packing density compared to heuristics that only look at local fit.
    
    We use an O(N) algorithm to compute the variance for each candidate bin by
    updating the sum and sum of squares of the bin capacities.
    """
    n_bins = len(bins)
    if n_bins == 0:
        return np.array([])
        
    # Calculate initial statistics
    sum_bins = np.sum(bins)
    sum_sq_bins = np.sum(bins ** 2)
    
    # If we place 'item' in a bin, the new sum of capacities is (sum_bins - item)
    new_sum = sum_bins - item
    # The new mean is constant for all choices
    new_mean = new_sum / n_bins
    
    # Pre-compute the constant part of the variance formula: - (new_mean)^2
    # Var = E[X^2] - (E[X])^2
    # We want to minimize Var, so we maximize -Var.
    # Score_j = - (S2'_j / N - new_mean^2) = -S2'_j / N + new_mean^2
    # Since new_mean^2 is constant, we can just maximize -S2'_j / N
    # Or simply minimize S2'_j
    
    # S2'_j = sum_sq_bins - bins[j]^2 + (bins[j] - item)^2
    #       = sum_sq_bins - bins[j]^2 + bins[j]^2 - 2*bins[j]*item + item^2
    #       = sum_sq_bins - 2*bins[j]*item + item^2
    
    # So minimizing S2'_j is equivalent to minimizing -2*bins[j]*item + item^2
    # Which is equivalent to maximizing 2*bins[j]*item
    # Which is equivalent to maximizing bins[j]
    
    # Wait, let's re-evaluate.
    # Minimizing variance of the *remaining* capacities.
    # If we just maximize bins[j], we are picking the bin with the most space.
    # That is "Worst Fit".
    # Let's check the math again.
    
    # S2'_j = S2 - 2*x_j*w + w^2
    # To minimize S2'_j, we need to maximize 2*x_j*w.
    # Assuming w > 0, we maximize x_j.
    # So pure variance minimization on the *remaining* capacities leads to Worst Fit?
    
    # Let's test with a simple example.
    # Bins: [10, 10]. Item: 5.
    # Option 1: Place in bin 0. New bins: [5, 10]. Mean: 7.5. Var: ((5-7.5)^2 + (10-7.5)^2)/2 = (6.25+6.25)/2 = 6.25.
    # Option 2: Place in bin 1. New bins: [10, 5]. Mean: 7.5. Var: 6.25.
    # Symmetric.
    
    # Bins: [8, 4]. Item: 4.
    # Option 1: Place in bin 0 (8). New: [4, 4]. Mean: 4. Var: 0.
    # Option 2: Place in bin 1 (4). New: [8, 0]. Mean: 4. Var: ((8-4)^2 + (0-4)^2)/2 = (16+16)/2 = 16.
    # Min variance picks Option 1 (Worst Fit).
    
    # Bins: [6, 4]. Item: 4.
    # Option 1: Place in bin 0 (6). New: [2, 4]. Mean: 3. Var: ((2-3)^2 + (4-3)^2)/2 = 1.
    # Option 2: Place in bin 1 (4). New: [6, 0]. Mean: 3. Var: ((6-3)^2 + (0-3)^2)/2 = 9.
    # Min variance picks Option 1 (Worst Fit).
    
    # It seems minimizing variance of remaining space pushes towards balancing the bins,
    # which often looks like Worst Fit or Average Fit.
    # However, "Best Fit" is usually better for packing.
    # Why? Because Best Fit reduces the *maximum* gap, or creates smaller gaps.
    # Variance measures spread.
    
    # Is the directive asking for variance of *remaining* or *used*?
    # "score bins based on the negative variance of the remaining capacity distribution"
    
    # If Worst Fit is the result, is that good?
    # Worst Fit is often worse than Best Fit for packing density.
    # Maybe I should look at the variance of the *used* capacity?
    # Or maybe the directive implies a different risk model?
    
    # Let's look at the current score 0.0375. It's low.
    # If I switch to pure Worst Fit (max remaining), it might be worse.
    # If I switch to Best Fit (min remaining), it's a different design.
    
    # Let's re-read carefully: "Model the bin selection as a probabilistic risk assessment by scoring bins based on the negative variance of the remaining capacity distribution across all currently open bins"
    
    # Perhaps the "risk" is not just variance, but something related to the probability of being able to fit future items?
    # But the directive is specific about the mechanism: "negative variance".
    
    # Let's consider that minimizing variance of the *remaining* capacities might be interpreted differently.
    # What if we consider the variance of the *fill* levels?
    # Fill = 1 - remaining/capacity.
    # Minimizing variance of remaining is equivalent to minimizing variance of fill (since it's a linear transform).
    
    # If the math holds that Min Var(remaining) == Max remaining (Worst Fit), then the directive leads to Worst Fit.
    # Is Worst Fit a good heuristic? Generally no, Best Fit Decreasing is SOTA.
    # However, the directive says "do not switch to a different design".
    # So I must implement the variance-based score.
    
    # Maybe I should add a small penalty for large remaining capacities to steer it away from pure Worst Fit?
    # No, "do not switch".
    
    # Let's just implement the O(N) variance minimization as derived.
    # If it turns out to be Worst Fit, so be it. It is a valid heuristic.
    
    # However, there is a nuance. The number of bins N might change?
    # "across all currently open bins".
    # In online bin packing, we usually have a fixed number of bins or open new ones.
    # The interface takes `bins` as an array. This implies the set of bins is fixed for this call.
    
    # Let's implement the formula:
    # Score_j = -Var_j
    # Var_j = (S2'_j / N) - (S1'_j / N)^2
    # S1'_j is constant for all j.
    # So minimizing Var_j is equivalent to minimizing S2'_j.
    # S2'_j = S2 - 2*bins[j]*item + item^2
    # Minimizing S2'_j is equivalent to maximizing bins[j]*item.
    # Since item is constant and positive, this is maximizing bins[j].
    
    # So the algorithm is:
    # 1. Identify valid bins (bins[i] >= item).
    # 2. Among valid bins, pick the one with the maximum value (Worst Fit).
    # 3. Return scores such that the max bin gets the highest score.
    
    # This is extremely simple.
    # Is there any other interpretation?
    # What if "variance of the remaining capacity distribution" refers to the variance of the *original* distribution, and we score based on how much that specific bin contributes to the variance?
    # Contribution to variance of bin i is (x_i - mean)^2.
    # If we remove x_i and add x_i - w, the change in variance is...
    # This gets complicated.
    
    # Let's stick to the "post-placement variance" interpretation as it's the most standard "look-ahead" risk metric.
    # And yes, it simplifies to Worst Fit.
    
    # Wait, is it possible that `bins` contains *used* capacity?
    # "score each bin with enough remaining capacity"
    # "bins: np.ndarray"
    # Usually `bins` in these interfaces represents the *remaining* capacity.
    # "remaining = bins - item" in the current code confirms `bins` is remaining capacity.
    
    # So, I will implement the logic that maximizes the remaining capacity (Worst Fit) via the variance minimization framework, but I will write the code to explicitly calculate the variance to be faithful to the "variance" keyword in the directive, even if it simplifies.
    # Actually, writing the simplified version is better for performance.
    # But to be safe and "research-like", I'll compute the variance explicitly for the top few candidates? No, just do the O(N) variance calc.
    
    # Let's refine the "Worst Fit" observation.
    # If I just return `bins` as scores, it's Worst Fit.
    # The current code has a score of 0.0375.
    # If Worst Fit is bad, the score will drop.
    # If the directive is misleading and "variance" should have been "sum of squares" or something else, I'm stuck.
    # But I must follow the directive.
    
    # Let's try a hybrid: The directive says "negative variance".
    # I will implement the O(N) calculation of the post-placement variance and return the negative of it.
    
    # One detail: If `item` is 0, variance is unchanged.
    
    # Implementation:
    # 1. Check valid bins.
    # 2. Compute S1, S2.
    # 3. Compute new_mean.
    # 4. For each bin, compute S2_new.
    # 5. Compute Var_new.
    # 6. Score = -Var_new.
    
    # This is robust and follows the text exactly.
    
    scores = np.full(n_bins, -np.inf)
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        return scores
        
    # Only compute for valid bins to save time, though O(N) is fast
    # We need S1 and S2 of the *entire* array because the variance is over *all* open bins.
    # If we only considered valid bins, the "distribution" would be different.
    # "across all currently open bins" implies the whole array.
    
    sum_bins = np.sum(bins)
    sum_sq_bins = np.sum(bins ** 2)
    
    new_sum = sum_bins - item
    new_mean = new_sum / n_bins
    
    # S2'_j = sum_sq_bins - 2*bins[j]*item + item**2
    # We only care about valid j.
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return scores
        
    # Vectorized calculation for valid indices
    x_valid = bins[valid_indices]
    s2_new = sum_sq_bins - 2 * x_valid * item + (item ** 2)
    
    # Var = S2'/N - Mean^2
    var_new = s2_new / n_bins - (new_mean ** 2)
    
    # Score = -Var
    scores[valid_indices] = -var_new
    
    return scores
# EVOLVE-BLOCK-END
