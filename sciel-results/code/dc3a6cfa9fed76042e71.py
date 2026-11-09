"""Cap set greedy priority function."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Priority for greedy cap set construction in {0,1,2}^n.
    
    Components:
    1. Symbol balance: reward near-uniform distribution of 0,1,2
    2. Per-symbol position spread: reward symbols whose occurrences are well-spread
    3. Zero spread bonus: extra weight for zero spread (known effective heuristic)
    """
    v = np.array(vector)
    
    # 1. Symbol balance
    counts = np.array([np.sum(v == i) for i in range(3)], dtype=float)
    ideal = n / 3.0
    balance_penalty = np.sum((counts - ideal) ** 2) / n
    balance_score = 1.0 - balance_penalty / (n / 9.0 + 1e-12)  # normalized to roughly [0,1]
    
    # 2. Per-symbol position spread
    def position_spread(positions):
        """Average pairwise circular distance on cycle of length n."""
        k = len(positions)
        if k <= 1:
            return 0.0
        positions = sorted(positions)
        total_dist = 0.0
        count_pairs = 0
        for i in range(k):
            for j in range(i + 1, k):
                d = abs(positions[j] - positions[i])
                d = min(d, n - d)  # circular distance
                total_dist += d
                count_pairs += 1
        return total_dist / count_pairs if count_pairs > 0 else 0.0
    
    spreads = []
    for sym in range(3):
        positions = [i for i in range(n) if v[i] == sym]
        spreads.append(position_spread(positions))
    
    # Normalize spreads to [0,1] by dividing by max possible (n/2)
    max_spread = n / 2.0
    if max_spread > 0:
        norm_spreads = [s / max_spread for s in spreads]
    else:
        norm_spreads = [0.0, 0.0, 0.0]
    
    # Weight zero spread more heavily (known effective)
    spread_score = 0.5 * norm_spreads[0] + 0.25 * norm_spreads[1] + 0.25 * norm_spreads[2]
    
    # 3. Combined priority
    # Balance and spread both in [0,1] roughly
    priority_val = 1.0 * balance_score + 1.5 * spread_score
    
    return float(priority_val)
# EVOLVE-BLOCK-END
