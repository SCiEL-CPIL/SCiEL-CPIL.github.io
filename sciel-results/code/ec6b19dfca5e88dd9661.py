# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
import random
import itertools

# Global state
_start_time = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill

def priority(vector, n):
    """
    Strategy: Weighted Sum of Characteristically Biased Coordinates (Zero-Position Spread)
    
    This strategy shifts from global digit balance to a position-aware heuristic.
    The core idea is that '0' is the additive identity in F_3. Vectors with zeros
    are structurally 'simpler' or 'more linear' in a sense, but the key insight
    from the observed improvements is that the *distribution* of zeros matters.
    
    Specifically, we want to prioritize vectors where:
    1. The total number of zeros is high (bias towards 0s).
    2. The zeros are spread out (high entropy/spread of zero positions).
    
    Why spread? If zeros are clustered, the vector is 'close' to a lower-dimensional
    subspace defined by those coordinates being 0. If they are spread, the vector
    is more 'generic' in the zero-identity sense, potentially reducing the likelihood
    of forming tight arithmetic progressions with other similarly biased vectors.
    
    Additionally, we include a small penalty for '1's and a bonus for '2's? 
    Actually, let's stick to the observed success: Zero-Position Spread.
    
    Let's formalize the score:
    1. Count zeros (c0).
    2. Calculate the "spread" of zero positions. 
       A simple measure of spread is the sum of squared distances from the mean position of zeros?
       Or simply the number of distinct "blocks" of zeros?
       Or, even simpler and computationally cheap: The variance of the indices where v_i == 0.
       
       Let indices = [i for i, v in enumerate(vector) if v == 0]
       If len(indices) < 2, spread = 0
       Else, spread = variance(indices)
       
    3. Combine: Score = alpha * c0 + beta * spread - gamma * (n - c0)
    
    Let's tune alpha, beta, gamma.
    The current program scored 1.23. The inspiration 2 (1.3477) used "spatial spread of 1-positions".
    The knowledge base says "Adding zero-position spread... improved scores from 1.23 -> 1.34".
    
    So, I will implement:
    Score = w0 * count_0 + w_spread * variance_zero_indices + w_bal * (balance_term)
    
    To ensure robustness and avoid overfitting to one specific metric, I will combine:
    1. Zero Count (primary bias)
    2. Zero Spread (variance of zero indices)
    3. A small global balance term to prevent degenerate cases (all zeros or all ones).
    
    Note: The evaluator calls this function repeatedly. We must be fast.
    Variance calculation is O(n).
    """
    
    # Time guard
    if time.time() - _start_time > _DEADLINE:
        # If we are running out of time, return a simple static score to avoid hanging
        # This is unlikely to be hit if n is small, but good for safety.
        return sum(vector) * 0.1
        
    c0 = 0
    c1 = 0
    c2 = 0
    zero_indices = []
    
    for i, v in enumerate(vector):
        if v == 0:
            c0 += 1
            zero_indices.append(i)
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
            
    # 1. Zero Bias
    # More zeros is generally better based on the "0 is identity" hypothesis.
    score = 2.0 * c0
    
    # 2. Zero Spread
    # We want zeros to be spread out, not clustered.
    # Variance is a good proxy for spread.
    if len(zero_indices) >= 2:
        mean_idx = sum(zero_indices) / len(zero_indices)
        var_idx = sum((x - mean_idx) ** 2 for x in zero_indices) / len(zero_indices)
        # Normalize variance by n to make it comparable across dimensions?
        # Or just use raw variance?
        # Let's use normalized variance: var / (n+1)
        score += 1.0 * (var_idx / (n + 1))
    else:
        # If 0 or 1 zeros, spread is 0 or undefined. 
        # 0 zeros: no spread.
        # 1 zero: no spread.
        # This naturally penalizes high zero counts if they are clustered? 
        # No, if c0 is high, variance is likely high unless clustered.
        # If c0 is 0, score is 0 from this term.
        pass
        
    # 3. Balance / Anti-Clustering of 1s and 2s?
    # The inspiration mentioned "spatial spread of 1-positions".
    # Let's add a small term for the spread of 1s as well, or just a penalty for 1s.
    # The current program had a penalty for 1s.
    score -= 0.5 * c1
    
    # 4. Global Balance
    # Ensure we don't just pick all-zero vectors (which might form APs? 
    # Actually, {0,0,0} is an AP. But {0,1,2} is not.
    # Wait, a cap set cannot contain {a, a+d, a+2d}.
    # If we have many 0s, we need to be careful not to have too many vectors that are "close" in a way that forms APs.
    # However, the empirical result was positive for zero-bias + spread.
    
    # Let's add a small penalty if the vector is "too balanced" in a bad way?
    # No, let's stick to the successful formula: Zero Bias + Zero Spread.
    
    # To differentiate from simple zero count, the spread term is key.
    # Let's refine the weights.
    # If c0 is high, the spread term becomes more important.
    
    return score

# EVOLVE-BLOCK-END
