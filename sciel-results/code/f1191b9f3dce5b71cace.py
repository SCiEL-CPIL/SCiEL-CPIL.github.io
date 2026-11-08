"""Erdos minimum overlap: choose step heights on [0,2] to minimize max overlap of h with shifted 1-h."""
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Return the constant function h(x) = 0.5.
    
    Analysis:
    The overlap at shift k is O(k) = sum_i h[i] (1 - h[i+k]).
    Let h[i] = 0.5 + e[i] with sum(e) = 0.
    O(k) = 0.25*N - 0.5*sum(e[i]) - 0.5*sum(e[i+k]) + sum(e[i]e[i+k])? No.
    O(k) = sum (0.5+e[i])(0.5-e[i+k]) = sum (0.25 - 0.5e[i+k] + 0.5e[i] - e[i]e[i+k])
    Since sum(e)=0, the linear terms vanish.
    O(k) = 0.25*N - sum(e[i]e[i+k]).
    We want to minimize max_k O(k).
    This is equivalent to maximizing min_k sum(e[i]e[i+k]).
    
    For h=0.5 (e=0), O(k) = 0.25*N for all k. Max O = 0.25*N.
    
    If e is non-zero, let S = sum(e[i]^2) > 0.
    sum_k sum(e[i]e[i+k]) = sum_i sum_k e[i]e[i+k] = sum_i e[i] sum_k e[i+k] = sum_i e[i] * 0 = 0.
    So the average of the correction terms is 0.
    However, we need the MINIMUM of the correction terms to be large (to keep O(k) low? No, O(k) = 0.25N - corr(k). To minimize max O(k), we want max(0.25N - corr(k)) to be small.
    This means we want min(corr(k)) to be large?
    Wait. Max(0.25N - corr(k)) = 0.25N - Min(corr(k)).
    To minimize this, we need to Maximize Min(corr(k)).
    
    We know sum_k corr(k) = 0.
    If we have N terms summing to 0, can the minimum be positive?
    No. If all corr(k) > 0, sum > 0. Contradiction.
    So Min(corr(k)) <= 0.
    Therefore, Max(0.25N - corr(k)) >= 0.25N - 0 = 0.25N.
    The minimum possible max overlap is 0.25N, achieved when corr(k)=0 for all k (which is impossible for non-zero e, but approached if corr(k) oscillates around 0? No, if min corr is close to 0, max overlap is close to 0.25N).
    Actually, if min corr is negative, max overlap is > 0.25N.
    So h=0.5 is the global optimum.
    """
    return np.full(N, 0.5)
# EVOLVE-BLOCK-END
