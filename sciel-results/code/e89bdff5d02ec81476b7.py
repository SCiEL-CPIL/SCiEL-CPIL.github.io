"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _value(h):
    """Compute the objective value (lower is better)."""
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return np.inf
    c = np.convolve(h, h)
    return abs(2 * N * np.max(c) / s ** 2)


def _spectral_candidate(frequencies, amplitudes, phases):
    """
    Construct h as a sum of sinusoids with zero mean.
    h[k] = sum_i amplitudes[i] * cos(frequencies[i] * k + phases[i])
    Then subtract the mean to enforce zero-sum... wait, zero-sum would make denominator zero.
    
    Actually, we need sum(h) != 0. Let's use:
    h[k] = base + sum_i amplitudes[i] * cos(frequencies[i] * k + phases[i])
    where base is chosen so that the oscillation has a specific property.
    
    Better approach: Let h = a + b * u where u has zero mean and we want to minimize
    2*N*max(convolve(h,h)) / sum(h)^2
    
    sum(h) = N*a (since sum(u) = 0)
    convolve(h,h) = convolve(a+bu, a+bu) = N*a^2 + 2ab*sum(u) + b^2*convolve(u,u)
                  = N*a^2 + b^2*convolve(u,u)  [since sum(u)=0]
    
    So value = 2*N * max(N*a^2 + b^2 * convolve(u,u)) / (N*a)^2
             = 2*N * max(N*a^2 + b^2 * cu) / (N^2 * a^2)
             = 2 * max(N*a^2 + b^2 * cu) / (N * a^2)
             = 2 * [N + (b^2/a^2) * max(cu)] / N
             = 2 * [1 + (b^2/a^2) * max(cu) / N]
    
    If max(cu) is negative, this is less than 2! But cu is autoconvolution of u, and the center
    element is sum(u^2) >= 0. So max(cu) >= sum(u^2)/... actually the center is sum(u^2) which is positive.
    
    Hmm, so max(cu) >= 0 always. So value >= 2.
    
    Wait, but the task says "the maximum is taken over its signed entries, with no absolute value inside np.max".
    And the center of the full convolution is sum(h^2) which is always >= 0. So np.max(convolve(h,h)) >= sum(h^2) >= 0.
    
    And by Cauchy-Schwarz, for any lag k: |convolve(h,h)[k]| <= sum(h^2). So the max is exactly sum(h^2).
    
    Therefore value = 2*N*sum(h^2)/(sum(h))^2 >= 2 by Cauchy-Schwarz, with equality iff h is constant.
    
    So the seed (ones) is provably optimal with value 2.0.
    """
    t = np.arange(N, dtype=np.float64)
    h = np.zeros(N, dtype=np.float64)
    for freq, amp, phase in zip(frequencies, amplitudes, phases):
        h += amp * np.cos(2 * np.pi * freq * t / N + phase)
    # Add a constant to ensure sum != 0
    h += 1.0
    return h


def construct():
    """Construct step heights minimizing the objective value."""
    # Theoretical minimum is 2.0, achieved by any constant vector.
    # We return the seed which achieves this exactly.
    return np.ones(N)

# EVOLVE-BLOCK-END
