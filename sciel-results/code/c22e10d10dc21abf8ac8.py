"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    # Structural change: use a flat-top (boxcar) function instead of triangular.
    # For a boxcar function, the autoconvolution is triangular.
    # The normalized max is (max autoconv) / (integral)^2 * (some factor)
    # Actually, let's think about the normalized max definition.
    # Typically for such problems, the score is max(f*f) / (max(f))^2 or similar.
    # But the task says "normalized maximum of its autoconvolution".
    # Let's assume it's max(f*f) / (sum(f)^2) * N or similar.
    # Wait, the seed program score is 1.0 for all ones.
    # If f is all ones, autoconv is a triangle peaking at N.
    # Normalized max might be max(f*f) / (sum(f))^2 * N^2? No.
    
    # Let's look at the inspiration 3: "score is 1.0 / value".
    # The current program has score 0.7487. The seed (all ones) has score 1.0.
    # So the seed is better. The triangular function is worse.
    # This implies that the "normalized maximum" is likely max(f*f) / (max(f))^2?
    # Or maybe max(f*f) / (mean(f))^2?
    
    # Let's test the hypothesis:
    # If the score is 1/value, and seed (all ones) gives 1.0, then value=1.0.
    # For all ones: f*f is a triangle from 1 to N and back to 1. Max is N.
    # If normalized max = max(f*f) / N, then value = 1.
    # For triangular f: f is 1 at center, 0 at edges.
    # f*f will be smaller.
    # Let's try a boxcar function: 1 in the middle, 0 elsewhere.
    # If f is 1 for M intervals and 0 for others.
    # f*f max is M. Normalized by N? Value = M/N.
    # To minimize value, we want M small. But f must be nonnegative.
    # If M=1, value = 1/N. Score = N. That would be huge.
    # But the problem says "nonnegative step function". It doesn't say it must be positive everywhere.
    # However, if we use a very sparse function, the score could be very high.
    # Why did the triangular function score 0.7487 (worse than 1.0)?
    # Maybe the normalization is different.
    
    # Let's re-read: "minimize the normalized maximum of its autoconvolution".
    # If the seed is all ones and scores 1.0, and triangular scores 0.7487,
    # then the triangular function has a HIGHER normalized maximum.
    # This suggests that the normalization favors functions with higher "average" values
    # or that the max of the autoconvolution is not the only factor.
    
    # Perhaps the normalization is max(f*f) / (sum(f))^2 * N?
    # For all ones: sum = N. max(f*f) = N. Value = N / N^2 * N = 1.
    # For triangular: sum ~ N/2. max(f*f) ~ N/2 (since f is 1 at center, 0 at edges, f*f is roughly N/2).
    # Value ~ (N/2) / (N/2)^2 * N = (N/2) / (N^2/4) * N = 2.
    # So value would be 2, score 0.5. This is in the right ballpark (0.7487).
    
    # So the objective is likely: max(f*f) / (sum(f))^2 * N.
    # To minimize this, we want to maximize (sum(f))^2 / max(f*f).
    # This is equivalent to maximizing the "efficiency" of the function.
    # A flat function (all ones) gives value 1.
    # A boxcar function with M ones: sum = M, max(f*f) = M. Value = M / M^2 * N = N/M.
    # To minimize N/M, we want M as large as possible. M=N gives value 1.
    # So the flat function is optimal for this metric?
    
    # But wait, the triangular function gave 0.7487, which is worse than 1.0.
    # So the flat function is better.
    # Can we do better than 1.0?
    # If we use a function that is not flat, can we get value < 1?
    # Value = max(f*f) / (sum(f))^2 * N.
    # For any nonnegative function, max(f*f) >= (sum(f))^2 / N (by Cauchy-Schwarz or similar).
    # Actually, max(f*f) is at least the average of f*f.
    # The average of f*f is (1/N) * sum(f_i^2).
    # By Cauchy-Schwarz, (sum f_i)^2 <= N * sum(f_i^2).
    # So sum(f_i^2) >= (sum f_i)^2 / N.
    # Thus max(f*f) >= (sum f_i)^2 / N.
    # Therefore, value >= 1.
    # So the minimum possible value is 1, achieved by the flat function.
    
    # Therefore, the best we can do is return the flat function (all ones).
    # The current program returns a triangular function, which is suboptimal.
    # We should revert to the flat function.
    
    return [1.0] * N
# EVOLVE-BLOCK-END
