"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 100

# EVOLVE-BLOCK-START

def construct():
    # Use a multi-harmonic Fourier series to flatten the autocorrelation profile.
    # This approach uses multiple frequency components to reduce the peak overlap
    # at specific shifts (particularly N/2) compared to a single harmonic.
    i = np.arange(N)
    # Coefficients chosen to minimize the maximum overlap
    a1 = 0.45
    a2 = 0.15
    a3 = 0.05
    h = 0.5 + a1 * np.cos(2 * np.pi * i / N) + a2 * np.cos(4 * np.pi * i / N) + a3 * np.cos(6 * np.pi * i / N)
    
    # Ensure h is within [0, 1] and sum is exactly N/2
    # The sum of cosines is 0, so sum(h) = N * 0.5 = N/2 is exact.
    # Clip to [0, 1] to handle numerical errors or amplitude spikes, then renormalize if necessary.
    # However, with these coefficients, max h = 0.5 + 0.45 + 0.15 + 0.05 = 1.15.
    # This violates the [0, 1] constraint. We must scale down.
    
    # Max possible value is 0.5 + sum(|a_k|).
    # Sum of amps = 0.65. Max = 1.15. Min = 0.5 - 0.65 = -0.15.
    # We need to scale amplitudes so that max <= 1 and min >= 0.
    # Let scale factor s such that 0.5 + s * 0.65 <= 1 => s * 0.65 <= 0.5 => s <= 0.5/0.65 = 0.769.
    # And 0.5 - s * 0.65 >= 0 => s <= 0.769.
    # So we can use s = 0.769.
    
    s = 0.5 / (a1 + a2 + a3)
    h = 0.5 + s * (a1 * np.cos(2 * np.pi * i / N) + a2 * np.cos(4 * np.pi * i / N) + a3 * np.cos(6 * np.pi * i / N))
    
    # Due to floating point, sum might be slightly off, but cos sums are 0.
    # Return h
    return h

# EVOLVE-BLOCK-END
