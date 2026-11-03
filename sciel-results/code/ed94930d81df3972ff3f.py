"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _compute_overlap(h):
    """
    Compute the maximum overlap of h with shifted 1-h.
    
    The overlap at shift k is sum_i h[i] * (1 - h[(i+k) % N]).
    We want to minimize the maximum over all shifts k.
    """
    N = len(h)
    one_minus_h = 1.0 - h
    max_overlap = 0.0
    for k in range(N):
        shifted = np.roll(one_minus_h, k)
        overlap = np.dot(h, shifted)
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap


def _soft_binary_from_boundaries(boundaries, N, temperature=0.1):
    """
    Convert boundary positions to a soft-binary sequence using a smooth step.
    
    boundaries: array of length 2 (start and end of the '1' region in [0, N])
    We create a sequence that is approximately 0 or 1, with smooth transitions.
    
    Actually, let's think more carefully. We have N positions. We want to place
    a contiguous block of 1s (or multiple blocks). Let's use a different approach:
    represent the sequence as a sum of box functions.
    
    For simplicity, let's start with a single block of 1s of length L, starting at position s.
    The heights are: h[i] = 1 if s <= i < s+L, else 0.
    
    But we need sum(h) = N/2, so L = N/2 = 50.
    
    So we just need to optimize the starting position s (and possibly make it soft).
    
    Let's use a soft representation: h[i] = sigmoid((i - s + 0.5) / temp) * sigmoid((s + L - 0.5 - i) / temp)
    This creates a smooth box from s to s+L.
    
    Actually, let's use a simpler approach: optimize the boundary positions directly
    with a soft-max / sigmoid representation.
    """
    # Let's try a different formulation: 
    # We have N positions. We want to choose which are 1 and which are 0.
    # Let's parameterize by a set of boundary positions.
    
    # For now, let's just optimize a single contiguous block of 1s.
    # h[i] = 1 for i in [s, s+L), 0 otherwise, with soft edges.
    
    L = N // 2  # 50
    s = boundaries[0]  # start position, in [0, N]
    
    # Create indices
    i = np.arange(N)
    
    # Soft box function: sigmoid((i - s + 0.5) / temp) * sigmoid((s + L - 0.5 - i) / temp)
    # This is approximately 1 for i in (s, s+L) and 0 outside, with smooth transitions.
    temp = 0.5
    left = 1.0 / (1.0 + np.exp(-(i - s + 0.5) / temp))
    right = 1.0 / (1.0 + np.exp(-(s + L - 0.5 - i) / temp))
    h = left * right
    
    # Normalize to ensure sum is exactly N/2
    current_sum = np.sum(h)
    if current_sum > 0:
        h = h * (N / 2.0) / current_sum
    
    # Clip to [0, 1]
    h = np.clip(h, 0.0, 1.0)
    
    return h


def _objective(boundaries):
    """Objective function to minimize: the maximum overlap."""
    N = 100
    h = _soft_binary_from_boundaries(boundaries, N)
    
    # Ensure sum constraint is satisfied
    current_sum = np.sum(h)
    target_sum = N / 2.0
    penalty = 100.0 * (current_sum - target_sum) ** 2
    
    overlap = _compute_overlap(h)
    
    return overlap + penalty


def construct():
    """
    Construct the sequence of step heights.
    
    Returns: sequence of N step heights in [0, 1] whose sum is exactly N/2.
    """
    N = 100
    
    # Optimize the start position of a contiguous block of 1s
    # Initial guess: start at 0
    x0 = np.array([0.0])
    
    # Bounds: s in [0, N]
    bounds = [(0.0, float(N))]
    
    # Use Nelder-Mead since it doesn't require gradients
    result = minimize(_objective, x0, method='Nelder-Mead', 
                      bounds=bounds,
                      options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
    
    s_opt = result.x[0]
    h = _soft_binary_from_boundaries([s_opt], N)
    
    # Ensure exact sum constraint
    current_sum = np.sum(h)
    if current_sum > 0:
        h = h * (N / 2.0) / current_sum
    
    # Final clip
    h = np.clip(h, 0.0, 1.0)
    
    return h
# EVOLVE-BLOCK-END
