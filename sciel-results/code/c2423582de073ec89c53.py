import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    # Convert to numpy array for vectorization
    d = np.array(demands, dtype=float)
    
    # Calculate all pair sums
    pair_sums = d[:, None] + d[None, :]
    
    # Initialize heuristic matrix with the minimum allowed value (for invalid pairs)
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # Identify pairs that fit in the bin
    valid_mask = pair_sums <= capacity
    
    # For valid pairs, calculate the heuristic
    # We prefer pairs that fill the bin as much as possible (minimize remaining space)
    # Heuristic = 1 / remaining_space
    # To avoid division by zero if sum == capacity, we use max(remaining, epsilon)
    # Since demands are integers >= 20, and capacity 150, the minimum remaining space for a valid pair is at least 1 (e.g., 99+51=150, rem=0? No, 99+51=150 is valid. Rem=0.
    # Wait, if rem=0, 1/0 is inf.
    # The prompt says "Heuristic entries that are not positive become 1e-9". Inf is not positive? Or is it positive but infinite?
    # Usually "not finite" makes the result invalid.
    # So we must avoid inf.
    
    # Calculate remaining space
    remaining = capacity - pair_sums
    
    # Only update where valid
    # We want to penalize large remaining space.
    # Use a small epsilon for the denominator to handle rem=0
    # If rem=0, it's a perfect fit. Heuristic should be very high.
    # Let's set the floor for remaining space to 1.0.
    # If rem=0, we treat it as 1. If rem=100, it's 100.
    
    valid_remaining = np.maximum(remaining[valid_mask], 1.0)
    eta[valid_mask] = 1.0 / valid_remaining
    
    return eta
