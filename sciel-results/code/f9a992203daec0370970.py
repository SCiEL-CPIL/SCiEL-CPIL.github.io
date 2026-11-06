import time
import numpy as np
import random

def construct():
    t0 = time.time()
    deadline = t0 + 55  # Safety margin for 90s limit, evaluator might have overhead
    
    n = 256
    
    # 1. Precompute all Arithmetic Progressions (APs)
    # An AP is defined by start, step, length.
    # We need all APs contained in [0, n-1].
    # "Any start, positive step and length including one."
    # The task says "contained in [1, 256]". In 0-indexed, [0, 255].
    # AP: start, start+step, ..., start+(k-1)*step < n
    
    aps = []
    # Map from index to list of AP indices containing it
    # This allows efficient updates when a sign is flipped
    index_to_aps = [[] for _ in range(n)]
    
    ap_id = 0
    for start in range(n):
        for step in range(1, n - start):
            # Generate indices for this AP
            idx = list(range(start, n, step))
            if len(idx) < 1:
                continue
            
            # Register this AP
            aps.append(idx)
            
            # Update reverse mapping
            for pos in idx:
                index_to_aps[pos].append(ap_id)
            
            ap_id += 1
            
    num_aps = len(aps)
    
    # 2. Initialize
    # Use a fixed seed for reproducibility in this function, 
    # but the problem asks to minimize. Random restarts or SA usually suffice.
    rng = np.random.RandomState(42)
    
    # Signs: -1 or 1
    signs = rng.choice([-1.0, 1.0], size=n)
    
    # Initialize sums for each AP
    sums = np.zeros(num_aps)
    for i, ap in enumerate(aps):
        # Sum of signs at positions in ap
        # Using numpy sum for initialization
        sums[i] = np.sum(signs[ap])
        
    current_max = np.max(np.abs(sums))
    best_max = current_max
    best_signs = signs.copy()
    
    # 3. Simulated Annealing
    # Parameters
    start_temp = 1.0
    end_temp = 0.01
    # Number of iterations depends on time budget
    # We will run until deadline
    
    # Pre-compute the delta magnitude for a flip
    # If we flip sign at index k from v to -v:
    # The sum of any AP containing k changes by -2*v.
    # Since v is either 1 or -1, the change is either -2 or 2.
    
    # SA Loop
    temp = start_temp
    # Estimate steps: 60s / 10us per step = 6,000,000 steps max.
    # Let's just loop with time check.
    
    while time.time() < deadline:
        # Decay temperature
        # Simple geometric decay
        temp *= 0.99999
        
        # Pick a random index to flip
        k = rng.randint(0, n)
        
        # Current sign at k
        v = signs[k]
        
        # Identify APs containing k
        ap_indices = index_to_aps[k]
        
        # Calculate change in sums
        # new_sum = old_sum - 2*v
        delta = -2.0 * v
        
        # Apply change
        old_sums_subset = sums[ap_indices].copy()
        sums[ap_indices] += delta
        
        # Calculate new max
        # We only need to check if the max changed.
        # The max can only increase if one of the updated APs became the new max,
        # or decrease if the previous max was one of the updated APs and its abs value decreased.
        # However, checking max of the whole array is O(num_aps) ~ 1400, which is very fast.
        # 1400 operations is negligible compared to Python overhead.
        new_max = np.max(np.abs(sums))
        
        delta_E = new_max - current_max
        
        # Acceptance criterion
        if delta_E < 0 or rng.rand() < np.exp(-delta_E / temp):
            # Accept
            signs[k] = -v
            current_max = new_max
            
            if current_max < best_max:
                best_max = current_max
                best_signs = signs.copy()
        else:
            # Reject, revert sums
            sums[ap_indices] = old_sums_subset
            
    # Convert to Python integers
    # Ensure values are exactly -1 or 1
    result = [int(s) for s in best_signs]
    
    return result
