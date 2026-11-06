import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 85  # Safety margin for 90s limit
    
    n = 256
    
    # 1. Precompute all Arithmetic Progressions (APs)
    # An AP is defined by start, step, length.
    # We need all APs contained in [0, n-1].
    
    aps = []
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
    rng = np.random.RandomState(42)
    
    # Signs: -1 or 1
    signs = rng.choice([-1.0, 1.0], size=n)
    
    # Initialize sums for each AP
    sums = np.zeros(num_aps)
    for i, ap in enumerate(aps):
        sums[i] = np.sum(signs[ap])
        
    current_max = np.max(np.abs(sums))
    best_max = current_max
    best_signs = signs.copy()
    
    # 3. Simulated Annealing with optimized acceptance
    # We track the maximum and second maximum to avoid full scans if possible,
    # but given num_aps is ~1400, a full scan is fast enough in numpy.
    # The main cost is Python loop overhead.
    
    start_temp = 2.0
    end_temp = 0.05
    # We will run until deadline
    
    # Pre-compute the list of APs for each index as a numpy array for faster indexing?
    # index_to_aps is a list of lists. Converting to numpy arrays might help with sums[ap_indices]
    ap_indices_arr = [np.array(idx, dtype=np.int32) for idx in index_to_aps]
    
    temp = start_temp
    # Estimate steps: 85s / 10us per step = 8,500,000 steps max.
    # Let's just loop with time check.
    
    # To speed up, we can batch the time check
    # But time.time() is cheap.
    
    while time.time() < deadline:
        # Decay temperature
        # Simple geometric decay
        temp *= 0.99999
        
        # Pick a random index to flip
        k = rng.randint(0, n)
        
        # Current sign at k
        v = signs[k]
        
        # Identify APs containing k
        ap_indices = ap_indices_arr[k]
        
        # Calculate change in sums
        # new_sum = old_sum - 2*v
        delta = -2.0 * v
        
        # Apply change
        old_sums_subset = sums[ap_indices].copy()
        sums[ap_indices] += delta
        
        # Calculate new max
        new_max = np.max(np.abs(sums))
        
        delta_E = new_max - current_max
        
        # Acceptance criterion
        # If delta_E < 0, always accept
        # If delta_E >= 0, accept with probability exp(-delta_E / temp)
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
