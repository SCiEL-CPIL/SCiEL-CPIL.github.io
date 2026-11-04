import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (10s before 150s limit)
    
    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        c = np.convolve(h, h)
        return abs(2.0 * N * np.max(c) / (s * s))

    # Initialize with a few diverse candidates
    # 1. Constant (seed)
    h1 = np.ones(N)
    # 2. Cosine taper (smooth, zero mean perturbation)
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    h2 = 1.0 + 0.5 * np.cos(x)
    # 3. High frequency oscillation
    h3 = 1.0 + 0.5 * np.cos(10 * x)
    # 4. Random smoothed
    rng = np.random.default_rng(123)
    r = rng.normal(0, 1, N)
    # Simple smoothing to avoid high freq noise which spikes max conv
    r = np.convolve(r, np.ones(5)/5, mode='same')
    r = r - np.mean(r)
    h4 = 1.0 + 0.5 * r / (np.std(r) + 1e-9)
    
    candidates = [h1, h2, h3, h4]
    best_val = np.inf
    best_h = h1
    
    for h in candidates:
        v = compute_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
            
    # Simulated Annealing
    # We fix the sum to N to remove scaling ambiguity.
    # Value = 2N * max(conv(h,h)) / N^2 = 2 * max(conv(h,h)) / N
    # So minimizing value is equivalent to minimizing max(conv(h,h)) given sum(h)=N.
    
    current_h = best_h.copy()
    # Normalize current_h to have sum N
    current_h = current_h * (N / np.sum(current_h))
    current_val = compute_value(current_h)
    
    if current_val < best_val:
        best_val = current_val
        best_h = current_h.copy()
        
    T = 1.0  # Initial temperature
    cooling_rate = 0.999
    min_T = 1e-4
    
    # Move generators
    def move_swap(h):
        """Swap two random elements. Preserves sum."""
        h_new = h.copy()
        i, j = rng.integers(0, N, 2)
        if i != j:
            h_new[i], h_new[j] = h_new[j], h_new[i]
        return h_new

    def move_block_shift(h):
        """Shift a block of elements. Preserves sum if we rotate the whole array? 
           No, let's do a local rearrangement.
           Pick a segment and reverse it? Or just swap neighbors?
           Let's try swapping a block of length L with another block of length L."""
        h_new = h.copy()
        L = rng.integers(1, 10)
        start1 = rng.integers(0, N - L)
        start2 = rng.integers(0, N - L)
        if abs(start1 - start2) >= L:
            block1 = h_new[start1:start1+L].copy()
            block2 = h_new[start2:start2+L].copy()
            h_new[start1:start1+L] = block2
            h_new[start2:start2+L] = block1
        return h_new

    def move_gaussian_pair(h):
        """Add noise to two elements and subtract from others to keep sum?
           Simpler: Add noise to one, subtract from another. Preserves sum."""
        h_new = h.copy()
        i, j = rng.integers(0, N, 2)
        if i != j:
            delta = rng.normal(0, 0.5)
            h_new[i] += delta
            h_new[j] -= delta
        return h_new

    moves = [move_swap, move_block_shift, move_gaussian_pair]
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        if T < min_T:
            T = min_T
            
        # Select a move
        move_fn = moves[rng.integers(0, len(moves))]
        h_new = move_fn(current_h)
        
        # Ensure finite
        if not np.all(np.isfinite(h_new)):
            continue
            
        new_val = compute_value(h_new)
        
        if new_val < current_val:
            current_h = h_new
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
        else:
            # Accept with probability exp(-(new_val - current_val) / T)
            # Note: new_val and current_val are around 1.0-2.0.
            # Difference can be small.
            delta_e = new_val - current_val
            if delta_e < 0:
                accept = True
            else:
                # Prevent overflow in exp
                if delta_e > 100 * T:
                    accept = False
                else:
                    accept = rng.random() < np.exp(-delta_e / T)
            
            if accept:
                current_h = h_new
                current_val = new_val
                
        T *= cooling_rate
        
        # Periodic restart from best if stuck?
        # Or just let it cool.
        
    return best_h
