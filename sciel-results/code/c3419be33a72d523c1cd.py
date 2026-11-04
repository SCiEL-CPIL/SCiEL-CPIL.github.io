import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # Initialize with random +/- 1
    rng = np.random.RandomState(42)
    h = rng.choice([-1, 1], size=N).astype(np.float64)
    
    def get_max_sidelobe(h):
        # Compute aperiodic autocorrelation
        c = np.correlate(h, h, mode='full')
        # c[N-1] is c[0] = N
        # Sidelobes are c[0:N-1] and c[N:2N-1]
        # We want max absolute value of these
        # Note: c is real for real h
        left = c[:N-1]
        right = c[N:]
        return np.max(np.abs(np.concatenate([left, right])))
        
    current_cost = get_max_sidelobe(h)
    best_cost = current_cost
    best_h = h.copy()
    
    # Local search
    # We use a simple hill climbing approach with random restarts if stuck
    no_improve_count = 0
    max_no_improve = 5000  # Restart if stuck for a while
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Restart with a new random sequence or a perturbed one
            # Let's try a new random start to escape deep local minima
            h = rng.choice([-1, 1], size=N).astype(np.float64)
            current_cost = get_max_sidelobe(h)
            if current_cost < best_cost:
                best_cost = current_cost
                best_h = h.copy()
            no_improve_count = 0
            continue
            
        # Pick a random index to flip
        i = rng.randint(N)
        old_val = h[i]
        h[i] = -old_val
        
        new_cost = get_max_sidelobe(h)
        
        if new_cost < current_cost:
            current_cost = new_cost
            no_improve_count = 0
            if new_cost < best_cost:
                best_cost = new_cost
                best_h = h.copy()
        else:
            h[i] = old_val
            no_improve_count += 1
            
    return best_h
# EVOLVE-BLOCK-END
