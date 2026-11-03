import numpy as np
import time

N = 100

def compute_max_overlap_excl0(h):
    """Compute max C(k) for k != 0, where C(k) = sum_i h_i*(1-h_{(i+k)%N})."""
    h_comp = 1.0 - h
    H = np.fft.fft(h)
    Hc = np.fft.fft(h_comp)
    c = np.fft.ifft(H * np.conj(Hc)).real
    # c[k] = sum_i h[i] * h_comp[(i+k) % N] = C(k)
    # Exclude k=0 (index 0)
    c[0] = -np.inf
    return np.max(c)

def compute_max_overlap_incl0(h):
    """Compute max C(k) for all k including k=0."""
    h_comp = 1.0 - h
    H = np.fft.fft(h)
    Hc = np.fft.fft(h_comp)
    c = np.fft.ifft(H * np.conj(Hc)).real
    return np.max(c)

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 35s buffer for safety
    
    # The uniform solution
    h_uniform = np.full(N, 0.5)
    score_uniform_excl0 = compute_max_overlap_excl0(h_uniform)
    score_uniform_incl0 = compute_max_overlap_incl0(h_uniform)
    
    best_h = h_uniform.copy()
    best_score = score_uniform_excl0  # Track the excl0 score as primary
    
    def random_initial():
        """Generate random h with sum exactly N/2, all in [0,1]."""
        # Use a method that guarantees feasibility
        # Start with uniform, add random perturbations that sum to zero
        h = np.full(N, 0.5)
        # Add random zero-sum perturbations
        for _ in range(50):
            i, j = np.random.choice(N, 2, replace=False)
            max_d = min(0.5 - abs(h[i] - 0.5), 0.5 - abs(h[j] - 0.5))
            # Actually simpler: just use random and project
        # Simpler approach: random, then project
        h = np.random.uniform(0, 1, N)
        # Project to sum = N/2 with bounds [0,1]
        for _ in range(20):
            s = np.sum(h)
            if abs(s - N/2) < 1e-12:
                break
            diff = N/2 - s
            # Distribute diff to elements with room
            if diff > 0:
                room = 1.0 - h
                if np.sum(room) > 1e-15:
                    # Proportional distribution
                    h += diff * room / np.sum(room)
                else:
                    break
            else:
                h += diff / N
            h = np.clip(h, 0, 1)
        # Final exact fix
        s = np.sum(h)
        diff = N/2 - s
        if abs(diff) > 1e-15:
            # Find elements that can absorb the difference
            if diff > 0:
                room = 1.0 - h
                idx = np.where(room > 1e-10)[0]
                if len(idx) > 0:
                    h[idx] += diff / len(idx)
                    h = np.clip(h, 0, 1)
            else:
                idx = np.where(h > 1e-10)[0]
                if len(idx) > 0:
                    h[idx] += diff / len(idx)
                    h = np.clip(h, 0, 1)
        return h
    
    # Simulated Annealing
    T_start = 0.1
    T_end = 0.0001
    cooling_rate = 0.9995
    steps_per_temp = 50
    
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if r == 0:
            h = best_h.copy()
        else:
            h = random_initial()
        
        current_score = compute_max_overlap_excl0(h)
        T = T_start
        
        while T > T_end:
            if time.time() - start_time > time_limit:
                break
                
            for _ in range(steps_per_temp):
                if time.time() - start_time > time_limit:
                    break
                    
                # Pick two indices for swap
                i, j = np.random.choice(N, 2, replace=False)
                
                # Max possible delta preserving bounds
                max_delta = min(1.0 - h[i], h[j])
                if max_delta <= 1e-12:
                    continue
                
                # Choose delta
                delta = np.random.uniform(-max_delta, max_delta)
                
                new_h = h.copy()
                new_h[i] += delta
                new_h[j] -= delta
                
                # Check bounds (floating point safety)
                if new_h[i] < -1e-12 or new_h[i] > 1.0 + 1e-12 or new_h[j] < -1e-12 or new_h[j] > 1.0 + 1e-12:
                    continue
                
                new_score = compute_max_overlap_excl0(new_h)
                delta_E = new_score - current_score
                
                # Accept if better, or with probability exp(-delta_E / T)
                if delta_E < 0:
                    accept = True
                else:
                    if T > 1e-15:
                        accept = np.random.random() < np.exp(-delta_E / T)
                    else:
                        accept = False
                
                if accept:
                    h = new_h
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_h = h.copy()
            
            T *= cooling_rate
    
    # Final feasibility check and correction
    s = np.sum(best_h)
    diff = N/2 - s
    if abs(diff) > 1e-12:
        best_h += diff / N
        best_h = np.clip(best_h, 0, 1)
    
    # Verify constraints
    assert np.all(best_h >= -1e-9) and np.all(best_h <= 1.0 + 1e-9), "Bounds violated"
    assert abs(np.sum(best_h) - N/2.0) < 1e-9, "Sum constraint violated"
    
    # Clip to exact [0,1] for safety
    best_h = np.clip(best_h, 0.0, 1.0)
    
    # One final sum correction after clip
    s = np.sum(best_h)
    diff = N/2 - s
    if abs(diff) > 1e-12:
        # Distribute to elements with room
        if diff > 0:
            room = 1.0 - best_h
            mask = room > 1e-10
            if np.sum(mask) > 0:
                best_h[mask] += diff / np.sum(mask)
        else:
            mask = best_h > 1e-10
            if np.sum(mask) > 0:
                best_h[mask] += diff / np.sum(mask)
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
