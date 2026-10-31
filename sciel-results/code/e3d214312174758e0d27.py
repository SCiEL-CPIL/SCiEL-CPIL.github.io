import numpy as np
import time

N = 100

def construct():
    # Strategy: Discrete Binary Search with Simulated Annealing
    # The problem asks for h in [0,1] summing to N/2.
    # The uniform solution h=0.5 gives max_overlap = N/2 (normalized 1.0).
    # Binary solutions (0/1) with exactly N/2 ones are valid.
    # For binary h, overlap(s) = count of (h[i]=1 and h[i+s]=0).
    # Minimizing max overlap for binary sequences is equivalent to finding
    # a set of indices with minimal "gaps" or "clumps" in a circular sense.
    # We use Simulated Annealing to find a binary configuration that minimizes
    # the maximum overlap.
    
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin for 150s limit
    
    # Initial state: random binary vector with exactly N/2 ones
    def random_binary_state():
        ones = np.zeros(N, dtype=int)
        ones[:N//2] = 1
        np.random.shuffle(ones)
        return ones

    def compute_overlap_profile(h):
        # h is binary array of 0s and 1s
        # overlap[s] = sum(h[i] * (1 - h[(i+s)%N]))
        # Using FFT for speed
        # H = FFT(h)
        # Correlation C[s] = IFFT(H * conj(H)) -> gives sum h[i]h[i+s]
        # overlap[s] = sum h[i] - sum h[i]h[i+s] = N/2 - C[s]
        H = np.fft.fft(h.astype(float))
        power = H * np.conj(H)
        r = np.fft.ifft(power).real
        overlaps = N/2 - r
        return overlaps

    def get_max_overlap(h):
        overlaps = compute_overlap_profile(h)
        return np.max(overlaps)

    # Simulated Annealing
    current_h = random_binary_state()
    current_max_ov = get_max_overlap(current_h)
    
    best_h = current_h.copy()
    best_max_ov = current_max_ov
    
    # Parameters
    T = 50.0  # Initial temperature (scale of overlap differences)
    cooling = 0.995
    
    # Precompute shifts for fast evaluation? 
    # Actually, FFT is O(N log N) ~ 700 ops. 100000 iters * 700 = 70M ops.
    # In Python/NumPy, this might be slow.
    # Let's optimize the evaluation.
    # Since h is binary, we can use bit operations if N is small, but N=100.
    # FFT is fine if we vectorize the loop or just run it.
    # 100,000 iterations of FFT(100) might take ~10-20 seconds.
    # Let's aim for 50,000 iterations.
    
    rng = np.random.default_rng(42)
    
    for i in range(50000):
        if time.time() > deadline:
            break
            
        # Move: flip two random bits (one 1 to 0, one 0 to 1) to maintain sum
        # Find indices of 1s and 0s
        # This is O(N) per step if we use np.where, which is slow.
        # Optimization: Keep track of lists of 1s and 0s?
        # Or just pick two random indices and check if they are different.
        
        idx1 = rng.integers(0, N)
        idx2 = rng.integers(0, N)
        
        if idx1 == idx2:
            continue
            
        if current_h[idx1] == current_h[idx2]:
            continue
            
        # Propose flip
        new_h = current_h.copy()
        new_h[idx1] = 1 - new_h[idx1]
        new_h[idx2] = 1 - new_h[idx2]
        
        # Evaluate new max overlap
        new_max_ov = get_max_overlap(new_h)
        
        delta = new_max_ov - current_max_ov
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_h = new_h
            current_max_ov = new_max_ov
            
            if current_max_ov < best_max_ov:
                best_h = current_h.copy()
                best_max_ov = current_max_ov
        
        T *= cooling
        
        # If T gets too small, restart with higher T to escape local minima
        if T < 1.0:
            T = 20.0
            # Optional: restart from best or random
            # Let's stick with current to refine, but periodic random restarts help
            if i % 10000 == 0 and i > 0:
                # 10% chance to restart from best
                if rng.random() < 0.1:
                    current_h = best_h.copy()
                    current_max_ov = best_max_ov

    # Convert best binary h to float for return
    # The task requires h in [0,1] and sum N/2.
    # Binary h satisfies this.
    h_final = best_h.astype(float)
    
    # Verify constraints
    if not np.all((h_final >= 0) & (h_final <= 1)):
        return np.full(N, 0.5)
    if not np.isclose(np.sum(h_final), N/2):
        return np.full(N, 0.5)
        
    return h_final
