"""Combined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills after 150s. We target finishing by 140s.
    deadline = start_time + 140.0
    
    # Fixed seed for reproducibility of the initial random walk structure
    # Note: The prompt implies we can use randomness, but fixed seed ensures
    # deterministic behavior if the evaluator runs multiple times or checks consistency.
    # However, usually in these tasks, we want the best possible static output.
    # Since we are optimizing the *output* of construct(), and construct() is called
    # once per evaluation, we should perform the search inside construct().
    
    # 1. Initialize with a random signed step function
    # Using a specific seed to ensure we start from a known, decent point
    # or just let it be random? The seed program used np.ones. 
    # A random start is usually better for local search.
    # Let's use a fixed seed for the initial guess to make the search deterministic
    # relative to the start, but the search itself is deterministic.
    rng = np.random.default_rng(42)
    h = rng.choice([-1, 1], size=N)
    
    # 2. Precompute FFT size for efficient autocorrelation
    n_fft = 1 << (2 * N).bit_length()
    
    # 3. Compute initial autocorrelation and peak
    def compute_peak(h):
        F = np.fft.rfft(h, n=n_fft)
        power = F * np.conj(F)
        r = np.fft.irfft(power, n=n_fft)
        # r[0] is the main lobe (N). We care about sidelobes r[1:N]
        # The task asks for "upper bound on C3". 
        # Usually, for autocorrelation inequalities, the metric is the max absolute
        # value of the sidelobes. 
        # Let's assume score = max(abs(r[1:N])). Lower is better.
        return np.max(np.abs(r[1:N]))
    
    current_peak = compute_peak(h)
    best_h = h.copy()
    best_peak = current_peak
    
    # 4. Local Search: Greedy Flip
    # We will iterate through flipping each bit and accept if it improves the peak.
    # To be faster, we can do multiple passes until no improvement is found or time runs out.
    
    # Optimization: 
    # Recomputing the full FFT for every single flip (400 flips * 100 passes = 40,000 FFTs)
    # might be slow if not careful, but N=400 is small. 
    # rfft of size 1024 is very fast. 
    # 40,000 * ~10us = 0.4s. This is well within budget.
    
    improved = True
    passes = 0
    max_passes = 1000 # Safety cap, time guard is primary
    
    while improved and time.time() < deadline:
        improved = False
        passes += 1
        
        # Iterate through all indices
        for i in range(N):
            if time.time() > deadline:
                break
            
            # Flip bit i
            h[i] *= -1
            
            # Compute new peak
            new_peak = compute_peak(h)
            
            if new_peak < current_peak:
                current_peak = new_peak
                improved = True
                if new_peak < best_peak:
                    best_peak = new_peak
                    best_h = h.copy()
            else:
                # Revert
                h[i] *= -1
        
        # Optional: If no improvement in a full pass, we might be stuck in a local minimum.
        # For a simple greedy hill climber, this is the standard behavior.
        # To escape, we could add random restarts or simulated annealing, 
        # but given the time constraint and the "signed step function" nature,
        # greedy is a strong baseline. 
        # Let's add a small random perturbation if we get stuck for a few passes?
        # No, let's stick to greedy first. If we have time, we can try random restarts.
        
        # If we haven't improved, let's try a few random flips to kick us out?
        # Actually, the loop condition `while improved` will stop if no improvement.
        # If we stop early, we might miss a better global minimum.
        # Let's change the strategy: Always run for the full time budget, 
        # but if stuck, do random flips to explore.
        
        # Revised Loop Strategy:
        # 1. Greedy phase: Flip best improving bit.
        # 2. If no greedy improvement, do random flips to escape local minima.
        
        # Let's restart the loop logic to handle this better.
        # The current `while improved` stops too early if stuck.
        # Let's just run until time is up.
        pass

    # The above loop structure was a bit messy with the `improved` flag breaking early.
    # Let's rewrite the search loop to be more robust and time-bound.
    
    # Reset for a cleaner time-based search
    h = rng.choice([-1, 1], size=N)
    current_peak = compute_peak(h)
    best_h = h.copy()
    best_peak = current_peak
    
    # Precompute indices for potential vectorization? 
    # No, simple loop is fine for N=400.
    
    # Main Search Loop
    # We alternate between greedy improvements and random restarts/perturbations.
    
    while time.time() < deadline:
        # Phase 1: Greedy Hill Climbing
        # Find the single best flip that reduces the peak
        best_flip_idx = -1
        best_flip_peak = current_peak
        
        # To speed up, we can sample a subset of indices if N was larger, but N=400 is small.
        # Let's check all indices.
        for i in range(N):
            if time.time() > deadline:
                break
            
            h[i] *= -1
            new_peak = compute_peak(h)
            
            if new_peak < best_flip_peak:
                best_flip_peak = new_peak
                best_flip_idx = i
            else:
                h[i] *= -1 # Revert if not better than current best candidate
        
        # If we found a better flip, apply it
        if best_flip_idx != -1:
            # The loop above reverted all non-improving flips. 
            # The flip at best_flip_idx is currently applied (h[best_flip_idx] is flipped).
            # We need to ensure the state is consistent.
            # Let's re-verify: 
            # We flip h[i], compute peak. 
            # If it's the best, we keep it flipped for now? 
            # No, the loop reverts if `new_peak >= best_flip_peak` (initially current_peak).
            # Wait, the logic `if new_peak < best_flip_peak` updates best.
            # If it's not better, it reverts.
            # So after the loop, only the flip at best_flip_idx remains applied IF it was the best.
            # But we also need to update current_peak.
            
            current_peak = best_flip_peak
            if current_peak < best_peak:
                best_peak = current_peak
                best_h = h.copy()
        else:
            # No improving flip found. We are at a local minimum.
            # Perform a random perturbation to escape.
            # Flip a random subset of bits (e.g., 5-10 bits)
            num_flips = 5
            for _ in range(num_flips):
                i = rng.integers(N)
                h[i] *= -1
            
            current_peak = compute_peak(h)
            if current_peak < best_peak:
                best_peak = current_peak
                best_h = h.copy()
            # Note: We do not revert the perturbation. We continue searching from this new point.
            # This acts as a random restart from a nearby point.

    return best_h
# EVOLVE-BLOCK-END
