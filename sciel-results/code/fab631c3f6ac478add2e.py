"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s kill

    rng = np.random.RandomState(42)

    def compute_corr_aperiodic(h):
        """Compute aperiodic autocorrelation c[k] for k=0..N-1 using FFT."""
        size = 1 << (2 * N - 1).bit_length()
        H = np.fft.rfft(h, size)
        C = np.fft.irfft(H * np.conj(H), size)
        c = C[:N]
        return c

    def cost_from_corr(c):
        """Max |c[k]| for k >= 1 (sidelobe max)."""
        if len(c) <= 1:
            return 0.0
        return float(np.max(np.abs(c[1:])))

    def fast_local_search(h, time_limit):
        """
        Greedy local search using O(N^2) direct delta calculation.
        Much faster than batch FFT for N=400.
        """
        h = h.copy().astype(np.float64)
        c = compute_corr_aperiodic(h)
        current_cost = cost_from_corr(c)
        best_cost = current_cost
        best_h = h.copy()
        
        # Precompute indices for delta calculation
        # We want to evaluate flip at j for all k.
        # delta_c[j, k] = -2 * h[j] * (h[j+k] (if j+k<N) + h[j-k] (if j-k>=0))
        # We can build a matrix of deltas.
        
        stall = 0
        max_stall = 100
        
        while time.time() < time_limit:
            # Calculate delta for all j simultaneously
            # h is shape (N,)
            # We need a matrix D of shape (N, N) where D[j, k] is the change in c[k] if we flip h[j]
            
            # h[j+k] part:
            # For a fixed j, h[j+1:N] corresponds to k=1..N-1-j
            # h[j-k] part:
            # For a fixed j, h[j::-1] corresponds to k=1..j
            
            # Let's build D
            D = np.zeros((N, N))
            
            # Vectorized construction of D
            # For each j, D[j, 1:N-j] += -2 * h[j] * h[j+1:N]
            # D[j, 1:j+1] += -2 * h[j] * h[j::-1]
            
            # This loop is O(N^2) but with simple operations.
            for j in range(N):
                hc = h[j]
                # Positive lags
                if N - j > 1:
                    D[j, 1:N-j] = -2.0 * hc * h[j+1:N]
                # Negative lags (lag k corresponds to h[j-k])
                if j >= 1:
                    D[j, 1:j+1] += -2.0 * hc * h[j::-1]
            
            # New correlation matrix C_new[j, k] = c[k] + D[j, k]
            # c is shape (N,), so c[None, :] is (1, N)
            C_new = c[None, :] + D
            
            # Cost for each j: max |C_new[j, 1:]|
            costs = np.max(np.abs(C_new[:, 1:]), axis=1)
            
            best_idx = int(np.argmin(costs))
            best_new_cost = float(costs[best_idx])
            
            if best_new_cost < current_cost - 1e-12:
                # Apply flip
                h[best_idx] *= -1.0
                # Update c efficiently? Or just recompute?
                # Since we have D, we can update c directly:
                c = c + D[best_idx, :]
                current_cost = best_new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_h = h.copy()
                stall = 0
            else:
                stall += 1
                if stall >= max_stall:
                    break
                    
        return best_h, best_cost

    # Generate diverse initial candidates
    candidates = []

    # 1. Random
    candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # 2. Chirp-like (quadratic phase)
    t = np.linspace(0, 1, N)
    chirp = np.sign(np.cos(2 * np.pi * t * t * N / 2))
    candidates.append(chirp)

    # 3. Legendre-like (using simple modulo)
    # Just use random for now to keep it simple and fast, 
    # or a simple periodic sequence
    seq = np.tile([1.0, 1.0, -1.0, -1.0], N // 4)
    candidates.append(seq)

    best_h = candidates[0].copy()
    best_cost = np.inf

    # Run local search on each candidate
    time_per_candidate = (deadline - 10 - time.time()) / len(candidates)
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 10:
            break
        t_limit = time.time() + max(1.0, time_per_candidate)
        h_local, cost_local = fast_local_search(cand, t_limit)
        if cost_local < best_cost:
            best_cost = cost_local
            best_h = h_local.copy()

    # Final polish on the best found
    if time.time() < deadline - 1:
        h_polished, cost_polished = fast_local_search(best_h.copy(), deadline - 0.5)
        if cost_polished < best_cost:
            best_cost = cost_polished
            best_h = h_polished.copy()

    # CRITICAL FIX: Ensure sum is not zero
    # If sum is 0, flip the first element
    if np.sum(best_h) == 0:
        best_h[0] *= -1.0
        # Recalculate cost to be safe, though we return the heights
        # The evaluator will calculate the score.
        
    return best_h
# EVOLVE-BLOCK-END
