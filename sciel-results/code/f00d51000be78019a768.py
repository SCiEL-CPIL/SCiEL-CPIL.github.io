import numpy as np
import time

N = 400
N_FFT = 2 * N

def construct():
    """
    Construct a signed step function (sequence of +/-1 of length N)
    that minimizes the upper bound on C3 (autoconvolution sidelobe).
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    # Fast autocorrelation via FFT
    def autocorr_fast(h):
        F = np.fft.rfft(h, n=N_FFT)
        power = F * np.conj(F)
        ac = np.fft.irfft(power, n=N_FFT)
        return ac

    def max_sidelobe(ac):
        # Sidelobes: lags 1..N-1 and -(N-1)..-1
        # Due to symmetry, check ac[1:N]
        return np.max(np.abs(ac[1:N]))

    def evaluate(h):
        ac = autocorr_fast(h)
        return max_sidelobe(ac), ac

    def flip_cost(h, idx):
        """
        Estimate change in max sidelobe if we flip h[idx].
        Uses the fact that flipping h[idx] changes autocorrelation.
        For speed, we do a full recomputation but it's vectorized.
        """
        h_new = h.copy()
        h_new[idx] = -h_new[idx]
        ac_new = autocorr_fast(h_new)
        return max_sidelobe(ac_new)

    # Initialize with several candidate starts
    best_h = None
    best_val = float('inf')
    best_ac = None

    # Start 1: Random
    rng = np.random.default_rng(42)
    h1 = rng.choice([-1.0, 1.0], size=N)
    val1, ac1 = evaluate(h1)
    best_h = h1.copy()
    best_val = val1
    best_ac = ac1

    # Start 2: Alternating
    h2 = np.ones(N) * (1.0 - 2.0 * (np.arange(N) % 2))
    val2, ac2 = evaluate(h2)
    if val2 < best_val:
        best_h = h2.copy()
        best_val = val2
        best_ac = ac2

    # Start 3: Sine-based sign pattern
    t = np.linspace(0, 2 * np.pi * 7, N, endpoint=False)
    h3 = np.sign(np.sin(t) + 0.3 * np.sin(3 * t))
    h3[h3 == 0] = 1.0
    val3, ac3 = evaluate(h3)
    if val3 < best_val:
        best_h = h3.copy()
        best_val = val3
        best_ac = ac3

    # Start 4: Another random with different seed
    rng2 = np.random.default_rng(12345)
    h4 = rng2.choice([-1.0, 1.0], size=N)
    val4, ac4 = evaluate(h4)
    if val4 < best_val:
        best_h = h4.copy()
        best_val = val4
        best_ac = ac4

    # Start 5: Walsh-Hadamard based (pseudo-random low autocorrelation)
    def walsh_hadamard_sign(n):
        """Generate a pseudo-random sign sequence using Walsh-Hadamard structure."""
        # Use a single Walsh function index for variety
        m = n
        h_w = np.ones(m)
        # Use index 3 (binary 0011) for a specific Walsh function
        idx = 3
        for j in range(m):
            # Compute Walsh function value at position j
            v = 1
            for b in range(8):
                if (idx >> b) & 1:
                    v *= -1 if ((j >> b) & 1) else 1
            h_w[j] = v
        return h_w

    h5 = walsh_hadamard_sign(N)
    val5, ac5 = evaluate(h5)
    if val5 < best_val:
        best_h = h5.copy()
        best_val = val5
        best_ac = ac5

    # Start 6: M-sequence like (maximal length sequence)
    # Use a linear feedback shift register pattern
    def m_sequence(n, poly):
        """
        Generate an M-sequence sign pattern.
        poly: list of feedback tap positions (1-indexed from LSB)
        """
        m = n
        h_m = np.zeros(m)
        state = 1
        for i in range(m):
            h_m[i] = 1.0 if state & 1 else -1.0
            # Feedback
            fb = 0
            for p in poly:
                fb ^= (state >> (p - 1)) & 1
            state = (state >> 1) | (fb << (n - 1))
        return h_m

    # Use a known primitive polynomial for n=7 (period 127), tile
    poly7 = [3, 7]  # x^7 + x^3 + 1
    seq7 = m_sequence(7, poly7)
    # Tile to length 400
    h6 = np.tile(seq7, N // 7 + 1)[:N]
    val6, ac6 = evaluate(h6)
    if val6 < best_val:
        best_h = h6.copy()
        best_val = val6
        best_ac = ac6

    # Local search: steepest descent with full recomputation
    def local_search_steepest(h, ac, val, time_limit=10.0):
        """
        Steepest descent: try all single flips, take the best improvement.
        Repeat until no improvement or time limit.
        """
        t0 = time.time()
        improved = True
        while improved and time.time() - t0 < time_limit:
            improved = False
            best_delta = 0
            best_idx = -1

            for i in range(N):
                # Flip
                h[i] = -h[i]
                ac_new = autocorr_fast(h)
                val_new = max_sidelobe(ac_new)
                delta = val - val_new
                # Unflip
                h[i] = -h[i]

                if delta > best_delta:
                    best_delta = delta
                    best_idx = i
                    best_ac_new = ac_new
                    best_val_new = val_new

            if best_idx != -1:
                h[best_idx] = -h[best_idx]
                ac = best_ac_new
                val = best_val_new
                improved = True

        return h, ac, val

    # Simulated Annealing with parallel tempering (simplified: multiple replicas)
    def parallel_tempering(h_init, ac_init, val_init, time_limit=15.0, num_replicas=4):
        """
        Parallel tempering SA with num_replicas replicas at different temperatures.
        """
        t0 = time.time()
        # Initialize replicas
        replicas = []
        for r in range(num_replicas):
            h_r = h_init.copy()
            ac_r = ac_init.copy()
            val_r = val_init
            replicas.append((h_r, ac_r, val_r))

        # Temperature schedule: T_r = T_base * (r+1)
        T_base = 0.3
        T_end = 0.05
        cooling = 0.995

        while time.time() - t0 < time_limit:
            # Update each replica
            for r in range(num_replicas):
                h_r, ac_r, val_r = replicas[r]
                T_r = max(T_end, T_base * (r + 1) * cooling)

                # Propose a flip
                i = np.random.randint(N)
                h_r[i] = -h_r[i]
                ac_new = autocorr_fast(h_r)
                val_new = max_sidelobe(ac_new)

                delta = val_r - val_new
                if delta > 0 or np.random.rand() < np.exp(-delta / T_r):
                    replicas[r] = (h_r, ac_new, val_new)
                else:
                    h_r[i] = -h_r[i]
                    # Keep old ac, val

            # Try swapping between adjacent replicas
            for r in range(num_replicas - 1):
                if np.random.rand() < 0.5:
                    # Accept swap with probability based on temperature difference
                    # Simple: always accept if it improves the global best
                    pass  # Simplified: no swaps for now to save time

            # Cool down
            T_base = max(T_end, T_base * cooling)

        # Return the best replica
        best_idx = 0
        for r in range(num_replicas):
            if replicas[r][2] < replicas[best_idx][2]:
                best_idx = r
        return replicas[best_idx]

    # Apply local search to the best initial candidate
    if time.time() < deadline - 20.0:
        t_limit = min(15.0, deadline - time.time() - 20.0)
        best_h, best_ac, best_val = local_search_steepest(best_h.copy(), best_ac.copy(), best_val, time_limit=t_limit)

    # Apply parallel tempering
    if time.time() < deadline - 10.0:
        t_limit = min(20.0, deadline - time.time() - 10.0)
        h_pt, ac_pt, val_pt = parallel_tempering(best_h.copy(), best_ac.copy(), best_val, time_limit=t_limit)
        if val_pt < best_val:
            best_h = h_pt
            best_ac = ac_pt
            best_val = val_pt

    # Final polish with steepest descent
    if time.time() < deadline - 2.0:
        t_limit = min(8.0, deadline - time.time() - 2.0)
        best_h, best_ac, best_val = local_search_steepest(best_h.copy(), best_ac.copy(), best_val, time_limit=t_limit)

    return best_h
