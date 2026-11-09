import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin for 120s limit

    # Pre-compute FFT size
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2

    def get_autocorr(seq):
        """Compute non-zero lag autocorrelations using FFT."""
        f = np.zeros(fft_size, dtype=np.complex128)
        f[:N] = seq.astype(np.float64)
        F = np.fft.fft(f)
        ac = np.fft.ifft(F * np.conj(F)).real[:N]
        return ac[1:]

    def score(seq):
        """Max absolute autocorrelation over non-zero lags."""
        ac = get_autocorr(seq)
        return np.max(np.abs(ac))

    def fast_flip_update(seq, idx, ac):
        """
        Update autocorrelation array ac after flipping seq[idx].
        Returns new ac array.
        """
        new_ac = ac.copy()
        old_val = seq[idx]
        # Flipping seq[idx] changes it by -2*old_val
        # For lag k:
        #   term where i=idx: seq[idx]*seq[idx+k] -> -seq[idx]*seq[idx+k], change = -2*old_val*seq[idx+k]
        #   term where i+k=idx (i=idx-k): seq[idx-k]*seq[idx] -> -seq[idx-k]*seq[idx], change = -2*seq[idx-k]*old_val
        for k in range(1, N):
            delta = 0.0
            if idx < N - k:
                delta += -2.0 * old_val * seq[idx + k]
            if idx - k >= 0:
                delta += -2.0 * seq[idx - k] * old_val
            new_ac[k - 1] += delta
        return new_ac

    # Generate LFSR m-sequence for good initial structure
    def lfsr_sequence(n, poly=0x199, init=1 << 8, length=511):
        lfsr = init
        seq = []
        for _ in range(length):
            seq.append(1 if (lfsr & 1) else -1)
            feedback = (lfsr & 1)
            lfsr >>= 1
            if feedback:
                lfsr ^= poly
        # Take first n bits, pad or truncate
        if len(seq) >= n:
            return np.array(seq[:n], dtype=np.float64)
        else:
            # Pad with random
            remaining = n - len(seq)
            pad = np.random.choice([-1, 1], size=remaining)
            return np.concatenate([seq, pad])

    rng = np.random.default_rng(42)

    # Try multiple initializations: LFSR and random
    best_a = None
    best_score = np.inf

    # Initialize with LFSR
    a_lfsr = lfsr_sequence(N)
    s_lfsr = score(a_lfsr)
    best_a = a_lfsr.copy()
    best_score = s_lfsr

    # Also try a few random inits
    for _ in range(5):
        a_rand = rng.choice([-1, 1], size=N).astype(np.float64)
        s_rand = score(a_rand)
        if s_rand < best_score:
            best_score = s_rand
            best_a = a_rand.copy()

    # Simulated annealing with fast flip updates
    T = 20.0
    T_min = 0.01
    cooling = 0.995

    current_a = best_a.copy()
    current_ac = get_autocorr(current_a)
    current_score = np.max(np.abs(current_ac))
    best_a = current_a.copy()
    best_score = current_score
    best_ac = current_ac.copy()

    while time.time() < deadline and T > T_min:
        # Try multiple flips per temperature step
        n_flips = 200
        for _ in range(n_flips):
            if time.time() > deadline:
                break
            idx = rng.integers(N)
            old_val = current_a[idx]
            current_a[idx] *= -1

            # Fast update
            new_ac = fast_flip_update(current_a, idx, current_ac)
            new_score = np.max(np.abs(new_ac))

            delta = new_score - current_score
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_score = new_score
                current_ac = new_ac
                if new_score < best_score:
                    best_score = new_score
                    best_a = current_a.copy()
                    best_ac = new_ac.copy()
            else:
                # Revert
                current_a[idx] = old_val
        T *= cooling

    # Local search polish with fast updates
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            old_val = best_a[idx]
            best_a[idx] *= -1
            new_ac = fast_flip_update(best_a, idx, best_ac)
            new_score = np.max(np.abs(new_ac))
            if new_score < best_score:
                best_score = new_score
                best_ac = new_ac
                improved = True
            else:
                best_a[idx] = old_val

    # Final verification
    final_score = score(best_a)
    if final_score < best_score:
        best_score = final_score

    return best_a
