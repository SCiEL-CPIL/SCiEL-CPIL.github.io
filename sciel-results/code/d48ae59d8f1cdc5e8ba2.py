"""Improved program for the third autocorrelation inequality task."""
import numpy as np
import time

N = 400


def _fft_autocorr_max_abs(s):
    """Compute max |autocorr| over non-zero lags for a length-N integer sequence.

    Returns the maximum of |r[k]| for k=1..N-1 where r = ifft(|fft(s, n_fft)|^2).real
    and r[k] is the circular autocorrelation at lag k (with n_fft large enough to
    capture the linear autocorrelation).
    """
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.fft(s_f, n=n_fft)
    acf = np.fft.ifft(F * np.conj(F)).real[:N]
    # Normalize by N so that lag-0 is ~sum(s^2)/N
    acf /= N
    return float(np.max(np.abs(acf[1:])))


def _score(s):
    """Score = upper bound on C3; lower is better."""
    return _fft_autocorr_max_abs(s)


def _random_walk_sequence():
    """Generate a random +/-1 sequence."""
    return np.random.choice([-1, 1], size=N)


def _construct_seed():
    """Construct the all-ones seed (score 1.0 baseline)."""
    return np.ones(N, dtype=np.int8)


def construct():
    """
    Search for a signed step function (400 intervals of +/-1) that minimizes
    the max absolute off-peak autocorrelation, which upper-bounds C3.

    Strategy:
    - Start with the all-ones seed.
    - Run a local search (flip random positions) with restarts, using a time budget.
    - Keep the best sequence found.
    """
    deadline = time.time() + 110  # safety margin before 120s kill
    best_seq = _construct_seed()
    best_score = _score(best_seq)  # should be ~1.0 (all ones has autocorr ~1 at all lags)

    # Actually for all-ones, autocorr at lag k is (N-k)/N, max is (N-1)/N ≈ 0.9975
    # So best_score ≈ 0.9975

    rng = np.random.default_rng(42)

    # Local search: flip one random position at a time
    # Use a temperature-like schedule: accept if improves, or with small probability if not
    seq = best_seq.copy().astype(np.int8)
    current_score = best_score

    while time.time() < deadline:
        # Flip a random position
        idx = rng.integers(0, N)
        old_val = seq[idx]
        seq[idx] = -old_val
        new_score = _score(seq)

        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_seq = seq.copy()
        else:
            # Revert (simple hill climbing)
            seq[idx] = old_val

    return best_seq
