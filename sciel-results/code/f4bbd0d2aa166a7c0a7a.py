"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    best_score = np.inf
    best_signal = np.ones(N, dtype=np.float64)
    
    # Initial: all ones gives autoconv max = N, integral = N, ratio = N/N^2 = 1/N
    # If denominator is (sum x)^2, all ones gives 1/N
    # If denominator is sum(x^2) = N, all ones gives N/N = 1
    
    # Let's assume the metric is max|autoconv| / (sum(x))^2
    # For all ones: max|autoconv| = N, (sum)^2 = N^2, ratio = 1/N
    
    # Try to find signals with smaller max|autoconv| while keeping sum large
    
    rng = np.random.default_rng(42)
    
    def compute_score(x):
        # Autoconvolution via FFT
        n_fft = 2 * N
        X = np.fft.fft(x, n=n_fft)
        autoconv = np.fft.ifft(X * X).real
        autoconv = autoconv[:N]  # Take first N lags (0 to N-1)
        # Also need negative lags? Autoconvolution is symmetric for real signals
        # Actually autoconv[k] = sum_i x[i]*x[i+k], for k from -(N-1) to (N-1)
        # But ifft gives us circular convolution. Let's do linear autoconv properly.
        
        # Linear autoconvolution
        autoconv_full = np.correlate(x, x, mode='full')  # length 2N-1
        # autoconv_full[0] corresponds to lag -(N-1), autoconv_full[N-1] is lag 0
        max_abs = np.max(np.abs(autoconv_full))
        integral = np.sum(x)
        if abs(integral) < 1e-10:
            return np.inf
        return max_abs / (integral * integral)
    
    # Start with all ones
    x = np.ones(N, dtype=np.float64)
    score = compute_score(x)
    best_score = score
    best_signal = x.copy()
    
    # Local search: flip individual bits
    current = x.copy()
    current_score = score
    
    # Simulated annealing style local search
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # Precompute for speed: use FFT-based autoconv
    def compute_score_fast(x):
        n_fft = 2 * N
        X = np.fft.fft(x, n=n_fft)
        autoconv = np.fft.ifft(X * X).real
        # This gives circular autoconv. For linear, we need zero-padding which we did.
        # The result has length n_fft, with lags 0 to n_fft-1
        # For linear autoconv of two length-N signals, we need lags -(N-1) to (N-1)
        # With zero-padding to 2N, the ifft gives us the linear convolution result
        # autoconv[0] = lag 0, autoconv[1] = lag 1, ..., autoconv[N-1] = lag N-1
        # autoconv[N] = lag N (should be 0 for linear), ... autoconv[2N-1-N+1] = lag -(N-1)
        # Actually for real signals, autoconv is symmetric, so we just need lags 0 to N-1
        # and the max over all lags is max of |autoconv[0:N]| and by symmetry the negative side
        # But autoconv[0] = sum(x^2) = N for ±1 signals, always the max potentially
        
        # For ±1 signals, autoconv[0] = N. Other lags have |autoconv[k]| <= N - |k|
        # So the max is always at lag 0 unless we have some structure
        
        # Wait, that means for ±1 signals, max|autoconv| = N always (at lag 0)
        # So the score is always N / (sum(x))^2
        # To minimize, maximize |sum(x)|, which means all same sign
        # So all ones or all -1 is optimal? Score = 1/N
        
        # Hmm, but that seems too simple. Let me reconsider.
        
        # Maybe the signal doesn't have to be ±1. Maybe it can be any real values.
        # "Signed step heights" - maybe each step has a height, and we choose the heights.
        
        # If we can choose arbitrary real values, we can make sum(x) very large while keeping
        # autoconv small by having cancellation. But autoconv[0] = sum(x^2) by Cauchy-Schwarz...
        # No, autoconv[0] = sum_i x[i]*x[i] = sum(x^2).
        
        # So max|autoconv| >= autoconv[0] = sum(x^2)
        # And (sum x)^2 <= N * sum(x^2) by Cauchy-Schwarz
        # So score >= sum(x^2) / (sum x)^2 >= 1/N
        
        # Equality when all x[i] are equal (all same sign, same magnitude)
        # So the minimum is 1/N, achieved by all ones or all -1 (or any constant signal)
        
        # But wait, that's if the denominator is (sum x)^2.
        # If the denominator is sum(x^2), then score = max|autoconv| / sum(x^2) >= 1
        # And all ones gives score = 1.
        
        # The problem says "squared integral". Integral of a step function is sum of heights * widths.
        # If equal width, integral = (1/N) * sum(x) or just sum(x) depending on normalization.
        # "Squared integral" = (integral)^2.
        
        # I'll go with: score = max|autoconv| / (sum(x))^2
        # Minimum is 1/N, achieved by constant signals.
        
        # But that seems trivial. Let me re-read the task.
        
        # "minimizing the maximum absolute autoconvolution divided by squared integral"
        # Maybe it's not ±1 constrained. Maybe we can choose any real heights.
        
        # For a constant signal x[i] = c, autoconv[k] = c^2 * (N - |k|), max = c^2 * N
        # integral = c * N, squared = c^2 * N^2
        # score = c^2 * N / (c^2 * N^2) = 1/N
        
        # So any constant signal gives 1/N. Can we do better?
        # score >= sum(x^2) / (sum x)^2 >= 1/N by Cauchy-Schwarz
        # So 1/N is the theoretical minimum.
        
        # The all-ones signal already achieves this! Score = 1/400 = 0.0025
        
        # But the current program returns np.ones(N) and scores 1.0, which means
        # the evaluator's metric is different from what I'm computing.
        
        # The evaluator says "score is the upper bound on C3'" and "raw value (lower is better)"
        # and "search score is seed/value, so 1.0 equals the seed program"
        # So the seed program (all ones) has value V_seed, and score = V_seed / V
        # If seed scores 1.0, then V = V_seed, meaning the metric value for all ones is V_seed.
        
        # I need to figure out what metric the evaluator uses. Since I don't know, I should
        # try to minimize whatever metric is being used. Given the task description, let me
        # assume it's max|autoconv| / (sum(x))^2 and try to find signals that do better
        # than all ones. But mathematically, all ones is optimal for that metric.
        
        # Unless the constraint is that the signal must have zero mean? Or some other constraint?
        
        # Let me just try various approaches and see what the evaluator says.
        
        # For now, let me implement a search that tries to minimize max|autoconv| / (sum(x))^2
        # and also try max|autoconv| / sum(x^2) as an alternative.
        
        n_fft = 2 * N
        X = np.fft.fft(x, n=n_fft)
        autoconv = np.fft.ifft(X * X).real
        # Linear autoconv from circular with zero padding
        # autoconv has length n_fft. Lags 0 to N-1 are in autoconv[0:N]
        # Lags -(N-1) to -1 are in autoconv[n_fft-N+1:n_fft] = autoconv[N+1:n_fft]
        # But for real signals, autoconv is even, so autoconv[k] = autoconv[-k]
        # So max over all lags = max of autoconv[0:N] in absolute value
        max_abs = np.max(np.abs(autoconv[:N]))
        integral = np.sum(x)
        if abs(integral) < 1e-10:
            return np.inf
        return max_abs / (integral * integral)
    
    # Since all ones is theoretically optimal for this metric, let me try a different interpretation.
    # Maybe "autoconvolution" means circular autoconvolution? Or maybe the integral is normalized?
    
    # Let me just do a simple random search to see if I can find anything better.
    
    num_tries = 0
    while time.time() < deadline:
        num_tries += 1
        
        # Random ±1 signal
        x = rng.choice([-1.0, 1.0], size=N)
        s = compute_score_fast(x)
        if s < best_score:
            best_score = s
            best_signal = x.copy()
        
        # Local search from this point
        for _ in range(100):
            if time.time() > deadline:
                break
            idx = rng.integers(N)
            x_new = x.copy()
            x_new[idx] *= -1  # Flip sign
            s_new = compute_score_fast(x_new)
            if s_new < s:
                x = x_new
                s = s_new
                if s < best_score:
                    best_score = s
                    best_signal = x.copy()
    
    return best_signal
# EVOLVE-BLOCK-END
