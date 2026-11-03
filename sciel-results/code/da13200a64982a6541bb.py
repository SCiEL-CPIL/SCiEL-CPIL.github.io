import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_x = np.ones(N)
    best_score = np.inf
    
    def get_autoconv(x):
        # Compute autoconvolution using FFT
        # Length of autoconv is 2N-1
        # We only care about lags 1 to N-1
        n_fft = 1 << (2 * N - 1).bit_length()
        X = np.fft.rfft(x, n=n_fft)
        C = np.fft.irfft(X * np.conj(X), n=n_fft)[:N]
        # C[0] is sum of squares. We want max abs of C[1:]
        return C
    
    def eval_score(x):
        C = get_autoconv(x)
        # Exclude lag 0
        return np.max(np.abs(C[1:]))
    
    # 1. Random restarts with local search
    rng = np.random.default_rng(42)
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        # Initialize
        # Mix of random and structured
        if num_restarts % 3 == 0:
            x = rng.choice([-1, 1], size=N)
        elif num_restarts % 3 == 1:
            # Blocky random
            x = np.repeat(rng.choice([-1, 1], size=N//10), 10)
            x = x[:N]
        else:
            # High frequency
            x = np.sign(np.sin(np.arange(N) * rng.uniform(0.1, 1.0)))
            x[x == 0] = 1
            
        current_score = eval_score(x)
        
        # Local Search: Greedy flip
        # We try to reduce the maximum peak
        # To speed up, we only look at the top few peaks
        improved = True
        while improved and time.time() < deadline:
            improved = False
            C = get_autoconv(x)
            peaks = np.abs(C[1:])
            max_val = np.max(peaks)
            k_max = np.argmax(peaks) + 1 # Lag index
            
            # Find best flip
            best_gain = 0
            best_flip_idx = -1
            
            # We can approximate the effect of flipping x[i] on C[k_max]
            # C[k] = sum x[j] x[j+k]
            # If we flip x[i], terms involving i change.
            # i is involved in C[k] if i = j or i = j+k
            # j = i -> term x[i]x[i+k] becomes -x[i]x[i+k] (change -2 x[i]x[i+k])
            # j+k = i -> j = i-k -> term x[i-k]x[i] becomes -x[i-k]x[i] (change -2 x[i-k]x[i])
            
            # Calculate new C[k_max] for each i?
            # That's O(N) per i, O(N^2) total. N=400, 160k ops. Fast.
            
            x_old = x
            for i in range(N):
                # Flip x[i]
                x[i] *= -1
                # Recompute only C[k_max]? Or full max?
                # Recomputing full max is safer but slower.
                # Let's just compute the new value at k_max and check if it's better
                # But it might increase another peak.
                # Heuristic: If it reduces the current max, accept.
                
                # Efficient update for C[k_max]:
                # We don't have the old C[k_max] stored separately from the array C?
                # We have C from the previous iteration.
                # Let's just do a full FFT for simplicity if N is small?
                # 400 is small. FFT is ~us.
                # 1000 iterations * 400 flips * FFT(400) ~ 400k FFTs. Might be slow in Python.
                
                # Better: Analytical update for the specific lag.
                # But we need to ensure we don't blow up other lags.
                # Strategy: Flip if it reduces the *current* max peak significantly.
                
                # Let's try a simpler loop:
                # Try flipping each i, compute new max autocorrelation, keep best.
                # This is O(N * CostFFT).
                # CostFFT ~ N log N ~ 400 * 9 ~ 3600 ops.
                # Total per step: 400 * 3600 ~ 1.4M ops.
                # 1000 steps: 1.4B ops. Too slow for 120s?
                # Python overhead is high.
                
                # Alternative: Only check the effect on the top K peaks.
                # Or use the analytical delta for the main peak.
                
                # Let's use the analytical delta for the main peak k_max.
                # Delta = -2 * x[i] * (x[i+k_max] + x[i-k_max]) (with bounds)
                # If Delta < 0, it reduces the positive peak.
                # If Delta > 0, it reduces the negative peak (if C[k_max] < 0).
                
                # We want to minimize |C[k_max]|.
                # Let val = C[k_max]
                # new_val = val + Delta
                # If |new_val| < |val|, it's a good move for this peak.
                
                x_i = x[i]
                term1 = 0
                term2 = 0
                if i + k_max < N:
                    term1 = x[i + k_max]
                if i - k_max >= 0:
                    term2 = x[i - k_max]
                
                delta = -2 * x_i * (term1 + term2)
                current_val = C[k_max]
                new_val = current_val + delta
                
                if abs(new_val) < abs(current_val) - 1e-9: # Strict improvement
                    # Check if this flip worsens other top peaks?
                    # For now, accept if it improves the main peak.
                    # This is a greedy hill climb on the main peak.
                    best_gain = abs(current_val) - abs(new_val)
                    best_flip_idx = i
                    break # Take the first improvement? Or best improvement?
                    # Taking first is faster.
            
            if best_flip_idx != -1:
                x[best_flip_idx] *= -1
                improved = True
            else:
                # No single flip improves the main peak.
                # Try random perturbation to escape
                if rng.random() < 0.1:
                    idx = rng.integers(N)
                    x[idx] *= -1
                    improved = True # Force continue
                else:
                    break
        
        score = eval_score(x)
        if score < best_score:
            best_score = score
            best_x = x.copy()
            
        # Print progress if needed (disabled)
        # print(f"Restart {num_restarts}: Score {score:.2f}, Best {best_score:.2f}")

    return best_x

# EVOLVE-BLOCK-END
