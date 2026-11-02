import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 600

def construct():
    start_time = time.time()
    deadline = 115  # Safety margin for 120s limit (or 150s if evaluator is looser, but 120 is stated)
    
    # Helper: Compute the score (max sidelobe / peak)
    # x: array of length N
    # Autoconvolution c_k = sum_i x_i x_{i+k}
    # c_0 = sum x_i^2
    # We want to minimize max_{k!=0} |c_k| / c_0
    # Note: Since x_i >= 0, c_k can be positive or negative? 
    # Wait, autoconvolution of non-negative sequence:
    # c_k = sum_i x_i x_{i+k}. Since x_i >= 0, c_k >= 0 for all k.
    # So we just need max_{k!=0} c_k / c_0.
    
    def get_score(x):
        # x is 1D array
        # Compute full autoconvolution via FFT
        # Length of full conv is 2N - 1
        # Zero pad to next power of 2 for FFT speed? 2N-1 = 1199. Next pow2 is 2048.
        n_fft = 2048
        X = np.fft.rfft(x, n=n_fft)
        C = np.fft.irfft(X * np.conj(X), n=n_fft)
        # The correlation is in the first N values and the last N-1 values?
        # np.correlate(x, x, 'full') gives:
        # [x0x0, x0x1+x1x0, ..., x0x_{N-1}+...+x_{N-1}x0, ..., x_{N-1}x_{N-1}]
        # Actually, irfft of |X|^2 gives circular correlation.
        # For zero-padded x, the first N samples of irfft are the linear correlation c_0, c_1, ..., c_{N-1}
        # And the last N-1 samples are c_{-(N-1)}, ..., c_{-1}.
        # Since x is real, c_k = c_{-k}.
        # c_0 is C[0].
        # c_1 is C[1].
        # ...
        # c_{N-1} is C[N-1].
        # c_{-1} is C[N_fft-1].
        
        c0 = C[0]
        if c0 == 0:
            return float('inf')
        
        # Extract sidelobes
        # Positive lags: 1 to N-1
        # Negative lags: N to N_fft-1 (which correspond to -1 to -(N-1))
        # Due to symmetry, we can just take max of C[1:N] and C[N:N_fft-1]
        # But C[N:N_fft-1] is actually the negative lags.
        # Let's just take the max of all C except C[0] and the zero-padding tail?
        # The zero-padding ensures that the circular correlation matches linear correlation for lags < N.
        # So C[1:N] are lags 1..N-1.
        # C[N:N_fft-1] are lags -(N-1)..-1.
        # We want max of these.
        
        # Optimization: C is symmetric around 0. 
        # C[k] = C[N_fft-k].
        # So max(C[1:N], C[N:N_fft-1]) is just max(C[1:N])? 
        # Wait. C[N_fft-1] corresponds to lag -1. C[1] corresponds to lag 1.
        # Yes, for real signals, autocorrelation is even.
        # So we only need to check lags 1 to N-1.
        
        max_side = np.max(C[1:N])
        
        return max_side / c0

    # Initial guesses
    # 1. Uniform
    x_uniform = np.ones(N)
    
    # 2. Two blocks (from inspiration 1, which scored 1.125)
    # Let's construct a specific two-block sequence to give L-BFGS a good start.
    # Block width W, gap G.
    # Total width 2W+G <= N.
    # Let's try W=N//4, G=N//2.
    x_two_block = np.zeros(N)
    W = N // 4
    G = N // 2
    start1 = N//4
    end1 = start1 + W
    x_two_block[start1:end1] = 1.0
    
    start2 = end1 + G
    end2 = start2 + W
    if end2 < N:
        x_two_block[start2:end2] = 1.0
    else:
        # Adjust if it goes out of bounds
        x_two_block = np.zeros(N)
        W = N // 3
        G = N // 6
        start1 = 0
        end1 = W
        x_two_block[start1:end1] = 1.0
        start2 = end1 + G
        end2 = start2 + W
        x_two_block[start2:end2] = 1.0

    # 3. Gaussian
    x_gauss = np.exp(-((np.arange(N) - N/2)**2) / (2 * (N/6)**2))

    # Evaluate initial scores
    best_seq = x_uniform
    best_score = get_score(x_uniform)
    
    if get_score(x_two_block) < best_score:
        best_score = get_score(x_two_block)
        best_seq = x_two_block
        
    if get_score(x_gauss) < best_score:
        best_score = get_score(x_gauss)
        best_seq = x_gauss

    # Objective function for L-BFGS-B
    # We minimize f(x) = max_{k!=0} c_k(x) / c_0(x)
    # To make it smoother, we could use a log-sum-exp approximation, but L-BFGS-B often works with raw max if the landscape isn't too jagged.
    # However, max is non-differentiable. 
    # Alternative: Minimize max_{k} c_k subject to c_0 = 1.
    # This is equivalent to minimizing the ratio.
    # Let's stick to the ratio.
    
    def objective(x):
        return get_score(x)
    
    # Constraints: x_i >= 0
    bounds = [(0, None) for _ in range(N)]
    
    # Run L-BFGS-B
    # maxiter and maxfun need to be chosen carefully to fit in time.
    # Each evaluation involves an FFT of size 2048. This is very fast (~microseconds).
    # L-BFGS-B does many evaluations.
    # Let's set a time limit inside the loop? No, L-BFGS-B doesn't support easy time-outs.
    # We can run it with a generous maxiter and check time after.
    # Or use a smaller maxiter.
    # N=600. 1000 iterations might be safe.
    
    try:
        res = minimize(
            objective, 
            best_seq, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 2000, 'maxfun': 20000, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if res.success or res.fun < best_score:
            if res.fun < best_score:
                best_score = res.fun
                best_seq = res.x
            else:
                # Even if not strictly better, maybe res.x is better than initial?
                if res.fun < get_score(best_seq):
                    best_score = res.fun
                    best_seq = res.x
    except Exception as e:
        print(f"Optimization error: {e}")
        # Fallback to best_seq found so far

    # Check time
    if time.time() > deadline - 10:
        return list(best_seq)

    # Post-processing: 
    # L-BFGS-B might leave very small negative values or values close to 0.
    # Clip to ensure non-negativity.
    best_seq = np.clip(best_seq, 0, None)
    
    # Final refinement: Coordinate Descent / Simple Hill Climb
    # Sometimes L-BFGS-B gets stuck in a local minimum where one element is slightly suboptimal.
    # Let's do a few rounds of simple coordinate descent on the best_seq.
    # Pick random indices and try to adjust them? 
    # Or just accept the L-BFGS result.
    
    # Let's try a quick local search: 
    # For a few iterations, pick a random index i, and try to optimize x_i 
    # holding others fixed? This is hard because c_k depends on x_i in multiple terms.
    # Actually, c_k = sum_j x_j x_{j+k}.
    # If we change x_i, it affects c_k for k in [-i, N-1-i] etc.
    # This is complex. Let's stick to the L-BFGS result.
    
    # One more check: 
    # The problem is minimizing the MAX of sidelobes.
    # Often, the optimal solution has multiple sidelobes at the same peak level (equisidlobal).
    # L-BFGS-B should find this if the objective is smooth enough.
    
    # Ensure the output is a list of floats
    return list(best_seq)

if __name__ == "__main__":
    res = construct()
    print(len(res))
    # print(construct.__doc__)
