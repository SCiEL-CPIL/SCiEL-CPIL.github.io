"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog

def _compute_score(x):
    """
    Compute the ratio: max(|autocorr|) / (integral)^2
    where autocorr is the full linear autocorrelation of x.
    """
    # x is a 1D array of length N
    # Autocorrelation (linear convolution of x with x reversed, or just convolve x with x)
    # The task says "autoconvolution". Usually autocorrelation is <x, shift(x)>.
    # Linear autocorrelation r[k] = sum x[n] x[n-k].
    # This is equivalent to np.convolve(x, x, mode='full')?
    # Let's check: conv(x, x) gives sum x[i] x[j-i].
    # If we define autocorr[k] = sum x[n] x[n-k], this is the same as convolution if we handle indices carefully.
    # Actually, np.convolve(x, x, 'full') computes sum_{i} x[i] x[n-i].
    # For real signals, autocorrelation is often defined as sum x[n] x[n-k].
    # Let's use np.correlate(x, x, mode='full') which gives sum x[n+k] x[n].
    # The max absolute value is the same for correlation and convolution of x with itself?
    # conv(x,x)[k] = sum x[i] x[k-i].
    # corr(x,x)[k] = sum x[n+k] x[n].
    # They are related by reversal. The set of absolute values is identical.
    
    auto = np.correlate(x, x, mode='full')
    max_abs = np.max(np.abs(auto))
    
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e100  # Avoid division by zero, effectively infinite penalty
    
    return max_abs / (integral * integral)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial guess: Alternating signs to cancel out integral?
    # If integral is 0, the score is undefined/infinite. We want integral != 0.
    # We want to minimize max(|auto|) / integral^2.
    # This is equivalent to minimizing max(|auto|) subject to integral = 1 (scaling invariance).
    # Let's fix integral = 1 and minimize max(|auto|).
    
    # Strategy:
    # 1. Start with a simple pattern.
    # 2. Use gradient descent or scipy minimize to reduce the max absolute autocorrelation.
    #    Objective: f(x) = max(|auto(x)|) subject to sum(x) = 1.
    #    This is a non-smooth optimization problem.
    #    We can use a smooth approximation: sum(exp(beta * |auto|)) or similar.
    #    Or simply minimize the L-infinity norm using iterative reweighting (like L1 -> Linf).
    
    # Let's try a few random initializations and polish.
    
    best_x = None
    best_score = float('inf')
    
    # Candidate 1: All ones (normalized)
    x0 = np.ones(N) / N
    score0 = _compute_score(x0)
    if score0 < best_score:
        best_score = score0
        best_x = x0.copy()
    
    # Candidate 2: Alternating +1/-1 (sum is 0, bad).
    # Candidate 3: Random
    np.random.seed(42)
    for i in range(50):
        if time.time() > deadline:
            break
        x = np.random.randn(N)
        x = x / np.sum(x)  # Normalize to sum=1
        s = _compute_score(x)
        if s < best_score:
            best_score = s
            best_x = x.copy()
        
        # Try a local polish from this random start
        # Use scipy minimize with a smooth surrogate for max(abs(auto))
        # Surrogate: soft_max = (1/beta) * log(sum(exp(beta * |auto|)))
        # We want to minimize this.
        
        beta = 10.0
        
        def objective(x_flat):
            x = x_flat
            auto = np.correlate(x, x, mode='full')
            abs_auto = np.abs(auto)
            # Soft max
            # Numerical stability: subtract max
            m = np.max(abs_auto)
            if m == 0:
                return 0
            soft_max = m + np.log(np.sum(np.exp(beta * (abs_auto - m)))) / beta
            return soft_max
        
        # Constraint: sum(x) = 1
        # We can parametrize x to satisfy this? Or use penalty.
        # Let's use penalty method to keep it simple and robust.
        
        def penalized_objective(x_flat, mu=100.0):
            x = x_flat
            obj = objective(x)
            penalty = mu * (np.sum(x) - 1.0)**2
            return obj + penalty
        
        # Gradient is hard to compute analytically for soft max of abs.
        # Let's use BFGS or Nelder-Mead? Nelder-Mead is derivative-free but slow for high dim.
        # BFGS requires gradient.
        # Let's approximate gradient numerically? Too slow for N=400.
        
        # Alternative: Use the fact that we just want to reduce peak autocorrelation.
        # This is similar to Barker sequences or Zadoff-Chu, but real-valued.
        # For real-valued, we can use "Golay sequences" concepts?
        
        # Let's try a simple iterative projection / gradient step.
        # Gradient of max(abs(auto)) is tricky.
        # Let's try minimizing sum(auto^2) first (Wiener filter style) then polish peaks?
        # Min sum(auto^2) is equivalent to minimizing energy in frequency domain?
        # sum(auto^2) = sum |X(f)|^4. Minimizing this flattens the spectrum?
        
        # Let's try minimizing the L2 norm of the autocorrelation sidelobes.
        # Fix main lobe?
        
        # Actually, let's just use `minimize` with `method='L-BFGS-B'` and numerical gradients?
        # No, N=400 is too large for numerical gradients.
        
        # Let's try a specific structure:
        # x = [1, -1, 1, -1 ...] has auto [N, 0, -N, 0 ...]. Max abs = N. Sum=0.
        # If we perturb to make sum=1, e.g., x = [1, -1, ..., 1, -1, 2].
        
        # Let's try a "sinc-like" or "windowed" approach.
        
        # A known good sequence for low autocorrelation is the Barker sequence (complex).
        # For real, we can use "Zadoff-Chu" (complex) or specific real sequences.
        
        # Let's try to optimize x directly with a simple gradient descent on the smooth surrogate.
        # We need the gradient of soft_max(beta * |auto|).
        # auto[k] = sum x[n] x[n-k]
        # d auto[k] / d x[m] = x[m-k] + x[m+k] (with boundary checks)
        
        # This is getting complex. Let's try a simpler heuristic:
        # Randomize, then do "peak shaving".
        # Identify the index k with max |auto[k]|.
        # Adjust x to reduce auto[k].
        # auto[k] = sum x[n] x[n-k].
        # To reduce auto[k], we can flip signs of x where x[n] and x[n-k] have the same sign.
        
        # Let's implement a simple local search:
        # 1. Start with random x (sum=1).
        # 2. Repeat:
        #    a. Compute auto.
        #    b. Find k with max |auto[k]|.
        #    c. If k=0, it's the main lobe (energy). We can't reduce it without reducing energy (which increases score if sum fixed? No, sum fixed, energy min is 1/N for constant. But constant has huge sidelobes? No, constant has triangular auto.
        #       Wait, constant x: auto is triangular. Max is at 0.
        #       Score = N / 1^2 = N.
        #       Alternating x (sum~0): auto is spikes at 0 and N.
        
        # Let's look at the structure of the objective again.
        # Minimize max(|auto|) / sum(x)^2.
        # Scale x so sum(x)=1. Minimize max(|auto|).
        
        # If we use x = delta (impulse), auto = delta. max=1. sum=1. score=1.
        # If we use x = ones/N, auto[0]=1/N, auto[k]=(N-k)/N^2. max=1/N. score=(1/N)/1 = 1/N = 0.0025.
        # Wait, sum(x)=1. auto[0] = sum x^2 = 1/N.
        # So score is 1/N for uniform distribution.
        
        # Can we do better?
        # If we spread x more? No, x is discrete length N.
        # What if x has negative values?
        # Example: x = [1, -1, 1, -1, ...] normalized.
        # Sum is 0. Bad.
        # x = [1, -1, 1, -1, ..., 1, -1, 1] (N=400, odd number of 1s?).
        
        # Let's try to minimize the peak sidelobe.
        # The main lobe (k=0) is sum(x^2). By Cauchy-Schwarz, sum(x^2) >= (sum(x))^2 / N = 1/N.
        # So auto[0] >= 1/N.
        # The max absolute value is at least 1/N.
        # So the score is at least 1/N = 0.0025.
        # The uniform distribution achieves exactly 1/N at k=0, and the sidelobes are (N-k)/N^2 which is < 1/N.
        # So the uniform distribution achieves the theoretical lower bound for the main lobe!
        # Is it possible to have max(|auto|) < 1/N?
        # No, because auto[0] = sum(x^2) >= 1/N.
        # So the minimum possible score is 1/N.
        # The uniform distribution `np.ones(N)/N` achieves this.
        
        # Let's verify.
        # x = ones(N)/N.
        # sum(x) = 1.
        # auto[0] = sum(1/N^2) = N/N^2 = 1/N.
        # auto[k] for k>0: (N-k)/N^2.
        # Max abs is 1/N.
        # Score = (1/N) / 1 = 1/N.
        
        # So the optimal solution is simply `np.ones(N)/N`?
        # Or any distribution that minimizes sum(x^2) given sum(x)=1.
        # That is the uniform distribution.
        
        # Wait, is there a constraint that x must be "signed step heights"?
        # The task says "sequence of signed step heights".
        # It doesn't say x must be non-negative.
        # But if x has negative values, sum(x^2) is still >= (sum(x))^2 / N.
        # So the lower bound 1/N holds for any real x.
        # And the uniform distribution (all positive) achieves it.
        
        # So the answer is just `np.ones(N) / N`?
        # Let's check the seed program.
        # Seed returns `np.ones(N)`.
        # Score of seed:
        # x = ones(N). sum = N.
        # auto[0] = N.
        # Score = N / N^2 = 1/N.
        
        # So the seed program already achieves the optimal score 1/N?
        # If so, the score is 0.0000 in the prompt?
        # "score is the upper bound on C3' ... evaluator measures the raw value (lower is better). The search score is seed/value".
        # If value = 1/N, and seed value = 1/N, then score = 1.0.
        # But the prompt says "CURRENT PROGRAM (score 0.0000)".
        # This implies the seed program scored 0.
        # Why?
        # Maybe an exception? Or timeout? Or invalid?
        # `np.ones(N)` is valid.
        # Maybe the definition of autoconvolution is different?
        # "maximum absolute autoconvolution".
        # Maybe it's circular autocorrelation?
        # If circular, auto[0] = sum(x^2). Same bound.
        
        # Let's re-read carefully.
        # "Signed equal-width step functions minimizing the maximum absolute autoconvolution divided by squared integral."
        
        # If the seed scored 0, maybe it crashed?
        # `def construct(): return np.ones(N)`
        # This is valid Python.
        
        # Perhaps the evaluator expects a specific format?
        # "sequence of signed step heights".
        
        # Maybe the issue is that `np.ones(N)` is not considered "signed" in some strict sense? No.
        
        # Let's assume the seed is valid but maybe the score calculation in the evaluator is different or there's a bug in my understanding of the score.
        # However, if the lower bound is 1/N, and the seed achieves 1/N, it's optimal.
        
        # What if N is not 400? The code sets N=400.
        
        # Let's consider if "autoconvolution" means something else.
        # Convolution of x with itself?
        # conv(x, x) has length 2N-1.
        # Max abs of conv(x,x).
        # For x=ones(N), conv is triangular. Max is N.
        # Integral is N.
        # Score = N / N^2 = 1/N.
        
        # Is it possible to get a score < 1/N?
        # No, because auto[0] (or conv[N-1]) is sum(x^2) >= 1/N * (sum x)^2.
        
        # So 1/N is the global minimum.
        
        # Why did the seed score 0?
        # Maybe the evaluator timed out? No, it's instant.
        # Maybe the evaluator expects a list, not a numpy array?
        # "return sequence".
        
        # Let's try returning a list.
        # Also, let's add a tiny bit of noise or a different pattern just in case "ones" is penalized for some reason (e.g., not "signed" enough?).
        # But mathematically, ones is optimal.
        
        # Let's return `np.ones(N) / N` to normalize?
        # The score is invariant to scaling.
        # `np.ones(N)` -> score 1/N.
        # `np.ones(N)/N` -> score 1/N.
        
        # I will return `np.ones(N) / N` to be safe and clean.
        
        return np.ones(N) / N

# EVOLVE-BLOCK-END
