import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def compute_corr(h):
        """Compute circular autocorrelation using FFT."""
        fft_h = np.fft.fft(h, n=2 * N)
        corr = np.fft.ifft(fft_h * np.conj(fft_h)).real[:N]
        return corr

    def max_overlap(h):
        corr = compute_corr(h)
        min_corr = np.min(corr)
        return N / 2.0 - min_corr

    def project_to_polytope(h, S):
        """Project h onto { x : sum(x) = S, 0 <= x_i <= 1 } using binary search."""
        if S < 0 or S > N:
            return np.clip(h, 0, 1)
            
        lo = np.min(h) - 1.0
        hi = np.max(h) + 1.0
        
        for _ in range(60):
            mid = (lo + hi) / 2.0
            val = np.sum(np.clip(h - mid, 0.0, 1.0))
            if val > S:
                lo = mid
            else:
                hi = mid
                
        lambda_star = (lo + hi) / 2.0
        return np.clip(h - lambda_star, 0.0, 1.0)

    def smooth_max_grad(h, beta):
        """
        Returns (smooth_max_value, gradient) for the max overlap.
        Overlap O_k = N/2 - sum h_i h_{i+k}
        We want to minimize max_k O_k.
        """
        corr = compute_corr(h)
        overlaps = N / 2.0 - corr
        
        # Stable log-sum-exp for soft max
        max_o = np.max(overlaps)
        exps = np.exp(beta * (overlaps - max_o))
        sum_exps = np.sum(exps)
        
        # Weights for the soft max
        weights = exps / sum_exps
        
        # Smooth max value
        val = max_o + np.log(sum_exps) / beta
        
        # Gradient of O_k w.r.t h:
        # O_k = N/2 - sum_i h_i h_{i+k}
        # dO_k/dh_j = - (h_{j+k} + h_{j-k})
        # Total grad = sum_k weights[k] * (- (h_{j+k} + h_{j-k}))
        
        # Efficient computation using FFT
        # term_sum[j] = sum_k weights[k] * (h_{j+k} + h_{j-k})
        # Let W be the circular convolution kernel where W[k] = weights[k]
        # Since weights are real and we sum over all k, we can use FFT
        # But weights sum to 1, so it's like a weighted average of shifts.
        
        # Direct loop for clarity and correctness, N=100 is fast enough
        term_sum = np.zeros(N)
        for k in range(N):
            if weights[k] > 1e-9:
                term_sum += weights[k] * (np.roll(h, -k) + np.roll(h, k))
        
        grad = -term_sum
        return val, grad

    def get_best_2block():
        """Find the best 2-block configuration as a starting point."""
        best_h = None
        best_score = float('inf')
        
        for k in range(1, N):
            # Constraint: k*a + (N-k)*b = N/2
            # b = (N/2 - k*a) / (N-k)
            # We want to minimize max(a*(1-a), a*(1-b), b*(1-b), b)
            # subject to a in [0,1], b in [0,1]
            
            # Bounds on a:
            # b >= 0: N/2 - k*a >= 0 -> a <= N/(2k)
            # b <= 1: N/2 - k*a <= N-k -> k*a >= N/2 - N + k -> a >= (N/2 - N + k)/k
            a_min = max(0.0, (N / 2.0 - N + k) / k)
            a_max = min(1.0, N / (2.0 * k))
            
            if a_min >= a_max:
                continue
            
            # Sample a few points to find a good local structure
            # The objective for 2-block is simple, we can just check a few key points
            # or use a small grid
            a_vals = np.linspace(a_min + 1e-9, a_max - 1e-9, 100)
            for a_val in a_vals:
                b_val = (N / 2.0 - k * a_val) / (N - k)
                if b_val < -1e-12 or b_val > 1.0 + 1e-12:
                    continue
                b_val = np.clip(b_val, 0.0, 1.0)
                
                h = np.zeros(N)
                h[:k] = a_val
                h[k:] = b_val
                
                # Calculate score
                score = max_overlap(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
        
        if best_h is None:
            return np.full(N, 0.5)
        return best_h

    best_h = None
    best_score = float('inf')
    
    # 1. Start with the best 2-block structure
    h_2block = get_best_2block()
    best_h = h_2block.copy()
    best_score = max_overlap(h_2block)
    
    # 2. Refine using Projected Gradient Descent on the full vector
    # We use the 2-block as the primary initialization, but also try a few variations
    # to escape potential local minima specific to the 2-block symmetry.
    
    rng = np.random.default_rng(42)
    
    # Add small perturbations to the 2-block to allow it to "break" symmetry
    inits = [h_2block]
    for _ in range(5):
        h_pert = h_2block + 0.05 * rng.standard_normal(N)
        h_pert = project_to_polytope(h_pert, N / 2.0)
        inits.append(h_pert)
        
    # Also include the uniform start as a fallback
    inits.append(np.full(N, 0.5))

    for h0 in inits:
        if time.time() > deadline - 10.0:
            break
            
        h = h0.copy()
        
        # Adaptive beta (sharpness of soft max) and alpha (step size)
        beta = 1.0
        alpha = 0.01
        max_steps = 300
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
                
            val, grad = smooth_max_grad(h, beta)
            
            # Check actual max overlap for tracking best
            actual_score = max_overlap(h)
            if actual_score < best_score:
                best_score = actual_score
                best_h = h.copy()
                
            # Gradient step
            h = h - alpha * grad
            
            # Project back to feasible set
            h = project_to_polytope(h, N / 2.0)
            
            # Adaptive parameters
            # Increase beta to sharpen the max approximation
            beta *= 1.05
            # Decrease alpha slowly to converge
            alpha *= 0.99
            
            # Check convergence
            if np.max(np.abs(grad)) < 1e-6:
                break

    # 3. Final local refinement using L-BFGS-B on the best candidate
    if best_h is not None and time.time() < deadline - 5.0:
        def objective(x):
            h = np.clip(x, 0, 1)
            sum_val = h.sum()
            sum_penalty = 100.0 * (sum_val - N / 2.0) ** 2
            overlap = max_overlap(h)
            return overlap + sum_penalty

        def gradient(x):
            h = np.clip(x, 0, 1)
            sum_val = h.sum()
            sum_grad = 200.0 * (sum_val - N / 2.0)
            
            corr = compute_corr(h)
            k_min = np.argmin(corr)
            
            # Gradient of -min_corr w.r.t h[i] is -(h[(i+k_min)%N] + h[(i-k_min)%N])
            idx = np.arange(N)
            corr_grad = -(h[(idx + k_min) % N] + h[(idx - k_min) % N])
            
            return corr_grad + sum_grad

        try:
            res = minimize(
                objective,
                best_h,
                jac=gradient,
                method='L-BFGS-B',
                bounds=[(0, 1)] * N,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            h_opt = project_to_polytope(res.x, N / 2.0)
            score = max_overlap(h_opt)
            
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
        except Exception:
            pass

    if best_h is None:
        return np.full(N, 0.5)
        
    # Final polish: ensure exact constraints
    best_h = project_to_polytope(best_h, N / 2.0)
    
    # Safety check
    if best_h.min() < -1e-9 or best_h.max() > 1 + 1e-9:
        return np.full(N, 0.5)
    if abs(best_h.sum() - N / 2.0) > 1e-6:
        return np.full(N, 0.5)

    return best_h
