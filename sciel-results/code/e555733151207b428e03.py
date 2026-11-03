import numpy as np
import time
from scipy.optimize import minimize

N = 600
TIME_LIMIT = 145  # Safety margin before 150s kill

# Precompute grid
x_edges = np.linspace(-0.25, 0.25, N + 1)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0

# Precompute FFT size for fast convolution
# Convolution of two length-N sequences results in length 2N-1.
# We need FFT size >= 2N-1. Next power of 2 for 1199 is 2048.
FFT_SIZE = 2048

def compute_score(f):
    """
    Compute the normalized maximum of the autoconvolution.
    f: array of shape (N,)
    Returns: max(conv) / (sum(f)^2)
    """
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    
    # Use FFT for fast convolution
    f_padded = np.zeros(FFT_SIZE)
    f_padded[:N] = f
    conv_fft = np.fft.ifft(np.fft.fft(f_padded) ** 2)
    conv = np.real(conv_fft[:2*N-1])
    
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    
    return np.max(conv) / (s * s)

def construct():
    start_time = time.time()
    
    best_score = np.inf
    best_f = None
    
    # Helper to evaluate a shape function
    def eval_shape(f):
        f = np.maximum(f, 0.0)
        return compute_score(f)

    # --- Strategy: FISTA on 600-dim vector with log-sum-exp smoothing ---
    
    # We want to minimize J(f) = max(conv(f)) / (sum(f))^2
    # Subject to f >= 0.
    # Scaling invariance: J(c*f) = J(f).
    # We can fix sum(f) = 1 to remove the scaling degree of freedom.
    # Then minimize max(conv(f)) subject to f >= 0, sum(f) = 1.
    
    # To handle the max, we use a smooth approximation:
    # max(x) ≈ (1/t) * log(sum(exp(t*x)))
    # Let conv = C(f). We minimize (1/t) * log(sum(exp(t * conv(f)))).
    # However, conv(f) depends quadratically on f.
    # Let's define the objective function for the optimizer.
    
    # We will use scipy.optimize.minimize with method='Nelder-Mead' 
    # or 'L-BFGS-B' if we can compute gradients. 
    # Computing gradients of the smoothed max is possible but complex.
    # Let's try Nelder-Mead first as it is derivative-free and robust for small dims, 
    # but 600 dims is too high for NM.
    
    # Better approach: Use a parametric representation that is still flexible 
    # but lower dimensional? No, the directive says "MATERIALLY DIFFERENT".
    # Direct high-dim optimization is the key.
    
    # Since 600 dims is high for NM, let's use a **Gradient-based** method.
    # We need gradients.
    # Let g(f) = log(sum(exp(t * conv(f)))) / t
    # grad g = (1/t) * sum(exp(t * conv(f)) * grad conv(f)) / sum(exp(t * conv(f)))
    # conv(f)_k = sum_{i+j=k} f_i f_j (with appropriate indexing for linear conv)
    # This is the autoconvolution.
    
    # Alternative: Use the fact that we can project onto the simplex (f>=0, sum=1).
    # We can use FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) 
    # which works for convex problems. But our problem is non-convex.
    # However, if we fix the "shape" and optimize scale? No.
    
    # Let's try a different structural change:
    # **Randomized Coordinate Descent with Non-negative Projection**
    # Or **Stochastic Gradient Descent** on the smoothed objective.
    
    # Let's implement a simple SGD on the 600-dim vector.
    # Objective: LSE smoothing of max conv.
    # Constraint: f >= 0. We can project onto f>=0 after each step.
    # Constraint: sum(f) = 1. We can normalize after each step.
    
    # Initialize with a random non-negative vector
    def random_init():
        f = np.random.rand(N)
        f /= np.sum(f)
        return f

    def smooth_max_conv(f, t=10.0):
        """
        Smooth approximation of max(conv(f)).
        f: array of shape (N,), assumed non-negative.
        """
        f_padded = np.zeros(FFT_SIZE)
        f_padded[:N] = f
        conv_fft = np.fft.ifft(np.fft.fft(f_padded) ** 2)
        conv = np.real(conv_fft[:2*N-1])
        
        # Log-sum-exp for numerical stability
        max_conv = np.max(conv)
        if max_conv > 0:
            # exp(t * (conv - max_conv))
            shifted = t * (conv - max_conv)
            # clip to avoid overflow/underflow issues in exp, though shifted <= 0
            exp_shifted = np.exp(shifted)
            log_sum_exp = max_conv + np.log(np.sum(exp_shifted)) / t
            return log_sum_exp
        else:
            return max_conv # or 0 if all negative, but conv is non-neg if f is non-neg

    def objective_and_grad(f, t=10.0):
        """
        Compute objective and gradient for SGD.
        f: array of shape (N,)
        """
        # Ensure non-negative for gradient calculation? 
        # We will project after the step.
        f = np.maximum(f, 0.0)
        
        f_padded = np.zeros(FFT_SIZE)
        f_padded[:N] = f
        F = np.fft.fft(f_padded)
        conv_fft = np.fft.ifft(F ** 2)
        conv = np.real(conv_fft[:2*N-1])
        
        max_conv = np.max(conv)
        shifted = t * (conv - max_conv)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        
        obj = max_conv + np.log(sum_exp) / t
        
        # Gradient of obj w.r.t conv:
        # d obj / d conv_k = (1/t) * exp(t * (conv_k - max_conv)) / sum_exp
        #                   = exp_shifted[k] / sum_exp
        # Note: max_conv depends on conv, but in LSE, the derivative simplifies to softmax weights.
        weights = exp_shifted / sum_exp
        
        # Gradient of conv w.r.t f:
        # conv_k = sum_{i} f_i f_{k-i} (with bounds)
        # d conv_k / d f_j = f_{k-j} + f_{j-k} (if indices valid)
        # This is the autoconvolution gradient.
        # It's essentially 2 * cross-correlation of f with itself?
        # For linear convolution, grad_conv = 2 * (f * f) but with specific alignment.
        # Actually, d/d f_j (sum_i f_i f_{k-i}) = f_{k-j} + f_{j-k} (if k-j and j-k are in [0, N-1])
        
        # We can compute this efficiently using FFT?
        # The gradient of the LSE objective is sum_k weights[k] * grad conv_k.
        # Let's compute grad_conv as a matrix-vector product or similar.
        # Since N=600, a dense matrix is 600x600 = 360k elements. That's small.
        # We can precompute the gradient matrix structure?
        # G_{j, k} = d conv_k / d f_j.
        
        # Let's build G once.
        # conv_k = sum_{i=0}^{N-1} f_i f_{k-i} where f_m = 0 if m not in [0, N-1].
        # For a fixed j, d conv_k / d f_j = f_{k-j} + f_{j-k} (if k-j in [0,N-1] or j-k in [0,N-1])
        # Wait, the sum is over i. The term is f_i f_{k-i}.
        # If i=j, term is f_j f_{k-j}. Deriv is f_{k-j}.
        # If k-i=j => i=k-j, term is f_{k-j} f_j. Deriv is f_j.
        # So d conv_k / d f_j = f_{k-j} + f_{j-k} (with f_m=0 outside).
        
        # We need grad_f = sum_k weights[k] * (f_{k-j} + f_{j-k})
        # This is a linear operation on f, weighted by weights.
        # Let's compute this efficiently.
        # grad_f_j = sum_k weights[k] f_{k-j} + sum_k weights[k] f_{j-k}
        # Let h_k = weights[k].
        # First term: (h * f)_j ? No, it's cross-correlation.
        # sum_k h_k f_{k-j}. Let m = k-j => k = m+j. sum_m h_{m+j} f_m.
        # This is the cross-correlation of h and f.
        
        # We can use FFT for this too if needed, but N=600 is small.
        # Let's just use numpy broadcasting or loops if fast enough.
        # 600*600 = 360,000 operations. Python loop might be slow (0.1s+).
        # Vectorized:
        # Create indices for k and j.
        # Actually, we can just compute the gradient by backpropagating through the FFT?
        # No, simpler:
        # grad_conv = weights (shape 2N-1)
        # We want grad_f (shape N).
        # grad_f_j = sum_{k} weights[k] * (f_{k-j} + f_{j-k})
        
        # Let's use the property that conv = f * f.
        # grad (w^T conv) = 2 * (w * f) ? 
        # If conv was circular, yes. For linear, it's a bit different.
        # Let's stick to the explicit formula with vectorization.
        
        # Precompute index matrix?
        # k ranges 0..2N-2. j ranges 0..N-1.
        # We need to sum over k where k-j is valid.
        
        # Efficient way:
        # term1_j = sum_k weights[k] f_{k-j}
        # This is the convolution of weights with f?
        # If we define weights_ext of length 2N-1, and f of length N.
        # It's a cross-correlation.
        
        # Let's just do it with a loop over j? 600 iterations, each O(N). 360k ops.
        # In numpy, a loop of 600 slices is fast.
        
        grad_f = np.zeros(N)
        for j in range(N):
            # k-j in [0, N-1] => k in [j, j+N-1]
            k_start = j
            k_end = j + N
            if k_end > 2*N - 1:
                k_end = 2*N - 1
            if k_start < 2*N - 1:
                # f_{k-j} for k in [k_start, k_end]
                # indices of f: 0..N-1
                # k-j ranges from 0 to k_end-1-j
                idx_f1 = np.arange(0, k_end - k_start)
                idx_w1 = np.arange(k_start, k_end)
                grad_f[j] += np.dot(weights[idx_w1], f[idx_f1])
                
            # j-k in [0, N-1] => k in [j-N+1, j]
            k_start2 = j - N + 1
            k_end2 = j + 1
            if k_start2 < 0:
                k_start2 = 0
            if k_end2 > 2*N - 1:
                k_end2 = 2*N - 1
            if k_start2 < k_end2:
                # f_{j-k} for k in [k_start2, k_end2]
                # j-k ranges from j-k_end2 to j-k_start2
                # j-k_end2 = j - (j+1) = -1 (if k_end2=j+1) -> wait.
                # If k=j, j-k=0. If k=j-N+1, j-k=N-1.
                # So f indices are 0..N-1.
                # k goes from k_start2 to k_end2-1.
                # j-k goes from j-k_start2 down to j-k_end2+1.
                idx_w2 = np.arange(k_start2, k_end2)
                # f index = j - k
                idx_f2 = j - idx_w2
                # Check validity
                valid = (idx_f2 >= 0) & (idx_f2 < N)
                if np.any(valid):
                    grad_f[j] += np.dot(weights[idx_w2[valid]], f[idx_f2[valid]])
        
        # The factor of 2 is included in the sum of both terms?
        # d conv_k / d f_j = f_{k-j} + f_{j-k}.
        # Yes, the loop covers both terms.
        
        # Normalize gradient?
        # We are minimizing obj.
        return obj, grad_f

    # SGD Loop
    # We will run SGD for a fixed number of iterations or time limit.
    # Learning rate needs to be small.
    # Objective is scale invariant? No, we normalized f to sum=1.
    # So f is on the simplex.
    # We project onto f>=0 and sum(f)=1 after each step.
    
    def project_simplex(v):
        # Project v onto simplex {x | x>=0, sum(x)=1}
        # This is equivalent to projecting onto l1 ball then scaling?
        # No, just standard simplex projection.
        # Algorithm: 
        # 1. Sort v in descending order.
        # 2. Find rho such that sum_{i<=rho} (v_i - t) = 1, where t = (sum_{i<=rho} v_i - 1)/rho.
        
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, N + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1.0)
        return np.maximum(v - theta, 0)

    # Run SGD
    best_sgd_score = np.inf
    best_sgd_f = None
    
    # Initialize with a few random starts
    n_restarts = 5
    iters_per_restart = 2000
    lr = 0.01 # Initial guess
    
    for r in range(n_restarts):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        f = random_init()
        
        # Adaptive learning rate?
        # Let's try a fixed small lr first.
        # The objective is roughly quadratic near the optimum?
        
        for it in range(iters_per_restart):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
                
            obj, grad = objective_and_grad(f, t=20.0)
            
            # Normalize gradient by sum(f)^2? 
            # Our objective is max(conv) / sum(f)^2.
            # We fixed sum(f)=1, so objective is max(conv).
            # But the gradient we computed is for max(conv) (smoothed).
            # So we can use it directly.
            
            f_new = f - lr * grad
            
            # Project onto simplex
            f_new = project_simplex(f_new)
            
            # Check improvement
            # To save time, don't compute exact score every iter.
            # But we need to track best.
            if it % 100 == 0 or it == iters_per_restart - 1:
                score = eval_shape(f_new)
                if score < best_sgd_score:
                    best_sgd_score = score
                    best_sgd_f = f_new.copy()
            
            f = f_new
            
            # Decay learning rate
            lr *= 0.995

    if best_sgd_f is not None and best_sgd_score < best_score:
        best_score = best_sgd_score
        best_f = best_sgd_f

    # --- Fallback: Parametric search if SGD fails or is slow ---
    # Re-use the efficient parametric search from the original code as a backup
    # to ensure we have a valid result.
    
    def make_cosine_plateau(H, W):
        f = np.zeros(N)
        mask = np.abs(x_centers) < W
        f[mask] = H * 0.5 * (1 + np.cos(np.pi * x_centers[mask] / W))
        return f

    def make_gaussian(H, sigma):
        f = H * np.exp(-0.5 * (x_centers / sigma) ** 2)
        return f

    # Quick grid search for Cosine Plateau
    best_cos_score = np.inf
    best_cos_f = None
    for W in np.linspace(0.05, 0.24, 50):
        f = make_cosine_plateau(1.0, W)
        s = eval_shape(f)
        if s < best_cos_score:
            best_cos_score = s
            best_cos_f = f

    if best_cos_f is not None and best_cos_score < best_score:
        best_score = best_cos_score
        best_f = best_cos_f

    # Quick grid search for Gaussian
    best_gauss_score = np.inf
    best_gauss_f = None
    for sigma in np.linspace(0.02, 0.2, 30):
        f = make_gaussian(1.0, sigma)
        s = eval_shape(f)
        if s < best_gauss_score:
            best_gauss_score = s
            best_gauss_f = f

    if best_gauss_f is not None and best_gauss_score < best_score:
        best_score = best_gauss_score
        best_f = best_gauss_f

    # Final check
    if best_f is None:
        best_f = np.ones(N)
        best_score = eval_shape(best_f)
        
    best_f = np.maximum(best_f, 0.0)
    
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
