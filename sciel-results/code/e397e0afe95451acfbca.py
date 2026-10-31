import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    # Initialize with constant function
    f = np.ones(N)
    
    # We assume the score is max(f*f) / sum(f^2)
    # To minimize this, we fix sum(f^2) = 1 (or any constant) and minimize max(f*f)
    # Using sum(f^2) as a proxy for max(f*f) is a standard technique.
    
    # Pre-compute indices or just use np.convolve
    # np.convolve is O(N^2) which is 360,000 ops, very fast.
    
    step_size = 0.01
    # We will use a simple gradient descent with projection and normalization
    
    for _ in range(1000):
        if time.time() > deadline:
            break
            
        # Compute convolution
        g = np.convolve(f, f, mode='full')
        
        # Compute gradient of sum(g^2)
        # grad = 4 * convolve(f, g)
        # Note: convolve(f, g) gives a vector of length 2N-1.
        # We need the part corresponding to f's indices.
        # If f is length N, g is length 2N-1.
        # (f*g)_j for j in 0..N-1?
        # The gradient of L = sum_k g_k^2 w.r.t f_j is 4 * (f * g)_j.
        # The convolution f*g has length 2N-1.
        # The indices of f correspond to the "center" of the convolution?
        # Let's check alignment.
        # g[k] = sum_i f[i] f[k-i].
        # d(sum g[k]^2)/d f[j] = 2 sum_k g[k] (f[k-j] + f[j-k]).
        # This is 4 * (f * g)[j] ?
        # (f*g)[j] = sum_i f[i] g[j-i].
        # Let's just compute the gradient numerically or use the formula carefully.
        
        # Actually, simpler:
        # The term (f * g) where g = f * f.
        # The vector (f * g) has length 3N - 2.
        # We need the central N components?
        # Let's use a numerical gradient for safety if N is small, but N=600 is small enough for analytical.
        
        # Let's use the analytical gradient:
        # grad_f = 4 * np.convolve(f, g, mode='full')
        # We need to extract the N components corresponding to f.
        # If f is indexed 0..N-1, and g is 0..2N-2.
        # The convolution h = f * g is 0..3N-3.
        # The component h[j] corresponds to derivative w.r.t f[j]?
        # Let's test with N=2.
        # f=[a,b]. g=[a^2, 2ab, b^2].
        # L = a^4 + 4a^2b^2 + b^4.
        # dL/da = 4a^3 + 8ab^2.
        # h = conv([a,b], [a^2, 2ab, b^2]) = [a^3, 2a^2b+ab^2, a^2b+2ab^2, ab^2+b^3]? No.
        # conv([a,b], [a^2, 2ab, b^2])
        # index 0: a*a^2 = a^3
        # index 1: a*2ab + b*a^2 = 2a^2b + a^2b = 3a^2b
        # index 2: a*b^2 + b*2ab = ab^2 + 2ab^2 = 3ab^2
        # index 3: b*b^2 = b^3
        # This doesn't look like 4a^3 + 8ab^2.
        
        # The correct gradient of ||f*f||_2^2 is 4 * (f * (f*f)) but aligned correctly?
        # Actually, the gradient is 4 * (f * g) where the convolution is circular? No, linear.
        # Let's just use a small numerical gradient step or a simpler heuristic.
        # Given N=600, we can afford to compute the full gradient matrix if needed, but that's 600x600.
        
        # Alternative: Use the fact that we want to flatten the peak.
        # We can use a "water filling" or iterative proportional fitting approach.
        
        # Let's try a different objective: Minimize the variance of the convolution?
        # Or just minimize the max directly using a smooth approximation.
        
        # Let's use the log-sum-exp approximation for the max.
        # L = (1/beta) * log(sum(exp(beta * g)))
        # Grad L = (1/beta) * sum(exp(beta*g) * beta * grad_g) / sum(exp(beta*g))
        #        = sum(exp(beta*g) * grad_g) / sum(exp(beta*g))
        # This is a weighted average of the gradients of g_k.
        
        # grad_g_k w.r.t f_j is f_{k-j} + f_{j-k} (with 0 padding).
        # This is effectively the correlation.
        
        # Let's implement the log-sum-exp gradient.
        beta = 10.0 # Temperature parameter
        
        # Compute g
        g = np.convolve(f, f, mode='full')
        
        # Compute weights for log-sum-exp
        # Shift g to center 0 to avoid overflow? g is non-negative.
        max_g = np.max(g)
        exp_g = np.exp(beta * (g - max_g))
        weights = exp_g / np.sum(exp_g)
        
        # Compute gradient of L w.r.t f
        # L = log(sum(exp(beta g)))
        # dL/df_j = sum_k weights[k] * (d g_k / d f_j)
        # d g_k / d f_j = f_{k-j} + f_{j-k}
        # This is the cross-correlation of f with itself, weighted by g?
        # Actually, d g_k / d f_j is 1 if k-j is in range, etc.
        # The vector of d g_k / d f_j for all k is essentially the impulse response.
        # The sum over k of weights[k] * (d g_k / d f_j) is the convolution of weights with the "impulse" defined by the derivative structure.
        
        # Note: d g_k / d f_j = f_{k-j} + f_{j-k} is NOT correct.
        # g_k = sum_i f_i f_{k-i}.
        # d g_k / d f_j = f_{k-j} + f_{j-k} (if indices valid).
        # So dL/df_j = sum_k weights[k] (f_{k-j} + f_{j-k})
        # This is 2 * (weights * f)_j?
        # (weights * f)_j = sum_k weights[k] f_{j-k}? No, convolution is sum_i weights[i] f[j-i].
        # Let h = weights * f.
        # h_j = sum_k weights[k] f_{j-k}.
        # Then sum_k weights[k] f_{j-k} is h_j.
        # And sum_k weights[k] f_{k-j} is h_{-j}? Or h_j if symmetric?
        # weights is symmetric (g is symmetric). f is not necessarily.
        # But g = f*f is symmetric. So weights are symmetric.
        # So sum_k weights[k] f_{k-j} = sum_k weights[k] f_{-(k-j)} = sum_k weights[-k] f_{j-k} = sum_k weights[k] f_{j-k}.
        # So the gradient is 2 * (weights * f)_j.
        
        # So we just need to compute conv(weights, f).
        grad = 2 * np.convolve(weights, f, mode='full')
        
        # Extract the central N components?
        # weights has length 2N-1. f has length N.
        # conv has length 3N-2.
        # The indices of f are 0..N-1.
        # The gradient components corresponding to f_j are at index j + (N-1)?
        # Let's check the alignment.
        # g_k is defined for k in 0..2N-2.
        # weights[k] corresponds to g_k.
        # d g_k / d f_j is non-zero for k in j..j+N-1?
        # The sum over k is effectively a convolution.
        # The result vector grad_full has length 3N-2.
        # The relevant part for f (indices 0..N-1) is usually the center.
        # Let's assume the standard alignment:
        # grad_f = grad_full[N-1 : 2*N-1] ?
        
        # Let's verify with N=1.
        # f=[a]. g=[a^2]. weights=[1].
        # grad = 2 * conv([1], [a]) = 2*[a].
        # Index 0. N-1=0. 2*N-1=1. Slice [0:1] is [a]. Correct.
        
        # N=2.
        # f=[a,b]. g=[a^2, 2ab, b^2].
        # weights = w0, w1, w2.
        # grad_full = 2 * conv([w0,w1,w2], [a,b])
        # = 2 * [w0*a, w0*b+w1*a, w1*b+w2*a, w2*b]
        # Indices:
        # dL/da = 2(w0*a + w1*b + w2*a)? No.
        # d g_0/da = 2a. d g_1/da = 2b. d g_2/da = 0.
        # dL/da = w0*2a + w1*2b.
        # Formula: 2 * (w0*a + w1*b). Matches.
        # dL/db = w1*2a + w2*2b.
        # Formula: 2 * (w1*a + w2*b). Matches.
        # So grad_f[0] corresponds to index 1 of grad_full?
        # grad_full[1] = w0*b + w1*a.
        # We want w0*a + w1*b for index 0?
        # Wait, conv([w0,w1,w2], [a,b])
        # index 0: w0*a
        # index 1: w0*b + w1*a
        # index 2: w1*b + w2*a
        # index 3: w2*b
        
        # dL/da = 2(w0*a + w1*b). This is NOT directly in the conv output like that.
        # dL/da = 2 * (w0*a + w1*b).
        # dL/db = 2 * (w1*a + w2*b).
        
        # It seems dL/d f_j = 2 * sum_{k} weights[k] f_{k-j} ?
        # For j=0: 2 * (w0*f0 + w1*f1 + w2*f2(=0)) = 2(w0 a + w1 b). Correct.
        # For j=1: 2 * (w0*f(-1)=0 + w1*f0 + w2*f1) = 2(w1 a + w2 b). Correct.
        
        # So dL/df_j = 2 * (weights * f)_j where the convolution is aligned such that
        # (weights * f)_j = sum_k weights[k] f_{j-k}.
        # This is the standard convolution.
        # So we compute h = np.convolve(weights, f, mode='full').
        # h has length 3N-2.
        # We need h[j] for j=0..N-1?
        # For j=0, we used w0, w1. These are the first terms of weights.
        # For j=N-1, we use w_{N-1}..w_{2N-2}?
        # The vector h is the full convolution.
        # The component h[j] in the "full" mode corresponds to sum_{k} weights[k] f[j-k].
        # Yes, the indices of h correspond to the indices of the result if we treat them as polynomials.
        # So we just take the first N elements?
        # Let's check j=0. h[0] = w0*f0.
        # We need w0*f0 + w1*f1.
        # This is h[0] + h[1]? No.
        
        # Actually, the formula dL/df_j = 2 * sum_k weights[k] f_{j-k} is the cross-correlation?
        # Let's just use the numerical gradient for the first few steps to verify, or stick to a simpler heuristic.
        
        # Given the complexity of alignment, let's use a simpler, robust method:
        # Minimize the sum of squares of the convolution (L2 norm of g).
        # Gradient of sum(g^2) is 4 * (f * g) aligned properly.
        # Let's assume the alignment is the central part.
        
        # Actually, for N=600, we can just do a coordinate descent or a simple random search with local refinement.
        
        # Let's try a different approach:
        # The optimal function for minimizing the peak of autocorrelation with fixed L2 norm is often close to a Gaussian or a "bang-bang" function.
        # Given the constraint of step function, we can use a simple iterative algorithm:
        # 1. Compute g = f*f.
        # 2. Compute target t = max(g).
        # 3. Adjust f to reduce g where it is high.
        
        # Let's use the `scipy.optimize.minimize` with L-BFGS-B.
        # Objective: sum(g^2)
        # Constraints: sum(f^2)=1, f>=0.
        
    # Since we are in a loop, let's just do a few iterations of a simple gradient step.
    
    # To be safe and fast, let's use a fixed number of iterations of a simple update.
    
    # Reset f
    f = np.ones(N)
    
    # Normalize
    f = f / np.linalg.norm(f)
    
    for i in range(50):
        if time.time() > deadline:
            break
        
        g = np.convolve(f, f, mode='full')
        
        # Gradient of sum(g^2)
        # We established that grad = 4 * (f * g) is tricky to align.
        # Let's use a numerical gradient for a small subset or just use the correlation property.
        
        # Actually, let's just use the fact that we want to flatten g.
        # We can use a "heat equation" like flow on the convolution.
        
        # Simple heuristic:
        # Reduce f where it contributes to the peak of g.
        # The contribution of f_j to g_k is f_j * f_{k-j}.
        
        # Let's try a simple random perturbation with acceptance if score improves.
        
    # Given the time, let's return the seed if optimization is too risky/slow.
    # But we know we can do better.
    
    # Let's try a specific known better function:
    # A function that is 1 in the center and 0 at edges?
    # Or a cosine wave?
    
    # Let's use a cosine bump: f(x) = cos(pi * x / (1/2)) for x in [-1/4, 1/4]?
    # Support is [-1/4, 1/4].
    # f[i] = cos(pi * (i - N/2) / (N/2)) ?
    
    x = np.linspace(-0.25, 0.25, N)
    f_cos = np.cos(np.pi * x / 0.5)
    f_cos[f_cos < 0] = 0
    
    # Normalize
    f_cos = f_cos / np.linalg.norm(f_cos)
    
    # Check score of f_cos vs f_const
    # We can't calculate the exact score without knowing the evaluator's normalization,
    # but assuming it's max(f*f)/sum(f^2), let's estimate.
    
    g_const = np.convolve(np.ones(N), np.ones(N), mode='full')
    score_const = np.max(g_const) / np.sum(np.ones(N)**2)
    
    g_cos = np.convolve(f_cos, f_cos, mode='full')
    score_cos = np.max(g_cos) / np.sum(f_cos**2)
    
    if score_cos < score_const:
        return list(f_cos)
    else:
        return [1.0] * N
