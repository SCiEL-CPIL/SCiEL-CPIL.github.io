"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    # Deadline guard: must finish well before 150s
    start = time.time()
    deadline = start + 130.0

    # Try to find a good pattern by optimizing the max overlap.
    # The overlap at shift k is: sum_{i=0}^{N-k-1} h[i] * (1 - h[i+k])
    # We want to minimize the maximum over all k of this quantity.
    
    # Start with a good initial guess: alternating high/low pattern
    # If we alternate between a and b where a+b=1, then:
    # At shift 0: sum h[i]*(1-h[i]) = N/2 * a*b (if a=b=0.5, this is N/4)
    # At shift 1: sum h[i]*(1-h[i+1]) = sum h[i]*h[i] (since 1-h[i+1] = h[i] for alternating)
    #   = N/2 * (a^2 + b^2)
    # At shift 2: same as shift 0
    # So the max overlap depends on the ratio a:b.
    
    # Actually let's think more carefully. With h alternating between a and b:
    # h = [a, b, a, b, ...] with a+b=1, sum = N/2 (since N is even)
    # Shift 0: sum h[i]*(1-h[i]) = N/2 * a*(1-a) + N/2 * b*(1-b) = N/2*(a*b + b*a) = N*a*b
    #   = N*(1/4) if a=b=0.5
    # Shift 1: sum_{i=0}^{N-2} h[i]*(1-h[i+1])
    #   For even i: h[i]=a, h[i+1]=b, so term = a*(1-b) = a*a = a^2
    #   For odd i: h[i]=b, h[i+1]=a, so term = b*(1-a) = b*b = b^2
    #   Sum = (N/2)*a^2 + (N/2)*b^2 = N/2*(a^2+b^2) = N/2*(1-2ab)
    # Shift 2: same as shift 0 = N*a*b
    # etc.
    # 
    # So max overlap = max(N*a*b, N/2*(1-2ab))
    # Let x = a*b. Then max = max(N*x, N/2*(1-2x))
    # Minimize over x in [0, 1/4] (since a+b=1, max ab = 1/4)
    # N*x = N/2*(1-2x) => 2x = 1-2x => x = 1/4, which gives max = N/4
    # At x=1/4: both terms = N/4
    # At x=0: max = N/2
    # So the minimum is at x=1/4, giving max overlap = N/4 = 25 for N=100.
    # Wait, that's the same as uniform! Let me recheck.
    
    # Hmm, for uniform h=0.5:
    # Shift 0: sum 0.5*0.5 = N/4 = 25
    # Shift k: sum 0.5*0.5 = N/4 = 25
    # So all shifts give 25.
    
    # For alternating a, b:
    # Shift 0: N*a*b
    # Shift 1: N/2*(a^2+b^2)
    # These are equal when 2ab = a^2+b^2, i.e., (a-b)^2=0, i.e., a=b=0.5
    # So uniform is actually optimal among alternating patterns!
    
    # Let's try a different approach: use scipy optimization to find the best h.
    # The objective is min max_k sum_i h[i]*(1-h[i+k])
    # subject to 0<=h<=1, sum(h)=N/2
    
    try:
        from scipy.optimize import minimize
    except ImportError:
        # Fallback: return a reasonable pattern
        h = np.full(N, 0.5)
        return h
    
    def max_overlap(h):
        """Compute the maximum overlap over all shifts."""
        h = np.asarray(h)
        max_val = 0.0
        for k in range(N):
            if k == 0:
                overlap = np.sum(h * (1 - h))
            else:
                overlap = np.sum(h[:N-k] * (1 - h[k:]))
            if overlap > max_val:
                max_val = overlap
        return max_val
    
    # Use SLSQP to minimize the max overlap
    # This is a min-max problem, which is harder. Let's try a different approach.
    # We can use a smooth approximation or just try to minimize the max directly.
    
    # Alternative: try various structured patterns and pick the best.
    best_h = np.full(N, 0.5)
    best_score = max_overlap(best_h)
    
    # Try sinusoidal patterns
    for freq in range(1, 20):
        for phase in np.linspace(0, 2*np.pi, 10):
            h = 0.5 + 0.3 * np.sin(2*np.pi*freq/N * np.arange(N) + phase)
            h = np.clip(h, 0, 1)
            # Adjust to satisfy sum constraint
            diff = N/2 - np.sum(h)
            h += diff / N
            h = np.clip(h, 0, 1)
            # Re-adjust
            for _ in range(20):
                diff = N/2 - np.sum(h)
                if abs(diff) < 1e-10:
                    break
                h += diff / N
                h = np.clip(h, 0, 1)
            
            score = max_overlap(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Try using scipy's SLSQP on a transformed problem
    if time.time() < deadline:
        # Use a barrier/augmented Lagrangian approach for min-max
        # Or just use minimize on the max with smoothing
        from scipy.optimize import minimize
        
        def objective(h):
            # Smooth approximation of max using log-sum-exp
            h = np.clip(h, 0, 1)
            max_val = 0.0
            overflows = []
            for k in range(N):
                if k == 0:
                    overlap = np.sum(h * (1 - h))
                else:
                    overlap = np.sum(h[:N-k] * (1 - h[k:]))
                overflows.append(overlap)
            overflows = np.array(overflows)
            # Use a high-temperature softmax-like approach
            T = 1.0
            return np.max(overflows)
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2}]
        bounds = [(0, 1)] * N
        
        h0 = best_h.copy()
        result = minimize(objective, h0, method='SLSQP', bounds=bounds,
                         constraints=constraints, options={'maxiter': 50, 'ftol': 1e-6})
        
        if result.success:
            h_opt = result.x
            h_opt = np.clip(h_opt, 0, 1)
            # Ensure sum constraint
            for _ in range(20):
                diff = N/2 - np.sum(h_opt)
                if abs(diff) < 1e-10:
                    break
                h_opt += diff / N
                h_opt = np.clip(h_opt, 0, 1)
            
            score = max_overlap(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt
    
    # Final validation
    h_final = np.clip(best_h, 0, 1)
    # Ensure exact sum
    for _ in range(50):
        diff = N/2 - np.sum(h_final)
        if abs(diff) < 1e-12:
            break
        h_final += diff / N
        h_final = np.clip(h_final, 0, 1)
    
    return h_final
# EVOLVE-BLOCK-END
