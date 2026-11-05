"""Combined program for third autocorrelation inequality optimization."""
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

def _compute_value(h):
    """Compute the evaluation metric."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return float('inf')
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def _square_wave_mode(k, phase):
    """
    Generate a square wave mode of frequency k (cycles per N samples).
    k=0 is constant (all ones).
    phase shifts the wave.
    """
    if k == 0:
        return np.ones(N)
    t = np.arange(N)
    # Square wave: sign of sin(2*pi*k*t/N + phase)
    # We want it to be a valid step function mode.
    # A pure square wave has 50% duty cycle.
    # Let's use a standard square wave definition.
    angle = 2.0 * np.pi * k * t / N + phase
    # sign(sin(angle)) gives +1/-1
    # But we want it to be orthogonal to the constant vector (k=0).
    # For integer k, the sum of a square wave over one period is 0.
    # However, discrete sampling might cause slight imbalance if N is not a multiple of the period.
    # N=400. If k divides 400, it's perfectly balanced.
    # Let's just use sign(sin).
    sw = np.sign(np.sin(angle))
    # Ensure it's strictly +1 or -1. If sin is 0, sign is 0. Let's avoid 0.
    # Actually, for optimization, we want a smooth-ish parameterization or just discrete.
    # The strategy says "linear combination of a constant vector and a single high-frequency square wave mode".
    # So h = a * ones + b * sw.
    # We optimize a and b (or ratio r = b/a).
    # Since scale is invariant, we can fix a=1 and optimize b, or fix sum=1 and optimize the shape.
    # Let's fix sum(h) = 1.
    # h = a * ones + b * sw.
    # sum(h) = a*N + b*sum(sw).
    # If sw is balanced (sum=0), then sum(h) = a*N => a = 1/N.
    # Then h = (1/N)*ones + b*sw.
    # Value depends on b.
    
    # To handle the "high-frequency" part, we should try various k.
    return sw

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    # Phase 1: Structured Search (Square Wave Modes)
    # The hypothesis is that mixing a DC component with a high-frequency square wave
    # might reduce the peak of the autocorrelation relative to the sum squared.
    # Although the previous analysis suggested 2.0 is the global minimum, 
    # numerical quirks or specific discrete structures might allow slight improvements
    # or the previous proof might have missed a nuance in the "max over signed entries" 
    # allowing negative peaks to dominate? No, max takes the largest algebraic value.
    # If we can make the peak at lag 0 (sum h^2) smaller while keeping sum(h) large?
    # No, sum h^2 >= sum(h)^2 / N is a hard bound.
    # So value >= 2.0.
    
    # Wait, is it possible that np.max(conv) is NOT at lag 0?
    # For positive h, conv[0] = sum h^2 is the max.
    # If h has negative parts, conv[0] is still sum h^2.
    # Is it possible that some other lag k has a value LARGER than conv[0]?
    # conv[k] = sum h[i]h[i+k].
    # By Cauchy-Schwarz, |conv[k]| <= sqrt(sum h^2 * sum h^2) = sum h^2.
    # So conv[k] <= conv[0] for all k.
    # Therefore max(conv) = conv[0] = sum(h_i^2).
    # So the objective is always abs(2 * N * sum(h^2) / sum(h)^2).
    # Let S = sum(h), Q = sum(h^2).
    # Value = 2 * N * Q / S^2.
    # We know Q >= S^2 / N.
    # So Value >= 2 * N * (S^2/N) / S^2 = 2.
    # Equality holds if and only if h is constant.
    # So the minimum is exactly 2.0.
    
    # Why did the inspiration get 1.2797?
    # Score = seed / value. Seed value = 2.0.
    # If score is 1.2797, then value = 2.0 / 1.2797 = 1.5628.
    # This is < 2.0.
    # How is this possible?
    # Did I misinterpret the formula?
    # "value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)"
    # Maybe np.max is not conv[0]?
    # Let's check the Cauchy-Schwarz bound again.
    # conv[k] = sum_{i} h_i h_{i+k}.
    # This is the inner product of h and a shifted version of h.
    # Let u = h, v = shifted h.
    # <u, v> <= ||u|| ||v||.
    # ||u|| = sqrt(Q). ||v|| = sqrt(Q) (since it's just a permutation/shift, sum of squares is same).
    # So conv[k] <= Q.
    # So max(conv) <= Q.
    # Wait, the inequality is conv[k] <= Q.
    # So max(conv) <= Q.
    # But we want to MINIMIZE max(conv).
    # We established max(conv) = Q is the UPPER bound on any conv[k].
    # But is max(conv) always Q?
    # conv[0] = Q.
    # So max(conv) >= conv[0] = Q.
    # And conv[k] <= Q.
    # So max(conv) = Q.
    # So the value is ALWAYS 2 * N * Q / S^2.
    # And min is 2.
    
    # So how did the other solution get 1.56?
    # Maybe the evaluator code is different?
    # "The evaluator computes exactly this value... value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)"
    # Maybe I am misinterpreting "np.convolve(h, h)"?
    # In NumPy, np.convolve(h, h) is full mode by default.
    # Length 2N-1.
    # Maybe the "max" is taken over a specific range? No, "over its signed entries".
    
    # Is it possible that sum(h) is allowed to be complex? No, "real".
    # Is it possible that the heights are not the vector h?
    # "Choose real... step heights h[0]...h[N-1]"
    
    # Let's look at the inspiration 2 score: 1.2797.
    # If the score is seed/value, and seed is 2.0, value is 1.56.
    # This implies max(conv) / sum(h)^2 < 1/N.
    # Which implies Q < S^2 / N.
    # This violates Cauchy-Schwarz.
    # Unless... h is not a vector of 400 numbers?
    # Or maybe the evaluator uses a different N? No, N=400.
    
    # Let's re-read the inspiration 1 analysis.
    # "So 2.0 is the global minimum?"
    # "Why would the task exist if the seed is optimal?"
    # Inspiration 1 scored 1.1250.
    # Value = 2.0 / 1.125 = 1.777.
    # Still < 2.0.
    
    # There must be a flaw in my Cauchy-Schwarz application or the problem statement interpretation.
    # Let's test N=2 again.
    # h = [1, -1]. S=0. Invalid.
    # h = [1, 0]. S=1. Q=1. Conv=[1, 0, 0]. Max=1. Val = 2*2*1/1 = 4.
    # h = [0.5, 0.5]. S=1. Q=0.5. Conv=[0.25, 0.5, 0.25]. Max=0.5. Val = 2*2*0.5/1 = 2.
    # h = [1, 1]. S=2. Q=2. Conv=[1, 2, 1]. Max=2. Val = 2*2*2/4 = 2.
    
    # What if h = [1, 2]? S=3. Q=5. Conv=[1, 4, 4]. Max=4. Val = 2*2*4/9 = 16/9 = 1.77.
    # 1.77 < 2.0!
    # My proof was WRONG.
    # Where?
    # "conv[k] <= Q"
    # For h=[1,2], Q=5.
    # Conv[0] = 1^2 + 2^2 = 5.
    # Conv[1] = 1*2 + 2*1 = 4.
    # Conv[2] = 2^2 = 4.
    # Max is 5.
    # Val = 2*2*5/9 = 20/9 = 2.22.
    # Wait, my manual calculation for h=[1,2] above:
    # Conv = np.convolve([1,2], [1,2]) = [1, 4, 4].
    # Wait, np.convolve([1,2], [1,2]):
    # 1*1 = 1
    # 1*2 + 2*1 = 4
    # 2*2 = 4
    # Max is 4? No, Conv[0] is 1?
    # np.convolve is cross-correlation? No, convolution.
    # For 1D, conv[a, b][k] = sum a[i] b[k-i].
    # For h=[1,2]:
    # k=0: h[0]h[0] = 1.
    # k=1: h[0]h[1] + h[1]h[0] = 1*2 + 2*1 = 4.
    # k=2: h[1]h[1] = 4.
    # So conv = [1, 4, 4]. Max = 4.
    # My previous "Conv[0] = sum h^2" is WRONG for np.convolve.
    # np.convolve(h, h)[0] is h[0]*h[0].
    # np.convolve(h, h)[N-1] is sum h[i]h[N-1-i].
    # The peak is at N-1, which is the "auto-correlation at lag 0" in signal processing terms?
    # No, in signal processing, auto-correlation R[k] = sum x[n]x[n+k].
    # np.convolve(h, h) is NOT auto-correlation. It is convolution.
    # Convolution of a signal with itself is NOT the same as auto-correlation.
    # Auto-correlation is convolution with the REVERSED signal.
    # np.convolve(h, h[::-1]) would be auto-correlation.
    # But the task says np.convolve(h, h).
    # So we are minimizing the peak of the CONVOLUTION of h with itself.
    
    # Let's re-evaluate the bound.
    # We want to minimize max_k (sum_i h_i h_{k-i}) / (sum h_i)^2.
    # For h=[1,2], max conv = 4. S=3. S^2=9.
    # Val = 2*2*4/9 = 16/9 = 1.777.
    # This is < 2.
    # So it IS possible to go below 2.
    # My previous proof assumed max(conv) >= sum(h^2).
    # But max(conv) is NOT necessarily sum(h^2).
    # sum(h^2) appears at lag N-1?
    # conv[N-1] = sum_{i=0}^{N-1} h_i h_{N-1-i}.
    # If h is symmetric, this is sum h_i^2.
    # But if h is NOT symmetric, conv[N-1] is sum h_i h_{N-1-i}.
    # And conv[0] = h[0]^2.
    # So we can potentially make conv[N-1] small by making h "anti-symmetric" with respect to the center?
    # If h_i = - h_{N-1-i}, then conv[N-1] = sum h_i (-h_i) = -sum h_i^2.
    # Then max(conv) might be determined by other lags.
    # And S = sum h_i. If h is anti-symmetric, S=0.
    # So we need a slight asymmetry to keep S != 0.
    
    # This confirms the direction: Use asymmetry to reduce the central peak (lag N-1)
    # which is usually the largest for positive symmetric signals.
    
    # Strategy:
    # 1. Start with a base vector that has a large sum but small central convolution peak.
    # 2. The square wave idea:
    #    h = a * ones + b * sw.
    #    If sw is anti-symmetric (sw[i] = -sw[N-1-i]), then the cross term in conv[N-1] vanishes?
    #    conv(h,h)[N-1] = sum (a + b sw_i)(a + b sw_{N-1-i})
    #    = sum (a^2 + ab sw_i + ab sw_{N-1-i} + b^2 sw_i sw_{N-1-i})
    #    = N a^2 + ab sum(sw_i + sw_{N-1-i}) + b^2 sum(sw_i sw_{N-1-i})
    #    If sw is anti-symmetric, sum(sw_i + sw_{N-1-i}) = 0.
    #    sum(sw_i sw_{N-1-i}) = sum(sw_i (-sw_i)) = -sum sw_i^2 = -N (since sw is +/- 1).
    #    So conv[N-1] = N a^2 - N b^2 = N(a^2 - b^2).
    #    Sum(h) = N a + b sum(sw). If sw is balanced (sum 0), S = N a.
    #    Value = 2 N max(conv) / (N a)^2 = 2 max(conv) / (N a^2).
    #    If we can make max(conv) small...
    #    conv[N-1] = N(a^2 - b^2).
    #    If b is close to a, conv[N-1] is close to 0.
    #    What about other lags?
    #    For a square wave, the convolution has a specific shape.
    #    High frequency square waves have convolution peaks that are lower relative to DC?
    
    # Let's implement this specific search.
    
    # Optimization over a and b (or ratio r=b/a).
    # Fix a=1. Optimize b.
    # Also try different frequencies k for the square wave.
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    # Try a range of frequencies for the square wave
    # k=1 is lowest freq (period 2N? No, period N/k).
    # High frequency means large k.
    # Let's try k from 1 to N/2.
    
    for k in [1, 2, 4, 8, 16, 32, 64, 100, 128, 200]:
        if time.time() > deadline:
            break
        # Generate square wave
        # We want it anti-symmetric: sw[i] = -sw[N-1-i].
        # A standard square wave sin(2*pi*k*t/N) is not necessarily anti-symmetric.
        # Let's construct an anti-symmetric square wave.
        # Define sw[i] for i < N/2, and sw[N-1-i] = -sw[i].
        # We can use a random sign pattern for the first half, mirrored with sign flip.
        # Or use a specific high-frequency pattern.
        
        # Let's use the "alternating" pattern which is k=N/2 (if N even).
        # h_i = (-1)^i.
        # This is anti-symmetric for N even?
        # h[0]=1, h[N-1]=-1.
        # h[i] = (-1)^i. h[N-1-i] = (-1)^{N-1-i} = (-1)^{N-1} (-1)^{-i} = (-1) (-1)^i = -h[i].
        # Yes, alternating is anti-symmetric.
        
        # Let's optimize h = a * ones + b * (-1)^i.
        # This is a 2-parameter family.
        
        def obj_r(r):
            # h = ones + r * alt
            h = np.ones(N) + r * np.array([(-1)**i for i in range(N)], dtype=float)
            return _compute_value(h)
        
        # Minimize obj_r over r
        # r=0 gives seed.
        # Try to find r that minimizes.
        try:
            res = minimize_scalar(obj_r, bounds=(-2.0, 2.0), method='bounded', options={'xatol': 1e-6})
            if res.success:
                h_opt = np.ones(N) + res.x * np.array([(-1)**i for i in range(N)], dtype=float)
                val_opt = _compute_value(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt
        except Exception:
            pass
        
        # Also try other anti-symmetric patterns?
        # Maybe a random anti-symmetric pattern?
        # Let's try a few random anti-symmetric patterns.
        for _ in range(5):
            if time.time() > deadline:
                break
            # Generate random signs for first half
            half = np.random.choice([-1, 1], size=N//2)
            # Mirror with sign flip
            sw = np.zeros(N)
            sw[:N//2] = half
            sw[N//2:] = -half[::-1]
            
            # Optimize h = ones + r * sw
            def obj_r2(r):
                h = np.ones(N) + r * sw
                return _compute_value(h)
            
            try:
                res = minimize_scalar(obj_r2, bounds=(-2.0, 2.0), method='bounded', options={'xatol': 1e-6})
                if res.success:
                    h_opt = np.ones(N) + res.x * sw
                    val_opt = _compute_value(h_opt)
                    if val_opt < best_val:
                        best_val = val_opt
                        best_h = h_opt
            except Exception:
                pass

    # Phase 2: Local refinement using L-BFGS on the best found structure
    if time.time() < deadline - 10:
        # Use the best_h found so far as initial guess
        h0 = best_h.copy()
        
        # We need to be careful with the objective function for L-BFGS.
        # The objective is non-smooth due to max().
        # But L-BFGS can often handle it.
        
        def objective(h):
            return _compute_value(h)
        
        try:
            res = minimize(objective, h0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-9})
            if res.success:
                val_res = _compute_value(res.x)
                if val_res < best_val:
                    best_val = val_res
                    best_h = res.x
        except Exception:
            pass

    return best_h

# EVOLVE-BLOCK-END
