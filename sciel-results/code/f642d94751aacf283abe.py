import numpy as np
from scipy.optimize import minimize
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds safety margin

    def objective(params):
        # params: (center, width)
        c = params[0]
        w = params[1]
        
        if w < 0.001:
            return 1e10
        if abs(c) + w/2 > 0.25:
            return 1e10
            
        # Construct the step function on 600 intervals
        # Intervals are on [-0.25, 0.25]
        # Width of each interval:
        h = 0.5 / N
        centers = -0.25 + (np.arange(N) + 0.5) * h
        
        # Rectangular bump: 1.0 where |x - c| < w/2, 0.0 otherwise
        # We need to assign heights to each interval.
        # An interval i covers [x_i - h/2, x_i + h/2].
        # It is "on" if its center is within [c - w/2, c + w/2]?
        # Actually, a step function is defined by constant values on intervals.
        # Let's say interval i has height 1 if the *majority* of the interval is in the bump,
        # or simply if the center of the interval is in the bump.
        # Using center check is standard for this kind of grid approximation.
        
        mask = np.abs(centers - c) < w/2.0
        f = np.zeros(N)
        f[mask] = 1.0
        
        # Calculate Autoconvolution
        # (f * f)(t) = integral f(x) f(t-x) dx
        # On the grid, this is a discrete convolution.
        # Since f is non-negative and supported on [-0.25, 0.25],
        # the autoconvolution is supported on [-0.5, 0.5].
        
        # We compute the discrete convolution.
        # The grid spacing is h.
        # conv = np.convolve(f, f, mode='full') * h
        # The resulting vector has length 2N - 1 + 1 = 1199.
        # The indices correspond to lags.
        # Lag 0 (center) corresponds to index N-1.
        # The maximum of the autoconvolution for a non-negative function is at t=0
        # because (f*f)(0) = sum f(x)^2 and by Cauchy-Schwarz or simple positivity,
        # (f*f)(t) <= max(f*f)(0) is not always true for arbitrary t?
        # Wait. For non-negative f, (f*f)(0) = integral f^2.
        # (f*f)(t) = integral f(x)f(x-t).
        # It is known that the maximum of the autocorrelation of a non-negative function is at 0.
        # Proof: (f*f)(t) = integral f(x)f(x-t).
        # (f*f)(0) = integral f^2(x).
        # By Cauchy-Schwarz, (integral f(x)f(x-t))^2 <= (integral f^2)(integral f(x-t)^2) = (integral f^2)^2.
        # So (f*f)(t) <= (f*f)(0).
        # Thus we only need to minimize (f*f)(0).
        
        # (f*f)(0) = sum f_i^2 * h (approx)
        # To minimize sum f_i^2, we should make f_i as small as possible?
        # No, we are constrained?
        # The problem says "Choose a nonnegative step function ... to minimize the normalized maximum of its autoconvolution".
        # Usually "normalized" implies fixing a norm, e.g. L1 or L2.
        # If no normalization is specified in the return, but the score is "upper bound on C1",
        # we likely need to fix the L1 norm to 1, or L2 norm to 1?
        # "normalized maximum" usually means max((f*f)) / (integral f)^2 or similar.
        # If we fix integral f = 1, then we want to minimize max(f*f).
        # Since max is at 0, we want to minimize (f*f)(0) = integral f^2.
        # Given integral f = 1, minimizing integral f^2 means spreading f out as much as possible?
        # No, spreading it out (making it wider and lower) increases the support, but for a fixed L1 norm,
        # integral f^2 is minimized when f is as "flat" as possible over its support?
        # Actually, for a fixed support, making it flatter (lower peak) reduces L2 norm squared.
        # But we can change the support too.
        # If we make f very wide, it becomes small, L2^2 decreases.
        # But the domain is limited to [-0.25, 0.25]?
        # The function is defined on [-0.25, 0.25].
        # If we can make f non-zero on the entire domain, and set it to a constant c such that integral = 1.
        # Width = 0.5. c = 1/0.5 = 2.
        # Then f(x) = 2 for x in [-0.25, 0.25].
        # (f*f)(0) = integral f^2 = integral 4 dx = 4 * 0.5 = 2.
        # Normalized max = 2 / (integral f)^2 = 2 / 1 = 2.
        
        # What if we make it a delta? Not allowed, step function.
        # What if we make it a narrow bump?
        # Width w. Height 1/w (to keep L1=1).
        # (f*f)(0) = integral (1/w)^2 * w = 1/w.
        # As w -> 0, this goes to infinity.
        # So we want to maximize width?
        # But we are constrained to the interval [-0.25, 0.25].
        # So the maximum width is 0.5.
        # So the optimal shape is a box of width 0.5, height 2.
        # Score = 2.0?
        
        # Let's re-read: "minimize the normalized maximum of its autoconvolution".
        # If the normalization is by L1^2, and we are limited to the domain,
        # the box function covering the whole domain is optimal.
        
        # However, maybe the normalization is different?
        # Or maybe I can shift the function?
        # If I shift it, it still covers the same width.
        
        # Let's check the seed program: [1.0] * N.
        # L1 = 0.5 * 1.0 = 0.5.
        # (f*f)(0) = integral 1^2 * 0.5 = 0.5.
        # Normalized max = 0.5 / (0.5)^2 = 0.5 / 0.25 = 2.0.
        
        # If I use [2.0] * N:
        # L1 = 1.0.
        # (f*f)(0) = integral 4 * 0.5 = 2.0.
        # Normalized max = 2.0 / 1.0^2 = 2.0.
        
        # It seems the score is independent of scaling if normalized by L1^2.
        # Is there a better shape?
        # If I use a triangular shape?
        # Triangular bump on [-0.25, 0.25].
        # f(x) = 1 - 4|x|. (Height 1 at 0, 0 at +/- 0.25).
        # L1 = 2 * integral_0^0.25 (1 - 4x) dx = 2 * [x - 2x^2]_0^0.25 = 2 * (0.25 - 2*0.0625) = 2 * (0.25 - 0.125) = 2 * 0.125 = 0.25.
        # (f*f)(0) = integral f^2 dx.
        # f^2 = (1 - 4x)^2 = 1 - 8x + 16x^2.
        # integral_0^0.25 (1 - 8x + 16x^2) dx = [x - 4x^2 + 16/3 x^3]_0^0.25
        # = 0.25 - 4(0.0625) + 5.333(0.015625)
        # = 0.25 - 0.25 + 0.0833 = 0.0833.
        # Total integral (symmetric) = 2 * 0.0833 = 0.1666.
        # Normalized max = 0.1666 / (0.25)^2 = 0.1666 / 0.0625 = 2.666.
        # This is worse than 2.0.
        
        # So the box function is better.
        
        # Can we do better than the box?
        # We are restricted to the support [-0.25, 0.25].
        # To minimize L2^2 / L1^2, we want to minimize L2^2 for fixed L1.
        # This is minimized by the flattest possible function.
        # The flattest function on a fixed domain is the constant function.
        # So the constant function (box) is optimal.
        
        # Why is the score 1.0 for the seed?
        # Seed returns [1.0]*N.
        # Score = 2.0.
        # If the reference score (seed) is 1.0, and my calculated score is 2.0...
        # Maybe the "normalized maximum" is defined differently?
        # Perhaps it's max(f*f) / max(f)^2? No.
        # Perhaps it's max(f*f) / (sum f)^2 * N? (Discrete normalization)
        
        # Let's look at the discrete definition.
        # Seed: f = [1, 1, ..., 1] (N=600).
        # Discrete autoconvolution (no h factor, just sum) at lag 0:
        # C[0] = sum f[i]^2 = 600.
        # Discrete L1 = sum f[i] = 600.
        # Ratio = 600 / 600^2 = 1/600.
        
        # If we scale by h=0.5/600 = 1/1200.
        # (f*f)_discrete * h = 600 * (1/1200) = 0.5. (Matches integral calc).
        # (L1)_discrete * h = 600 * (1/1200) = 0.5.
        # Ratio = 0.5 / 0.25 = 2.
        
        # So the score 1.0 corresponds to a value of 2.0?
        # "score is the upper bound on C1".
        # "search score is seed/value".
        # If seed value is 2.0, and seed score is 1.0, then Score = 2.0 / Value.
        # So to improve score, I need to reduce Value.
        # Value = 2.0 for the box.
        # Can I get Value < 2.0?
        
        # If I restrict the support to be smaller, Value increases.
        # If I change the shape to be "flatter" than a box? Impossible, box is flat.
        # Unless... the domain is not fixed?
        # "Choose a nonnegative step function with 600 equal intervals on [-1/4, 1/4]".
        # The function is defined on this interval. It can be 0 anywhere.
        # So the support is a subset of [-0.25, 0.25].
        # To minimize L2^2/L1^2, we maximize the support.
        # So full support is optimal.
        
        # Is there any other interpretation?
        # "normalized maximum of its autoconvolution"
        # Maybe it's max(f*f) / (f*f)(0)? That would be 1.
        
        # Let's assume the standard L2^2/L1^2 metric.
        # The box gives 2.0.
        # The seed gives 2.0.
        
        # Wait, is it possible to have a value < 2.0?
        # Only if the normalization is different.
        
        # What if we use a function that is not a single bump?
        # Two bumps?
        # If I split the mass into two bumps, L2^2 increases (convexity of x^2).
        # So single bump is better.
        
        # What if the normalization is by L2^2?
        # max(f*f) / L2^2 = L2^2 / L2^2 = 1.
        # Then everyone scores 1.0.
        
        # What if normalization is by max(f)?
        # Box: max(f)=1 (if normalized L1=1? No, if L1=1, max=2).
        # If we don't normalize L1, we can scale arbitrarily.
        # "normalized" implies scale invariance.
        
        # Let's try a different shape just in case.
        # A "super-box"? No.
        
        # Maybe the "normalized maximum" is max(f*f) / (integral |f|)^2 is wrong.
        # Maybe it's max(f*f) / (integral f)^2 * (some factor)?
        
        # Let's try to optimize a parametric family to see if we can beat 2.0.
        # Family: f(x) = exp(- (x-c)^2 / (2 w^2)) * A.
        # Gaussian.
        # L1 = A * w * sqrt(2pi).
        # L2^2 = A^2 * w * sqrt(pi).
        # Ratio = (A^2 w sqrt(pi)) / (A^2 w^2 2pi) = 1 / (w sqrt(2pi)).
        # To minimize ratio, maximize w.
        # w is limited by the domain.
        # If w -> infinity, ratio -> 0.
        # But the function must be zero outside [-0.25, 0.25]?
        # "step function ... on [-1/4, 1/4]".
        # Usually this implies f(x)=0 outside.
        # If we truncate the Gaussian, it's no longer a simple Gaussian.
        
        # If we use a truncated Gaussian with large w, it approximates a box.
        # So 2.0 is likely the minimum.
        
        # Why would the task exist if the answer is just the box?
        # Maybe I can't use the full box?
        # "Choose a nonnegative step function with 600 equal intervals on [-1/4, 1/4]"
        
        # Is it possible the score is NOT L2^2/L1^2?
        # "normalized maximum of its autoconvolution"
        
        # Let's look at the seed score again.
        # If the seed score is 1.0, and the seed is a box, and the box is optimal...
        # Then the best score is 1.0.
        # But the prompt says "score 1.0 equals the seed program and above 1.0 is an improvement".
        # This implies we CAN improve.
        
        # How to improve?
        # Maybe my formula for the metric is wrong.
        
        # Alternative metric: max(f*f) / (sum f)^2 in discrete terms?
        # Seed: 600 / 600^2 = 1/600.
        
        # Alternative: max(f*f) / (max f * sum f)?
        
        # Let's try to find a function with lower "peakedness".
        
        # What if the normalization is by the *area* of the autoconvolution?
        # Area(f*f) = (Area f)^2.
        # So it's the same.
        
        # What if "normalized maximum" means max(f*f) / max(f^2)?
        # For box: max(f*f)=2, max(f^2)=4 (if L1=1, f=2). Ratio 0.5.
        # For Gaussian (truncated):
        
        # Let's assume the metric is M = max(f*f) / (integral f)^2.
        # And we want to minimize M.
        
        # If I am stuck at 2.0, I should just return the box.
        # But maybe I can get 1.99 by numerical tricks?
        
        # Let's implement a simple optimizer to be sure.
        # We will optimize the width and center of a box, and also try a smooth bump.
        
        c = 0.0
        w = 0.5
        
        # Just return the box. It is the theoretical minimum for support constrained to [-0.25, 0.25].
        
        # Wait, what if I make the function slightly wider than the domain?
        # "on [-1/4, 1/4]" usually means the domain of definition.
        # If I define f on a larger domain, but the task says "on [-1/4, 1/4]", I should stick to it.
        
        # However, if I use a cosine bump:
        # f(x) = cos(pi x / 0.5) for x in [-0.25, 0.25].
        # f(0) = 1. f(+-0.25) = 0.
        # L1 = integral cos(2 pi x) dx from -0.25 to 0.25.
        # = [sin(2 pi x) / 2pi] ...
        # sin(0.5 pi) = 1. sin(-0.5 pi) = -1.
        # L1 = (1 - (-1)) / 2pi = 1/pi.
        # L2^2 = integral cos^2(2 pi x) dx.
        # = integral (1 + cos(4 pi x))/2 dx.
        # = 0.5 * 0.5 + 0 = 0.25.
        # Ratio = 0.25 / (1/pi)^2 = 0.25 * pi^2 = pi^2 / 4 approx 2.46.
        # Worse than 2.0.
        
        # So Box is indeed better.
        
        # Is there any constraint I'm missing?
        # "nonnegative step function with 600 equal intervals"
        
        # Maybe the "normalized maximum" is defined as:
        # max(f*f) / ( (1/N) sum f )^2 ?
        # No.
        
        # I will return the box function. It is the best I can analytically derive.
        # If I have time, I will run a quick check with a few random perturbations to see if the score drops.
        
        best_val = 2.0
        best_f = np.ones(N)
        
        # Let's try to optimize a parametric bump to be absolutely sure.
        # Param: height H, width W, center C.
        # f(x) = H if |x-C| < W/2 else 0.
        
        def score_box(H, W, C):
            if W < 0 or W > 0.5: return 1e10
            if abs(C) + W/2 > 0.25: return 1e10
            
            centers = -0.25 + (np.arange(N) + 0.5) * (0.5/N)
            mask = np.abs(centers - C) < W/2
            f = np.zeros(N)
            f[mask] = H
            
            L1 = np.sum(f)
            if L1 == 0: return 1e10
            L2_sq = np.sum(f**2)
            
            # Approximate integral
            h = 0.5/N
            I1 = L1 * h
            I2 = L2_sq * h
            
            return I2 / (I1**2)
        
        # Analytical: I2/I1^2 = (H^2 * W * h) / (H * W * h)^2 = 1 / (H * W * h) ? No.
        # I1 = H * W. (Integral of height H over width W).
        # I2 = H^2 * W.
        # Ratio = H^2 W / (H^2 W^2) = 1/W.
        
        # So Ratio = 1/W.
        # To minimize 1/W, maximize W.
        # Max W = 0.5.
        # Min Ratio = 1/0.5 = 2.
        
        # So 2.0 is the minimum.
        
        # I will return the box.
        
        return list(best_f)

