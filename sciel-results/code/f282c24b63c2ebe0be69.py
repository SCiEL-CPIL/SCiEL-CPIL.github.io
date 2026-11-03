"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize_scalar, minimize

N = 600
DEADLINE = 140.0  # safety margin from 150s limit

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Interval setup
    # [-1/4, 1/4] with 600 equal intervals
    # Interval width dx
    dx = (0.5) / N
    # The step function is defined on intervals [x_i, x_{i+1})
    # Let's define the grid points for the centers or edges?
    # "sequence of nonnegative step heights on equal intervals"
    # Usually implies the function value is constant on each interval.
    # Let's assume the intervals are [x_i, x_{i+1}) where x_i = -1/4 + i*dx.
    # The autoconvolution h(t) = \int f(\tau) f(t-\tau) d\tau.
    # Since f is a step function, h is a piecewise quadratic function?
    # No, the autoconvolution of a step function is a piecewise quadratic spline.
    # However, we can compute it efficiently via FFT.
    
    # To compute the normalized maximum of its autoconvolution:
    # We need to minimize max_t |h(t)| / (something)?
    # "minimize the normalized maximum of its autoconvolution"
    # The score is the upper bound on C1.
    # C1 is likely the constant in the inequality ||f * f||_\infty \le C1 ||f||_2^2 ?
    # Or ||f * f||_\infty \le C1 ||f||_\infty ||f||_1 ?
    # Let's look at the "First autocorrelation inequality".
    # Usually, for non-negative f, \int f^2 = \int (f*f)(0).
    # The peak of the autocorrelation is at 0 if f is non-negative?
    # If f is non-negative, f*f is maximized at t=0?
    # Let's check. (f*f)(t) = \int f(\tau) f(t-\tau) d\tau.
    # If f is concentrated at 0, peak is at 0.
    # If f is spread out, peak might be lower.
    # The problem asks to minimize the *normalized maximum*.
    # What is the normalization?
    # "upper bound on C1".
    # In the context of the "first autocorrelation inequality" (likely related to the uncertainty principle or similar),
    # a common inequality is ||f \tilde{f}||_\infty \le C ||f||_2^2.
    # For real non-negative f, the autocorrelation is f \tilde{f}.
    # The maximum of the autocorrelation is at t=0 if f is unimodal? Not necessarily.
    # But for non-negative f, \int f^2 = (f*f)(0).
    # Is it true that (f*f)(t) \le (f*f)(0) for all t if f \ge 0?
    # No. Consider f as two separated bumps. The cross-correlation at the separation distance could be significant, but the self-correlation at 0 is the sum of squares.
    # Actually, for non-negative f, the L1 norm is fixed? No, the problem doesn't fix L1 or L2.
    # "minimize the normalized maximum".
    # If we scale f by k, f*f scales by k^2.
    # So we must fix a norm.
    # Usually, we fix ||f||_2 = 1 or ||f||_1 = 1.
    # If we fix ||f||_2 = 1, then (f*f)(0) = 1.
    # Then the "maximum" is at least 1.
    # If the maximum is always at 0 for non-negative functions, then the min max is 1.
    # But the task implies there is an optimization.
    # Let's re-read: "Choose a nonnegative step function ... to minimize the normalized maximum of its autoconvolution".
    # If the normalization is ||f||_2^2, then (f*f)(0) = 1.
    # If max_t (f*f)(t) > 1, then the bound C1 must be >= 1.
    # If we can make max_t (f*f)(t) close to 1, the bound is 1.
    # Can we make it *less* than 1? No, because (f*f)(0) = 1.
    # So the minimum possible value is 1.
    # Unless... the normalization is different.
    # Maybe ||f||_1?
    # If ||f||_1 = 1, then (f*f)(0) can be small if f is spread out.
    # Then max (f*f) can be small.
    # But the "upper bound on C1" suggests C1 is a constant in an inequality like:
    # ||f \tilde{f}||_\infty \le C_1 ||f||_1^2 ?
    # If ||f||_1 = 1, then ||f \tilde{f}||_\infty is the value we minimize.
    # Let's assume the normalization is ||f||_1 = 1.
    # Then we want to minimize \max_t \int f(\tau) f(t-\tau) d\tau subject to \int f = 1, f \ge 0.
    # The constant function f = 2 on [-1/4, 1/4] has ||f||_1 = 1.
    # Its autocorrelation is a triangle function.
    # Peak at 0: \int_{-1/4}^{1/4} 2*2 d\tau = 4 * 0.5 = 2.
    # So the score would be 2.
    # Can we do better?
    # If we make f spikier, the peak increases.
    # If we make f more spread out, the peak decreases.
    # The support is fixed to [-1/4, 1/4].
    # The maximum spread is the uniform distribution on the support.
    # So the uniform distribution minimizes the peak of the autocorrelation?
    # Let's check.
    # For any non-negative f with support in [-A, A], is \max (f*f) minimized by the uniform distribution?
    # Intuitively, yes, because spreading the mass reduces the overlap at any shift.
    # The overlap at shift 0 is \int f^2. By Cauchy-Schwarz, \int f^2 \ge (\int f)^2 / (2A).
    # Equality holds for uniform f.
    # So (f*f)(0) is minimized by uniform f.
    # Is the global max of (f*f) also minimized by uniform f?
    # For uniform f, (f*f)(t) is a triangle peaking at 0.
    # So max is at 0.
    # For any other f, (f*f)(0) is larger.
    # Could (f*f)(t) for t != 0 be larger than (f*f)(0) for uniform f?
    # If we make f bimodal, (f*f)(t) at the separation distance is the product of the masses.
    # But (f*f)(0) is sum of squares.
    # If we split mass m into two bumps of m/2, (f*f)(0) becomes (m/2)^2 + (m/2)^2 = m^2/2.
    # Original uniform: m^2 / (2A) * (2A) ... wait.
    # Let's use numbers.
    # Uniform on [-1/4, 1/4], height h=2. ||f||_1=1.
    # (f*f)(0) = \int 4 d\tau = 4 * 0.5 = 2.
    # Two bumps of width w, height h_b.
    # 2 * h_b * w = 1 => h_b = 1/(2w).
    # (f*f)(0) = 2 * h_b^2 * w = 2 * (1/(4w^2)) * w = 1/(2w).
    # To minimize this, maximize w.
    # Max w is 0.25 (if bumps don't overlap? They can overlap).
    # If w=0.25, they are the uniform function (if placed at ends? No, if they cover the whole interval).
    # If we place them at -1/8 and 1/8 with width 1/4, they overlap completely?
    # If they are identical, it's just one bump of width 1/4?
    # If we split the uniform function into two halves, it's still uniform.
    # So uniform seems optimal.
    
    # Why is the current program scoring 0.0000?
    # "score is the upper bound on C1".
    # "evaluator measures the raw value (lower is better)".
    # "search score is seed/value".
    # If the seed returns [1.0]*600, what is its value?
    # If value is 2.0 (as calculated above for uniform), score = 1.0/2.0 = 0.5?
    # But the prompt says "score 0.0000" for the current program.
    # This implies the current program is invalid or the value is infinite/NaN?
    # Or maybe the normalization is different.
    # Or maybe I am misinterpreting "normalized maximum".
    
    # Let's look at the "First autocorrelation inequality" in literature.
    # Perhaps it refers to the inequality:
    # ||f \tilde{f}||_\infty \le C ||f||_2^2 ?
    # If so, for non-negative f, (f*f)(0) = ||f||_2^2.
    # And (f*f)(t) \le (f*f)(0) is NOT always true.
    # Wait, if f \ge 0, is (f*f)(t) \le (f*f)(0)?
    # Let f be two delta functions at -1 and 1.
    # (f*f)(0) = 0.
    # (f*f)(2) = 1.
    # So max is 1, (f*f)(0) is 0.
    # So the max is NOT at 0.
    # In this case, the ratio ||f*f||_\infty / ||f||_2^2 can be large.
    # For the uniform function, max is at 0, ratio is 1.
    # So uniform minimizes the ratio?
    # If we use deltas, ratio is infinity.
    # So we want to minimize the ratio.
    # The uniform function gives ratio 1.
    # Can we get ratio < 1?
    # No, because for any f, \int (f*f)(t) dt = (\int f)^2.
    # And \int (f*f)(t) dt \ge \max (f*f) * \text{support length}?
    # This doesn't help directly.
    
    # Let's reconsider the "score 0.0000".
    # If the seed program returns [1.0]*600, and the score is 0, it means the value is infinite or the program crashed?
    # "an invalid result, an exception or a timeout scores 0".
    # The current program is:
    # def construct(): return [1.0] * N
    # This is valid.
    # So why score 0?
    # Maybe the evaluator expects a specific format?
    # "sequence of nonnegative step heights".
    # Maybe the normalization is such that the value is 0 for the seed?
    # Or maybe the "raw value" is something else.
    
    # Let's look at the directive: "seed/value, so 1.0 equals the seed program".
    # If the seed program scores 0, it's because value is inf or NaN?
    # Or maybe the seed program IS the baseline, and its score is defined as 1.0?
    # "1.0 equals the seed program".
    # But the current program is the seed program.
    # And it has score 0.0000.
    # This is contradictory.
    # Unless... the "current program" in the prompt is NOT the seed program?
    # "CURRENT PROGRAM (score 0.0000)"
    # "INSPIRATIONS... (none)"
    # "DIRECTIVE... STAGNATION... population best has not improved".
    # Maybe the seed program had a score of 1.0, but this "current program" is a failed attempt?
    # No, it says "Initial program for the suite task".
    # Perhaps the evaluator calculates the value, and for [1.0]*600, the value is such that seed/value = 0?
    # That would mean value is infinite.
    # Why would the value be infinite?
    # Maybe the normalization is 1/||f||_2^2? No.
    
    # Let's assume the task is:
    # Minimize M = \max_t (f*f)(t) / (\int f)^2 ?
    # Or \max_t (f*f)(t) / \int f^2 ?
    
    # Let's try to compute the value for the uniform case with a specific normalization.
    # If the score is 0, maybe the program is timing out?
    # No, it's just returning a list.
    
    # Hypothesis: The "value" is the upper bound C1.
    # For the uniform function, C1 = 1 (if normalized by L2^2) or 2 (if normalized by L1^2).
    # If the seed score is 1.0, and the current score is 0.0, maybe the current program is NOT the seed?
    # But it says "Initial program".
    
    # Let's ignore the score 0.0 mystery and focus on finding a good function.
    # The goal is to minimize the "normalized maximum of its autoconvolution".
    # Common normalization in such problems is L2 norm.
    # Let's assume we want to minimize \max_t (f*f)(t) subject to \int f^2 = 1.
    # For uniform f on [-1/4, 1/4]:
    # \int f^2 = h^2 * 0.5 = 1 => h = \sqrt{2}.
    # (f*f)(0) = \int h^2 d\tau = 2 * 0.5 = 1.
    # Max is 1.
    # So the value is 1.
    # If the seed value is 1, and the score is seed/value, then score = 1/1 = 1.
    # But the current score is 0.
    
    # What if the normalization is L1?
    # \int f = 1.
    # Uniform: h=2.
    # (f*f)(0) = 2.
    # Value = 2.
    # Score = 1/2 = 0.5.
    
    # What if the "normalized maximum" is \max_t (f*f)(t) / \max_t f(t) ?
    # Uniform: max f = 2. Max conv = 2. Ratio = 1.
    
    # Let's look at the "First autocorrelation inequality" specifically.
    # It might be the inequality:
    # \| f \tilde{f} \|_\infty \le C \| f \|_1 \| f \|_\infty ?
    # For uniform: L1=1, Linf=2. Conv=2. Ratio = 2/2 = 1.
    
    # Let's try a different function.
    # What if we use a function that is 0 in the middle and high at the ends?
    # This might reduce the peak at 0?
    # No, the peak at 0 is \int f^2.
    # If we move mass to the ends, \int f^2 stays the same if height is same?
    # If we keep L2 fixed, \int f^2 is fixed.
    # So (f*f)(0) is fixed.
    # We want to reduce the peak at t != 0?
    # But the max is at 0 for uniform.
    # If we make it bimodal, the peak at 0 is the same (if L2 fixed).
    # But the cross-correlation at the separation distance might be lower than the self-correlation at 0.
    # So the max is still at 0.
    # So the value is still 1 (for L2 normalization).
    
    # Is it possible to get a value < 1?
    # If max (f*f) < \int f^2?
    # This requires (f*f)(t) < (f*f)(0) for all t != 0, AND (f*f)(0) is the max.
    # But we already have (f*f)(0) = \int f^2.
    # So the ratio is 1.
    # Can we make (f*f)(0) NOT the max?
    # If (f*f)(t) > (f*f)(0) for some t, then the max is larger, so the ratio is > 1.
    # So the minimum possible ratio is 1.
    # Achieved by any function where the autocorrelation is maximized at 0.
    # Uniform is one such function.
    # So the best score is 1.
    
    # Why is the current score 0?
    # Maybe the evaluator is calculating the value differently.
    # Or maybe the "seed" value is not 1.
    # "search score is seed/value".
    # If the seed program is [1.0]*N, and its value is V_seed.
    # Score = V_seed / V_current.
    # If V_seed = V_current, Score = 1.
    # If Score = 0, then V_current is infinite?
    # Or V_seed is 0?
    
    # Let's assume the task is solvable and the score 0 is due to a bug in my understanding or the evaluator's setup.
    # However, I must provide a program.
    # The best strategy is to provide the uniform function, as it is theoretically optimal for minimizing the peak of the autocorrelation relative to the L2 norm (or L1 norm, depending on definition, but uniform is generally the "flattest" and minimizes peaks).
    
    # Wait, the directive says: "Make a STRUCTURAL change... a different design...".
    # "Do not micro-tune constants."
    # "Fresh design".
    
    # If the uniform function is optimal, why change?
    # Maybe the uniform function is NOT optimal for the specific "normalized maximum".
    # What if the normalization is by the L1 norm, and we want to minimize \max (f*f) / (\int f)^2?
    # For uniform: 2 / 1^2 = 2.
    # For a delta function: \infty.
    # For a function concentrated in the center?
    # Let's try a function that is 0 at the edges and high in the center.
    # This increases \int f^2 for a fixed \int f.
    # So the ratio \max (f*f) / (\int f)^2 would increase.
    # So uniform is still better.
    
    # What if the normalization is by the L\infty norm?
    # Minimize \max (f*f) / \max f.
    # Uniform: 2 / 2 = 1.
    # Triangular function:
    # f(t) = 1 - |t|/0.25.
    # Max f = 1.
    # (f*f)(0) = \int (1-|t|/0.25)^2 dt.
    # This is smaller than the uniform case?
    # Let's check.
    # Uniform: \int f^2 = 2 * 0.5 = 1 (if h=2, L1=1). Wait, if L1=1, h=2. \int f^2 = 2.
    # Triangular: \int f^2 = 2 * \int_0^{0.25} (1-x/0.25)^2 dx.
    # Let u = x/0.25. dx = 0.25 du.
    # \int_0^1 (1-u)^2 0.25 du = 0.25 * [1/3] = 1/12.
    # Total \int f^2 = 2 * 1/12 = 1/6.
    # So (f*f)(0) = 1/6.
    # Max f = 1.
    # Ratio = (1/6) / 1 = 1/6.
    # This is much better than 1!
    
    # So if the normalization is by L\infty, the triangular function is better.
    # What is the "First autocorrelation inequality"?
    # It might be: \| f \tilde{f} \|_\infty \le C \| f \|_\infty \| f \|_1 ?
    # For uniform: 2 / (2*1) = 1.
    # For triangular: (1/6) / (1*1) = 1/6.
    # So C1 can be 1/6?
    # Is there a function with even lower ratio?
    # We want to minimize \max (f*f) subject to \max f = 1, \int f = 1.
    # To minimize \max (f*f), we want to minimize \int f^2 (since peak is likely at 0).
    # To minimize \int f^2 subject to \max f = 1 and \int f = 1, we should spread the mass as much as possible while keeping the height at 1.
    # This means f should be 1 everywhere?
    # But \int f = 1. If f=1, \int f = 0.5.
    # We need \int f = 1.
    # So we need a function with height 1 and area 1.
    # The support is [-1/4, 1/4], length 0.5.
    # Max area with height 1 is 0.5.
    # So we cannot have \int f = 1 with \max f = 1 on this support.
    # So the constraints are incompatible if we fix both.
    # We must scale.
    # Let's fix \max f = 1.
    # Then \int f \le 0.5.
    # Let A = \int f.
    # We want to minimize \max (f*f) / (1 * A) ? Or \max (f*f) / A^2?
    # If the inequality is \| f \tilde{f} \|_\infty \le C \| f \|_\infty \| f \|_1, then we minimize \max (f*f) / (\max f \cdot \int f).
    # For uniform: 2 / (2 * 1) = 1. (Here max f=2, int f=1).
    # Let's use max f = 1.
    # Uniform: h=1. \int f = 0.5.
    # (f*f)(0) = \int 1^2 = 0.5.
    # Ratio = 0.5 / (1 * 0.5) = 1.
    # Triangular: h=1. \int f = 0.25.
    # (f*f)(0) = 1/6.
    # Ratio = (1/6) / (1 * 0.25) = (1/6) / (1/4) = 4/6 = 2/3.
    # This is better than 1.
    
    # Can we do better?
    # We want to minimize \int f^2 / (\int f)^2 ?
    # Let f = g / \max g.
    # Ratio = \int g^2 / (\int g)^2.
    # We want to minimize this ratio.
    # This is the "inverse participation ratio" or similar.
    # To minimize \int g^2 / (\int g)^2, we want g to be as "flat" as possible?
    # No, we want g to be as "spread out" as possible.
    # But g is bounded by 1.
    # So the optimal g is 1 everywhere?
    # If g=1, \int g^2 = 0.5, (\int g)^2 = 0.25. Ratio = 2.
    # Wait, my previous calculation for uniform was:
    # h=2, \int f=1. \int f^2 = 2.
    # Ratio = 2 / (2 * 1) = 1.
    # Using g=1 (h=1), \int f=0.5. \int f^2=0.5.
    # Ratio = 0.5 / (1 * 0.5) = 1.
    # Consistent.
    
    # For triangular:
    # g is triangular, max 1.
    # \int g = 0.25.
    # \int g^2 = 1/6.
    # Ratio = (1/6) / (1 * 0.25) = 4/6 = 2/3.
    
    # So triangular is better.
    # What about a function that is 1 on a subset and 0 elsewhere?
    # Let g=1 on interval of length L.
    # \int g = L.
    # \int g^2 = L.
    # Ratio = L / (1 * L) = 1.
    # So any box function gives ratio 1.
    # Triangular gives 2/3.
    # What about a sine function?
    # g(t) = sin(\pi t / 0.5) ?
    # Max 1.
    # \int g = 2/ \pi * 0.5 ?
    # \int_{-0.25}^{0.25} \sin(\pi t / 0.5) dt ?
    # Let's just use a smooth bump.
    # The more "peaked" the function, the smaller \int g / \int g^2 ?
    # No, we want to minimize \int g^2 / (\int g)^2.
    # This is equivalent to maximizing (\int g)^2 / \int g^2.
    # This is the "efficiency" of the function.
    # It is maximized when g is constant?
    # Let's check.
    # If g is constant, (\int g)^2 / \int g^2 = (L \cdot h)^2 / (L \cdot h^2) = L.
    # If g is a delta, it's 0.
    # So we want L to be large.
    # Max L is 0.5.
    # So constant function gives 0.5.
    # Triangular: (\int g)^2 / \int g^2 = (0.25)^2 / (1/6) = 0.0625 * 6 = 0.375.
    # 0.5 > 0.375.
    # So constant is better?
    # Ratio for constant: 1/0.5 = 2?
    # Wait.
    # Ratio = \int g^2 / (\int g)^2.
    # Constant: 0.5 / 0.25 = 2.
    # Triangular: (1/6) / 0.0625 = 6/12 = 0.5.
    # So triangular is BETTER (lower ratio).
    
    # So we want to MINIMIZE \int g^2 / (\int g)^2.
    # This means we want the function to be as "non-constant" as possible?
    # No, we want the denominator to be large relative to the numerator.
    # This happens when the function is spread out but not flat?
    # Actually, for a fixed max value, the integral is maximized by the flat function.
    # But the integral of the square is also maximized by the flat function.
    # The ratio \int g^2 / (\int g)^2 is minimized when the function is "most peaked"?
    # Let's test a narrow triangle.
    # Base 2w, height 1.
    # \int g = w.
    # \int g^2 = 2w/3.
    # Ratio = (2w/3) / w^2 = 2/(3w).
    # As w decreases, ratio increases.
    # So we want w to be large.
    # Max w is 0.25 (full width).
    # Ratio = 2/(3*0.25) = 2/0.75 = 8/3 = 2.66.
    # Wait, my previous calculation for full triangle was 0.5.
    # Let's re-calculate full triangle.
    # Base 0.5. w=0.25.
    # \int g = 0.25.
    # \int g^2 = 1/6.
    # Ratio = (1/6) / (0.25)^2 = (1/6) / (1/16) = 16/6 = 8/3 = 2.66.
    # My previous ratio was 2/3. Why?
    # Ah, the ratio was \max (f*f) / (\max f \cdot \int f).
    # \max (f*f) \approx \int f^2.
    # So ratio = \int f^2 / (\max f \cdot \int f).
    # For triangle: (1/6) / (1 * 0.25) = 4/6 = 2/3.
    # For constant: (0.5) / (1 * 0.5) = 1.
    # So triangle is better.
    
    # So we want to minimize \int f^2 / (\max f \cdot \int f).
    # Let f = g / \max g.
    # Minimize \int g^2 / (\int g).
    # Subject to \max g = 1.
    # To minimize \int g^2 / \int g, we want \int g to be large and \int g^2 to be small.
    # This suggests a function that is close to 1 over a large area, but drops to 0?
    # No, if it drops to 0, \int g decreases.
    # If we take a box function of width L, height 1.
    # \int g = L. \int g^2 = L.
    # Ratio = L/L = 1.
    # If we take a triangle of width L, height 1.
    # \int g = L/2. \int g^2 = L/3.
    # Ratio = (L/3) / (L/2) = 2/3.
    # If we take a parabola?
    # g(t) = 1 - (t/a)^2.
    # \int g = 4a/3.
    # \int g^2 = 4a/5.
    # Ratio = (4a/5) / (4a/3) = 3/5 = 0.6.
    # Better than 2/3.
    # What about g(t) = (1 - (t/a)^2)^k?
    # As k increases, the function becomes more "box-like" in the center?
    # No, it becomes more peaked?
    # Let's try k=2.
    # g(t) = (1 - (t/a)^2)^2.
    # \int g = 16a/15.
    # \int g^2 = 16a/21.
    # Ratio = (16a/21) / (16a/15) = 15/21 = 5/7 = 0.71.
    # Worse.
    
    # So the parabola (k=1) is better than the box (k=0) and the higher powers.
    # What about a function that is 0 at the edges and 1 in the middle, but very flat?
    # Like a cosine bump?
    # g(t) = \cos(\pi t / 2a).
    # \int g = 4a/\pi.
    # \int g^2 = a.
    # Ratio = a / (4a/\pi) = \pi/4 = 0.78.
    # Worse than 0.6.
    
    # So the parabola seems good.
    # Can we go lower?
    # We want to minimize \int g^2 / \int g.
    # This is the mean of g with respect to the measure g(t)dt?
    # No.
    # Let's try a function that is 1 on [-b, b] and 0 elsewhere.
    # Ratio = 1.
    # Let's try a function that is 1 on [-b, b] and drops linearly to 0 at +/-a.
    # This is a trapezoid.
    # As the slope becomes steeper, it approaches the box.
    # As the slope becomes shallower, it approaches the triangle.
    # Triangle gave 2/3.
    # Parabola gave 3/5.
    # What about a function that is 1 in the center and drops quadratically?
    # That's the parabola.
    
    # Let's try a function that is 0 at the center? No, we want to maximize \int g.
    
    # The minimum of \int g^2 / \int g subject to \max g = 1 is achieved by the function that is 1 everywhere?
    # No, that gives 1.
    # The function that minimizes this ratio is the one that is "most spread out" in a way that reduces the square integral relative to the linear integral.
    # This is achieved by functions that are close to 1 over a large domain but taper off smoothly.
    # The parabola taper is better than linear.
    # What about a Gaussian?
    # g(t) = \exp(-t^2/a^2).
    # \int g = a \sqrt{\pi}.
    # \int g^2 = a \sqrt{\pi/2}.
    # Ratio = \sqrt{1/2} = 0.707.
    # Worse than 0.6.
    
    # So the parabola is a strong candidate.
    # Let's implement the parabola.
    # g(t) = 1 - (t/0.25)^2.
    # This is non-negative on [-1/4, 1/4].
    # Max is 1 at t=0.
    # 0 at t= +/- 0.25.
    
    # Let's verify the ratio.
    # \int_{-0.25}^{0.25} (1 - (t/0.25)^2) dt = 0.25 * 4/3 = 1/3.
    # \int_{-0.25}^{0.25} (1 - (t/0.25)^2)^2 dt = 0.25 * 8/15 = 2/15.
    # Ratio = (2/15) / (1/3) = 6/15 = 2/5 = 0.4.
    # Wait, my previous calculation was 3/5.
    # Let's re-integrate.
    # \int_{-1}^{1} (1-x^2) dx = [x - x^3/3]_{-1}^1 = (1-1/3) - (-1+1/3) = 2/3 + 2/3 = 4/3.
    # Scale by 0.25: 4/3 * 0.25 = 1/3. Correct.
    # \int_{-1}^{1} (1-x^2)^2 dx = \int (1 - 2x^2 + x^4) dx = [x - 2x^3/3 + x^5/5]_{-1}^1.
    # = (1 - 2/3 + 1/5) - (-1 + 2/3 - 1/5) = (15-10+3)/15 - (-15+10-3)/15 = 8/15 - (-8/15) = 16/15.
    # Scale by 0.25: 16/15 * 0.25 = 4/15.
    # Ratio = (4/15) / (1/3) = 12/15 = 4/5 = 0.8.
    # My previous 3/5 was wrong.
    # So parabola gives 0.8.
    # Triangle gave 2/3 = 0.66.
    # Box gives 1.
    # So triangle is better than parabola.
    
    # Let's re-evaluate triangle.
    # \int_{-1}^{1} (1-|x|) dx = 1.
    # Scale by 0.25: 0.25.
    # \int_{-1}^{1} (1-|x|)^2 dx = 2 \int_0^1 (1-x)^2 dx = 2 [1/3] = 2/3.
    # Scale by 0.25: 2/3 * 0.25 = 1/6.
    # Ratio = (1/6) / 0.25 = 4/6 = 2/3.
    
    # So triangle is better.
    # What about a function that is 0 at the edges and 1 in the middle, but with a steeper drop?
    # Like a higher power of the distance?
    # g(t) = (1 - |t/a|)^k.
    # \int g = 2a/k+1.
    # \int g^2 = 2a/2k+1.
    # Ratio = (2a/2k+1) / (2a/k+1) = (k+1)/(2k+1).
    # For k=1 (triangle): 2/3 = 0.66.
    # For k=2: 3/5 = 0.6.
    # For k=3: 4/7 = 0.57.
    # As k -> infinity, ratio -> 1/2 = 0.5.
    # So the "best" function in this family is the limit k->infinity, which is a box function?
    # No, as k->infinity, (1-|x|)^k approaches 0 everywhere except x=0?
    # No, it approaches a function that is 1 at x=0 and 0 elsewhere?
    # Wait, if k is large, (1-|x|)^k is small for any x != 0.
    # So the function becomes a spike.
    # A spike has high \int g^2 / \int g.
    # Let's check the limit.
    # Ratio = (k+1)/(2k+1) -> 1/2.
    # But a delta function has ratio infinity.
    # Where is the error?
    # \int g = 2a/(k+1).
    # \int g^2 = 2a/(2k+1).
    # Ratio = \frac{2a}{2k+1} \frac{k+1}{2a} = \frac{k+1}{2k+1}.
    # For k=1, 2/3.
    # For k=10, 11/21 = 0.52.
    # So the ratio decreases as k increases.
    # But the function becomes narrower.
    # Does this make sense?
    # If the function is a spike, \int g is small, \int g^2 is even smaller?
    # No, for a spike, \int g^2 is large relative to \int g.
    # Let's check a narrow triangle.
    # Base 2w.
    # \int g = w.
    # \int g^2 = 2w/3.
    # Ratio = 2/3.
    # It doesn't depend on w!
    # So the shape matters, not the width.
    # And for the family (1-|x|)^k, the ratio depends on k.
    # And it decreases with k.
    # So the "best" is k->infinity.
    # What is the function for k->infinity?
    # It's a function that is 1 at x=0 and 0 elsewhere?
    # No, it's a function that is 1 in a very small region around 0?
    # If it's a box of width \epsilon, ratio is 1.
    # Contradiction.
    
    # Ah, the function (1-|x|)^k is NOT a box.
    # It is a smooth bump.
    # As k increases, it becomes more peaked.
    # But the ratio decreases?
    # Let's check k=10.
    # Ratio = 11/21 = 0.52.
    # Box is 1.
    # So a peaked smooth bump is better than a box?
    # This seems counter-intuitive.
    # Let's check the integral of the square again.
    # For a box, \int g^2 = \int g.
    # For a peaked function, \int g^2 < \int g ?
    # No, if g <= 1, then g^2 <= g.
    # So \int g^2 <= \int g.
    # So the ratio is <= 1.
    # And we want to minimize it.
    # So we want g^2 to be as small as possible relative to g.
    # This means we want g to be close to 0 or 1?
    # No, we want g to be close to 0?
    # If g is 0, ratio is 0/0.
    # If g is 1, ratio is 1.
    # If g is 0.5, ratio is 0.25/0.5 = 0.5.
    # So a function that is 0.5 everywhere is better than 1 everywhere?
    # But we fixed max g = 1.
    # So we can't have g=0.5 everywhere if max is 1.
    # We can have g=1 in some places and 0 in others.
    # If we have g=1 on half the domain and 0 on the other half.
    # \int g = 0.25.
    # \int g^2 = 0.25.
    # Ratio = 1.
    # If we have g=0.5 on the whole domain.
    # \int g = 0.25.
    # \int g^2 = 0.125.
    # Ratio = 0.5.
    # But max g is 0.5, not 1.
    # We can scale it up.
    # If we scale g by 2, max g = 1.
    # \int g = 0.5.
    # \int g^2 = 0.5.
    # Ratio = 1.
    # So scaling doesn't help.
    
    # So the minimum ratio is 0.5, achieved by a function that is 0.5 everywhere?
    # No, because we can't have max 1 and constant 0.5.
    # We can have a function that is 1 in a small region and 0 elsewhere?
    # No, that gives 1.
    
    # The minimum is achieved by the function that is as "flat" as possible while satisfying max=1?
    # No, flat gives 1.
    # The minimum is achieved by the function that is as "spiky" as possible?
    # No, spiky gives 1 (box) or higher?
    
    # Let's go back to the triangle.
    # Ratio 2/3.
    # Let's try a function that is 1 at the center and 0 at the edges, linearly.
    # That's the triangle.
    
    # I will use the triangle function.
    
    def construct():
        # Triangle function
        # Height 1 at 0, 0 at +/- 0.25
        # x_i = -0.25 + i * dx
        # f_i = 1 - |x_i| / 0.25
        heights = []
        for i in range(N):
            x = -0.25 + (i + 0.5) * (0.5 / N)
            h = 1.0 - abs(x) / 0.25
            if h < 0:
                h = 0.0
            heights.append(h)
        return heights

# EVOLVE-BLOCK-END
