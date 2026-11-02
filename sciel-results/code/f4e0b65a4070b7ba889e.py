"""Optimized program for the first autocorrelation inequality task."""
import numpy as np

N = 600
W = 1.0 / (2.0 * N)

# EVOLVE-BLOCK-START
def construct():
    # We want to find a nonnegative step function h on N equal intervals of [-1/4, 1/4]
    # that minimizes the maximum of its autoconvolution.
    
    # The autoconvolution of a step function is a piecewise linear function (sum of triangles).
    # The maximum of this function occurs at one of the "peaks" of the constituent triangles,
    # which correspond to the midpoints of the intervals in the output domain.
    
    # Let the step heights be x = [x_0, ..., x_{N-1}].
    # The intervals are of width w = 1/(2N).
    # The autoconvolution c(t) is supported on [0, 1] (if we shift the domain to [0, 1/2]).
    # The peaks of the autoconvolution occur at t = (k + 0.5)w for k = 0, ..., 2N-2.
    
    # The value of the autoconvolution at these peaks can be related to the discrete convolution.
    # Specifically, if we define the discrete autoconvolution y = convolve(x, x, mode='full'),
    # the peak values of the continuous autoconvolution are proportional to the values of y.
    
    # Let's analyze the scaling.
    # The integral of h is I = sum(x_i) * w.
    # The "normalized maximum" usually refers to max(c(t)) / I^2 or similar.
    # However, the problem statement implies we just minimize the raw maximum, but the score is normalized?
    # "score is the upper bound on C1".
    # Let's assume the standard normalization: ||h||_1 = 1.
    # Then I = 1.
    # We want to minimize max(c(t)).
    
    # Let's check the uniform distribution.
    # If x_i = 1/N for all i, then I = N * (1/N) * w = w. This is not 1.
    # To get I = 1, we need sum(x_i) * w = 1 => sum(x_i) = 1/w = 2N.
    # So if we choose x_i = 2 for all i, then sum(x_i) = 2N, and I = 1.
    # Let's compute the max autoconvolution for x_i = 2.
    # The autoconvolution of a constant function (on its support) is a triangle.
    # The height of the triangle for a function of height H and width L is H^2 L.
    # Here H = 2, L = 1/2.
    # So max(c(t)) = 2^2 * (1/2) = 2.
    
    # Can we do better?
    # Consider the inequality ||h*h||_infty <= ||h||_2^2.
    # If ||h||_1 = 1, then ||h||_2^2 >= 1/L = 2.
    # So ||h*h||_infty >= 2? No, that's not right.
    # Actually, for non-negative functions, (h*h)(0) = 0 (if support is [0, L]).
    # The maximum is at the center.
    # (h*h)(L/2) = integral h(L/2 - s) h(s) ds.
    # By Cauchy-Schwarz, (integral h(s) ds)^2 <= (measure of support) * integral h(s)^2 ds.
    # 1 <= (1/2) * ||h||_2^2 => ||h||_2^2 >= 2.
    # Also, (h*h)(t) <= ||h||_2^2 for all t.
    # So max(h*h) <= ||h||_2^2.
    # And we know ||h||_2^2 >= 2.
    # So max(h*h) can be as low as 2?
    # If h is constant, ||h||_2^2 = H^2 L = 4 * 0.5 = 2.
    # And max(h*h) = 2.
    # So the uniform distribution achieves the lower bound of 2.
    
    # Therefore, the optimal solution is the uniform distribution.
    # We need to return the step heights x_i such that sum(x_i) * w = 1.
    # x_i = 2 for all i.
    
    # Let's double check the "normalized maximum" part.
    # If the score is just the raw max, then 2 is the value.
    # If the seed program returned something else, maybe it didn't normalize correctly.
    # The seed program used SLSQP to minimize the max autoconvolution subject to sum(x_i) w = 1.
    # It likely converged to the uniform distribution or close to it.
    # If it returned x_i = 2, the score would be 2.
    # If the score is defined as the value itself, and lower is better, then 2 is the minimum.
    
    # Let's construct the uniform step heights.
    x = np.full(N, 2.0)
    
    return x.tolist()

# EVOLVE-BLOCK-END
