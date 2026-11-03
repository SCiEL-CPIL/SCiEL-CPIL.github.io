"""Optimized program for the first autocorrelation inequality task."""
import numpy as np

N = 600

def construct():
    # We want to find a nonnegative step function h on N equal intervals of [-1/4, 1/4]
    # that minimizes the normalized maximum of its autoconvolution.
    # The score is max_t (h*h)(t) / (sum x_i^2 * w) where w = 1/(2N).
    # Since w is a constant, we can minimize max_t (h*h)(t) subject to sum x_i^2 = 1.
    # The continuous autoconvolution of a step function with heights x_i on intervals of width w
    # is a piecewise linear function. Its maximum occurs at one of the peaks of the constituent
    # triangles, which are at t = (i+j+1)*w for i, j in {0, ..., N-1}.
    # The value at t = (i+j+1)*w is w * sum_{p+q = i+j} x_p x_q.
    # Let k = i+j. The value is w * d_k where d_k = sum_{p+q=k} x_p x_q.
    # Note that d_k is the k-th coefficient of the autoconvolution of the sequence x.
    # Specifically, if we define the discrete autoconvolution as d = np.convolve(x, x, mode='full'),
    # then d_k = sum_{p+q=k} x_p x_q.
    # The maximum of the continuous autoconvolution is w * max(d).
    # So we want to minimize max(d) subject to sum x_i^2 = 1, x_i >= 0.
    
    # Let's use a gradient descent method with soft-thresholding to enforce non-negativity.
    # We will minimize f(x) = max(d) subject to g(x) = sum x_i^2 - 1 = 0.
    # Using Lagrange multipliers, we can write the Lagrangian as:
    # L(x, lambda) = max(d) + lambda * (sum x_i^2 - 1)
    # The gradient of max(d) with respect to x_i is the sum of x_j for all j such that
    # the term x_i x_j contributes to the maximum coefficient.
    # Specifically, if d_k is the maximum, then the gradient of d_k with respect to x_i is
    # sum_{j: i+j=k} x_j.
    # If there are multiple maximum coefficients, we can take the average of their gradients.
    
    # Let's implement this.
    x = np.ones(N) / np.sqrt(N)  # Initial guess: uniform distribution.
    x = np.maximum(x, 0)  # Ensure non-negativity.
    
    # Normalize to satisfy the constraint.
    norm = np.sqrt(np.sum(x**2))
    x /= norm
    
    # Number of iterations.
    n_iter = 5000
    lr = 0.01  # Learning rate.
    
    for it in range(n_iter):
        # Compute the autoconvolution.
        d = np.convolve(x, x, mode='full')
        max_d = np.max(d)
        
        # Find the indices of the maximum coefficients.
        max_indices = np.where(d == max_d)[0]
        
        # Compute the gradient of max(d) with respect to x.
        # The gradient of d_k with respect to x_i is sum_{j: i+j=k} x_j.
        # We can compute this by summing the gradients of all maximum coefficients.
        grad = np.zeros(N)
        for k in max_indices:
            # For a given k, the terms x_i x_j with i+j=k contribute to d_k.
            # The gradient of d_k with respect to x_i is x_{k-i}.
            # So we need to sum x_{k-i} for all valid i.
            # i ranges from max(0, k-N+1) to min(N-1, k).
            i_min = max(0, k - N + 1)
            i_max = min(N - 1, k)
            for i in range(i_min, i_max + 1):
                j = k - i
                grad[i] += x[j]
        
        # Average the gradients if there are multiple maximum coefficients.
        if len(max_indices) > 0:
            grad /= len(max_indices)
        
        # Compute the gradient of the constraint g(x) = sum x_i^2 - 1.
        grad_g = 2 * x
        
        # Update x using the gradient of the Lagrangian.
        # L(x, lambda) = max(d) + lambda * (sum x_i^2 - 1)
        # dL/dx = grad + 2 * lambda * x
        # We want to minimize L, so we move in the negative gradient direction.
        # x_new = x - lr * (grad + 2 * lambda * x)
        # But we also need to update lambda.
        # A simple approach is to use a fixed lambda or adapt it.
        # Let's use a fixed lambda for simplicity, or set it to 0 and just project onto the constraint.
        
        # Actually, let's use a projected gradient descent method.
        # We minimize max(d) subject to sum x_i^2 = 1 and x_i >= 0.
        # We can use the following update:
        # x_new = x - lr * grad
        # Then project x_new onto the feasible set: {x: sum x_i^2 <= 1, x_i >= 0}.
        # The projection onto the non-negative orthant is simply max(x_new, 0).
        # The projection onto the unit ball is x_new / max(1, ||x_new||_2).
        
        x_new = x - lr * grad
        x_new = np.maximum(x_new, 0)  # Project onto non-negative orthant.
        norm_new = np.sqrt(np.sum(x_new**2))
        if norm_new > 1:
            x_new /= norm_new  # Project onto the unit ball.
        
        x = x_new
        
        # Optional: restart or adapt learning rate.
        if it % 1000 == 0:
            lr = max(0.001, lr * 0.5)
    
    # The score is the normalized maximum of the autoconvolution.
    # d = np.convolve(x, x, mode='full')
    # max_d = np.max(d)
    # norm_sq = np.sum(x**2)
    # score = max_d / norm_sq  # This is the ratio.
    
    # But the task says the score is the upper bound on C1.
    # The seed program returns x, and the evaluator measures the raw value.
    # Let's check what the evaluator expects.
    # The task says: "score is the upper bound on C1".
    # And "the evaluator measures the raw value (lower is better)".
    # So the evaluator probably computes the score from the returned x.
    # We just need to return x.
    
    return x.tolist()

# EVOLVE-BLOCK-END
