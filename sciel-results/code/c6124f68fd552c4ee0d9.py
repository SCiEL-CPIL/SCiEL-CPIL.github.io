import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a local interpolation strategy.
    
    Strategy:
    Instead of a fixed global basis (polynomials, trig), we use a sum of 
    shifted Gaussian kernels centered at the training data points (or a 
    subset thereof if params length < n_train, but here params is length 10).
    
    Wait, the params array is FIXED length 10. The training data has 200 points.
    If I use 200 Gaussians, I need 200 params. But I only have 10.
    
    Re-evaluating the constraint:
    "The evaluator fits params, an array of length 10..."
    
    If I only have 10 parameters, I cannot fit a weight for each of the 200 training points.
    Therefore, a pure "sum of 200 Gaussians" approach is impossible with 10 params.
    
    Alternative Strategy:
    Use a "Mixture of Experts" or "Piecewise" approach where the 10 parameters
    define a more flexible global function than the current one, OR
    use the 10 parameters to define a function that is locally adaptive.
    
    However, the prompt says "rethinking the approach".
    The current approach is a linear model with fixed basis.
    
    Let's look at the score: 2654524. This is huge. The baseline (linear) got 1.0.
    Why did the complex basis fail?
    Likely because BFGS starting from all ones on a highly non-linear, potentially
    ill-conditioned landscape (mixing polynomials and trig/exp) got stuck or diverged.
    
    A different strategy: **Non-linear functional form with 10 params**.
    Instead of `sum(p_i * b_i(x))`, use a form like `p0 * sin(p1 * x + p2) + p3 * cos(p4 * x + p5) + p6 * x^2 + p7 * x + p8 * exp(-p9 * x^2)`.
    This allows the frequencies and widths to be optimized, not just the amplitudes.
    This is a much more powerful hypothesis class for symbolic regression.
    
    Let's try:
    y = p0 * sin(p1 * x + p2) + p3 * cos(p4 * x + p5) + p6 * exp(-p7 * x**2) + p8 * x + p9
    
    This uses 10 params.
    p0, p1, p2: Amplitude, Frequency, Phase for sine
    p3, p4, p5: Amplitude, Frequency, Phase for cosine
    p6, p7: Amplitude, Width for Gaussian
    p8, p9: Linear term and bias
    
    This is materially different: the basis functions themselves are parameterized.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Unpack 10 parameters
    # Group 1: Sine component
    # p0: amp, p1: freq, p2: phase
    # Group 2: Cosine component
    # p3: amp, p4: freq, p5: phase
    # Group 3: Gaussian component
    # p6: amp, p7: width_inv (inverse width to avoid division by zero issues if not careful, but exp(-p7*x^2) is safe if p7 is finite)
    # Group 4: Linear + Bias
    # p8: slope, p9: intercept
    
    p0, p1, p2 = params[0], params[1], params[2]
    p3, p4, p5 = params[3], params[4], params[5]
    p6, p7 = params[6], params[7]
    p8, p9 = params[8], params[9]
    
    # Time check: if we are taking too long (unlikely for this simple formula, 
    # but good practice per directive), we could simplify. 
    # However, this formula is O(N) and very fast.
    
    # Calculate components
    # Add small epsilon to frequencies to avoid potential issues if they hit exactly 0? 
    # Actually, sin(0) is fine.
    
    term_sin = p0 * np.sin(p1 * x + p2)
    term_cos = p3 * np.cos(p4 * x + p5)
    
    # Gaussian: exp(-p7 * x^2)
    # If p7 is negative, it becomes exp(+|p7|x^2) which can overflow.
    # To be safe against overflow during optimization, we can clamp p7 or use a stable exp.
    # However, BFGS might explore negative p7. 
    # Let's use a soft-clamp or just let it be, but handle non-finite outputs.
    # A common trick: exp(-abs(p7) * x^2) forces the Gaussian to be a bump, not a valley.
    # But maybe the function needs a valley? 
    # Given x in [-2, 2], x^2 is in [0, 4].
    # If p7 is -10, exp(40) is ~2e17. This might cause numerical issues in BFGS gradients.
    # Let's constrain the width parameter to be positive by using p7*p7 or abs(p7) in the exponent?
    # If we use p7*p7, we lose the sign information, but Gaussian width is physically positive.
    # Let's use exp(-(p7*p7) * x**2) to ensure stability. The amplitude p6 can handle the sign.
    width_sq = p7 * p7
    term_gauss = p6 * np.exp(-width_sq * x**2)
    
    term_lin = p8 * x + p9
    
    y = term_sin + term_cos + term_gauss + term_lin
    
    # Check for non-finite values
    if not np.all(np.isfinite(y)):
        # If overflow occurred, return a large penalty value to guide optimizer away
        # Or just return the values, but the prompt says non-finite error costs 1e30 during fitting.
        # It's better to return finite values if possible, or let the evaluator handle it.
        # However, if we return inf, the evaluator handles it. 
        # But to be safe for the "return best result" logic if we were doing internal search (we aren't), 
        # we just return the computed y.
        
        pass
    
    return y
