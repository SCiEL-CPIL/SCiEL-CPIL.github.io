import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a 10-parameter model with a rich basis:
    - Polynomial terms: 1, x, x^2, x^3, x^4
    - Trigonometric terms: sin(x), cos(x), sin(2x), cos(2x)
    - Adaptive Gaussian term: exp(-(x - c)^2 / (2*s^2))
    
    The last two parameters control the center c and width s of the
    Gaussian term. The first 8 parameters are linear coefficients for
    the fixed basis functions.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure params has exactly 10 elements
    if params.shape[0] != 10:
        return np.zeros_like(x)
        
    # Extract parameters
    c_poly = params[:5]       # Coeffs for 1, x, x^2, x^3, x^4
    c_trig = params[5:9]      # Coeffs for sin(x), cos(x), sin(2x), cos(2x)
    g_center = params[9]      # Center for Gaussian
    g_width = params[8]       # Width for Gaussian (reusing index 8)
    
    # Actually, let's be clearer:
    # params[0:5] -> poly coeffs
    # params[5:9] -> trig coeffs
    # params[9]   -> gaussian center
    # We need another param for gaussian width. 
    # The contract says 10 params. 
    # Let's use:
    # 0: 1
    # 1: x
    # 2: x^2
    # 3: x^3
    # 4: x^4
    # 5: sin(x)
    # 6: cos(x)
    # 7: sin(2x)
    # 8: cos(2x)
    # 9: tanh(x) ? No, the inspiration said Gaussian.
    
    # Let's stick to the structure that worked (score 4930) but maybe refine it.
    # The previous good score was 4930 with:
    # 1, x, x^2, x^3, sin(x), cos(x), sin(2x), cos(2x), tanh(s*x), coeff
    # Wait, the current program in the prompt is the one with score 3273535 (bad).
    # The inspiration 1 mentions a Gaussian RBF with score 4930.
    # The "Knowledge" section says: "switch from a single Gaussian-RBF family to a composite Poly+Trig+Tanh basis... jumps 236->4930".
    # So the 4930 score is from the Composite Poly+Trig+Tanh basis.
    # The current program in the prompt IS that composite basis, but it has a score of 3273535.
    # Why? Maybe the BFGS optimizer is getting stuck or the basis is ill-conditioned?
    # Or maybe the specific arrangement of parameters is bad for BFGS starting from all ones.
    
    # Let's try a different strategy: A purely linear model with a very rich, fixed basis.
    # No non-linear parameters (no 's' in tanh(s*x) or 'c' in Gaussian).
    # This makes the problem convex and easy for BFGS (which is for non-linear, but works for linear too).
    # Actually, if it's linear in params, we can just use least squares, but the evaluator uses BFGS.
    # BFGS on a quadratic (MSE of linear model) should converge quickly.
    
    # Let's use a basis that covers a wide range of functions:
    # Polynomials up to degree 8? Might be overfitting or ill-conditioned.
    # Let's try:
    # 1, x, x^2, x^3, x^4, x^5, x^6, x^7, x^8, x^9
    # This is a 10th order polynomial.
    # With 200 points, this might fit well if the function is smooth.
    # But if the function is periodic, polynomials are bad.
    
    # Let's mix:
    # 1, x, x^2, x^3, sin(x), cos(x), sin(2x), cos(2x), sin(3x), cos(3x)
    # This covers low-order polynomials and low-frequency periodicities.
    
    # Or, let's look at the "Gaussian RBF" inspiration again.
    # "The current program uses a fixed Gaussian RBF model with 10 linear weights... score 4930".
    # This implies a model: sum_{i=1}^{10} w_i * exp(-(x - c_i)^2 / (2*s_i^2))
    # where c_i and s_i are FIXED.
    # This is linear in w.
    # If we fix the centers and widths to a grid in [-2, 2], we get a universal approximator (locally).
    # Since it's linear in params, BFGS will find the global minimum (convex problem).
    # This avoids local minima of non-linear parameter fitting.
    
    # Let's implement this:
    # 10 fixed Gaussian RBFs.
    # Centers: evenly spaced in [-2, 2].
    # Widths: fixed, e.g., 0.5 or 1.0.
    
    centers = np.linspace(-2, 2, 10)
    width = 0.5  # Fixed width
    
    # Compute RBF values
    # x is (N,), centers is (10,)
    # diff is (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-(diff ** 2) / (2 * width ** 2))
    
    # Prediction is dot product of rbf_values (N, 10) and params (10,)
    prediction = rbf_values @ params
    
    return prediction
