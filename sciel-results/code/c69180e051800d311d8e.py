import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the function using a Piecewise Linear (PWL) basis with 10 fixed knots.
    The model is linear in the parameters (values at the knots).
    This ensures the optimization landscape is convex, allowing BFGS to find the
    global minimum reliably, unlike non-linear basis functions (like RBFs with
    variable centers) which can get stuck in local minima.
    """
    # 10 knots uniformly distributed in [-2, 2]
    knots = np.linspace(-2.0, 2.0, 10)
    
    # Calculate the PWL basis values for each point in x
    # For each x, we find the segment it falls into and interpolate linearly.
    # Since the model is sum(params[i] * basis[i]), and basis[i] is the hat function,
    # we can compute the prediction directly.
    
    # Efficient computation:
    # For each x, find the index of the right knot.
    # Use np.searchsorted to find the interval.
    
    # Handle boundaries
    # If x < knots[0], use knots[0] logic (or clamp)
    # If x > knots[-1], use knots[-1] logic
    
    # Let's use np.interp for simplicity and robustness, 
    # but we need to return a function of params.
    # Actually, np.interp takes xp (knots) and fp (params).
    # The definition of np.interp is exactly linear interpolation between knots.
    # This matches the PWL model perfectly.
    
    # np.interp(x, knots, params) returns the linear interpolation of params at knots x.
    # This is equivalent to sum(params[i] * hat_i(x)).
    
    return np.interp(x, knots, params)
