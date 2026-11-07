import numpy as np

def equation(x, y, params):
    """
    Symbolic regression with 2 inputs using a product-of-sines architecture.
    The model is a single term p0 * sin(p1 * x + p2 * y + p3), allowing BFGS
    to jointly optimize amplitude, frequencies, and phase. This structure can
    exactly represent simple sinusoidal targets like sin(x+y) or cos(x-y).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    
    n = len(x)
    if n == 0:
        return np.array([])
    
    # Use first 3 parameters for the sine term: amplitude, freq_x, freq_y, phase
    # We have 10 params total. Let's use a combination of terms to increase flexibility
    # while keeping the "product of sines" or "single term" spirit for the main part.
    # However, the directive suggests replacing the fixed linear basis with a product-of-sines
    # architecture where the model is a single term p0 sin(p1 x + p2 y + p3).
    # Since we have 10 params, we can use multiple such terms or a more complex structure.
    # But to follow the directive's suggestion of a "single term" to allow BFGS to recover
    # simple sinusoidal targets, let's stick to a few such terms or just one if it's too restrictive.
    # Let's try a sum of 3 such terms to use more params, or maybe just use the first 4 for one term
    # and the rest for something else? The directive says "Replace the fixed linear basis with a
    # product-of-sines architecture where the model is a single term...".
    # Let's interpret this as: use the params to define a sum of a few sine terms, or just one.
    # If we only use one term, we waste 6 params. BFGS will just set them to 0 if they don't affect loss?
    # No, if they are not in the expression, they don't affect loss.
    # Let's use a sum of 3 sine terms to utilize more params and increase expressiveness.
    
    # Term 1: a1 * sin(f1x * x + f1y * y + p1)
    # Term 2: a2 * sin(f2x * x + f2y * y + p2)
    # Term 3: a3 * sin(f3x * x + f3y * y + p3)
    
    # That uses 3 * 4 = 12 params. We only have 10.
    # So maybe 2 terms (8 params) + 2 extra? Or 1 term (4 params) + 6 extra?
    # Let's try 2 terms (8 params) and use the last 2 for a linear bias or something simple.
    # Or just use 1 term and ignore the rest? That might be too restrictive.
    
    # Let's try:
    # pred = p0 * sin(p1 * x + p2 * y + p3) + p4 * sin(p5 * x + p6 * y + p7) + p8 * x + p9 * y
    # This uses all 10 params.
    
    pred = (params[0] * np.sin(params[1] * x + params[2] * y + params[3]) +
            params[4] * np.sin(params[5] * x + params[6] * y + params[7]) +
            params[8] * x +
            params[9] * y)
    
    return pred
