import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Combined basis: mix of polynomial, trigonometric, and interaction terms
    # This aims to capture both smooth polynomial behavior and potential oscillatory components.
    
    # Precompute terms
    x2 = x * x
    y2 = y * y
    xy = x * y
    
    # Trigonometric
    sin_x = np.sin(x)
    sin_y = np.sin(y)
    cos_x = np.cos(x)
    cos_y = np.cos(y)
    
    # Mixed interaction terms (polynomial-trig)
    # These can capture phase-shifted oscillations modulated by position
    sin_x_y = sin_x * y
    cos_x_y = cos_x * y
    sin_y_x = sin_y * x
    cos_y_x = cos_y * x
    
    # Basis selection:
    # We have 10 params. Let's pick a diverse set.
    # 0: 1
    # 1: x
    # 2: y
    # 3: x*y
    # 4: x^2
    # 5: y^2
    # 6: sin(x)
    # 7: sin(y)
    # 8: cos(x)
    # 9: cos(y)
    
    # Alternative: Include higher order poly?
    # Inspiration 1 & 2 scored 7.02 (worse than 5.90).
    # Inspiration 3 scored 5.90 (same as current).
    # Current is essentially Inspiration 3.
    
    # The stagnation directive says: Make a STRUCTURAL change.
    # Current: 1, x, y, xy, x2, y2, sin(x), sin(y), cos(x), cos(y)
    # Insp 1: 1, x, y, xy, x2, y2, x3, y3, x2y, xy2 (Pure Poly) -> Score 7.02
    # Insp 2: 1, x, y, x2, xy, y2, x3, x2y, xy2, y3 (Pure Poly) -> Score 7.02
    
    # Since pure poly was worse, and current mixed is 5.90, maybe the function is NOT highly oscillatory?
    # Or maybe it's a specific type of function.
    # If it were a polynomial, pure poly would likely be better or equal.
    # If it were trig, current is good.
    
    # What if we use a different structural representation?
    # The evaluator uses BFGS. The landscape might be difficult.
    # However, we can only return the equation. We cannot change the optimizer.
    
    # Let's try a basis that includes radial distance or angle?
    # r = sqrt(x^2 + y^2)
    # theta = arctan2(y, x)
    # This is a structural change from Cartesian product basis to Polar.
    # Many 2D functions (e.g., r^2, sin(r), cos(theta)) are simpler in polar.
    
    r = np.sqrt(x2 + y2)
    # Avoid division by zero for theta, but arctan2 handles 0,0 safely (returns 0)
    theta = np.arctan2(y, x)
    
    # Basis:
    # 0: 1
    # 1: r
    # 2: r^2
    # 3: sin(r)
    # 4: cos(r)
    # 5: sin(theta)  -> y/r, problematic at 0
    # 6: cos(theta)  -> x/r, problematic at 0
    # 7: x
    # 8: y
    # 9: x*y
    
    # Using sin(theta) and cos(theta) directly is unstable at r=0.
    # But sin(theta) = y/r, cos(theta) = x/r.
    # We can use x, y instead of r*cos(theta), r*sin(theta).
    # So:
    # 0: 1
    # 1: r
    # 2: r^2
    # 3: sin(r)
    # 4: cos(r)
    # 5: x
    # 6: y
    # 7: x^2
    # 8: y^2
    # 9: x*y
    
    # This mixes radial and cartesian.
    
    pred = (params[0] * np.ones_like(x) + 
            params[1] * r + 
            params[2] * (r * r) + 
            params[3] * np.sin(r) + 
            params[4] * np.cos(r) + 
            params[5] * x + 
            params[6] * y + 
            params[7] * x2 + 
            params[8] * y2 + 
            params[9] * xy)
            
    return pred
