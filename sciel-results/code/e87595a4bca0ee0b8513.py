"""
Strategy: Hybrid Polynomial + Tunable Sigmoid Basis.

This approach combines a global linear component with local, sharp-featured
sigmoidal basis functions. Unlike Chebyshev polynomials (which are global and
struggle with localized features) or Gaussians (which are smooth bumps),
sigmoids can approximate step functions and sharp transitions, making them
effective for functions with high curvature or near-discontinuous behavior.

Params layout (length 10):
- params[0]: Constant term (bias)
- params[1]: Linear term coefficient
- params[2:6]: Centers of 4 sigmoid functions
- params[6:10]: Slopes (steepness) of 4 sigmoid functions

The function is:
y = p0 + p1*x + sum_{i=1}^{4} w_i * sigmoid(s_i * (x - c_i))

To keep the parameter count at 10 and allow BFGS to optimize effectively,
we fix the weights w_i to 1.0 for simplicity, or better, we can use the
centers and slopes as the primary degrees of freedom. However, with only
10 params, we can't have free weights for 4 sigmoids plus linear terms.
A better allocation for 10 params to maximize flexibility:
- 1 param: Global bias
- 1 param: Global linear slope
- 4 params: Sigmoid centers
- 4 params: Sigmoid slopes
We fix the sigmoid amplitudes to 1.0. This forces the model to find the
right locations and sharpness of transitions. If the target requires
different amplitudes, the linear term and the interaction of sigmoids might
partially compensate, but this is a risk. 

Alternative allocation:
Use 3 sigmoids.
- 1 bias
- 1 linear
- 3 centers
- 3 slopes
- 2 weights for the first two sigmoids? No, that's messy.

Let's stick to the most robust interpretation:
The prompt says "recovers an unknown function".
If I use 5 sigmoids:
- 5 centers
- 5 slopes
Total 10 params. No linear term.
This might fail on linear targets.

Let's try:
- 1 bias
- 1 linear
- 4 centers
- 4 slopes
Total 10.
We fix sigmoid weights to 1.0.
y = p0 + p1*x + sum( sigmoid(p_{6+i} * (x - p_{2+i})) )

This assumes the "height" of the step is 1. If the target has steps of height 2,
it can't represent it well unless two sigmoids overlap or the linear term helps.
To be safer, let's use a basis that is more flexible.
How about a combination of Chebyshev and one strong local feature?
No, the directive asks for a materially different family.

Let's refine the Sigmoid approach.
Instead of fixed weights, we can interpret the parameters differently.
What if we use a **Piecewise Linear** approximation via max/min or abs?
y = sum( a_i * |x - c_i| + b_i ) + c0*x + c1
This requires 2 params per segment.
With 10 params:
- 1 global bias
- 1 global slope
- 4 segments: 4 centers, 4 slopes?
y = p0 + p1*x + sum( p_{4+i} * |x - p_{2+i}| )
Here p_{4+i} is the slope of the V-shape.
This is a standard "ReLU" or "Hinge" basis. It's excellent for approximating
piecewise linear functions and smooth functions with high curvature.
It is materially different from Chebyshev (polynomial) and Gaussians (smooth bumps).
Sigmoid is also good, but Hinge/ReLU is often more efficient for symbolic regression
tasks that might involve abs() or piecewise definitions.

Let's go with the **Hinge Basis (Piecewise Linear)** strategy.
Params:
- p0: bias
- p1: global linear slope
- p2, p3: center, slope for hinge 1
- p4, p5: center, slope for hinge 2
- p6, p7: center, slope for hinge 3
- p8, p9: center, slope for hinge 4

y = p0 + p1*x + p2*|x-p3| + p4*|x-p5| + p6*|x-p7| + p8*|x-p9|

This allows the function to change slope abruptly at the centers, which is
very powerful for approximating many "unknown" functions that might be
piecewise or have sharp features. BFGS can handle the non-differentiability
of abs() reasonably well (subgradients), or we can use a smooth approximation
like sqrt(x^2 + eps) to ensure smoothness for BFGS.
Since BFGS assumes smoothness, using `np.sqrt(x**2 + 1e-6)` is safer than `np.abs(x)`.
"""

import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a Hybrid Basis:
    A global linear term plus four tunable hinge (absolute value) functions.
    
    y(x) = p0 + p1*x + sum_{i=1}^{4} w_i * sqrt((x - c_i)^2 + eps)
    
    Where:
    p0 = params[0] (bias)
    p1 = params[1] (linear slope)
    For i in 1..4:
      c_i = params[2 + 2*(i-1)] (center)
      w_i = params[3 + 2*(i-1)] (slope/weight)
    """
    
    # Extract parameters
    p0 = params[0]
    p1 = params[1]
    
    # Initialize result with global linear trend
    result = p0 + p1 * x
    
    # Add 4 hinge functions
    # Using smooth approximation of absolute value: sqrt(x^2 + eps)
    # eps is small to maintain differentiability for BFGS
    eps = 1e-6
    
    for i in range(4):
        center = params[2 + 2*i]
        weight = params[3 + 2*i]
        
        # Hinge component: weight * |x - center|
        # Smoothed: weight * sqrt((x - center)^2 + eps)
        diff = x - center
        hinge_val = np.sqrt(diff * diff + eps)
        
        result += weight * hinge_val
        
    return result
