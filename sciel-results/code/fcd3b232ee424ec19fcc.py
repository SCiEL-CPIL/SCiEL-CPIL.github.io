"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # We have 10 parameters. The parent (linear model) scored 1.0.
    # The current program (quadratic polynomial) scored 7.39 (worse).
    # This suggests the true function is likely linear, or close to it,
    # and the extra flexibility of the quadratic model caused overfitting
    # or optimization issues with BFGS starting from all ones.
    
    # Strategy:
    # 1. Keep the core linear terms which worked well (score 1.0).
    # 2. Add a small number of higher-order terms to capture potential non-linearity,
    #    but ensure they are well-conditioned and don't hinder the fit.
    # 3. Use a mix of linear and simple non-linear terms.
    #    
    # Let's try a model that is primarily linear but allows for some curvature.
    # Since BFGS starts at all ones, terms that are very large or very small
    # relative to the linear terms might cause issues.
    # Inputs are in [0.5, 2].
    # x, y, z are O(1).
    # x^2 is O(1).
    # x*y is O(1).
    
    # Let's try a basis that includes:
    # p0: bias
    # p1: x
    # p2: y
    # p3: z
    # p4: x^2
    # p5: y^2
    # p6: z^2
    # p7: x*y
    # p8: x*z
    # p9: y*z
    
    # Wait, the current program IS this quadratic model and it scored 7.39.
    # Why did it fail? 
    # 1. BFGS might have gotten stuck in a local minimum.
    # 2. The true function might be linear, and the quadratic terms introduced
    #    noise/overfitting on the training set, leading to poor generalization.
    # 3. Or, the quadratic terms made the Hessian ill-conditioned.
    
    # The inspiration (linear model) scored 1.0. This is a strong baseline.
    # Score 1.0 means normalized MSE is 1.0. 
    # If the true function is linear, the linear model should get close to 0.
    # A score of 1.0 implies the model explains 0% of the variance? 
    # No, normalized MSE = 1.0 means the model is as good as predicting the mean.
    # If the linear model gets 1.0, it's not fitting the data well at all.
    
    # Let's re-read the score definition.
    # "normalized MSE on the test points, mean((prediction - y)^2) / var(y)"
    # If the model predicts the mean of y, the MSE is var(y), so the ratio is 1.
    # So the linear model is failing to fit the data.
    
    # If the linear model fails, the function is likely non-linear.
    # The quadratic model scored 7.39, which is worse than the linear model.
    # This is strange. Usually, adding terms should not make things worse if the
    # optimizer is good. Unless the optimizer failed.
    
    # BFGS starting from all ones.
    # If the true function is, say, sin(x) * cos(y) * z, linear and quadratic
    # models will both fail.
    
    # Let's look at the inputs: [0.5, 2].
    # Maybe the function involves exponentials or other non-polynomial terms?
    # But we only have 10 params.
    
    # Let's try a different basis.
    # Perhaps the function is separable? f(x) + g(y) + h(z)?
    # Or multiplicative? f(x)*g(y)*h(z)?
    
    # Let's try a model that can represent both additive and multiplicative structures
    # by using log or exp transformations?
    # But we don't know the signs.
    
    # Let's try a richer basis with the 10 parameters.
    # Maybe the issue with the quadratic model was that BFGS couldn't handle
    # the scale differences or the non-convexity.
    
    # Let's try to make the model more robust.
    # How about using a mix of linear, quadratic, and some non-linear terms?
    
    # p0: 1
    # p1: x
    # p2: y
    # p3: z
    # p4: x*y
    # p5: x*z
    # p6: y*z
    # p7: x^2
    # p8: y^2
    # p9: z^2
    
    # This is the same as before.
    
    # What if we try:
    # p0: 1
    # p1: x
    # p2: y
    # p3: z
    # p4: x**2
    # p5: y**2
    # p6: z**2
    # p7: x*y
    # p8: x*z
    # p9: y*z
    
    # This is exactly what the current program does.
    
    # Why did it score 7.39?
    # Maybe the true function is something like exp(x) or sin(x)?
    # If the function is exp(x), a polynomial approximation might be poor
    # over [0.5, 2].
    
    # Let's try a basis that includes some non-polynomial terms.
    # But we don't know which ones.
    
    # Alternative idea:
    # The linear model scored 1.0.
    # The quadratic model scored 7.39.
    # This suggests that the quadratic model is actually *worse* than the linear model.
    # This can happen if the quadratic terms have large coefficients that are
    # poorly estimated, leading to high variance.
    
    # Let's try a model that is a hybrid:
    # Use linear terms for the main structure, and add a small correction term.
    # But we don't know the correction.
    
    # Let's try a different allocation of the 10 parameters.
    # Maybe the function is of the form:
    # a*x + b*y + c*z + d*x^2 + e*y^2 + f*z^2 + g*x*y + h*x*z + i*y*z + j
    # This is the full quadratic.
    
    # What if the function is:
    # a*x + b*y + c*z + d*sin(x) + e*sin(y) + f*sin(z) + g*x*y + h*x*z + i*y*z + j
    # Mixing polynomial and trigonometric might help if the function is periodic.
    
    # Or:
    # a*x + b*y + c*z + d*exp(x) + e*exp(y) + f*exp(z) + g*x*y + h*x*z + i*y*z + j
    
    # Let's try a mix of polynomial and exponential.
    # exp(x) for x in [0.5, 2] is [1.65, 7.39].
    # x is [0.5, 2].
    # The scales are different.
    
    # Let's try:
    # p0: 1
    # p1: x
    # p2: y
    # p3: z
    # p4: x**2
    # p5: y**2
    # p6: z**2
    # p7: x*y
    # p8: x*z
    # p9: y*z
    
    # And maybe we should normalize the inputs to improve conditioning?
    # x_norm = (x - 1.25) / 0.75
    # This centers the inputs around 0.
    # This might help BFGS converge better.
    
    # Let's try normalizing the inputs.
    # x_c = x - 1.25
    # y_c = y - 1.25
    # z_c = z - 1.25
    
    # Then use x_c, y_c, z_c in the quadratic terms.
    # This removes the large bias from the squared terms.
    
    # Let's test this hypothesis.
    # If the true function is linear, the centered quadratic terms should have
    # near-zero coefficients, and the model should behave like the linear model.
    # If the true function is non-linear, the centered terms might fit better.
    
    # Let's implement this.
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    
    # Center the inputs
    x_c = x - 1.25
    y_c = y - 1.25
    z_c = z - 1.25
    
    # Build features
    # p0: 1
    # p1: x_c
    # p2: y_c
    # p3: z_c
    # p4: x_c**2
    # p5: y_c**2
    # p6: z_c**2
    # p7: x_c*y_c
    # p8: x_c*z_c
    # p9: y_c*z_c
    
    p = params
    pred = (p[0] * np.ones_like(x) +
            p[1] * x_c +
            p[2] * y_c +
            p[3] * z_c +
            p[4] * x_c**2 +
            p[5] * y_c**2 +
            p[6] * z_c**2 +
            p[7] * x_c * y_c +
            p[8] * x_c * z_c +
            p[9] * y_c * z_c)
    
    return pred
# EVOLVE-BLOCK-END
