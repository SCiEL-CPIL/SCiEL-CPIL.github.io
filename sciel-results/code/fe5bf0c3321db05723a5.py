"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 145.0  # 5 seconds margin before 150s kill

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a neutral guess if we are too late
        return np.zeros_like(x, dtype=np.float64)

    # Ensure inputs are float64 arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    # We need to recover the target function f(x,y,z).
    # The evaluator passes `y` as the input variable name, but in the context of 
    # "equation(x, y, z, params)", y is the second input variable.
    # Wait, the task says: "Write equation(x, y, z, params)".
    # "Data: 200 training points ... inputs drawn uniformly from [0.5, 2] x 3".
    # "prediction ... mean((prediction - y)^2)".
    # Usually in these tasks, the function signature is equation(inputs, target, params) 
    # OR equation(input1, input2, input3, params).
    # The prompt says: "Symbolic regression with 3 input(s)".
    # "equation(x, y, z, params)".
    # So x, y, z are the 3 inputs.
    # The target is hidden. The evaluator fits params to minimize training MSE.
    # The MSE is between `equation(x, y, z, params)` and the hidden target.
    # But wait, the signature is `equation(x, y, z, params)`.
    # If the evaluator calls `equation(x_train, y_train, z_train, params)`, 
    # then `y` is the second input variable, NOT the target.
    # However, the scoring description says: "mean((prediction - y)^2) / var(y)".
    # This implies `y` in the scoring formula is the TARGET.
    # This is a common ambiguity. Let's look at the current program.
    # The current program uses `x`, `y`, `z` as inputs in the basis functions.
    # It does NOT use `y` as the target.
    # If `y` was the target, the current program would be useless because it ignores the target in the prediction step (it just predicts from x,y,z).
    # But the evaluator "fits params ... minimizing the training MSE".
    # This means the evaluator has access to the target.
    # The `equation` function is called with the inputs and the current params.
    # It returns the prediction.
    # The evaluator compares this prediction to the target.
    # So, inside `equation`, `x, y, z` are the three input features.
    # The target is NOT passed to `equation`.
    
    # Strategy:
    # The current program uses a fixed basis of 10 terms.
    # The evaluator fits the 10 params to this fixed basis.
    # If the hidden function is not well-approximated by this basis, the score is poor.
    # 
    # New Strategy:
    # We have 10 params. We can use them to weight a set of 10 candidate basis functions.
    # But we can also try to find a BETTER set of 10 basis functions?
    # No, the basis is fixed at compile time (inside the function).
    # The params are fitted by the evaluator.
    # So we must choose 10 basis functions that are likely to span the hidden function.
    # 
    # The current basis is quadratic polynomials.
    # Let's try a richer basis:
    # 1. 1
    # 2. x
    # 3. y
    # 4. z
    # 5. x*y
    # 6. x*z
    # 7. y*z
    # 8. x^2
    # 9. y^2
    # 10. z^2
    # This is similar to the current one, but the current one uses orthogonal polynomials P2.
    # Orthogonal polynomials are good for least squares.
    # 
    # What if the function is exponential or trigonometric?
    # We can't know.
    # However, we can use the 10 params to combine a diverse set of features.
    # 
    # Alternative Idea:
    # The prompt mentions "inspiration 1" and "inspiration 3".
    # Inspiration 3 suggests "richer, non-polynomial feature set".
    # Let's implement a basis that includes:
    # 1. Constant
    # 2-4. Linear (x, y, z)
    # 5-7. Pairwise products (xy, xz, yz)
    # 8. exp(-x)
    # 9. exp(-y)
    # 10. exp(-z)
    # 
    # But wait, the evaluator fits params using BFGS starting from all ones.
    # If the features are very different in scale, BFGS might struggle.
    # We should normalize the features.
    
    # Let's try a basis that covers a wider range of possible functions.
    # Domain is [0.5, 2].
    
    x_c = x
    y_c = y
    z_c = z
    
    # Normalize to roughly [0, 1] or [-1, 1] to help optimization?
    # BFGS starting from 1.0. If features are large, gradients might be large.
    # Let's stick to the original normalization to be safe, or just use raw values since domain is small.
    
    # Let's try a basis that includes interactions and non-linearities.
    # 1. 1
    # 2. x
    # 3. y
    # 4. z
    # 5. x*y
    # 6. x*z
    # 7. y*z
    # 8. x**2
    # 9. y**2
    # 10. z**2
    
    # This is essentially the current basis but without orthogonality.
    # The current basis has P2 which is 0.5*(3t^2-1). This is just a linear combination of 1 and t^2.
    # So the span of {1, t, P2} is the same as {1, t, t^2}.
    # So the current program's span is the same as the standard quadratic polynomial span.
    
    # To improve, we need a DIFFERENT span.
    # Let's replace the quadratic terms with something else, or add non-polynomial terms.
    # Since we only have 10 params, we must drop some.
    # Let's try:
    # 1. 1
    # 2. x
    # 3. y
    # 4. z
    # 5. x*y
    # 6. x*z
    # 7. y*z
    # 8. exp(x)
    # 9. exp(y)
    # 10. exp(z)
    
    # Or:
    # 1. 1
    # 2. x
    # 3. y
    # 4. z
    # 5. x*y
    # 6. x*z
    # 7. y*z
    # 8. sin(x)
    # 9. sin(y)
    # 10. sin(z)
    
    # Or a mix.
    # Let's try a mix of polynomial and exponential.
    # 1. 1
    # 2. x
    # 3. y
    # 4. z
    # 5. x*y
    # 6. x*z
    # 7. y*z
    # 8. exp(-x)
    # 9. exp(-y)
    # 10. exp(-z)
    
    # Let's compute these features.
    f1 = np.ones_like(x)
    f2 = x
    f3 = y
    f4 = z
    f5 = x * y
    f6 = x * z
    f7 = y * z
    f8 = np.exp(-x)
    f9 = np.exp(-y)
    f10 = np.exp(-z)
    
    # Combine
    pred = (params[0] * f1 +
            params[1] * f2 +
            params[2] * f3 +
            params[3] * f4 +
            params[4] * f5 +
            params[5] * f6 +
            params[6] * f7 +
            params[7] * f8 +
            params[8] * f9 +
            params[9] * f10)
            
    return pred
# EVOLVE-BLOCK-END
