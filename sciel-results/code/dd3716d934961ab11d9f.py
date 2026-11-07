"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure inputs are float64 arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    n = len(x)
    
    # Strategy:
    # Use a fixed, orthogonal polynomial basis (Legendre polynomials).
    # The domain is [0.5, 2]. We map this to [-1, 1] for standard Legendre polynomials.
    # Center = 1.25, Half-width = 0.75.
    # u = (x - 1.25) / 0.75
    
    # Legendre Polynomials P_k(u):
    # P_0(u) = 1
    # P_1(u) = u
    # P_2(u) = 0.5 * (3u^2 - 1)
    # P_3(u) = 0.5 * (5u^3 - 3u)
    
    # We have 10 parameters.
    # A full tensor product basis up to degree 3 has 4^3 = 64 terms, which is too many.
    # A symmetric/low-order basis is better.
    # Let's select 10 specific monomials in the scaled variables u, v, w.
    # To improve conditioning, we use the orthogonal polynomials directly.
    
    # Basis functions:
    # 0: 1
    # 1: u
    # 2: v
    # 3: w
    # 4: u*v
    # 5: u*w
    # 6: v*w
    # 7: u^2
    # 8: v^2
    # 9: w^2
    
    # Note: Using raw monomials u*v etc. is not orthogonal, but it is much better conditioned
    # than the original raw x,y,z because the range is [-1,1].
    # However, for maximum robustness against BFGS convergence issues, 
    # using the actual orthogonal polynomials P_k is superior.
    # But P_2(u)P_2(v) etc. are not in the list above.
    # Let's stick to the 10 terms above but calculated using the orthogonal polynomials where possible?
    # Actually, for a linear model, the choice of basis just changes the parameter values, 
    # not the function space spanned (if the basis is complete for that subspace).
    # The subspace spanned by {1, u, v, w, uv, uw, vw, u^2, v^2, w^2} 
    # is the same as {1, P_0, P_1, P_1, P_1, P_1, P_1, P_2, P_2, P_2} roughly.
    # The key is that the condition number of the design matrix is low.
    
    # Let's verify the condition number improvement.
    # Original x in [0.5, 2]. x^2 in [0.25, 4]. 
    # Scaled u in [-1, 1]. u^2 in [0, 1].
    # This is a significant improvement in scale.
    
    # Even better: Use the orthogonal polynomials directly as the basis vectors.
    # Let's define the 10 basis functions using P_k(u), P_k(v), P_k(w).
    # 0: 1
    # 1: P_1(u) = u
    # 2: P_1(v) = v
    # 3: P_1(w) = w
    # 4: P_1(u)*P_1(v) = u*v
    # 5: P_1(u)*P_1(w) = u*w
    # 6: P_1(v)*P_1(w) = v*w
    # 7: P_2(u) = 0.5*(3u^2 - 1)
    # 8: P_2(v) = 0.5*(3v^2 - 1)
    # 9: P_2(w) = 0.5*(3w^2 - 1)
    
    # This basis spans the same space as {1, u, v, w, uv, uw, vw, u^2, v^2, w^2}
    # (since P_2 is a linear combo of u^2 and 1).
    # But the vectors P_2(u) are orthogonal to 1 and u (in the continuous L2 sense over [-1,1]).
    # This orthogonality makes the normal matrix (X^T X) nearly diagonal, 
    # which is ideal for BFGS (or any optimizer) to find the exact solution in one step 
    # (if it were least squares) or converge very quickly.
    
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    # Compute Basis
    # P_0 = 1
    # P_1 = u, v, w
    # P_2 = 0.5 * (3*u^2 - 1)
    
    b0 = np.ones(n)
    b1 = u
    b2 = v
    b3 = w
    b4 = u * v
    b5 = u * w
    b6 = v * w
    b7 = 0.5 * (3 * u**2 - 1)
    b8 = 0.5 * (3 * v**2 - 1)
    b9 = 0.5 * (3 * w**2 - 1)
    
    # Stack into matrix (n, 10)
    # Note: We are not using P_3 terms because we only have 10 params.
    # This limits us to quadratic approximations.
    # If the function is cubic, this will be biased.
    # However, the current RBF approach (score 2.22) is likely overfitting or struggling with optimization.
    # A robust quadratic fit might be better than a struggling cubic+ fit.
    # Let's check if we can include P_3.
    # If we drop one quadratic term, we can add P_3(u), P_3(v), P_3(w)?
    # P_3(u) = 0.5 * (5u^3 - 3u)
    # The space spanned by {1, u, v, w, uv, uw, vw, u^2, v^2, w^2} is 10-dim.
    # If we replace b7 (u^2) with P_3(u), we lose the pure quadratic u^2 term?
    # No, P_3(u) is independent of u^2. But we lose the ability to fit a pure u^2 shape well if it's not in the basis.
    # Actually, the set {1, u, v, w, uv, uw, vw, u^2, v^2, w^2} is a standard quadratic basis.
    # It is better to keep this complete quadratic basis with good scaling/orthogonality.
    
    # Let's stick to the 10 terms defined above.
    
    design_matrix = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    # Safety check for finite values
    if not np.all(np.isfinite(design_matrix)):
        design_matrix = np.nan_to_num(design_matrix)
        
    pred = design_matrix @ params
    
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred)
        
    return pred

# EVOLVE-BLOCK-END
