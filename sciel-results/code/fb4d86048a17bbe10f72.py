import numpy as np
import scipy.optimize as opt
import time
from scipy.special import lpmv

def _get_quadrature():
    """
    Generate quadrature points for the sphere.
    70-point Gauss-Legendre for mu = cos(theta)
    130 equally spaced points for phi
    """
    # Gauss-Legendre for mu in [-1, 1]
    mu, w_mu = np.polynomial.legendre.leggauss(70)
    
    # Equally spaced phi in [0, 2*pi)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    w_phi = np.full(n_phi, 2 * np.pi / n_phi)
    
    # Meshgrid
    MU, PHI = np.meshgrid(mu, phi, indexing='ij')
    
    # Weights: w_mu * w_phi
    # Note: The integral is \int_0^{2\pi} d\phi \int_{-1}^1 d\mu ...
    # So the total weight is w_mu[i] * w_phi[j]
    W = np.outer(w_mu, w_phi)
    
    return MU, PHI, W

def _compute_f(c, MU, PHI):
    """
    Compute f = sum c_m Y_m on the grid.
    c is ordered m = -10 .. 10.
    Y_m definition:
    Y_0 = N_0 P_10^0(cos theta)
    Y_m = sqrt(2) N_m P_10^m(cos theta) cos(m phi) for m > 0
    Y_m = sqrt(2) N_|m| P_10^|m|(cos theta) sin(|m| phi) for m < 0
    N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
    P_10^m is associated Legendre with Condon-Shortley phase.
    """
    k = 10
    f = np.zeros_like(MU)
    
    # Precompute factorials or use gammaln for stability, but k=10 is small
    # N_m = sqrt( (2k+1)/(4*pi) * (k-m)! / (k+m)! )
    # We can compute N_m on the fly or precompute.
    
    # Let's precompute N_m for m in 0..10
    N = np.zeros(11)
    for m in range(11):
        # (2*10+1)/(4*pi) * (10-m)! / (10+m)!
        # Using math.factorial is fine for small numbers
        num = 21 * np.math.factorial(10 - m)
        den = 4 * np.pi * np.math.factorial(10 + m)
        N[m] = np.sqrt(num / den)
        
    # P_10^m(mu)
    # scipy.special.lpmv(m, l, x) returns P_l^m(x) with Condon-Shortley phase (-1)^m
    # The definition in the prompt says "with the Condon-Shortley phase", so lpmv is correct.
    
    for idx, m in enumerate(range(-10, 11)):
        c_m = c[idx]
        if c_m == 0:
            continue
            
        abs_m = abs(m)
        P_val = lpmv(abs_m, 10, MU)
        
        if m == 0:
            Y_val = N[0] * P_val
        elif m > 0:
            Y_val = np.sqrt(2) * N[m] * P_val * np.cos(m * PHI)
        else:
            Y_val = np.sqrt(2) * N[abs_m] * P_val * np.sin(abs_m * PHI)
            
        f += c_m * Y_val
        
    return f

def _objective_and_grad(c, MU, PHI, W):
    """
    Minimize J(c) = ||f||_4^4 subject to ||f||_2^2 = 1.
    We minimize J directly, but we must ensure normalization.
    It is often better to normalize c inside the objective or use a penalty.
    However, trust-constr can handle equality constraints.
    
    Let's define the function to return the objective value and gradient.
    f = Y c
    ||f||_4^4 = sum W * f^4
    ||f||_2^2 = sum W * f^2
    
    Gradient of J = 4 sum W * f^3 * df/dc_m
    df/dc_m = Y_m
    """
    f = _compute_f(c, MU, PHI)
    
    obj = np.sum(W * f**4)
    
    # Gradient
    # dJ/dc_m = 4 sum W f^3 Y_m
    # We need Y_m for all m. We can compute them in a loop or vectorize.
    # Since k=10, loop is fine.
    grad = np.zeros(21)
    
    # Precompute N again or pass it. Let's just recompute or store in closure if possible.
    # For efficiency in the loop, let's compute Y_m components.
    # Actually, computing _compute_f already loops. 
    # Let's modify _compute_f to also return the basis functions? 
    # Or just recompute. 21 * (70*130) operations is small.
    
    # Recompute basis for gradient
    N = np.zeros(11)
    for m in range(11):
        num = 21 * np.math.factorial(10 - m)
        den = 4 * np.pi * np.math.factorial(10 + m)
        N[m] = np.sqrt(num / den)
        
    for idx, m in enumerate(range(-10, 11)):
        abs_m = abs(m)
        P_val = lpmv(abs_m, 10, MU)
        
        if m == 0:
            Y_val = N[0] * P_val
        elif m > 0:
            Y_val = np.sqrt(2) * N[m] * P_val * np.cos(m * PHI)
        else:
            Y_val = np.sqrt(2) * N[abs_m] * P_val * np.sin(abs_m * PHI)
            
        grad[idx] = 4.0 * np.sum(W * f**3 * Y_val)
        
    return obj, grad

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 500  # Safety margin for 600s limit
    
    MU, PHI, W = _get_quadrature()
    
    # Initial guesses
    # 1. All ones
    # 2. Only m=0
    # 3. Random
    # 4. Symmetric patterns
    
    best_c = None
    best_val = np.inf
    
    # Helper to check validity and score
    def evaluate(c):
        # Normalize c to ensure ||f||_2 = 1 for consistent comparison?
        # The task asks for coefficients. The ratio is scale invariant.
        # The evaluator computes R = ||f||_4 / ||f||_2.
        # If we return c, the evaluator scales it implicitly by the ratio.
        # However, the validity check says max|c| < 1e-3 or > 1e5 is invalid.
        # So we should return a reasonably scaled vector.
        # Let's normalize the returned vector to have ||f||_2 = 1 (or similar).
        # If we minimize ||f||_4^4 subject to ||f||_2=1, the resulting c will have a specific scale.
        # Let's check the scale of the all-ones vector.
        
        f = _compute_f(c, MU, PHI)
        norm2_sq = np.sum(W * f**2)
        if norm2_sq == 0:
            return None, np.inf
            
        norm2 = np.sqrt(norm2_sq)
        norm4 = np.sum(W * f**4)
        norm4_val = norm4 ** 0.25
        
        R = norm4_val / norm2
        
        # Check validity of c itself
        if np.any(np.isnan(c)) or np.any(np.isinf(c)):
            return None, np.inf
        if np.max(np.abs(c)) < 1e-3 or np.max(np.abs(c)) > 1e5:
             # Try to rescale? 
             # If we rescale, R doesn't change.
             # We can return c * (1.0 / np.max(np.abs(c))) * 1.0 
             # But we need to be careful.
             pass
             
        return c, R

    # Optimization function
    def minimize_from_start(c0):
        # Normalize c0 to have ||f||_2 = 1 initially to help optimizer?
        # The constraint is ||f||_2^2 = 1.
        # Let's define the constraint function.
        
        def constraint(c):
            f = _compute_f(c, MU, PHI)
            return np.sum(W * f**2) - 1.0
        
        def constraint_jac(c):
            # Jacobian of constraint: 2 * Y^T W Y c
            # We can approximate or compute.
            # For trust-constr, analytic jac is better.
            # Let's compute it.
            f = _compute_f(c, MU, PHI) # Not needed for jac of constraint, only Y
            # Jac_i = 2 sum W Y_i Y_j c_j
            # This is 2 * (Y^T W Y) c
            # Let's build the matrix G = Y^T W Y (21x21)
            # Then jac = 2 G c
            
            # Build G
            # Y shape (21, 70*130)
            # W shape (70, 130) -> flatten to (70*130,)
            W_flat = W.flatten()
            
            Y_basis = np.zeros((21, 70*130))
            N = np.zeros(11)
            for m in range(11):
                num = 21 * np.math.factorial(10 - m)
                den = 4 * np.pi * np.math.factorial(10 + m)
                N[m] = np.sqrt(num / den)
                
            for idx, m in enumerate(range(-10, 11)):
                abs_m = abs(m)
                P_val = lpmv(abs_m, 10, MU).flatten()
                
                if m == 0:
                    Y_val = N[0] * P_val
                elif m > 0:
                    Y_val = np.sqrt(2) * N[m] * P_val * np.cos(m * PHI).flatten()
                else:
                    Y_val = np.sqrt(2) * N[abs_m] * P_val * np.sin(abs_m * PHI).flatten()
                    
                Y_basis[idx] = Y_val
                
            # G = Y_basis @ diag(W_flat) @ Y_basis.T
            # G = (Y_basis * W_flat[None, :]) @ Y_basis.T
            G = (Y_basis * W_flat[None, :]) @ Y_basis.T
            return 2.0 * (G @ c)

        # Objective and gradient
        def objective(c):
            obj, grad = _objective_and_grad(c, MU, PHI, W)
            return obj, grad

        # Bounds for c? 
        # To avoid huge values, we can set bounds [-10, 10] or similar.
        # Since we normalize to ||f||_2=1, the coefficients shouldn't be huge.
        bounds = [(-10, 10)] * 21
        
        res = opt.minimize(
            objective, 
            c0, 
            method='trust-constr', 
            constraints=[{'type': 'eq', 'fun': constraint, 'jac': constraint_jac}],
            bounds=bounds,
            options={'maxiter': 100, 'gtol': 1e-8, 'xtol': 1e-8, 'verbose': 0}
        )
        
        if res.success:
            return res.x
        else:
            # Return best found even if not fully converged?
            # Or return None?
            # Let's return res.x if it's better than current best
            return res.x

    # Starting points
    starts = []
    
    # 1. All ones
    starts.append(np.ones(21))
    
    # 2. Only m=0
    c0 = np.zeros(21)
    c0[10] = 1.0
    starts.append(c0)
    
    # 3. Only m=10, -10
    c1 = np.zeros(21)
    c1[0] = 1.0
    c1[20] = 1.0
    starts.append(c1)
    
    # 4. Random
    np.random.seed(42)
    for _ in range(10):
        c_r = np.random.randn(21)
        starts.append(c_r)
        
    # 5. Symmetric real harmonic (e.g. cos(10 phi) type)
    c2 = np.zeros(21)
    c2[20] = 1.0 # m=10
    c2[0] = 1.0 # m=-10
    starts.append(c2)
    
    # 6. Another symmetric
    c3 = np.zeros(21)
    c3[10] = 1.0 # m=0
    c3[15] = 1.0 # m=5
    c3[5] = 1.0 # m=-5
    starts.append(c3)

    for c0 in starts:
        if time.time() > deadline:
            break
            
        try:
            c_res = minimize_from_start(c0)
            if c_res is not None:
                # Normalize result to have reasonable scale
                # Let's check the norm of f for c_res
                f_res = _compute_f(c_res, MU, PHI)
                norm2_sq = np.sum(W * f_res**2)
                if norm2_sq > 1e-10:
                    # Scale c_res such that ||f||_2 = 1
                    # f is linear in c, so ||f||_2 scales linearly with c
                    # We want ||f_scaled||_2 = 1
                    # ||f_scaled||_2 = |scale| * ||f_res||_2
                    # scale = 1 / sqrt(norm2_sq)
                    scale = 1.0 / np.sqrt(norm2_sq)
                    c_scaled = c_res * scale
                    
                    # Check validity of c_scaled
                    if np.max(np.abs(c_scaled)) < 1e-3 or np.max(np.abs(c_scaled)) > 1e5:
                        # If invalid, try to adjust scale?
                        # If max|c| is too small, increase scale.
                        # If max|c| is too big, decrease scale.
                        # But R is invariant.
                        # Let's just pick a scale that puts max|c| around 1.0
                        max_c = np.max(np.abs(c_scaled))
                        if max_c > 0:
                            c_final = c_scaled * (1.0 / max_c)
                        else:
                            c_final = c_scaled
                    else:
                        c_final = c_scaled
                        
                    # Evaluate R for c_final (should be same as c_res)
                    f_final = _compute_f(c_final, MU, PHI)
                    n2 = np.sqrt(np.sum(W * f_final**2))
                    n4 = np.sum(W * f_final**4) ** 0.25
                    R = n4 / n2
                    
                    if R < best_val:
                        best_val = R
                        best_c = c_final.copy()
                        
        except Exception as e:
            print(f"Error during optimization: {e}")
            continue

    if best_c is None:
        # Fallback to all ones
        return np.ones(21)
        
    return best_c

# EVOLVE-BLOCK-START
# The code above is the complete program.
# We just need to ensure the entry point is correct.
# The function rudin_problem10 is defined.
# EVOLVE-BLOCK-END
