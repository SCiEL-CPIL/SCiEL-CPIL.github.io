import numpy as np
import scipy.optimize as opt
import time

def construct():
    start_time = time.time()
    deadline = 205.0  # Safety margin before 210s limit

    # Use exact Gauss-Hermite quadrature.
    # g is degree 12. g^2 is degree 24.
    # x^2 g^2 is degree 26.
    # (g' - xg)^2 involves derivatives, so degree increases.
    # g' is degree 11. xg is degree 13.
    # g' - xg is degree 13.
    # (g' - xg)^2 is degree 26.
    # So all integrands are polynomials of degree <= 26 times e^{-x^2}.
    # Gauss-Hermite with N points integrates polynomials up to degree 2N-1 exactly.
    # We need 2N-1 >= 26 => 2N >= 27 => N >= 13.5.
    # So N=14 is sufficient for exact integration.
    N_points = 14
    x_nodes, w_nodes = np.polynomial.hermite.hermgauss(N_points)
    
    # Pre-compute powers and Hermite polynomials at nodes
    x = x_nodes
    x2 = x**2
    x3 = x2 * x
    x4 = x2**2
    x5 = x4 * x
    x6 = x4 * x2
    x7 = x6 * x
    x8 = x4**2
    x9 = x8 * x
    x10 = x8 * x2
    x11 = x10 * x
    x12 = x6**2
    
    # Hermite polynomials (physicists' convention)
    H0 = np.ones_like(x)
    H4 = x4 - 6*x2 + 3
    H8 = x8 - 28*x6 + 210*x4 - 420*x2 + 105
    H12 = x12 - 66*x10 + 1260*x8 - 9240*x6 + 23100*x4 - 27720*x2 + 10395
    
    # Derivatives
    dH0 = np.zeros_like(x)
    dH4 = 4*x3 - 12*x
    dH8 = 8*x7 - 168*x5 + 840*x3 - 840*x
    dH12 = 12*x11 - 660*x9 + 10080*x7 - 55440*x5 + 92400*x3 - 55440*x

    # Pre-compute basis vectors for vectorized operations
    # We represent g as c0*H0 + c4*H4 + c8*H8 + c12*H12
    # c12 = -(c0 + 3*c4 + 105*c8) / 10395
    # Let's define the effective basis functions for the free parameters c4, c8
    # g = c0*H0 + c4*H4 + c8*H8 - (c0 + 3*c4 + 105*c8)/10395 * H12
    # g = c0*(H0 - H12/10395) + c4*(H4 - 3*H12/10395) + c8*(H8 - 105*H12/10395)
    
    B0 = H0 - H12/10395.0
    B4 = H4 - 3.0*H12/10395.0
    B8 = H8 - 105.0*H12/10395.0
    
    dB0 = dH0 - dH12/10395.0
    dB4 = dH4 - 3.0*dH12/10395.0
    dB8 = dH8 - 105.0*dH12/10395.0
    
    # Pre-compute weights for integrals
    w = w_nodes
    wx2 = w_nodes * x2
    
    # We want to minimize sqrt( (I_x2 / I_0) * (1/(4*pi^2) * I_deriv / I_0) )
    # This is equivalent to minimizing (I_x2 * I_deriv) / (I_0^2)
    # Let Q(c) = I_x2(c) * I_deriv(c) - lambda * I_0(c)^2 ? No, just minimize the ratio.
    # Let's use log-minimization or direct ratio.
    # Since the function is homogeneous of degree 0, we can fix c0=1.
    # Then we minimize f(c4, c8) = (I_x2(1, c4, c8) * I_deriv(1, c4, c8)) / (I_0(1, c4, c8)^2)
    
    def compute_integrals(c4, c8):
        # c0 is fixed to 1.0
        g = B0 + c4*B4 + c8*B8
        gp = dB0 + c4*dB4 + c8*dB8
        
        g2 = g**2
        I0 = np.sum(w * g2)
        
        if I0 < 1e-12:
            return np.inf, np.inf, np.inf
            
        Ix2 = np.sum(wx2 * g2)
        
        deriv_term = gp - x * g
        Id = np.sum(w * deriv_term**2)
        
        return I0, Ix2, Id

    def objective(params):
        c4, c8 = params
        I0, Ix2, Id = compute_integrals(c4, c8)
        if not np.isfinite(I0) or not np.isfinite(Ix2) or not np.isfinite(Id):
            return 1e10
        val = (Ix2 * Id) / (I0 * I0)
        return np.sqrt(val) / (2.0 * np.pi)

    # Gradient for optimization
    def objective_and_grad(params):
        c4, c8 = params
        I0, Ix2, Id = compute_integrals(c4, c8)
        if not np.isfinite(I0) or I0 < 1e-12:
            return 1e10, np.array([0.0, 0.0])
        
        # We need derivatives of I0, Ix2, Id with respect to c4, c8
        # I0 = sum w * g^2. dI0/dc = 2 sum w * g * B_c
        # Ix2 = sum w x2 * g^2. dIx2/dc = 2 sum w x2 * g * B_c
        # Id = sum w * (gp - xg)^2. Let D = gp - xg.
        # D = dB0 + c4*dB4 + c8*dB8 - x*(B0 + c4*B4 + c8*B8)
        # D = (dB0 - x*B0) + c4*(dB4 - x*B4) + c8*(dB8 - x*B8)
        # Let D0 = dB0 - x*B0, D4 = dB4 - x*B4, D8 = dB8 - x*B8
        # Id = sum w * D^2. dId/dc = 2 sum w * D * D_c
        
        D0 = dB0 - x*B0
        D4 = dB4 - x*B4
        D8 = dB8 - x*B8
        
        g = B0 + c4*B4 + c8*B8
        D = D0 + c4*D4 + c8*D8
        
        dI0_dc4 = 2.0 * np.sum(w * g * B4)
        dI0_dc8 = 2.0 * np.sum(w * g * B8)
        
        dIx2_dc4 = 2.0 * np.sum(wx2 * g * B4)
        dIx2_dc8 = 2.0 * np.sum(wx2 * g * B8)
        
        dId_dc4 = 2.0 * np.sum(w * D * D4)
        dId_dc8 = 2.0 * np.sum(w * D * D8)
        
        # f = sqrt( (Ix2 * Id) / I0^2 ) / (2*pi)
        # Let R = (Ix2 * Id) / I0^2. f = 0.5 * sqrt(R) / pi
        # df/dc = (1/(4*pi*sqrt(R))) * dR/dc
        # R = Ix2 * Id * I0^-2
        # dR/dc = (dIx2 * Id * I0^-2) + (Ix2 * dId * I0^-2) - 2 * Ix2 * Id * I0^-3 * dI0
        # dR/dc = (1/I0^2) * (dIx2 * Id + Ix2 * dId - 2 * Ix2 * Id * dI0 / I0)
        
        R = (Ix2 * Id) / (I0 * I0)
        if R <= 0:
             return 1e10, np.array([0.0, 0.0])
             
        dR_dc4 = (1.0/(I0*I0)) * (dIx2_dc4 * Id + Ix2 * dId_dc4 - 2.0 * Ix2 * Id * dI0_dc4 / I0)
        dR_dc8 = (1.0/(I0*I0)) * (dIx2_dc8 * Id + Ix2 * dId_dc8 - 2.0 * Ix2 * Id * dI0_dc8 / I0)
        
        grad_factor = 1.0 / (4.0 * np.pi * np.sqrt(R))
        
        grad = np.array([grad_factor * dR_dc4, grad_factor * dR_dc8])
        
        return 0.5 * np.sqrt(R) / np.pi, grad

    # Strategy:
    # 1. Coarse 2D Grid Search to find global minimum basin.
    # 2. Local Polish with L-BFGS-B using analytical gradients.
    
    # Determine bounds for c4, c8
    # The seed was roughly c4=0.04, c8=0.0003.
    # Let's search in a wider range to be safe.
    c4_min, c4_max = -1.0, 1.0
    c8_min, c8_max = -0.1, 0.1
    
    # Grid Search
    n_grid = 50
    c4_vals = np.linspace(c4_min, c4_max, n_grid)
    c8_vals = np.linspace(c8_min, c8_max, n_grid)
    
    best_score = np.inf
    best_params = np.array([0.0, 0.0])
    
    # Vectorized grid evaluation for speed
    # We can't easily vectorize the full objective with gradients, but we can evaluate values.
    # Let's do a simple loop for the grid, it's 2500 points, very fast.
    for c4 in c4_vals:
        for c8 in c8_vals:
            if time.time() > deadline - 50:
                break
            val, _ = objective_and_grad(np.array([c4, c8]))
            if val < best_score:
                best_score = val
                best_params = np.array([c4, c8])
        if time.time() > deadline - 50:
            break

    # Polish the best found
    if time.time() < deadline - 10:
        try:
            res = opt.minimize(
                objective_and_grad, 
                best_params, 
                jac=True, # Tells scipy to use the gradient returned
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    # Also try polishing from the original seed point (c4=0.04, c8=1/3240)
    seed_params = np.array([1.0/25.0, 1.0/3240.0])
    if time.time() < deadline - 5:
        try:
            res_seed = opt.minimize(
                objective_and_grad, 
                seed_params, 
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res_seed.fun < best_score and np.isfinite(res_seed.fun):
                best_score = res_seed.fun
                best_params = res_seed.x
        except Exception:
            pass

    # Also try polishing from Gaussian (c4=0, c8=0)
    gauss_params = np.array([0.0, 0.0])
    if time.time() < deadline - 2:
        try:
            res_gauss = opt.minimize(
                objective_and_grad, 
                gauss_params, 
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res_gauss.fun < best_score and np.isfinite(res_gauss.fun):
                best_score = res_gauss.fun
                best_params = res_gauss.x
        except Exception:
            pass

    # Construct final coefficients
    # c0 = 1.0
    # c4 = best_params[0]
    # c8 = best_params[1]
    # c12 is derived
    
    return np.array([1.0, best_params[0], best_params[1]])
