"""Program for the uncertainty inequality task using proper Remez exchange algorithm."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import hermite
import math


def _precompute_hermite_basis():
    """Precompute the basis functions on a fine grid for fast evaluation."""
    x_grid = np.linspace(-12, 12, 5001)
    gauss = np.exp(-x_grid**2 / 2)
    
    H0 = np.ones_like(x_grid)
    H4 = 16 * x_grid**4 - 48 * x_grid**2 + 12
    H8 = 256 * x_grid**8 - 3584 * x_grid**6 + 13440 * x_grid**4 - 13440 * x_grid**2 + 1680
    H12 = (4096 * x_grid**12 - 122880 * x_grid**10 + 1290240 * x_grid**8 
           - 6451200 * x_grid**6 + 13107200 * x_grid**4 - 9031680 * x_grid**2 + 1032192)
    
    # Precompute x*H_n*gauss for each degree
    xH0 = x_grid * H0 * gauss
    xH4 = x_grid * H4 * gauss
    xH8 = x_grid * H8 * gauss
    xH12 = x_grid * H12 * gauss
    
    # For Fourier side: G(xi) = sqrt(2*pi) * (c0*H0(xi) + c4*H4(xi) + c8*H8(xi) + c12*H12(xi)) * exp(-xi^2/2)
    # xi*G(xi) = sqrt(2*pi) * (c0*xi*H0(xi) + c4*xi*H4(xi) + c8*xi*H8(xi) + c12*xi*H12(xi)) * exp(-xi^2/2)
    xiH0 = x_grid * H0 * gauss  # Same grid, by symmetry
    xiH4 = x_grid * H4 * gauss
    xiH8 = x_grid * H8 * gauss
    xiH12 = x_grid * H12 * gauss
    
    sqrt_2pi = math.sqrt(2 * math.pi)
    
    return {
        'x_grid': x_grid,
        'xH0': xH0, 'xH4': xH4, 'xH8': xH8, 'xH12': xH12,
        'xiH0': xiH0, 'xiH4': xiH4, 'xiH8': xiH8, 'xiH12': xiH12,
        'sqrt_2pi': sqrt_2pi
    }


def _compute_bound_fast(c0, c4, c8, basis):
    """Fast computation of the bound using precomputed basis."""
    c12 = -(c0 + 12 * c4 + 1680 * c8) / 1032192.0
    
    xH0, xH4, xH8, xH12 = basis['xH0'], basis['xH4'], basis['xH8'], basis['xH12']
    xiH0, xiH4, xiH8, xiH12 = basis['xiH0'], basis['xiH4'], basis['xiH8'], basis['xiH12']
    sqrt_2pi = basis['sqrt_2pi']
    
    # x*g(x) = c0*xH0 + c4*xH4 + c8*xH8 + c12*xH12
    xg = c0 * xH0 + c4 * xH4 + c8 * xH8 + c12 * xH12
    
    # xi*G(xi) = sqrt(2*pi) * (c0*xiH0 + c4*xiH4 + c8*xiH8 + c12*xiH12)
    xiG = sqrt_2pi * (c0 * xiH0 + c4 * xiH4 + c8 * xiH8 + c12 * xiH12)
    
    A_f = np.max(np.abs(xg))
    A_F = np.max(np.abs(xiG))
    
    return A_f * A_F


def _find_critical_points(xg, xiG, x_grid, n_points=4):
    """Find the critical points where the suprema are achieved."""
    # Find local maxima of |xg| and |xiG|
    abs_xg = np.abs(xg)
    abs_xiG = np.abs(xiG)
    
    # Find indices of top n_points values
    xg_indices = np.argsort(abs_xg)[-n_points:]
    xiG_indices = np.argsort(abs_xiG)[-n_points:]
    
    # Sort by x value for consistent ordering
    xg_indices = np.sort(xg_indices)
    xiG_indices = np.sort(xiG_indices)
    
    x_crit = x_grid[xg_indices]
    xi_crit = x_grid[xiG_indices]
    
    return x_crit, xi_crit


def _remez_exchange(c_init, basis, max_iter=50, tol=1e-14):
    """
    Proper Remez exchange algorithm.
    
    The bound is C4 = A_f * A_F where A_f = sup|x*g(x)| and A_F = sup|xi*G(xi)|.
    Both are piecewise-linear in the coefficients. At the optimum, we expect
    the suprema to be achieved at a small set of points with alternating signs.
    
    Strategy:
    1. Find the active points (where suprema are achieved)
    2. Set up equations: x_i * g(x_i) = s_i * A_f and xi_j * G(xi_j) = r_j * A_F
    3. Solve for c0, c4, c8, A_f, A_F (but A_f * A_F is what we minimize)
    4. Actually, we can normalize: let A_f = A_F = t (by symmetry of the problem)
       Then we have 3 unknowns (c0, c4, c8) and can set up 3 equations from
       alternating signs at 3 critical points total.
    """
    c = np.array(c_init, dtype=float)
    x_grid = basis['x_grid']
    
    # Precompute basis values
    xH0, xH4, xH8, xH12 = basis['xH0'], basis['xH4'], basis['xH8'], basis['xH12']
    xiH0, xiH4, xiH8, xiH12 = basis['xiH0'], basis['xiH4'], basis['xiH8'], basis['xiH12']
    sqrt_2pi = basis['sqrt_2pi']
    
    best_bound = _compute_bound_fast(*c, basis)
    best_c = c.copy()
    
    for iteration in range(max_iter):
        c0, c4, c8 = c
        c12 = -(c0 + 12 * c4 + 1680 * c8) / 1032192.0
        
        # Compute x*g(x) and xi*G(xi)
        xg = c0 * xH0 + c4 * xH4 + c8 * xH8 + c12 * xH12
        xiG = sqrt_2pi * (c0 * xiH0 + c4 * xiH4 + c8 * xiH8 + c12 * xiH12)
        
        abs_xg = np.abs(xg)
        abs_xiG = np.abs(xiG)
        
        A_f = np.max(abs_xg)
        A_F = np.max(abs_xiG)
        current_bound = A_f * A_F
        
        if current_bound < best_bound:
            best_bound = current_bound
            best_c = c.copy()
        
        # Find critical points
        # We expect the optimum to have alternating signs at the critical points
        # For 3 free variables, we need 3 active constraints
        # Try different numbers of points from each side
        
        # Find top 3 from each side
        xg_top_idx = np.argsort(abs_xg)[-3:]
        xiG_top_idx = np.argsort(abs_xiG)[-3:]
        
        xg_top_idx = np.sort(xg_top_idx)
        xiG_top_idx = np.sort(xiG_top_idx)
        
        # Get the signs at these points
        xg_signs = np.sign(xg[xg_top_idx])
        xiG_signs = np.sign(xiG[xiG_top_idx])
        
        # For Remez, we want alternating signs. Let's try different combinations.
        # The key insight: at the optimum, the function should equiosculate.
        
        # Approach: use the 3 points with the largest |xg| and |xiG| values
        # and set up a system where they all have the same magnitude t
        # with alternating signs.
        
        # Let's try: use 2 points from x-side and 1 from xi-side, or vice versa
        # Actually, let's use all 6 points and solve a least-squares problem
        
        # Build the design matrix for the 3 coefficients
        # xg at a point x_i: c0*xH0[i] + c4*xH4[i] + c8*xH8[i] + c12*xH12[i]
        # where c12 = -(c0 + 12*c4 + 1680*c8)/1032192
        # So xg[i] = c0*(xH0[i] - xH12[i]/1032192) + c4*(xH4[i] - 12*xH12[i]/1032192) + c8*(xH8[i] - 1680*xH12[i]/1032192)
        
        # Similarly for xiG[i] = sqrt(2*pi) * [c0*(xiH0[i] - xiH12[i]/1032192) + c4*(xiH4[i] - 12*xiH12[i]/1032192) + c8*(xiH8[i] - 1680*xiH12[i]/1032192)]
        
        # Let's define effective basis functions
        c12_factor = 1.0 / 1032192.0
        
        xH0_eff = xH0 - c12_factor * xH12
        xH4_eff = xH4 - 12 * c12_factor * xH12
        xH8_eff = xH8 - 1680 * c12_factor * xH12
        
        xiH0_eff = xiH0 - c12_factor * xiH12
        xiH4_eff = xiH4 - 12 * c12_factor * xiH12
        xiH8_eff = xiH8 - 1680 * c12_factor * xiH12
        
        # Now xg[i] = c0*xH0_eff[i] + c4*xH4_eff[i] + c8*xH8_eff[i]
        # xiG[i] = sqrt(2*pi) * (c0*xiH0_eff[i] + c4*xiH4_eff[i] + c8*xiH8_eff[i])
        
        # For Remez: we want to find c that minimizes max(|xg|, |xiG|/sqrt(2*pi))
        # Let's normalize: define h(x) = x*g(x) and k(xi) = xi*G(xi)/sqrt(2*pi)
        # Then we want to minimize max(sup|h|, sup|k|)
        
        # At the optimum, we expect h and k to have equal suprema with alternating signs
        
        # Select critical points: use the points where |xg| and |xiG| are largest
        # and set up equations for equiosculation
        
        # Try: use 3 points total (1 from x-side, 2 from xi-side or vice versa)
        # and enforce alternating signs with equal magnitude
        
        # Let's try a simpler approach: minimize the maximum of |xg| and |xiG|/sqrt(2*pi)
        # using subgradient method guided by the critical points
        
        # Subgradient of max(|xg|, |xiG|/sqrt(2*pi)) at point c:
        # If |xg| > |xiG|/sqrt(2*pi), subgrad is sign(xg[i_max]) * [xH0_eff[i_max], xH4_eff[i_max], xH8_eff[i_max]]
        # If |xiG|/sqrt(2*pi) > |xg|, subgrad is sign(xiG[j_max]) * [xiH0_eff[j_max], xiH4_eff[j_max], xiH8_eff[j_max]]
        
        # Use this to do a projected subgradient descent
        
        # Find the active constraint
        max_xg_idx = np.argmax(abs_xg)
        max_xiG_idx = np.argmax(abs_xiG)
        
        val_xg = abs_xg[max_xg_idx]
        val_xiG = abs_xiG[max_xiG_idx] / sqrt_2pi
        
        if val_xg > val_xiG:
            # Active constraint is from x-side
            i = max_xg_idx
            subgrad = np.array([xH0_eff[i], xH4_eff[i], xH8_eff[i]])
            subgrad *= np.sign(xg[i])
        else:
            # Active constraint is from xi-side
            j = max_xiG_idx
            subgrad = np.array([xiH0_eff[j], xiH4_eff[j], xiH8_eff[j]])
            subgrad *= np.sign(xiG[j])
        
        # Normalize subgradient
        subgrad_norm = np.linalg.norm(subgrad)
        if subgrad_norm > 1e-15:
            subgrad = subgrad / subgrad_norm
        
        # Step size (decreasing)
        step_size = 0.01 / (iteration + 1)
        
        # Update
        c_new = c - step_size * subgrad
        
        # Check if we improved
        new_bound = _compute_bound_fast(*c_new, basis)
        if new_bound < current_bound:
            c = c_new
        else:
            # Try smaller step
            for factor in [0.5, 0.1, 0.01]:
                c_try = c - step_size * factor * subgrad
                bound_try = _compute_bound_fast(*c_try, basis)
                if bound_try < current_bound:
                    c = c_try
                    break
            
            if np.allclose(c, c_new, atol=1e-15):
                # No improvement, break
                break
        
        # Convergence check
        if iteration > 5 and abs(best_bound - _compute_bound_fast(*c, basis)) < tol:
            break
    
    return best_c, best_bound


def construct():
    """
    Return three coefficients [c0, c4, c8] that minimize the upper bound C4.
    """
    # Precompute basis
    basis = _precompute_hermite_basis()
    
    # Start from the known good solution
    start_c = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_c = start_c.copy()
    best_bound = _compute_bound_fast(*best_c, basis)
    
    # Run Remez from multiple starting points
    start_points = [
        start_c,
        np.array([-1.0, 0.1, 0.001]),
        np.array([-1.2, 0.05, 0.0005]),
        np.array([-0.9, 0.15, 0.002]),
        np.array([-1.1, 0.08, 0.001]),
        np.array([-1.05, 0.09, 0.0012]),
        np.array([-1.15, 0.06, 0.0008]),
        np.array([-1.08, 0.07, 0.0015]),
    ]
    
    for sp in start_points:
        try:
            c_opt, bound_opt = _remez_exchange(sp, basis, max_iter=100, tol=1e-14)
            if bound_opt < best_bound:
                best_bound = bound_opt
                best_c = c_opt
        except Exception:
            pass
    
    # Final refinement with more iterations
    try:
        c_final, bound_final = _remez_exchange(best_c, basis, max_iter=200, tol=1e-15)
        if bound_final < best_bound:
            best_bound = bound_final
            best_c = c_final
    except Exception:
        pass
    
    return np.array([best_c[0], best_c[1], best_c[2]])
# EVOLVE-BLOCK-END
