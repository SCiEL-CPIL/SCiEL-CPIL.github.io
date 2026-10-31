"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
import time

def _compute_objective(c):
    """Compute A(f)*A(F) for given coefficients c = [c0, c4, c8]."""
    c0, c4, c8 = c
    
    # Compute c12 to satisfy g(0) = 0
    H0_0 = eval_hermite(0, 0.0)  # = 1
    H4_0 = eval_hermite(4, 0.0)  # = 12
    H8_0 = eval_hermite(8, 0.0)  # = 168
    H12_0 = eval_hermite(12, 0.0)  # = 1440
    
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    # Coefficients in Hermite basis: indices 0, 4, 8, 12
    coeffs = {0: c0, 4: c4, 8: c8, 12: c12}
    degrees = [0, 4, 8, 12]
    coeff_vec = np.array([c0, c4, c8, c12])
    
    # Compute norm: ∫ |g(x)|² dx where g(x) = Σ c_k H_k(x) e^{-x²/2}
    # ∫ H_m(x) H_n(x) e^{-x²} dx = π^{1/2} 2^n n! δ_{mn}
    # So ∫ |g|² dx = π^{1/2} Σ |c_k|² 2^k k!
    
    from math import factorial
    norm_sq = 0.0
    for k in degrees:
        norm_sq += abs(coeff_vec[degrees.index(k)])**2 * (2.0**k) * factorial(k)
    norm_sq *= np.sqrt(np.pi)
    
    if norm_sq < 1e-15:
        return 1e10
    
    # Compute ∫ x² |g(x)|² dx
    # x H_n(x) = (1/2) H_{n+1}(x) + n H_{n-1}(x)
    # So x² H_n = (1/2) x H_{n+1} + n x H_{n-1}
    # = (1/2)[(1/2)H_{n+2} + (n+1)H_n] + n[(1/2)H_n + (n-1)H_{n-2}]
    # = (1/4)H_{n+2} + [(n+1)/2 + n/2]H_n + n(n-1)/2 H_{n-2}
    # = (1/4)H_{n+2} + (2n+1)/2 H_n + n(n-1)/2 H_{n-2}
    
    # ∫ x² H_m H_n e^{-x²} dx can be computed via the recurrence
    # Alternatively, compute directly using the basis expansion
    
    # Let's compute ∫ x² |g|² dx by expanding
    # x² g = Σ c_k x² H_k e^{-x²/2}
    # ∫ x² |g|² dx = Σ_{k,j} c_k c_j ∫ x² H_k H_j e^{-x²} dx
    # By orthogonality with the x² weight, we use:
    # ∫ x² H_k H_j e^{-x²} dx = 0 if k≠j (since x² H_k is a polynomial of degree k+2, 
    # and we need the coefficient of H_j in x² H_k)
    
    # Actually, x² H_k = (1/4)H_{k+2} + (2k+1)/2 H_k + k(k-1)/2 H_{k-2} (for k≥2)
    # For k=0: x² H_0 = x² = (1/4)H_2 + 1/2 H_0
    # For k=1: x² H_1 = x²(2x) = 2x³ = (1/2)H_3 + 1 H_1... let me verify
    # H_3 = 8x³ - 12x, so x³ = (H_3 + 12x)/8 = H_3/8 + 3H_1/4
    # 2x³ = H_3/4 + 3H_2... no, 2x³ = 2*(H_3/8 + 3H_1/4) = H_3/4 + 3H_1/2
    # Hmm, let me just use the general formula.
    
    # The recurrence: x² H_n = (1/4)H_{n+2} + (n + 1/2)H_n + (n(n-1)/2)H_{n-2}
    # This gives the overlap: ∫ x² H_k H_j e^{-x²} dx = π^{1/2} 2^k k! * [coefficient of H_j in x² H_k]
    # Wait, that's not right. Let me think again.
    
    # ∫ H_m H_n e^{-x²} dx = π^{1/2} 2^n n! δ_{mn}
    # So ∫ x² H_k H_j e^{-x²} dx = ∫ [Σ_i a_i H_i] H_j e^{-x²} dx where x² H_k = Σ_i a_i H_i
    # = a_j π^{1/2} 2^j j!
    
    # So I need the coefficient of H_j in x² H_k.
    # From x² H_k = (1/4)H_{k+2} + (k + 1/2)H_k + (k(k-1)/2)H_{k-2}
    
    # So the only nonzero overlaps are when j = k+2, j = k, or j = k-2.
    
    x2_weighted_norm = 0.0
    for i, k in enumerate(degrees):
        ck = coeff_vec[i]
        # Coefficient of H_k in x² H_k: (k + 1/2)
        x2_weighted_norm += abs(ck)**2 * (k + 0.5) * (2.0**k) * factorial(k)
        # Coefficient of H_{k+2} in x² H_k: 1/4
        if k + 2 in degrees:
            j_idx = degrees.index(k + 2)
            cj = coeff_vec[j_idx]
            x2_weighted_norm += 2 * ck * cj * 0.25 * (2.0**(k+2)) * factorial(k+2)
        # Coefficient of H_{k-2} in x² H_k: k(k-1)/2
        if k >= 2 and (k - 2) in degrees:
            j_idx = degrees.index(k - 2)
            cj = coeff_vec[j_idx]
            x2_weighted_norm += 2 * ck * cj * (k*(k-1)/2.0) * (2.0**(k-2)) * factorial(k-2)
    
    x2_weighted_norm *= np.sqrt(np.pi)
    
    A_f = x2_weighted_norm / norm_sq
    
    # For A(F): F(ξ) = Σ c_k (-i)^k H_k(ξ) e^{-ξ²/2}
    # The (-i)^k phase doesn't affect |F|², so A(F) = same computation as A_f
    # because the norm and x²-weighted norm are the same!
    # Wait, that means A(f) = A(F) always? That can't be right for the uncertainty product.
    
    # Actually, yes! For Hermite functions, the Fourier transform is a phase rotation in the 
    # Hermite basis, so |F|² has the same "shape" in the Hermite basis as |g|².
    # Therefore A(f) = A(F) and the product is A(f)².
    
    # So we just need to minimize A(f)², which is equivalent to minimizing A(f).
    
    A_F = A_f  # Same by symmetry
    return A_f * A_F


def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin
    
    # The objective is A(f)^2 where A(f) = ∫x²|g|²/∫|g|²
    # We want to minimize this subject to g(0)=0 (handled by c12 constraint)
    
    # Use multi-start optimization
    best_val = 1e10
    best_c = np.array([-113/100, 1/25, 1/3240])
    
    # Initial guess from seed
    val0 = _compute_objective(best_c)
    best_val = val0
    
    # Random starts
    np.random.seed(42)
    n_starts = 30
    
    for i in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        # Generate random initial point
        scale = 1.0
        c_init = np.random.uniform(-scale, scale, size=3)
        
        # Try L-BFGS-B
        try:
            result = minimize(_compute_objective, c_init, method='L-BFGS-B', 
                            options={'maxiter': 200, 'ftol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x
        except:
            pass
    
    # Grid search for refinement
    if time.time() < deadline - 60:
        # Coarse grid
        grid_range = 2.0
        n_grid = 15
        c0_vals = np.linspace(-grid_range, grid_range, n_grid)
        c4_vals = np.linspace(-grid_range, grid_range, n_grid)
        c8_vals = np.linspace(-grid_range, grid_range, n_grid)
        
        for c0 in c0_vals:
            if time.time() > deadline - 60:
                break
            for c4 in c4_vals:
                if time.time() > deadline - 60:
                    break
                for c8 in c8_vals:
                    val = _compute_objective(np.array([c0, c4, c8]))
                    if val < best_val:
                        best_val = val
                        best_c = np.array([c0, c4, c8])
        
        # Refine around best
        if time.time() < deadline - 30:
            for _ in range(3):
                result = minimize(_compute_objective, best_c, method='L-BFGS-B',
                                options={'maxiter': 500, 'ftol': 1e-14})
                if result.fun < best_val:
                    best_val = result.fun
                    best_c = result.x
    
    return best_c
# EVOLVE-BLOCK-END
