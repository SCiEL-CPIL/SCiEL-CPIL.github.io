"""Rewritten program for uncertainty inequality tuning."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _hermite_polynomials(x, n_max=12):
    """Compute Hermite polynomials H_0 through H_{n_max} evaluated at x."""
    H = np.zeros((n_max + 1, len(x)))
    H[0, :] = 1.0
    H[1, :] = 2 * x
    for n in range(2, n_max + 1):
        H[n, :] = 2 * x * H[n-1, :] - 2 * (n-1) * H[n-2, :]
    return H

def _compute_bound(a0, a4, a8, n_grid=4096):
    """
    Compute the upper bound C4 for given coefficients a0, a4, a8.
    The degree-12 coefficient is fixed by g(0) = 0.
    
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    g(0) = (a0 + a4*H4(0) + a8*H8(0) + a12*H12(0)) = 0
    H4(0) = -12, H8(0) = 1344, H12(0) = -1032192
    So a12 = -(a0 - 12*a4 + 1344*a8) / (-1032192) = (a0 - 12*a4 + 1344*a8) / 1032192
    """
    # Compute a12 from constraint g(0) = 0
    H4_0 = -12.0
    H8_0 = 1344.0
    H12_0 = -1032192.0
    a12 = -(a0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # Grid for numerical computation
    x = np.linspace(-8, 8, n_grid)
    H = _hermite_polynomials(x, 12)
    
    # g(x)
    g = (a0 * H[0, :] + a4 * H[4, :] + a8 * H[8, :] + a12 * H[12, :]) * np.exp(-x**2 / 2)
    
    # Fourier transform of g: ĝ(ξ) = Σ a_n (-i)^n H_n(ξ) exp(-ξ^2/2)
    # For real-valued bound, we use |ĝ(ξ)|
    i = 1j
    g_hat = (a0 * H[0, :] + a4 * (i**4) * H[4, :] + a8 * (i**8) * H[8, :] + a12 * (i**12) * H[12, :]) * np.exp(-x**2 / 2)
    # i^4 = 1, i^8 = 1, i^12 = -1
    g_hat = (a0 * H[0, :] + a4 * H[4, :] + a8 * H[8, :] - a12 * H[12, :]) * np.exp(-x**2 / 2)
    
    # The uncertainty bound: we need to find sup over f of |∫fg|·|∫fĝ| / ||f||²
    # This is equivalent to finding the largest eigenvalue of a certain operator
    # Numerically: discretize and find the max singular value of the matrix
    # M_{ij} = g(x_i) * conj(g_hat(x_j)) * dx
    dx = x[1] - x[0]
    
    # Construct the rank-1 operator matrix (but we can compute its norm analytically)
    # |∫fg|·|∫fĝ| ≤ (∫|g|²)^{1/2} (∫|ĝ|²)^{1/2} ||f||² ... no that's not right
    
    # Actually, by Cauchy-Schwarz: |∫fg| ≤ ||g||₂ ||f||₂, |∫fĝ| ≤ ||ĝ||₂ ||f||₂
    # So |∫fg|·|∫fĆ| ≤ ||g||₂ ||ĝ||₂ ||f||₂²
    # But this is a loose bound. The sharp bound involves the operator:
    # T(f) = (∫fg, ∫fĝ) mapped to product
    
    # The sharp constant C4 = sup_{||f||=1} |⟨f,g⟩|·|⟨f,ĝ⟩|
    # This equals the square root of the largest eigenvalue of the 2x2 Gram matrix
    # G = [[⟨g,g⟩, ⟨g,ĝ⟩], [⟨ĝ,g⟩, ⟨ĝ,ĝ⟩]] ... no
    
    # Actually for two vectors u, v in Hilbert space:
    # sup_{||f||=1} |⟨f,u⟩|·|⟨f,v⟩| = sqrt(⟨u,u⟩⟨v,v⟩ + |⟨u,v⟩|²) / 2 + ... 
    # Let me think more carefully.
    
    # max_{||f||=1} |⟨f,u⟩||⟨f,v⟩| where u=g, v=ĝ
    # This is a known result. Let α=⟨u,u⟩, β=⟨v,v⟩, γ=|⟨u,v⟩|
    # The max is (sqrt(αβ) + γ) / 2 ... let me verify
    
    # Actually, parametrize f = a*u/||u|| + b*v/||v|| + c*w where w⊥span{u,v}
    # For the max, c=0. Then |⟨f,u⟩|·|⟨f,v⟩| = |a||u|| · |b||v|| = |a||b|·||u||||v||
    # with constraint a²α + b²β + 2ab·Re(⟨u,v⟩/||u||||v||) = 1
    
    # This is getting complex. Let me just compute it numerically via eigenvalue.
    
    # Discretized approach: the operator is rank 2.
    # C4² = max_{||f||=1} (⟨f,g⟩⟨f,g⟩̄)(⟨f,ĝ⟩⟨f,ĝ⟩̄)
    # = max |⟨f,g⟩|² |⟨f,ĝ⟩|²
    
    # Let's compute the 2x2 matrix and find the relevant quantity
    alpha = np.sum(np.abs(g)**2) * dx  # ||g||²
    beta = np.sum(np.abs(g_hat)**2) * dx  # ||ĝ||²
    gamma = np.sum(np.conj(g) * g_hat) * dx  # ⟨g, ĝ⟩
    
    # The maximum of |⟨f,g⟩|·|⟨f,ĝ⟩| over ||f||=1
    # This equals sqrt of the largest eigenvalue of the matrix:
    # [[|⟨g,g⟩|·|⟨ĝ,ĝ⟩|, ...]] 
    # Actually, it's known that max = (||g||||ĝ|| + |⟨g,ĝ⟩|) / 2 if the vectors are aligned appropriately
    # More precisely: max_{||f||=1} |⟨f,u⟩||⟨f,v⟩| = (||u||||v|| + |⟨u,v⟩|) / 2
    
    # Let me verify with simple case: u=v. Then max |⟨f,u⟩|² = ||u||², and formula gives (||u||²+||u||²)/2 = ||u||². ✓
    # u⊥v: max |⟨f,u⟩||⟨f,v⟩|. Let f = (u/||u|| + v/||v||)/sqrt(2). Then |⟨f,u⟩| = ||u||/sqrt(2), same for v. Product = ||u||||v||/2. Formula gives (||u||||v|| + 0)/2. ✓
    
    C4 = (np.sqrt(alpha * beta) + np.abs(gamma)) / 2.0
    
    return C4

def _objective(coeffs):
    a0, a4, a8 = coeffs
    try:
        val = _compute_bound(a0, a4, a8)
        if not np.isfinite(val) or val < 0:
            return 1e10
        return val
    except:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 180  # Safety margin before 210s kill
    
    # Initial guess from seed
    best_coeffs = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_val = _objective(best_coeffs)
    
    # Multi-start local optimization
    n_starts = 0
    np.random.seed(42)
    
    while time.time() < deadline - 10:  # Leave 10s margin
        n_starts += 1
        
        # Generate a random starting point near the best
        if n_starts == 1:
            x0 = best_coeffs.copy()
        else:
            scale = 0.1 / (1 + 0.1 * n_starts)
            x0 = best_coeffs + np.random.randn(3) * scale
        
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            
            if res.success or res.fun < best_val:
                if np.all(np.isfinite(res.x)) and res.fun < best_val:
                    best_val = res.fun
                    best_coeffs = res.x.copy()
        except:
            pass
        
        # Try a few more diverse starts
        if n_starts % 5 == 0 and time.time() < deadline - 30:
            for _ in range(3):
                x0 = np.random.randn(3) * 0.5
                try:
                    res = minimize(_objective, x0, method='Nelder-Mead',
                                  options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                    if res.fun < best_val:
                        best_val = res.fun
                        best_coeffs = res.x.copy()
                except:
                    pass
    
    # Final polish with tight tolerance
    if time.time() < deadline - 5:
        try:
            res = minimize(_objective, best_coeffs, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except:
            pass
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
