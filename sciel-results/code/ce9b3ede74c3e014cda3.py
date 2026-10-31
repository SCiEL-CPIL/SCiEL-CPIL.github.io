"""Optimized program: analytic-gradient L-BFGS-B with smooth surrogate and polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 180.0
_MARGIN = 10.0  # Larger safety margin for polish phase

# Precompute Hermite polynomial values on a fixed grid
_NPTS = 2001
_XMIN = -12.0
_XMAX = 12.0
_X = np.linspace(_XMIN, _XMAX, _NPTS)
_DX = _X[1] - _X[0]
_EXP = np.exp(-0.5 * _X * _X)

def _hermite(n, x):
    """Physicist's Hermite polynomial H_n(x) via recurrence."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2.0 * x
    else:
        h_nm2 = np.ones_like(x)
        h_nm1 = 2.0 * x
        for k in range(2, n + 1):
            h_n = 2.0 * x * h_nm1 - 2.0 * (k - 1) * h_nm2
            h_nm2, h_nm1 = h_nm1, h_n
        return h_nm1

# Precompute Hermite polynomials and their derivatives
_H0 = _hermite(0, _X)
_H4 = _hermite(4, _X)
_H8 = _hermite(8, _X)
_H12 = _hermite(12, _X)
_H3 = _hermite(3, _X)
_H7 = _hermite(7, _X)
_H11 = _hermite(11, _X)

# Derivatives: H_n'(x) = 2n H_{n-1}(x)
_H0p = np.zeros_like(_X)
_H4p = 8.0 * _H3
_H8p = 16.0 * _H7
_H12p = 24.0 * _H11

# Fourier transform kernel (precompute FFT setup)
_N = _NPTS
_OMEGA = 2.0 * np.pi * np.fft.fftfreq(_N, d=_DX)
_OMEGA_SORTED = np.fft.fftshift(_OMEGA)
_G_IDX = np.argsort(_OMEGA_SORTED)
_OMEGA_SORTED_SORTED = _OMEGA_SORTED[_G_IDX]

def _compute_integrals(coeffs):
    """Compute the three integrals I_g, I_gp2, I_G2 for given coefficients."""
    a0, a4, a8 = coeffs
    
    # Compute a12 from g(0) = 0 constraint
    a12 = -(a0 * 1.0 + a4 * (-12.0) + a8 * 168.0) / (-4960.0)
    
    # Build g(x) = p(x) * exp(-x^2/2)
    poly = a0 * _H0 + a4 * _H4 + a8 * _H8 + a12 * _H12
    g = poly * _EXP
    
    # Compute g'(x) = (p'(x) - x*p(x)) * exp(-x^2/2)
    pp = a0 * _H0p + a4 * _H4p + a8 * _H8p + a12 * _H12p
    gp = (pp - _X * poly) * _EXP
    
    # Fourier transform via FFT
    G = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(g))) * _DX
    G_real = G.real[_G_IDX]
    
    # Interpolate to match _X grid
    Gx = np.interp(_X, _OMEGA_SORTED_SORTED, G_real)
    
    # Compute integrals using trapezoidal rule
    I_g = np.trapz(g, _X)
    I_gp2 = np.trapz(gp * gp, _X)
    I_G2 = np.trapz(Gx * Gx, _X)
    
    return I_g, I_gp2, I_G2

def _compute_bound_and_grad(coeffs, smooth_eps=1e-6):
    """Compute the bound value and analytic gradient for given coefficients."""
    a0, a4, a8 = coeffs
    
    # Compute a12 from g(0) = 0 constraint
    a12 = -(a0 * 1.0 + a4 * (-12.0) + a8 * 168.0) / (-4960.0)
    
    # Build g(x) = p(x) * exp(-x^2/2)
    poly = a0 * _H0 + a4 * _H4 + a8 * _H8 + a12 * _H12
    g = poly * _EXP
    
    # Compute g'(x) = (p'(x) - x*p(x)) * exp(-x^2/2)
    pp = a0 * _H0p + a4 * _H4p + a8 * _H8p + a12 * _H12p
    gp = (pp - _X * poly) * _EXP
    
    # Fourier transform via FFT
    G = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(g))) * _DX
    G_real = G.real[_G_IDX]
    
    # Interpolate to match _X grid
    Gx = np.interp(_X, _OMEGA_SORTED_SORTED, G_real)
    
    # Compute integrals using trapezoidal rule
    I_g = np.trapz(g, _X)
    I_gp2 = np.trapz(gp * gp, _X)
    I_G2 = np.trapz(Gx * Gx, _X)
    
    # Smooth surrogate for abs(I_g) * sqrt(I_gp2) * sqrt(I_G2)
    # Use soft-abs: |x|_eps = sqrt(x^2 + eps^2)
    # And soft-sqrt: sqrt(x)_eps = sqrt(x + eps) for x >= -eps
    I_g_smooth = np.sqrt(I_g * I_g + smooth_eps * smooth_eps)
    I_gp2_smooth = np.sqrt(max(I_gp2, 0.0) + smooth_eps)
    I_G2_smooth = np.sqrt(max(I_G2, 0.0) + smooth_eps)
    
    val = I_g_smooth * I_gp2_smooth * I_G2_smooth
    
    # Compute analytic gradient
    # d/dcoeff of I_g, I_gp2, I_G2
    # I_g = ∫ g dx, where g = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    # a12 = -(a0 - 12*a4 + 168*a8) / (-4960) = (a0 - 12*a4 + 168*a8) / 4960
    
    # Partial derivatives of a12
    da12_da0 = 1.0 / 4960.0
    da12_da4 = -12.0 / 4960.0
    da12_da8 = 168.0 / 4960.0
    
    # dI_g/da0 = ∫ (H0 + da12_da0 * H12) * exp(-x^2/2) dx
    # dI_g/da4 = ∫ (H4 + da12_da4 * H12) * exp(-x^2/2) dx
    # dI_g/da8 = ∫ (H8 + da12_da8 * H12) * exp(-x^2/2) dx
    
    dI_g_da0 = np.trapz((_H0 + da12_da0 * _H12) * _EXP, _X)
    dI_g_da4 = np.trapz((_H4 + da12_da4 * _H12) * _EXP, _X)
    dI_g_da8 = np.trapz((_H8 + da12_da8 * _H12) * _EXP, _X)
    
    # For I_gp2 = ∫ gp^2 dx, where gp = (pp - x*poly) * exp(-x^2/2)
    # This is more complex; use numerical differentiation for the gradient of I_gp2 and I_G2
    # But we can compute it analytically
    
    # Let's define f(x, coeffs) = gp(x) = (pp - x*poly) * exp(-x^2/2)
    # dI_gp2/dcoeff = 2 * ∫ gp * dgp/dcoeff dx
    
    # dgp/da0 = ((H0p + da12_da0*H12p) - x*(H0 + da12_da0*H12)) * exp(-x^2/2)
    dgp_da0 = ((_H0p + da12_da0 * _H12p) - _X * (_H0 + da12_da0 * _H12)) * _EXP
    dgp_da4 = ((_H4p + da12_da4 * _H12p) - _X * (_H4 + da12_da4 * _H12)) * _EXP
    dgp_da8 = ((_H8p + da12_da8 * _H12p) - _X * (_H8 + da12_da8 * _H12)) * _EXP
    
    dI_gp2_da0 = 2.0 * np.trapz(gp * dgp_da0, _X)
    dI_gp2_da4 = 2.0 * np.trapz(gp * dgp_da4, _X)
    dI_gp2_da8 = 2.0 * np.trapz(gp * dgp_da8, _X)
    
    # For I_G2 = ∫ Gx^2 dx, use Parseval: I_G2 = (1/(2π)) * ∫ |G(ω)|^2 dω
    # But we're interpolating, so let's use numerical differentiation for this term
    # Or compute it via the relation to g
    
    # Actually, by Parseval's theorem for the Fourier transform:
    # ∫ |G(ω)|^2 dω = 2π * ∫ |g(x)|^2 dx
    # So I_G2 ≈ (1/(2π)) * 2π * ∫ g^2 dx = ∫ g^2 dx
    # Wait, that's not quite right with the normalization. Let me use numerical diff for safety.
    
    # Use central differences for I_G2 gradient only (it's the most complex)
    h = 1e-6
    dI_G2 = np.zeros(3)
    for i in range(3):
        c_plus = coeffs.copy()
        c_plus[i] += h
        _, _, I_G2_p = _compute_integrals(c_plus)
        
        c_minus = coeffs.copy()
        c_minus[i] -= h
        _, _, I_G2_m = _compute_integrals(c_minus)
        
        dI_G2[i] = (I_G2_p - I_G2_m) / (2 * h)
    
    # Now compute the gradient of val = I_g_smooth * I_gp2_smooth * I_G2_smooth
    # dval/dcoeff = dI_g_smooth/dcoeff * I_gp2_smooth * I_G2_smooth + 
    #               I_g_smooth * dI_gp2_smooth/dcoeff * I_G2_smooth + 
    #               I_g_smooth * I_gp2_smooth * dI_G2_smooth/dcoeff
    
    # dI_g_smooth/dcoeff = I_g / I_g_smooth * dI_g/dcoeff
    # dI_gp2_smooth/dcoeff = 0.5 / I_gp2_smooth * dI_gp2/dcoeff
    # dI_G2_smooth/dcoeff = 0.5 / I_G2_smooth * dI_G2/dcoeff
    
    dI_g_smooth = np.array([
        I_g / I_g_smooth * dI_g_da0,
        I_g / I_g_smooth * dI_g_da4,
        I_g / I_g_smooth * dI_g_da8
    ])
    
    dI_gp2_smooth = np.array([
        0.5 / I_gp2_smooth * dI_gp2_da0,
        0.5 / I_gp2_smooth * dI_gp2_da4,
        0.5 / I_gp2_smooth * dI_gp2_da8
    ])
    
    dI_G2_smooth = 0.5 / I_G2_smooth * dI_G2
    
    grad = (dI_g_smooth * I_gp2_smooth * I_G2_smooth + 
            I_g_smooth * dI_gp2_smooth * I_G2_smooth + 
            I_g_smooth * I_gp2_smooth * dI_G2_smooth)
    
    return val, grad

def _objective_and_grad(coeffs):
    """Return (value, grad) for the upper bound C4 given (a0,a4,a8)."""
    return _compute_bound_and_grad(coeffs)

def _objective_only(coeffs):
    """Return just the value for the upper bound C4."""
    val, _ = _compute_bound_and_grad(coeffs, smooth_eps=0.0)
    return val

def construct():
    """Return the three Hermite coefficients (degrees 0,4,8)."""
    # Prior state of the art as a starting point.
    best_coeffs = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    
    try:
        val0, _ = _objective_and_grad(best_coeffs)
        best_val = val0
    except Exception:
        return best_coeffs
    
    # Multi-start L-BFGS-B with diverse starts
    rng = np.random.default_rng(42)
    starts = [best_coeffs.copy()]
    
    # Add starts with different scales
    for scale in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
        starts.append(best_coeffs + rng.normal(0, scale, size=3))
    
    # Add some random starts in a wider box
    for _ in range(5):
        starts.append(rng.uniform(-5, 5, size=3))
    
    # Phase 1: Coarse optimization with smooth surrogate
    for x0 in starts:
        if time.time() > _DEADLINE - _MARGIN - 30:
            break
        try:
            res = minimize(
                _objective_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 200, 'ftol': 1e-10, 'gtol': 1e-8},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            continue
    
    # Phase 2: Polish with tighter tolerance and smaller smooth_eps
    if time.time() < _DEADLINE - _MARGIN:
        try:
            # Use smaller smooth_eps for more accurate gradient
            def _objective_polish(coeffs):
                return _compute_bound_and_grad(coeffs, smooth_eps=1e-8)
            
            res = minimize(
                _objective_polish,
                best_coeffs,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-14, 'gtol': 1e-12},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Safety: ensure finite
    if not np.all(np.isfinite(best_coeffs)):
        return np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    return best_coeffs
# EVOLVE-BLOCK-END
