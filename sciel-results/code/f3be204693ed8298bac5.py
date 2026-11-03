"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hermite_poly(n, x):
    """Compute physicist's Hermite polynomial Hn(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    h0 = np.ones_like(x)
    h1 = 2 * x
    for k in range(2, n + 1):
        h2 = 2 * x * h1 - 2 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def construct():
    start_time = time.time()
    deadline = 175  # seconds, keep margin for evaluator kill at 210s

    # Setup grid
    # N=8192 is faster to compute Gram matrices than 16384 but more precise than 4096.
    # The Gaussian decays fast, so range [-20, 20] is sufficient.
    N = 8192
    x = np.linspace(-20, 20, N)
    dx = x[1] - x[0]
    freqs = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))

    # Precompute Hermite polynomials
    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    gauss = np.exp(-x**2 / 2)

    # Effective basis vectors incorporating the g(0)=0 constraint
    # c12 = -(c0 + 12c4 + 1680c8) / 40320
    v0 = (H0 - H12 / 40320.0) * gauss
    v4 = (H4 - 12.0 * H12 / 40320.0) * gauss
    v8 = (H8 - 1680.0 * H12 / 40320.0) * gauss

    # Precompute Gram matrices
    # M: Norm squared
    # Mx: x^2 weighted norm
    # Mf: |xi|^2 weighted norm (in frequency space)
    
    M = np.zeros((3, 3))
    Mx = np.zeros((3, 3))
    
    # Vectorized computation for speed
    # v0, v4, v8 are shape (N,)
    # We need inner products: v_i * v_j, x^2 * v_i * v_j
    
    # Flatten for easy dot products
    v0_f = v0.flatten()
    v4_f = v4.flatten()
    v8_f = v8.flatten()
    x2_f = (x**2).flatten()
    
    M[0, 0] = np.dot(v0_f, v0_f) * dx
    M[1, 1] = np.dot(v4_f, v4_f) * dx
    M[2, 2] = np.dot(v8_f, v8_f) * dx
    M[0, 1] = M[1, 0] = np.dot(v0_f, v4_f) * dx
    M[0, 2] = M[2, 0] = np.dot(v0_f, v8_f) * dx
    M[1, 2] = M[2, 1] = np.dot(v4_f, v8_f) * dx
    
    Mx[0, 0] = np.dot(x2_f * v0_f, v0_f) * dx
    Mx[1, 1] = np.dot(x2_f * v4_f, v4_f) * dx
    Mx[2, 2] = np.dot(x2_f * v8_f, v8_f) * dx
    Mx[0, 1] = Mx[1, 0] = np.dot(x2_f * v0_f, v4_f) * dx
    Mx[0, 2] = Mx[2, 0] = np.dot(x2_f * v0_f, v8_f) * dx
    Mx[1, 2] = Mx[2, 1] = np.dot(x2_f * v4_f, v8_f) * dx

    # Frequency space matrices
    # G = FFT(v) * dx
    G0 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v0))) * dx
    G4 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v4))) * dx
    G8 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v8))) * dx
    
    f2_f = (freqs**2).flatten()
    
    Mf = np.zeros((3, 3))
    # Note: Parseval's theorem allows computing in freq space directly
    # ||v||^2 = sum |V|^2 * dx / N * N? 
    # Standard discrete Parseval: sum |v_n|^2 = (1/N) sum |V_k|^2
    # Here we have continuous approximations.
    # The code in the prompt used: sum(freqs^2 * |G|^2) * dx
    # Let's stick to the prompt's working formulation to avoid scaling errors.
    
    Mf[0, 0] = np.sum(f2_f * np.abs(G0)**2) * dx
    Mf[1, 1] = np.sum(f2_f * np.abs(G4)**2) * dx
    Mf[2, 2] = np.sum(f2_f * np.abs(G8)**2) * dx
    Mf[0, 1] = Mf[1, 0] = np.sum(f2_f * np.real(G0 * np.conj(G4))) * dx
    Mf[0, 2] = Mf[2, 0] = np.sum(f2_f * np.real(G0 * np.conj(G8))) * dx
    Mf[1, 2] = Mf[2, 1] = np.sum(f2_f * np.real(G4 * np.conj(G8))) * dx

    def compute_bound_from_coeffs(c):
        """Compute bound for a specific coefficient vector c."""
        c = np.array(c)
        norm2 = c @ M @ c
        if norm2 <= 0 or not np.isfinite(norm2):
            return np.inf
        x2_norm = c @ Mx @ c
        freq2_norm = c @ Mf @ c
        
        if not np.isfinite(freq2_norm) or freq2_norm < 0 or not np.isfinite(x2_norm) or x2_norm < 0:
            return np.inf
            
        delta_x = np.sqrt(x2_norm / norm2)
        delta_xi = np.sqrt(freq2_norm / norm2)
        return delta_x * delta_xi

    def compute_bound_from_spherical(theta, phi):
        """Compute bound for spherical parameters."""
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        return compute_bound_from_coeffs([c0, c4, c8])

    # 1. Coarse Grid Search to find the basin of attraction
    # This is deterministic and fast.
    # Theta: 0 to pi, Phi: 0 to 2pi
    # 100x100 grid = 10,000 evaluations. Each is a 3x3 matrix mult. Very fast.
    theta_grid = np.linspace(0.01, np.pi - 0.01, 100)
    phi_grid = np.linspace(0.01, 2 * np.pi - 0.01, 100)
    
    best_bound = np.inf
    best_theta = 0.0
    best_phi = 0.0
    
    # Vectorized grid evaluation if possible, but loop is fine for 10k items
    # We can vectorize the matrix multiplications if we stack the vectors, 
    # but simple loop is safer for memory and speed in Python for 10k small ops.
    for t in theta_grid:
        sin_t = np.sin(t)
        cos_t = np.cos(t)
        for p in phi_grid:
            c0 = sin_t * np.cos(p)
            c4 = sin_t * np.sin(p)
            c8 = cos_t
            # Inline calculation for speed
            norm2 = c0*M[0,0]*c0 + c4*M[1,1]*c4 + c8*M[2,2]*c8 + 2*(c0*M[0,1]*c4 + c0*M[0,2]*c8 + c4*M[1,2]*c8)
            if norm2 <= 0: continue
            x2_norm = c0*Mx[0,0]*c0 + c4*Mx[1,1]*c4 + c8*Mx[2,2]*c8 + 2*(c0*Mx[0,1]*c4 + c0*Mx[0,2]*c8 + c4*Mx[1,2]*c8)
            freq2_norm = c0*Mf[0,0]*c0 + c4*Mf[1,1]*c4 + c8*Mf[2,2]*c8 + 2*(c0*Mf[0,1]*c4 + c0*Mf[0,2]*c8 + c4*Mf[1,2]*c8)
            
            if x2_norm < 0 or freq2_norm < 0: continue
            
            bound = np.sqrt(x2_norm * freq2_norm) / norm2
            
            if bound < best_bound:
                best_bound = bound
                best_theta = t
                best_phi = p

    # 2. Local Refinement with L-BFGS-B
    # L-BFGS-B is excellent for smooth, bounded, low-dimensional problems.
    bounds = [(1e-6, np.pi - 1e-6), (1e-6, 2 * np.pi - 1e-6)]
    
    if time.time() - start_time < deadline - 5:
        try:
            res_lbfgs = minimize(
                compute_bound_from_spherical, 
                x0=[best_theta, best_phi], 
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-15, 'gtol': 1e-15, 'maxiter': 1000, 'maxfun': 10000}
            )
            if res_lbfgs.fun < best_bound:
                best_bound = res_lbfgs.fun
                best_theta, best_phi = res_lbfgs.x
        except Exception:
            pass

    # 3. Final Polish with Nelder-Mead
    # NM is robust and can find slightly better minima if L-BFGS-B stopped early or got stuck on a gradient issue.
    if time.time() - start_time < deadline - 2:
        try:
            res_nm = minimize(
                compute_bound_from_spherical, 
                x0=[best_theta, best_phi], 
                method='Nelder-Mead',
                options={'xatol': 1e-15, 'fatol': 1e-15, 'maxiter': 20000, 'adaptive': True}
            )
            if res_nm.fun < best_bound:
                best_bound = res_nm.fun
                best_theta, best_phi = res_nm.x
        except Exception:
            pass

    # Convert back to Cartesian coefficients
    c0 = np.sin(best_theta) * np.cos(best_phi)
    c4 = np.sin(best_theta) * np.sin(best_phi)
    c8 = np.cos(best_theta)
    
    return np.array([c0, c4, c8])

# EVOLVE-BLOCK-END
