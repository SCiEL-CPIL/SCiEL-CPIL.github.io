"""
Strategy: Low-Rank Fourier Parameterization with Differential Evolution

The current program optimizes 400 individual heights. This approach restricts the
solution space to signals that can be represented by the first K Fourier modes.
This drastically reduces the dimensionality of the search space (from 400 to ~2*K)
and inherently produces smooth signals. 

We use Differential Evolution (a global optimization method) to find good initial
Fourier coefficients, which avoids the local minima traps common in gradient-based
methods on the raw height vector. Then, we perform a local gradient refinement on
the Fourier coefficients to polish the result.

The value to minimize is:
    V(h) = abs(2 * N * max(conv(h, h)) / sum(h)^2)

By parameterizing h via Fourier coefficients c[0..K-1], we ensure h is smooth and
periodic-like (though we don't enforce periodicity in the convolution, the basis
functions are periodic). This structure often leads to lower peak sidelobes in the
autocorrelation compared to arbitrary noise or window functions.
"""

import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.signal import windows as sp_windows

N = 400
_DEADLINE = 145.0  # Wall-clock limit with safety margin


def _evaluate(h):
    """Compute the task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _h_from_fourier(coeffs):
    """
    Reconstruct h from complex Fourier coefficients.
    coeffs: array of shape (K,) containing c[0], c[1], ..., c[K-1].
    We assume real h, so c[-k] = conj(c[k]).
    """
    K = len(coeffs)
    n = np.arange(N)
    h = np.zeros(N)
    # c[0] is the DC component (real)
    if K > 0:
        h += np.real(coeffs[0])
    
    # Add harmonics
    for k in range(1, K):
        ck = coeffs[k]
        # h += ck * exp(2*pi*i*k*n/N) + conj(ck) * exp(-2*pi*i*k*n/N)
        # h += 2 * Re(ck * exp(2*pi*i*k*n/N))
        angle = 2 * np.pi * k * n / N
        h += 2 * np.real(ck * np.exp(1j * angle))
        
    return h


def _objective_fourier(coeffs, beta=50.0):
    """
    Objective function for optimization in Fourier space.
    Uses a softmax surrogate for the max to ensure differentiability.
    """
    h = _h_from_fourier(coeffs)
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return 1e10
    
    conv = np.convolve(h, h)
    m = np.max(conv)
    
    # Softmax approximation of max
    conv_shifted = conv - m
    weights = np.exp(beta * conv_shifted)
    soft_max = m + np.log(np.sum(weights)) / beta
    
    val = 2.0 * N * soft_max / (s * s)
    if not np.isfinite(val):
        return 1e10
    return val


def construct():
    """
    Construct step heights to minimize the autocorrelation inequality value.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # Try different numbers of Fourier coefficients
    # K=1 is just a constant (value 2.0)
    # K=2 adds a cosine/sine
    # Higher K allows more complex shapes
    K_values = [5, 10, 15, 20]
    
    rng = np.random.default_rng(42)
    
    for K in K_values:
        if time.time() > deadline - 20:
            break
            
        # Bounds for Fourier coefficients
        # c[0] (DC) can be larger to scale the signal
        # Higher harmonics should be smaller to maintain smoothness
        bounds = [(0, 10)]  # c[0] real part
        # For k=1..K-1, coeffs are complex (real, imag)
        for k in range(1, K):
            bounds.append((-1.0, 1.0))  # Real part
            bounds.append((-1.0, 1.0))  # Imag part
        
        # Differential Evolution expects a flat vector of reals
        # Our coeffs vector in _h_from_fourier is [c0, c1, ..., cK-1] where c_k is complex?
        # No, _h_from_fourier takes a flat array? 
        # Let's adjust _h_from_fourier to take a flat real array of length 1 + 2*(K-1)
        
        def _h_from_flat_flat(flat_coeffs):
            """
            flat_coeffs: [c0_real, c1_real, c1_imag, c2_real, c2_imag, ...]
            """
            c0 = flat_coeffs[0]
            h = np.full(N, c0)
            n = np.arange(N)
            for k in range(1, K):
                idx = 1 + 2*(k-1)
                cr = flat_coeffs[idx]
                ci = flat_coeffs[idx+1]
                ck = cr + 1j*ci
                angle = 2 * np.pi * k * n / N
                h += 2 * np.real(ck * np.exp(1j * angle))
            return h

        def _obj_flat(flat_coeffs):
            h = _h_from_flat_flat(flat_coeffs)
            s = np.sum(h)
            if s == 0 or not np.isfinite(s):
                return 1e10
            conv = np.convolve(h, h)
            m = np.max(conv)
            conv_shifted = conv - m
            # Use a fixed beta for DE to avoid non-differentiability issues if any, 
            # though DE doesn't use gradients.
            beta = 50.0
            weights = np.exp(beta * conv_shifted)
            soft_max = m + np.log(np.sum(weights)) / beta
            val = 2.0 * N * soft_max / (s * s)
            if not np.isfinite(val):
                return 1e10
            return val

        # Initial population for DE
        # We can seed with some known good structures like windows
        # But DE generates its own population. We'll just run it.
        
        try:
            # Use a relatively small population for speed
            popsize = 15
            res = differential_evolution(
                _obj_flat,
                bounds,
                maxiter=50,
                popsize=popsize,
                tol=1e-4,
                seed=42,
                polish=False, # We will do our own polish
                init='latinhypercube'
            )
            
            if res.success or res.fun < best_val:
                best_flat = res.x
                best_h_de = _h_from_flat_flat(best_flat)
                best_val_de = _evaluate(best_h_de)
                
                if best_val_de < best_val:
                    best_val = best_val_de
                    best_h = best_h_de.copy()
                    
        except Exception:
            continue
        
        # Local refinement using L-BFGS-B on the flat Fourier coefficients
        if time.time() < deadline - 10:
            def _obj_refine(flat_coeffs):
                if time.time() > deadline - 5:
                    raise ValueError("Timeout")
                return _obj_flat(flat_coeffs)
            
            try:
                res_refine = minimize(
                    _obj_refine,
                    best_flat,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 200}
                )
                if res_refine.success:
                    best_flat_ref = res_refine.x
                    best_h_ref = _h_from_flat_flat(best_flat_ref)
                    best_val_ref = _evaluate(best_h_ref)
                    
                    if best_val_ref < best_val:
                        best_val = best_val_ref
                        best_h = best_h_ref.copy()
                        
            except Exception:
                pass

    # Final fallback check
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
        
    return best_h
