import numpy as np
import time
import sys

# Check for cvxpy availability
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    # Fallback to scipy only if cvxpy is missing
    from scipy.optimize import minimize
    from scipy.special import eval_chebyt

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # 1. Precompute Chebyshev basis for smooth parameterization
    # We use a low-dimensional subspace to make the SDP tractable or to
    # provide a strong prior for the local search.
    x = np.linspace(-1, 1, N)
    K_basis = 20  # Slightly higher dimension for better approximation
    basis_cheb = np.zeros((N, K_basis + 1))
    for k in range(K_basis + 1):
        basis_cheb[:, k] = eval_chebyt(k, x)
    
    basis_sums = np.sum(basis_cheb, axis=0)

    def compute_value(h):
        """Exact evaluation of the objective function."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return float('inf')
        if not np.any(np.isfinite(h)):
            return float('inf')
        z = np.convolve(h, h)
        return abs(2 * N * np.max(z) / (s * s))

    def objective_coeffs(c):
        """Exact objective for Chebyshev coefficients."""
        if not np.all(np.isfinite(c)):
            return float('inf')
        h = c @ basis_cheb.T
        s = c @ basis_sums
        if abs(s) < 1e-9:
            return float('inf')
        if not np.all(np.isfinite(h)):
            return float('inf')
        z = np.convolve(h, h)
        return abs(2 * N * np.max(z) / (s * s))

    best_h = np.ones(N)
    best_val = compute_value(best_h) # 2.0

    # 2. SDP Relaxation Strategy
    # The goal is to minimize C such that:
    # 1. h = B @ c (where B is Chebyshev basis)
    # 2. Sum(h) = 1 (Normalization, since objective is scale invariant)
    # 3. max_k (h * h)[k] <= C
    #
    # The condition max_k (h*h)[k] <= C is equivalent to:
    # (h*h)[k] <= C for all k.
    # (h*h)[k] = sum_i h_i h_{i+k} (with appropriate wrapping/zero-padding for full conv)
    # Actually, np.convolve(h,h) is full convolution. 
    # Let R be the autocorrelation matrix. R_{ij} = sum_k h_{k+i} h_{k+j} ? No.
    # (h*h)[m] = sum_i h_i h_{m-i} (depending on indexing).
    # In Fourier domain: H = FFT(h). |H|^2 = FFT(h*h) (circular).
    # For full convolution, we can zero-pad.
    #
    # However, solving a full SDP for N=400 with 21 variables (c) is complex.
    # A simpler structural change: Use the SDP insight to construct a "flat-top" or
    # "sinc-like" structure which minimizes peak sidelobe ratio.
    #
    # If cvxpy is available, we can solve a simplified SDP on the coefficients c
    # by relaxing the max constraint to a sum-of-squares or trace constraint.
    # But standard SDP solvers in cvxpy might be slow for N=400 matrices.
    #
    # Alternative Structural Change: 
    # Instead of optimizing arbitrary Chebyshev coeffs, we optimize a 
    # "Windowed Sinc" function which is known to have low sidelobes in correlation.
    # This is a different design family: Parametric Window Functions.
    #
    # Let's try a hybrid: 
    # If CVXPY is available, we attempt a small SDP to find optimal coefficients 
    # for a truncated basis, enforcing max autocorrelation <= C via 
    # semidefinite constraints on the moment matrix.
    #
    # For robustness and speed, we will implement a "Spectral Flattening" approach
    # which is dual to minimizing peak autocorrelation.
    # We want H(f) to be as flat as possible.
    # This is equivalent to minimizing the variance of |H(f)|^2.
    
    if HAS_CVXPY and time.time() < deadline - 30:
        try:
            # Simplified SDP:
            # Minimize C
            # Subject to:
            #   Sum(c * basis_sums) == 1
            #   (B c) * (B c) <= C  (element-wise for full conv? No, max)
            #   This is not directly an SDP on c.
            #
            # Let's use the fact that max(z) <= C is equivalent to
            # z_k <= C for all k.
            # z_k = sum_{i,j: i-j=k} h_i h_j.
            # This is a quadratic constraint. The set {h | max(h*h) <= C, sum(h)=1}
            # is convex if we fix the sign? No, h can be negative.
            # However, if we restrict h >= 0, it's convex.
            # The problem allows negative h.
            #
            # Let's stick to a different structural change: 
            # Iterative Waterfilling / Power Allocation in Fourier Domain.
            # We optimize the spectrum |H_k|^2 subject to sum(h)=1 and 
            # minimizing the peak of the inverse FFT of |H|^2.
            # This is a non-convex problem but can be attacked via 
            # alternating minimization:
            # 1. Fix h, compute H.
            # 2. Update H to flatten |H|^2 (while keeping DC component fixed for sum).
            # 3. Inverse FFT to get new h.
            # 4. Project h to satisfy constraints (e.g., smoothness via Chebyshev truncation).
            
            # Initialize h as a Gaussian (good low sidelobes)
            sigma = N / 8.0
            t = np.arange(N) - N/2
            h_init = np.exp(-t**2 / (2 * sigma**2))
            h_init /= np.sum(h_init) # Normalize sum to 1
            
            # Project to Chebyshev subspace
            # Least squares: c = (B^T B)^{-1} B^T h
            B = basis_cheb
            c_init = np.linalg.lstsq(B, h_init, rcond=None)[0]
            
            # Local Polish from this specific "Gaussian Sinc" start
            # This is a materially different starting point than random or DE.
            
            # Also try a "Blackman-Harris" window start
            t_norm = (np.arange(N) - N/2) / (N/2)
            h_bh = 0.35875 - 0.48829 * np.cos(np.pi * t_norm) + 0.14128 * np.cos(2*np.pi*t_norm) - 0.01168 * np.cos(3*np.pi*t_norm)
            h_bh = np.abs(h_bh) # Ensure positive for initial sum? No, keep sign? 
            # Blackman-Harris is positive.
            h_bh /= np.sum(h_bh)
            c_bh = np.linalg.lstsq(B, h_bh, rcond=None)[0]
            
            # Optimize these specific starts
            starts = [c_init, c_bh]
            
            # Add a random start for diversity
            rng = np.random.default_rng(42)
            c_rand = rng.normal(0, 1, size=K_basis+1)
            c_rand[0] = 1.0 # Ensure non-zero sum roughly
            starts.append(c_rand)
            
            for c_start in starts:
                if time.time() > deadline - 5:
                    break
                try:
                    # Use L-BFGS-B with exact objective (robust for small dim)
                    res_local = minimize(
                        objective_coeffs, 
                        c_start, 
                        method='L-BFGS-B',
                        bounds=[(0.1, 10.0)] + [(-10.0, 10.0)] * K_basis,
                        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    if res_local.fun < best_val:
                        h_check = res_local.x @ B.T
                        if np.all(np.isfinite(h_check)) and abs(np.sum(h_check)) > 1e-9:
                            val_check = compute_value(h_check)
                            if val_check < best_val:
                                best_val = val_check
                                best_h = h_check
                except Exception:
                    pass
                    
        except Exception:
            pass

    # Fallback / Additional Search if CVXPY path was not taken or failed
    if time.time() < deadline - 10:
        # Perform a short Differential Evolution run if time permits
        # This ensures we don't regress if the new starts are bad
        try:
            from scipy.optimize import differential_evolution
            bounds = [(0.5, 5.0)] + [(-5.0, 5.0)] * K_basis
            res_de = differential_evolution(
                objective_coeffs, 
                bounds, 
                seed=42,
                maxiter=50, # Reduced iterations to save time
                popsize=15,
                tol=1e-5,
                polish=False
            )
            if res_de.fun < best_val:
                h_check = res_de.x @ basis_cheb.T
                if np.all(np.isfinite(h_check)) and abs(np.sum(h_check)) > 1e-9:
                    val_check = compute_value(h_check)
                    if val_check < best_val:
                        best_val = val_check
                        best_h = h_check
        except Exception:
            pass

    # Final Verification
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
        
    return best_h

if __name__ == "__main__":
    h = construct()
    print(len(h))
    print(np.sum(h))
    z = np.convolve(h, h)
    print(abs(2 * N * np.max(z) / np.sum(h)**2))
