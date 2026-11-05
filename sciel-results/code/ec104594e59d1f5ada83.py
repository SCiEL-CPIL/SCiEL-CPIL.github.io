"""Optimized program for the third autocorrelation inequality task.
Uses IDFT sparse spectrum representation to directly shape autocorrelation sidelobes.
"""
import numpy as np
import time
import sys

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    def compute_value(h):
        """Compute the objective value (lower is better)."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = np.convolve(h, h)
        return abs(2.0 * N * np.max(conv) / (s * s))
    
    def smooth_max(x, tau=0.01):
        """Soft approximation of max using log-sum-exp."""
        x_max = np.max(x)
        return x_max + tau * np.log(np.sum(np.exp((x - x_max) / tau)))
    
    def spectrum_to_h(coeffs):
        """Convert complex spectrum coefficients to time-domain h.
        coeffs: complex array of length N (the DFT coefficients)
        h = IDFT(coeffs), but we want real-valued h, so we enforce conjugate symmetry.
        Actually, let's just use the full IDFT and take real part, or better:
        represent h via its DFT and ensure h is real by construction.
        
        We'll parameterize as: h = IDFT(S) where S has conjugate symmetry.
        To keep it simple and differentiable, we'll use a parameterization where
        we specify the first N//2+1 complex coefficients and enforce symmetry.
        
        Alternative: just use N real parameters for the DFT of a real signal.
        For a real signal of length N, the DFT has N/2+1 independent complex values
        (with the DC and Nyquist being real). Let's use that.
        """
        # coeffs shape: (N//2 + 1,) complex
        # Build full spectrum with conjugate symmetry
        n_half = N // 2
        S = np.zeros(N, dtype=complex)
        # DC component (index 0) - real
        S[0] = coeffs[0].real
        # For k=1 to n_half-1: S[k] = coeffs[k], S[N-k] = conj(coeffs[k])
        for k in range(1, n_half):
            S[k] = coeffs[k]
            S[N - k] = np.conj(coeffs[k])
        # Nyquist (index n_half) - real
        if N % 2 == 0:
            S[n_half] = coeffs[n_half].real
        # IDFT
        h = np.fft.ifft(S).real
        return h
    
    def spectrum_objective(params, tau=0.05):
        """Objective for spectrum parameters.
        params: real array of length 2*(N//2+1) representing real and imag parts of coeffs
        """
        n_half = N // 2
        n_coeffs = n_half + 1
        # Reshape params to complex coeffs
        coeffs = params[:2*n_coeffs].reshape(n_coeffs, 2)
        coeffs_c = coeffs[:, 0] + 1j * coeffs[:, 1]
        h = spectrum_to_h(coeffs_c)
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)
    
    def sparse_spectrum_objective(params, tau=0.05):
        """Objective for sparse spectrum: only a few frequency bins are non-zero.
        params: [k1, a1r, a1i, k2, a2r, a2i, ..., kM, aMr, aMi]
        where k_j are integer frequency indices (0 to N-1), a_j are complex amplitudes.
        We'll use a fixed set of M bins to keep dimensionality manageable.
        """
        # Use M = 10 bins
        M = 10
        # params: 3*M floats: (k, ar, ai) for each bin
        S = np.zeros(N, dtype=complex)
        for j in range(M):
            k = int(round(params[3*j])) % N
            ar = params[3*j + 1]
            ai = params[3*j + 2]
            S[k] += ar + 1j * ai
        # Ensure conjugate symmetry for real h
        # For each non-zero bin k, add conjugate at N-k
        S2 = np.zeros(N, dtype=complex)
        for k in range(N):
            if abs(S[k]) > 1e-15:
                S2[k] += S[k]
                if k != 0 and k != N // 2:
                    S2[N - k] += np.conj(S[k])
        S = S2
        h = np.fft.ifft(S).real
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)
    
    def objective_full(h, tau=0.05):
        """Objective for direct h optimization."""
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    rng = np.random.RandomState(42)
    from scipy.optimize import minimize
    
    # Phase 1: Sparse spectrum optimization
    # Try different numbers of sparse bins
    M_list = [5, 8, 12]
    for M in M_list:
        if time.time() > deadline - 80.0:
            break
        
        n_params = 3 * M
        n_trials = 3
        for trial in range(n_trials):
            if time.time() > deadline - 80.0:
                break
            
            # Initialize: random bins with random amplitudes
            params0 = np.zeros(n_params)
            for j in range(M):
                params0[3*j] = rng.randint(0, N)
                params0[3*j + 1] = rng.uniform(-1.0, 1.0)
                params0[3*j + 2] = rng.uniform(-1.0, 1.0)
            
            try:
                result = minimize(
                    sparse_spectrum_objective,
                    params0,
                    args=(0.05,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                # Extract h
                S = np.zeros(N, dtype=complex)
                for j in range(M):
                    k = int(round(result.x[3*j])) % N
                    ar = result.x[3*j + 1]
                    ai = result.x[3*j + 2]
                    S[k] += ar + 1j * ai
                S2 = np.zeros(N, dtype=complex)
                for k in range(N):
                    if abs(S[k]) > 1e-15:
                        S2[k] += S[k]
                        if k != 0 and k != N // 2:
                            S2[N - k] += np.conj(S[k])
                S = S2
                h_opt = np.fft.ifft(S).real
                val_opt = compute_value(h_opt)
                if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                    best_h = h_opt.copy()
                    best_val = val_opt
                    print(f"Sparse M={M} trial {trial}: improved to {best_val:.6f}", file=sys.stderr)
            except Exception as e:
                print(f"Sparse M={M} trial {trial}: error {e}", file=sys.stderr)
    
    # Phase 2: Full spectrum optimization (all N/2+1 bins)
    if time.time() < deadline - 40.0:
        n_half = N // 2
        n_coeffs = n_half + 1
        n_params = 2 * n_coeffs
        
        # Initialize with flat spectrum (which gives h = ones)
        params0 = np.zeros(n_params)
        params0[0] = N  # DC component for sum(h) = N
        # Actually, for h = ones, the DFT is: S[0] = N, S[k] = 0 for k > 0
        # Let's start with a small perturbation
        params0[0] = float(N)
        
        try:
            result = minimize(
                spectrum_objective,
                params0,
                args=(0.05,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            # Extract h
            coeffs = result.x[:2*n_coeffs].reshape(n_coeffs, 2)
            coeffs_c = coeffs[:, 0] + 1j * coeffs[:, 1]
            h_opt = spectrum_to_h(coeffs_c)
            val_opt = compute_value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"Full spectrum: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Full spectrum: error {e}", file=sys.stderr)
    
    # Phase 3: Local refinement in time domain
    if time.time() < deadline - 20.0:
        for tau in [0.02, 0.01, 0.005, 0.001]:
            if time.time() > deadline - 10.0:
                break
            try:
                result = minimize(
                    objective_full,
                    best_h,
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                h_opt = result.x
                val_opt = compute_value(h_opt)
                if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                    best_h = h_opt.copy()
                    best_val = val_opt
                    print(f"Polish tau={tau}: improved to {best_val:.6f}", file=sys.stderr)
            except Exception as e:
                print(f"Polish tau={tau}: error {e}", file=sys.stderr)
    
    # Final safety check
    if time.time() > deadline - 2.0:
        return np.ones(N)
    
    if best_val < 2.0 and np.all(np.isfinite(best_h)) and abs(np.sum(best_h)) > 1e-6:
        return best_h
    else:
        return np.ones(N)
    
# EVOLVE-BLOCK-END
