import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    def compute_value(h):
        """Exact evaluation of the objective function."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return float('inf')
        if not np.all(np.isfinite(h)):
            return float('inf')
        z = np.convolve(h, h)
        return abs(2 * N * np.max(z) / (s * s))

    def qp_objective(h):
        """
        Objective: Minimize ||h * h||_2^2 subject to sum(h) = 1.
        This is a convex function of h.
        """
        # Compute autoconvolution via FFT for speed
        fft_len = 2 * N
        H = np.fft.fft(h, n=fft_len)
        Z = np.fft.ifft(H * H)
        z = Z.real[:2 * N - 1]
        return np.dot(z, z)

    def qp_gradient(h):
        """
        Gradient of ||h * h||_2^2 with respect to h.
        Let z = h * h.
        d/dh_j (sum_k z_k^2) = 2 * sum_k z_k * (d z_k / d h_j)
        z_k = sum_i h_i h_{k-i}
        d z_k / d h_j = h_{k-j} + h_{j-k} (with appropriate boundary handling, effectively 2 * (h * h)_k if we consider the full symmetric convolution structure, but let's use the FFT derivative trick).
        
        Actually, it's easier to note that:
        ||h * h||_2^2 = sum_k (sum_i h_i h_{k-i})^2
        Using Parseval's theorem in the discrete sense or simply differentiating:
        Let H = FFT(h). Z = H^2.
        Energy = sum |Z|^2 (if we consider complex, but here real).
        Actually, for real signals, the L2 norm of the convolution is related to the L2 norm of the product of spectra.
        
        Let's derive the gradient properly.
        f(h) = ||h * h||_2^2
        Let z = h * h.
        f = z^T z
        df = 2 z^T dz
        dz_k = sum_i (h_{k-i} dh_i + h_i dh_{k-i}) = 2 sum_i h_{k-i} dh_i (by symmetry of convolution)
        So dz = 2 (h * dh)
        df = 2 z^T (2 (h * dh)) = 4 (z * h)^T dh (using commutativity of convolution)
        So grad = 4 * (z * h)
        
        Let's verify this.
        If h is a delta function, z is delta. z*h is delta. Grad is 4*delta.
        If h is constant c, z is triangle. z*h is pyramid.
        
        We can compute z*h via FFT.
        """
        fft_len = 2 * N
        H = np.fft.fft(h, n=fft_len)
        Z = np.fft.ifft(H * H)
        z = Z.real[:2 * N - 1]
        
        # Compute z * h
        # z has length 2N-1, h has length N.
        # Convolution z * h has length 3N-2.
        # We only need the first N components? No, the gradient is a vector of length N.
        # The term (z * h)_j for j=0..N-1.
        
        # Let's compute the full convolution z * h using FFT.
        # Length of z is 2N-1, length of h is N. Result length 3N-2.
        # Next power of 2 for 3N-2 = 1198 is 2048.
        
        fft_len2 = 4 * N # Safe upper bound
        
        Z_fft = np.fft.fft(z, n=fft_len2)
        H_fft = np.fft.fft(h, n=fft_len2)
        
        ZH = np.fft.ifft(Z_fft * H_fft)
        zh = ZH.real[:N]
        
        return 4.0 * zh

    # 1. Initial Guess
    # We need sum(h) = 1.
    # Start with a flat distribution (uniform) or a Gaussian-like bump.
    
    candidates = []
    
    # Candidate 1: Uniform (normalized)
    h_uni = np.ones(N) / N
    candidates.append(h_uni)
    
    # Candidate 2: Gaussian bump (normalized)
    x = np.linspace(-3, 3, N)
    h_gauss = np.exp(-x**2 / 2)
    h_gauss = h_gauss / np.sum(h_gauss)
    candidates.append(h_gauss)
    
    # Candidate 3: Sinc-like (windowed)
    # A sinc function has flat spectrum, but we want to minimize energy.
    # A raised cosine is good.
    t = np.linspace(0, 1, N)
    h_cos = 0.5 * (1 + np.cos(2 * np.pi * t))
    h_cos = h_cos / np.sum(h_cos)
    candidates.append(h_cos)

    best_h = None
    best_val = float('inf')

    # Evaluate initial candidates with exact metric to pick best start for QP
    for h_cand in candidates:
        v = compute_value(h_cand)
        if v < best_val:
            best_val = v
            best_h = h_cand.copy()

    # 2. Optimization using SLSQP
    # SLSQP can handle the constraint sum(h)=1.
    # We will minimize the convex QP objective.
    
    constraint = {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0, 'jac': lambda h: np.ones(N)}
    
    # We might also want to ensure h is finite, which is handled by the solver.
    # Bounds? No bounds specified in task, but real values.
    # SLSQP might wander, but the convex objective should keep it bounded if sum is fixed.
    
    starts_to_optimize = [best_h]
    
    # Add a few more diverse starts if time permits
    if time.time() < deadline - 60:
        rng = np.random.default_rng(42)
        for _ in range(3):
            h_rand = rng.normal(0, 1, N)
            h_rand = h_rand / np.sum(h_rand)
            starts_to_optimize.append(h_rand)

    for h_start in starts_to_optimize:
        if time.time() > deadline - 10:
            break
            
        try:
            # SLSQP with gradient
            res = minimize(
                qp_objective, 
                h_start, 
                jac=qp_gradient,
                method='SLSQP',
                constraints=[constraint],
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.success and res.fun < float('inf'):
                h_check = res.x
                if np.all(np.isfinite(h_check)) and abs(np.sum(h_check)) > 1e-9:
                    val_check = compute_value(h_check)
                    if val_check < best_val:
                        best_val = val_check
                        best_h = h_check.copy()
        except Exception:
            pass

    # 3. Refinement
    # The QP minimizes L2 norm of autocorrelation. 
    # It might not minimize the MAX. 
    # We can take the result of the QP and use it as a start for a local search on the actual objective (max) if time permits.
    # But the directive says "Formulate ... as a convex QP ... Solve this QP".
    # The QP solution is a strong candidate.
    # Let's try one more step: use the QP result as start for a few iterations of the smooth max objective if time allows.
    
    if best_h is not None and time.time() < deadline - 20:
        def smooth_objective(h):
            """
            Differentiable surrogate for the objective function.
            Uses log-sum-exp to approximate max(autoconv) smoothly.
            """
            if not np.all(np.isfinite(h)):
                return float('inf')
            
            s = np.sum(h)
            if abs(s) < 1e-9:
                return float('inf')
                
            fft_len = 2 * N
            H_lin = np.fft.fft(h, n=fft_len)
            Z_lin = np.fft.ifft(H_lin * H_lin)
            z_lin = Z_lin.real[:2 * N - 1]
            
            tau = 0.5  # Sharper than before
            z_scaled = z_lin / tau
            z_max_scaled = np.max(z_scaled)
            exp_terms = np.exp(z_scaled - z_max_scaled)
            sum_exp = np.sum(exp_terms)
            
            if sum_exp == 0:
                log_sum_exp = z_max_scaled
            else:
                log_sum_exp = z_max_scaled + np.log(sum_exp)
                
            max_z_smooth = log_sum_exp * tau
            val = abs(2 * N * max_z_smooth / (s * s))
            return val

        def smooth_grad(h):
            # Gradient of the smooth objective is complex to derive analytically for LSE.
            # We can use numerical gradient or just skip this if QP is good enough.
            # Given time constraints, we might just rely on the QP result.
            # However, if the QP result is local for the max, a quick L-BFGS-B on the smooth max might help.
            # Let's try L-BFGS-B with numerical gradient? No, too slow for 400 dims.
            # Let's just return the QP result.
            pass

    # Ensure validity
    if best_h is None or not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
        
    return best_h

if __name__ == "__main__":
    h = construct()
    print(len(h))
    print(np.sum(h))
    z = np.convolve(h, h)
    print(abs(2 * N * np.max(z) / np.sum(h)**2))
