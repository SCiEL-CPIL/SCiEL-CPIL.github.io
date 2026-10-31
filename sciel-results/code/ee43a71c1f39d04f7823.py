import numpy as np
from scipy import optimize
from scipy.optimize import linprog
import time

N = 600

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    conv = np.convolve(h, h, 'full')
    total = np.sum(h)
    if total == 0:
        return 0.0
    return np.max(conv) / (total * total)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, leaving 10s margin
    
    # Build Legendre polynomial basis on [0, N)
    # P_k(x) for k=0,...,K-1 where x = (j + 0.5)/N * 2 - 1
    K = 20  # Number of Legendre polynomials
    x = (np.arange(N) + 0.5) / N * 2 - 1  # Points in [-1, 1]
    
    # Compute Legendre polynomials P_0, P_1, ..., P_{K-1} at all N points
    # P_0 = 1
    # P_1 = x
    # P_{n+1} = ((2n+1)*x*P_n - n*P_{n-1})/(n+1)
    P = np.zeros((K, N))
    P[0, :] = 1.0
    if K > 1:
        P[1, :] = x
    for n in range(1, K - 1):
        P[n+1, :] = ((2*n+1) * x * P[n, :] - n * P[n-1, :]) / (n+1)
    
    # Orthonormalize: Legendre polynomials are orthogonal but not orthonormal
    # <P_m, P_n> = delta_mn * 2/(2n+1) for continuous, but we have discrete
    # For discrete: compute Gram matrix and use Cholesky
    Gram = P @ P.T
    # Regularize
    Gram += 1e-10 * np.eye(K)
    L = np.linalg.cholesky(Gram)
    # Orthonormal basis: Q = P @ inv(L).T so that Q @ Q.T = I
    Q = P @ np.linalg.inv(L).T
    
    # Initial coefficients: start with uniform (only c_0 nonzero)
    # h = Q @ c, we want h >= 0
    # For uniform h: h_j = 1/N for all j
    # c_0 = <h, Q_0> = sum_j (1/N) * Q_0[j]
    c = np.zeros(K)
    c[0] = np.sum(Q[0, :] / N)
    
    def h_from_c(c):
        """Convert coefficients to h vector"""
        return Q @ c
    
    def soft_objective_c(c, T=0.01):
        """Soft-max objective in coefficient space"""
        h = h_from_c(c)
        # Project to nonnegative
        h = np.maximum(h, 0)
        total = np.sum(h)
        if total < 1e-10:
            return 1e10
        conv = np.convolve(h, h, 'full')
        conv_max = np.max(conv)
        shifted = (conv - conv_max) / T
        shifted = np.clip(shifted, -50, 50)
        soft_max = conv_max + T * np.log(np.sum(np.exp(shifted)))
        return soft_max / (total * total)
    
    def exact_objective_c(c):
        """Exact objective in coefficient space"""
        h = h_from_c(c)
        h = np.maximum(h, 0)
        return compute_autoconv_max(h)
    
    # Projected gradient descent on coefficients
    # h = Q @ c, project h >= 0, then map back to c-space
    def project_to_nn(c):
        """Project h = Q@c to nonnegative, return updated c"""
        h = h_from_c(c)
        h_nn = np.maximum(h, 0)
        # Project back to coefficient space: c_new = Q.T @ h_nn (since Q is orthonormal)
        c_new = Q.T @ h_nn
        return c_new
    
    def projected_gradient_step(c, lr, T=0.01):
        """One projected gradient step"""
        h = h_from_c(c)
        h = np.maximum(h, 0)
        total = np.sum(h)
        if total < 1e-10:
            return c
        h_norm = h / total
        conv = np.convolve(h_norm, h_norm, 'full')
        conv_max = np.max(conv)
        
        # Gradient of soft-max w.r.t. h
        shifted = (conv - conv_max) / T
        shifted = np.clip(shifted, -50, 50)
        weights = np.exp(shifted) / np.sum(np.exp(shifted))
        # d(soft_max)/d(conv_j) = weights_j
        # d(conv_j)/d(h_i) = h_norm[j-i] (for appropriate indices)
        # This is complicated; use finite differences or FFT
        
        # Simpler: use FFT-based gradient
        H = np.fft.fft(h_norm)
        power = np.abs(H)**2
        pmax = np.max(power)
        shifted_p = (power - pmax) / T
        shifted_p = np.clip(shifted_p, -50, 50)
        w = np.exp(shifted_p) / np.sum(np.exp(shifted_p))
        
        # Gradient of max|H|^2 w.r.t. h: 
        # d|H(k)|^2/dh_j = 2*Re{conj(H(k)) * exp(-2*pi*i*j*k/N)}
        grad_h = np.zeros(N)
        for k in range(N):
            if w[k] > 1e-8:
                phase = np.exp(-2j * np.pi * np.arange(N) * k / N)
                grad_h += w[k] * 2 * np.real(np.conj(H[k]) * phase)
        
        # Map gradient to coefficient space
        grad_c = Q.T @ grad_h
        c_new = c - lr * grad_c
        c_new = project_to_nn(c_new)
        return c_new
    
    # Adaptive learning rate projected gradient
    best_c = c.copy()
    best_score = exact_objective_c(best_c)
    
    # Try multiple initial conditions
    np.random.seed(42)
    initializations = [c.copy()]
    
    # Add some random perturbations
    for _ in range(5):
        c_rand = c.copy()
        for i in range(1, K):
            c_rand[i] += 0.01 * np.random.randn()
        initializations.append(c_rand)
    
    # Also try different K values
    for K_try in [10, 15, 25, 30]:
        if K_try == K:
            continue
        x = (np.arange(N) + 0.5) / N * 2 - 1
        P_try = np.zeros((K_try, N))
        P_try[0, :] = 1.0
        if K_try > 1:
            P_try[1, :] = x
        for n in range(1, K_try - 1):
            P_try[n+1, :] = ((2*n+1) * x * P_try[n, :] - n * P_try[n-1, :]) / (n+1)
        Gram_try = P_try @ P_try.T + 1e-10 * np.eye(K_try)
        L_try = np.linalg.cholesky(Gram_try)
        Q_try = P_try @ np.linalg.inv(L_try).T
        
        c_try = np.zeros(K_try)
        c_try[0] = np.sum(Q_try[0, :] / N)
        
        def h_from_c_try(c):
            return Q_try @ c
        
        def exact_obj_try(c):
            h = h_from_c_try(c)
            h = np.maximum(h, 0)
            return compute_autoconv_max(h)
        
        def project_try(c):
            h = h_from_c_try(c)
            h_nn = np.maximum(h, 0)
            return Q_try.T @ h_nn
        
        def pgd_step_try(c, lr, T=0.01):
            h = h_from_c_try(c)
            h = np.maximum(h, 0)
            total = np.sum(h)
            if total < 1e-10:
                return c
            h_norm = h / total
            H = np.fft.fft(h_norm)
            power = np.abs(H)**2
            pmax = np.max(power)
            shifted_p = (power - pmax) / T
            shifted_p = np.clip(shifted_p, -50, 50)
            w = np.exp(shifted_p) / np.sum(np.exp(shifted_p))
            
            grad_h = np.zeros(N)
            for k in range(N):
                if w[k] > 1e-8:
                    phase = np.exp(-2j * np.pi * np.arange(N) * k / N)
                    grad_h += w[k] * 2 * np.real(np.conj(H[k]) * phase)
            
            grad_c = Q_try.T @ grad_h
            c_new = c - lr * grad_h @ np.linalg.inv(Q_try.T @ Q_try)  # approximate
            c_new = c - lr * grad_c
            c_new = project_try(c_new)
            return c_new
        
        score_try = exact_obj_try(c_try)
        if score_try < best_score:
            best_score = score_try
            best_h = np.maximum(h_from_c_try(c_try), 0)
        
        # Run PGD
        lr = 0.1
        for it in range(2000):
            if time.time() > deadline - 15:
                break
            c_try = pgd_step_try(c_try, lr, T=0.01)
            if it % 100 == 0:
                s = exact_obj_try(c_try)
                if s < best_score:
                    best_score = s
                    best_h = np.maximum(h_from_c_try(c_try), 0)
            # Adaptive lr
            if it % 50 == 0 and it > 0:
                lr *= 0.95
    
    # Now optimize the main K=20 basis
    lr = 0.1
    for it in range(3000):
        if time.time() > deadline - 15:
            break
        c = projected_gradient_step(c, lr, T=0.01)
        if it % 50 == 0:
            s = exact_objective_c(c)
            if s < best_score:
                best_score = s
                best_h = np.maximum(h_from_c(c), 0)
        if it % 100 == 0 and it > 0:
            lr *= 0.95
    
    # Final polish with L-BFGS-B in full h-space
    if time.time() < deadline - 10:
        h0 = best_h.copy()
        
        def objective_h(x, T=0.005):
            h = np.maximum(x, 0)
            total = np.sum(h)
            if total < 1e-10:
                return 1e10
            conv = np.convolve(h, h, 'full')
            conv_max = np.max(conv)
            shifted = (conv - conv_max) / T
            shifted = np.clip(shifted, -50, 50)
            soft_max = conv_max + T * np.log(np.sum(np.exp(shifted)))
            return soft_max / (total * total)
        
        try:
            result = optimize.minimize(
                objective_h, h0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_new = np.maximum(result.x, 0)
            score_new = compute_autoconv_max(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new.copy()
        except Exception:
            pass
    
    return list(best_h)
