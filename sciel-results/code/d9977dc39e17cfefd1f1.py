import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 400

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin (evaluator kills at 150s)
    
    # Precompute Walsh-Hadamard basis of size N
    # W is N x N, W[i,j] = (-1)^(popcount(i & j))
    # This is the Hadamard matrix of order N (N must be power of 2, 400 is not)
    # We need to handle N=400. We can use a rectangular transform or just use FFT-based approach.
    # Since N=400 is not a power of 2, let's use a different structured basis.
    # Or we can pad to 512 and take first 400? No, that changes the problem.
    # Let's use a simpler structured approach: low-frequency DFT basis.
    # Represent x = H @ c where H is N x K with K << N, columns are first K DFT vectors.
    # Actually, the directive suggests Walsh-Hadamard. For N=400, we can use a block-Hadamard or just random orthogonal basis.
    # Let's use a random orthogonal basis but with low-rank constraint.
    # Or better: use the first K columns of a DFT matrix.
    
    # Let's use K = 20 basis functions (low-frequency DFT)
    K = 20
    
    # Build basis matrix B (N x K) where B[:, k] = [exp(2*pi*i*k*n/N) for n in range(N)]
    # For real-valued x, we can use cos and sin terms.
    # x[n] = sum_k (a_k * cos(2*pi*k*n/N) + b_k * sin(2*pi*k*n/N))
    # Then sign(x) gives our binary sequence.
    
    # Actually, let's follow the directive more closely: sparse low-rank combination.
    # We'll optimize continuous coefficients c of length K, then take sign of B @ c.
    
    # Build real-valued DFT basis
    n = np.arange(N)
    k = np.arange(K)
    # B[n, k] = cos(2*pi*k*n/N) for even k, sin for odd? 
    # Let's just use cos terms for simplicity, or both.
    # Let's use both cos and sin: 2*K coefficients
    K_total = 2 * K
    B = np.zeros((N, K_total))
    for kk in range(K):
        freq = 2 * np.pi * kk / N
        B[:, 2*kk] = np.cos(freq * n)
        B[:, 2*kk+1] = np.sin(freq * n)
    
    # Objective: given coefficients c, compute x = B @ c, then f = sign(x)
    # Compute autocorrelation of f, return max sidelobe + regularization
    def objective(c):
        x = B @ c
        f = np.sign(x)
        f[f == 0] = 1.0
        # Compute autocorrelation via FFT
        n_fft = 2 * N
        F = np.fft.fft(f, n=n_fft)
        R = np.fft.ifft(F * np.conj(F)).real
        # Sidelobes: all except index 0 and N (for linear autocorrelation, index 0 is main lobe)
        # Actually for linear autocorrelation of length N, R[0] is main lobe = sum(f^2) = N
        # R[1:N] are positive lags, R[N+1:] are negative lags
        # Max sidelobe
        max_sidelobe = np.max(np.abs(np.concatenate([R[1:N], R[N+1:2*N]])))
        # L1 regularization on c to promote sparsity
        reg = 0.01 * np.sum(np.abs(c))
        return max_sidelobe + reg
    
    # Gradient is hard to compute exactly due to sign(), but we can use numerical gradient
    # or use a soft-sign approximation.
    
    # Let's use a soft-sign: x_soft = tanh(x / temperature)
    # Then optimize over c, and at the end take sign.
    
    def soft_objective(c, temp=0.5):
        x = B @ c
        f_soft = np.tanh(x / temp)
        # Compute autocorrelation
        n_fft = 2 * N
        F = np.fft.fft(f_soft, n=n_fft)
        R = np.fft.ifft(F * np.conj(F)).real
        max_sidelobe = np.max(np.abs(np.concatenate([R[1:N], R[N+1:2*N]])))
        reg = 0.01 * np.sum(np.abs(c))
        return max_sidelobe + reg
    
    best_f = None
    best_score = float('inf')
    
    # Try multiple random initializations
    for seed in range(20):
        if time.time() > deadline - 10.0:
            break
            
        np.random.seed(seed)
        c0 = np.random.randn(K_total) * 0.5
        
        try:
            res = minimize(soft_objective, c0, method='L-BFGS-B', 
                          options={'maxiter': 50, 'ftol': 1e-4, 'gtol': 1e-4})
            c_opt = res.x
            x_opt = B @ c_opt
            f_candidate = np.sign(x_opt)
            f_candidate[f_candidate == 0] = 1.0
            
            # Evaluate
            n_fft = 2 * N
            F = np.fft.fft(f_candidate, n=n_fft)
            R = np.fft.ifft(F * np.conj(F)).real
            max_sidelobe = np.max(np.abs(np.concatenate([R[1:N], R[N+1:2*N]])))
            
            if max_sidelobe < best_score:
                best_score = max_sidelobe
                best_f = f_candidate.copy()
                
        except Exception:
            pass
        
        # Local search refinement
        if best_f is not None and time.time() < deadline - 5.0:
            f = best_f.copy()
            # Greedy hill climbing
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                n_fft = 2 * N
                F = np.fft.fft(f, n=n_fft)
                R = np.fft.ifft(F * np.conj(F)).real
                max_sidelobe = np.max(np.abs(np.concatenate([R[1:N], R[N+1:2*N]])))
                
                best_i = -1
                best_new_max = max_sidelobe
                
                # Try flipping each bit
                for i in range(N):
                    f_trial = f.copy()
                    f_trial[i] = -f_trial[i]
                    F_trial = np.fft.fft(f_trial, n=n_fft)
                    R_trial = np.fft.ifft(F_trial * np.conj(F_trial)).real
                    new_max = np.max(np.abs(np.concatenate([R_trial[1:N], R_trial[N+1:2*N]])))
                    
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_i = i
                
                if best_i != -1:
                    f[best_i] = -f[best_i]
                    improved = True
                else:
                    break
            
            # Check if improved
            n_fft = 2 * N
            F = np.fft.fft(f, n=n_fft)
            R = np.fft.ifft(F * np.conj(F)).real
            final_max = np.max(np.abs(np.concatenate([R[1:N], R[N+1:2*N]])))
            if final_max < best_score:
                best_score = final_max
                best_f = f.copy()
    
    if best_f is not None:
        return best_f
    else:
        # Fallback: random
        np.random.seed(0)
        return np.random.choice([-1.0, 1.0], size=N)
