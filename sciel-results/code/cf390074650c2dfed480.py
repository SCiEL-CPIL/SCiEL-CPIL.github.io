import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400


def compute_value(h):
    S = np.sum(h)
    if abs(S) < 1e-10:
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (S * S))


def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    idx = np.arange(N)

    best_h = np.ones(N)
    best_val = 2.0

    def try_candidate(h):
        nonlocal best_h, best_val
        if time.time() > deadline - 5:
            return
        if not np.all(np.isfinite(h)):
            return
        s = np.sum(h)
        if abs(s) < 1e-10:
            return
        v = compute_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # Phase 1: Spectral/Fourier approach
    # The autoconvolution in frequency domain: |H|^2 where H = FFT(h)
    # We want to minimize max of inverse FFT of |H|^2
    # Strategy: design H such that |H|^2 is as flat as possible (white noise spectrum)
    # but with DC component controlled
    
    # Start with white noise in frequency domain
    np.random.seed(42)
    
    # Try various spectral designs
    for design_type in ['flat', 'tapered', 'gaussian', 'power_law']:
        if time.time() > deadline - 40:
            break
        
        for alpha in np.linspace(0.1, 2.0, 10):
            if time.time() > deadline - 35:
                break
            
            H = np.zeros(N, dtype=complex)
            H[0] = 1.0  # DC component
            
            if design_type == 'flat':
                # Flat spectrum with some random phase
                for k in range(1, N // 2 + 1):
                    mag = alpha / np.sqrt(k)
                    phase = np.random.uniform(0, 2 * np.pi)
                    H[k] = mag * np.exp(1j * phase)
                    H[N - k] = np.conj(H[k])
                
            elif design_type == 'tapered':
                # Tapered spectrum
                for k in range(1, N // 2 + 1):
                    mag = alpha * np.exp(-k * 0.05)
                    phase = np.random.uniform(0, 2 * np.pi)
                    H[k] = mag * np.exp(1j * phase)
                    H[N - k] = np.conj(H[k])
                    
            elif design_type == 'gaussian':
                # Gaussian spectral shape
                for k in range(1, N // 2 + 1):
                    mag = alpha * np.exp(-k**2 / (2 * 50**2))
                    phase = np.random.uniform(0, 2 * np.pi)
                    H[k] = mag * np.exp(1j * phase)
                    H[N - k] = np.conj(H[k])
                    
            elif design_type == 'power_law':
                # Power law spectrum
                for k in range(1, N // 2 + 1):
                    mag = alpha / (k ** 0.5)
                    phase = np.random.uniform(0, 2 * np.pi)
                    H[k] = mag * np.exp(1j * phase)
                    H[N - k] = np.conj(H[k])
            
            h = np.real(np.fft.ifft(H))
            s = np.sum(h)
            if abs(s) > 1e-10:
                h = h / s  # normalize sum to 1
                try_candidate(h)
    
    # Phase 2: Direct optimization of the actual objective using subgradient/coordinate descent
    # The objective is |2*N*max(conv)/(sum(h))^2|
    # We can try to minimize max(conv) directly
    
    if time.time() < deadline - 20:
        # Use the current best as starting point
        h0 = best_h.copy()
        s0 = np.sum(h0)
        if abs(s0) > 1e-10:
            h0 = h0 / s0
        
        # Coordinate descent on individual elements
        for iteration in range(50):
            if time.time() > deadline - 15:
                break
            
            improved = False
            for i in range(N):
                if time.time() > deadline - 12:
                    break
                
                # Try perturbing element i
                for delta in [-0.5, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.5]:
                    h_test = h0.copy()
                    h_test[i] += delta
                    s_test = np.sum(h_test)
                    if abs(s_test) < 1e-10:
                        continue
                    
                    # Normalize to keep sum = 1
                    h_test = h_test / s_test
                    
                    conv_test = np.convolve(h_test, h_test)
                    max_test = np.max(conv_test)
                    
                    # Compare with current
                    conv_curr = np.convolve(h0, h0)
                    max_curr = np.max(conv_curr)
                    
                    if max_test < max_curr - 1e-10:
                        h0 = h_test
                        improved = True
                        break
            
            if not improved:
                break
        
        try_candidate(h0)
    
    # Phase 3: Random restarts with local search
    if time.time() < deadline - 10:
        for restart in range(20):
            if time.time() > deadline - 8:
                break
            
            # Random initial point
            h_rand = 1.0 + 0.3 * np.random.randn(N)
            s_rand = np.sum(h_rand)
            if abs(s_rand) < 1e-10:
                continue
            h_rand = h_rand / s_rand
            
            # Quick local search: gradient-free
            for step in range(30):
                if time.time() > deadline - 6:
                    break
                
                # Try small random perturbations
                best_local = h_rand
                best_local_val = compute_value(h_rand)
                
                for _ in range(5):
                    h_pert = h_rand + 0.05 * np.random.randn(N)
                    s_pert = np.sum(h_pert)
                    if abs(s_pert) < 1e-10:
                        continue
                    h_pert = h_pert / s_pert
                    v_pert = compute_value(h_pert)
                    if v_pert < best_local_val:
                        best_local = h_pert
                        best_local_val = v_pert
                
                if np.allclose(best_local, h_rand):
                    break
                h_rand = best_local
            
            try_candidate(h_rand)
    
    # Phase 4: Final refinement with smooth-max on the best solution
    if time.time() < deadline - 5:
        h0 = best_h.copy()
        s0 = np.sum(h0)
        if abs(s0) > 1e-10:
            h0 = h0 / s0
        
        for beta in [200.0, 500.0]:
            if time.time() > deadline - 3:
                break
            
            def obj_b(x, b=beta):
                conv = np.convolve(x, x)
                m = np.max(conv)
                sm = m + np.log(np.sum(np.exp((conv - m) * b))) / b
                s = np.sum(x)
                return sm + 1000.0 * (s - 1.0) ** 2
            
            try:
                res = minimize(obj_b, h0, method='L-BFGS-B',
                             options={'maxiter': 100, 'maxfun': 300, 'gtol': 1e-8})
                if time.time() > deadline - 2:
                    break
                h_new = res.x
                v_new = compute_value(h_new)
                if v_new < best_val:
                    best_val = v_new
                    best_h = h_new.copy()
                    h0 = h_new.copy() / np.sum(h_new)
            except Exception:
                pass
    
    return best_h
