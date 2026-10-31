import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    def compute_metric(h):
        """Compute normalized max of autoconvolution: max(f*f) / (sum(h))^2"""
        h = np.asarray(h, dtype=np.float64)
        total = np.sum(h)
        if total == 0:
            return float('inf')
        # Autoconvolution via FFT
        size = 2 * N
        H = np.fft.rfft(h, n=size)
        acorr = np.fft.irfft(H * np.conj(H), n=size)
        peak = np.max(acorr)
        return peak / (total * total)

    def spectral_equalize(h_init, max_iters=2000, gamma=0.5):
        """
        Structural change: Sinkhorn-like spectral equalization.
        Treats the step function as a discrete probability distribution.
        Iteratively rescales the Fourier spectrum to be as flat as possible.
        This is different from simple peak flattening; it attempts to solve
        for a 'flat-top' window design by equalizing spectral magnitude.
        """
        h = h_init.copy().astype(np.float64)
        # Normalize to sum 1 for stability during spectral ops, scale back at end if needed?
        # The metric is scale invariant: max(f*f)/(sum f)^2 is invariant to scaling f by c.
        # So we can work with sum=1.
        total = np.sum(h)
        if total > 0:
            h = h / total
            
        best_h = h.copy()
        best_m = compute_metric(h)
        
        size = 2 * N
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # FFT
            H = np.fft.rfft(h, n=size)
            
            # Magnitude spectrum
            mag = np.abs(H)
            
            # Target: Flat spectrum (constant magnitude)
            # But we must maintain the zero-frequency component (DC) consistent with sum(h)=1?
            # Actually, if we flatten the whole spectrum, the time-domain signal becomes an impulse train or similar.
            # We want to minimize the peak of the autoconvolution.
            # The autoconvolution peak is related to the integral of the power spectrum squared (by Parseval for the convolution?).
            # Let's stick to the "Sinkhorn" idea: iteratively project onto sets of constant spectral magnitude and non-negativity.
            
            # 1. Compute desired magnitude: Geometric mean or Mean?
            # Sinkhorn for matrix scaling uses row/col sums. Here, we want to equalize magnitudes.
            # A simple approach: Scale each frequency bin's magnitude towards the mean magnitude.
            mean_mag = np.mean(mag)
            # Soft equalization
            # mag_new = mag ** (1-gamma) * mean_mag ** gamma
            # This reduces contrast in the spectrum.
            mag_new = mag ** (1 - gamma) * (mean_mag ** gamma)
            
            # Preserve phase
            phase = np.angle(H)
            H_new = mag_new * np.exp(1j * phase)
            
            # Inverse FFT
            h_new = np.fft.irfft(H_new, n=size)[:N]
            
            # Project onto non-negativity
            h_new = np.maximum(h_new, 0.0)
            
            # Renormalize to sum 1 (since scaling in frequency domain changes DC)
            total_new = np.sum(h_new)
            if total_new < 1e-12:
                h_new = h.copy()
            else:
                h_new = h_new / total_new
            
            m_new = compute_metric(h_new)
            
            if m_new < best_m:
                best_m = m_new
                best_h = h_new.copy()
            
            h = h_new
            
        return best_h, best_m

    def local_refine(h_init, max_iters=20000):
        """
        Refinement machinery: Adaptive random perturbation.
        Takes the result of spectral equalization and refines it.
        """
        h = h_init.copy().astype(np.float64)
        best_h = h.copy()
        best_m = compute_metric(h)
        
        np.random.seed(42)
        
        # Adaptive step size
        step_size = 0.05
        no_improve = 0
        max_no_improve = 200
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick two random indices
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
            
            # Move mass
            # Ensure we don't make negative
            max_move = h[j] * 0.5
            delta = np.random.uniform(-max_move, max_move)
            
            new_h = h.copy()
            new_h[j] -= delta
            new_h[i] += delta
            
            if np.min(new_h) < 0:
                # Clip and renormalize? Or reject?
                # Rejecting is safer for strict non-negativity constraint during search
                # But clipping might be better for exploration.
                # Let's try clipping and renormalizing to sum 1.
                new_h = np.maximum(new_h, 0.0)
                s = np.sum(new_h)
                if s < 1e-12:
                    continue
                new_h = new_h / s
                
            m_new = compute_metric(new_h)
            
            if m_new < best_m:
                best_m = m_new
                best_h = new_h.copy()
                h = new_h
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    # Increase step size or reset?
                    step_size *= 1.1
                    if step_size > 0.2:
                        step_size = 0.2
                    no_improve = 0

        return best_h, best_m

    # Initializations
    i_arr = np.arange(N)
    mid = N / 2.0
    
    initializations = [
        np.ones(N),
        0.5 * (1.0 + np.cos(np.pi * (i_arr - mid) / (N / 2.0))),
        np.exp(-2.0 * ((i_arr - mid) / (N / 4.0)) ** 2),
    ]
    
    best_h = None
    best_m = float('inf')
    
    # Phase 1: Spectral Equalization (Structural Change)
    for init in initializations:
        if time.time() > deadline - 10:
            break
        init = np.maximum(init, 0.0)
        h_opt, m_opt = spectral_equalize(init, max_iters=1000, gamma=0.5)
        if m_opt < best_m:
            best_m = m_opt
            best_h = h_opt
    
    # Phase 2: Local Refinement (Refinement Machinery)
    if best_h is not None and time.time() < deadline - 5:
        h_ref, m_ref = local_refine(best_h, max_iters=15000)
        if m_ref < best_m:
            best_m = m_ref
            best_h = h_ref
            
    # Phase 3: Final spectral polish if time remains
    if best_h is not None and time.time() < deadline - 2:
        h_pol, m_pol = spectral_equalize(best_h, max_iters=500, gamma=0.3)
        if m_pol < best_m:
            best_m = m_pol
            best_h = h_pol

    if best_h is None:
        return [1.0] * N

    return list(best_h)
