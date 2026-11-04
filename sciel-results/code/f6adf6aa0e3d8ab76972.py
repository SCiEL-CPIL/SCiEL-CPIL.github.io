import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2
_N_RFFT = _FFT_SIZE // 2 + 1


def _autoconv_lin(x):
    """Compute linear autoconvolution using FFT."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    return np.fft.irfft(fx * fx, _FFT_SIZE)[:2 * N - 1]


def _metric(x):
    """max |autoconv| / (sum x)^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    return float(np.max(np.abs(c))) / (s * s)


def _reconstruct_from_mags(mags, phases):
    """Reconstruct time-domain signal from rFFT magnitudes and phases."""
    X = mags * np.exp(1j * phases)
    X[0] = mags[0]  # DC real and non-negative
    x = np.fft.irfft(X, _FFT_SIZE)[:N]
    return x


def _soft_max_abs(c, T=0.5):
    """Smooth approximation of max|c|."""
    ac = np.abs(c)
    return T * np.log(np.sum(np.exp(ac / T)))


def _objective_from_mags(mags):
    """
    Objective: soft_max|autoconv| / (sum x)^2
    Parameters: mags - array of shape (_N_RFFT,) with mags[0] fixed to sum target
    """
    # Reconstruct with random phases (fixed seed for reproducibility within DE)
    # Actually, for DE we need a deterministic objective. Use zero phases first,
    # then we'll do a phase-optimization step after DE.
    # Better: include phases in the parameterization? That's 2*_N_RFFT dims.
    # For DE, let's just optimize magnitudes with a fixed phase structure.
    
    # Use a simple approach: reconstruct with all-zero phases (real, even signal)
    # This gives us a real signal. Then we can do phase optimization later.
    phases = np.zeros(_N_RFFT)
    x = _reconstruct_from_mags(mags, phases)
    
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    
    c = _autoconv_lin(x)
    smax = _soft_max_abs(c, T=0.3)
    return smax / (s * s)


def _phase_optimize_fixed_mags(mags, n_iter=200, lr=0.05, deadline=None):
    """
    Optimize phases given fixed magnitudes to minimize max|autoconv|/(sum x)^2.
    """
    phases = np.random.randn(_N_RFFT) * 0.1
    phases[0] = 0.0
    
    best_x = _reconstruct_from_mags(mags, phases)
    best_metric = _metric(best_x)
    
    for i in range(n_iter):
        if deadline is not None and time.time() > deadline - 3:
            break
        
        x_cur = _reconstruct_from_mags(mags, phases)
        s = np.sum(x_cur)
        if abs(s) < 1e-12:
            phases += 0.01 * np.random.randn(_N_RFFT)
            phases[0] = 0.0
            continue
        
        c = _autoconv_lin(x_cur)
        metric = float(np.max(np.abs(c))) / (s * s)
        
        if metric < best_metric:
            best_metric = metric
            best_x = x_cur.copy()
        
        # Subgradient: perturb phase at argmax frequency
        k_max = np.argmax(np.abs(c))
        # Map time index to frequency: for autoconv, peak at lag k relates to freq
        # Simple heuristic: random phase perturbation weighted by |c|
        grad = np.zeros(_N_RFFT)
        # Compute gradient numerically for a few coords
        n_probe = min(15, _N_RFFT)
        for _ in range(n_probe):
            idx = np.random.randint(1, _N_RFFT)
            eps = 0.05
            phases[idx] += eps
            x_plus = _reconstruct_from_mags(mags, phases)
            s_plus = np.sum(x_plus)
            if abs(s_plus) > 1e-12:
                c_plus = _autoconv_lin(x_plus)
                m_plus = float(np.max(np.abs(c_plus))) / (s_plus * s_plus)
            else:
                m_plus = metric + 1.0
            phases[idx] -= 2 * eps
            x_minus = _reconstruct_from_mags(mags, phases)
            s_minus = np.sum(x_minus)
            if abs(s_minus) > 1e-12:
                c_minus = _autoconv_lin(x_minus)
                m_minus = float(np.max(np.abs(c_minus))) / (s_minus * s_minus)
            else:
                m_minus = metric + 1.0
            phases[idx] += eps
            grad[idx] = (m_plus - m_minus) / (2 * eps)
        
        phases -= lr * grad
        phases[0] = 0.0
    
    return best_x, best_metric


def _lbfgs_polish(x0, deadline=None):
    """L-BFGS-B polish in time domain using soft objective."""
    def obj(x):
        s = np.sum(x)
        if abs(s) < 1e-12:
            return 1e10
        c = _autoconv_lin(x)
        smax = _soft_max_abs(c, T=0.2)
        return smax / (s * s)
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', maxiter=100,
                       options={'maxfun': 200, 'ftol': 1e-12})
        if res.success or res.fun < _metric(x0):
            return res.x, _metric(res.x)
    except Exception:
        pass
    return x0, _metric(x0)


def construct():
    """
    Minimize max|autoconv| / (sum x)^2 using spectral parameterization + DE + phase opt + L-BFGS.
    """
    deadline = time.time() + 140  # Safety margin
    
    np.random.seed(42)
    
    best_x = np.ones(N)
    best_metric = _metric(best_x)
    
    # Phase 1: Differential Evolution on magnitude spectrum
    # Parameter: mags[1:] (skip DC which is determined by sum constraint)
    # We'll fix mags[0] to a target sum and optimize the rest
    
    target_sum = 1.0  # We'll normalize later
    
    def de_objective(mags_params):
        """mags_params: shape (_N_RFFT - 1,), the non-DC magnitudes."""
        if deadline is not None and time.time() > deadline - 20:
            return 1e10
        
        mags = np.zeros(_N_RFFT)
        mags[0] = target_sum  # DC component
        mags[1:] = mags_params
        
        # Reconstruct with zero phases (real signal)
        x = _reconstruct_from_mags(mags, np.zeros(_N_RFFT))
        
        s = np.sum(x)
        if abs(s) < 1e-12:
            return 1e10
        
        c = _autoconv_lin(x)
        smax = _soft_max_abs(c, T=0.3)
        return smax / (s * s)
    
    # DE bounds: magnitudes should be non-negative
    bounds = [(0, 1.0)] * (_N_RFFT - 1)
    
    try:
        de_res = differential_evolution(
            de_objective,
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False,
            updating='immediate',
            init='sobol'
        )
        
        if de_res.fun < best_metric:
            mags_opt = np.zeros(_N_RFFT)
            mags_opt[0] = target_sum
            mags_opt[1:] = de_res.x
            
            # Phase optimization on the best DE result
            x_phase, m_phase = _phase_optimize_fixed_mags(mags_opt, n_iter=100, deadline=deadline)
            if m_phase < best_metric:
                best_metric = m_phase
                best_x = x_phase
            
            # L-BFGS polish
            if time.time() < deadline - 15:
                x_lbfgs, m_lbfgs = _lbfgs_polish(best_x, deadline=deadline)
                if m_lbfgs < best_metric:
                    best_metric = m_lbfgs
                    best_x = x_lbfgs
    except Exception:
        pass
    
    # Phase 2: Try flat-spectrum initialization with phase optimization
    if time.time() < deadline - 10:
        # Flat magnitude spectrum
        mags_flat = np.ones(_N_RFFT) * 0.1
        mags_flat[0] = target_sum
        
        for trial in range(3):
            if time.time() > deadline - 5:
                break
            x_phase, m_phase = _phase_optimize_fixed_mags(mags_flat, n_iter=150, lr=0.03, deadline=deadline)
            if m_phase < best_metric:
                best_metric = m_phase
                best_x = x_phase
            
            # L-BFGS polish
            if time.time() < deadline - 5:
                x_lbfgs, m_lbfgs = _lbfgs_polish(best_x, deadline=deadline)
                if m_lbfgs < best_metric:
                    best_metric = m_lbfgs
                    best_x = x_lbfgs
    
    # Phase 3: Random spectral starts
    if time.time() < deadline - 10:
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            
            # Random magnitudes with decay
            freqs = np.linspace(0, 1, _N_RFFT)
            decay = np.random.uniform(0.5, 2.0)
            mags_rand = np.exp(-decay * freqs) * np.random.uniform(0.5, 1.5, _N_RFFT)
            mags_rand[0] = target_sum
            
            x_phase, m_phase = _phase_optimize_fixed_mags(mags_rand, n_iter=80, deadline=deadline)
            if m_phase < best_metric:
                best_metric = m_phase
                best_x = x_phase
            
            # L-BFGS polish
            if time.time() < deadline - 5:
                x_lbfgs, m_lbfgs = _lbfgs_polish(best_x, deadline=deadline)
                if m_lbfgs < best_metric:
                    best_metric = m_lbfgs
                    best_x = x_lbfgs
    
    return best_x
